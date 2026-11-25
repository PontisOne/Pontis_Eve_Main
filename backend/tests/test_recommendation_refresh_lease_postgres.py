"""Opt-in PostgreSQL integration tests for recommendation refresh leases.

These tests require an explicitly configured local test database.  They never
fall back to DATABASE_URL and isolate every test in a disposable schema.
"""

import asyncio
import os
import re
import uuid
from pathlib import Path

import pytest
from candidate_job_matching_service import _refresh_persistence_is_current
from recommendation_refresh_lease import (
    acquire_refresh_lease,
    release_refresh_lease,
    renew_refresh_lease,
)
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
RUN_POSTGRES_INTEGRATION = os.environ.get("RUN_POSTGRES_INTEGRATION") == "1"


def _safe_test_url(value: str | None) -> bool:
    if not value:
        return False
    try:
        url = make_url(value)
    except Exception:
        return False
    database = (url.database or "").lower()
    host = (url.host or "").lower()
    return (
        RUN_POSTGRES_INTEGRATION
        and url.drivername.startswith("postgresql")
        and host in {"localhost", "127.0.0.1", "::1"}
        and re.fullmatch(r"pontis_[a-z0-9_]*test[a-z0-9_]*", database) is not None
        and not any(
            token in database for token in ("prod", "production", "live", "railway")
        )
    )


if not _safe_test_url(TEST_DATABASE_URL):
    pytest.skip(
        "Recommendation lease PostgreSQL tests require RUN_POSTGRES_INTEGRATION=1 "
        "and an explicit local TEST_DATABASE_URL using a pontis_*test* database",
        allow_module_level=True,
    )
assert TEST_DATABASE_URL is not None


BACKEND = Path(__file__).parents[1]
LEASE_MIGRATION = (
    BACKEND / "migrations" / "20261120_candidate_recommendation_refresh_leases.sql"
)
async def _apply_sql_file(conn, path: Path) -> None:
    for statement in path.read_text(encoding="utf-8").split(";"):
        if statement.strip():
            await conn.execute(text(statement))


@pytest.fixture
def postgres():
    """Create one isolated schema and use independent connections per worker."""

    schema = "recommendation_lease_it_" + uuid.uuid4().hex
    engine = None
    schema_created = False

    async def setup():
        nonlocal engine, schema_created
        assert TEST_DATABASE_URL is not None
        bootstrap = None
        setup_exception = None
        try:
            bootstrap = create_async_engine(
                TEST_DATABASE_URL, pool_pre_ping=True, poolclass=NullPool
            )
            async with bootstrap.begin() as conn:
                await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            schema_created = True
        except BaseException as exc:
            setup_exception = exc
            raise
        finally:
            if bootstrap is not None:
                try:
                    await bootstrap.dispose()
                except BaseException:
                    if setup_exception is None:
                        raise

        engine = create_async_engine(
            TEST_DATABASE_URL,
            pool_pre_ping=True,
            poolclass=NullPool,
            connect_args={"server_settings": {"search_path": schema}},
        )
        async with engine.begin() as conn:
            await conn.execute(
                text("""
                CREATE TABLE candidates (
                    id UUID PRIMARY KEY,
                    last_full_recommendation_refresh_at TIMESTAMPTZ NULL,
                    matching_profile_revision BIGINT NOT NULL DEFAULT 0,
                    last_full_recommendation_revision BIGINT NULL
                )
                """)
            )
            await conn.execute(text("""
                CREATE TABLE candidate_job_recommendations (
                    id UUID PRIMARY KEY,
                    candidate_id UUID NOT NULL REFERENCES candidates(id),
                    job_id UUID NOT NULL,
                    match_score DOUBLE PRECISION,
                    recommendation_rank INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    generated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """))
            await _apply_sql_file(conn, LEASE_MIGRATION)
            await conn.execute(
                text("INSERT INTO candidates (id) VALUES (:candidate_id)"),
                {"candidate_id": str(uuid.uuid4())},
            )
        return engine

    async def teardown():
        cleanup_errors = []
        if engine is not None:
            try:
                await engine.dispose()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if schema_created:
            cleanup = None
            try:
                assert TEST_DATABASE_URL is not None
                cleanup = create_async_engine(
                    TEST_DATABASE_URL, pool_pre_ping=True, poolclass=NullPool
                )
                async with cleanup.begin() as conn:
                    await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            except BaseException as exc:
                cleanup_errors.append(exc)
            finally:
                if cleanup is not None:
                    try:
                        await cleanup.dispose()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
        if cleanup_errors:
            raise cleanup_errors[0]

    setup_exception = None
    try:
        engine = asyncio.run(setup())
        yield engine
    except BaseException as exc:
        setup_exception = exc
        raise
    finally:
        try:
            asyncio.run(teardown())
        except BaseException:
            if setup_exception is None:
                raise


def _run(coro):
    return asyncio.run(coro)


async def _candidate(engine) -> str:
    async with engine.connect() as conn:
        return str((await conn.execute(text("SELECT id FROM candidates"))).scalar())


async def _acquire(engine, candidate_id: str, owner: str, duration: int = 60):
    async with engine.begin() as conn:
        return await acquire_refresh_lease(conn, candidate_id, owner, duration)


def test_first_acquisition_and_active_lease_rejection(postgres):
    async def scenario():
        candidate_id = await _candidate(postgres)
        first = await _acquire(postgres, candidate_id, str(uuid.uuid4()))
        second = await _acquire(postgres, candidate_id, str(uuid.uuid4()))
        return first, second

    first, second = _run(scenario())
    assert first.acquired is True
    assert first.generation == 1
    assert second.acquired is False


def test_renew_release_and_stale_owner_rejection(postgres):
    async def scenario():
        candidate_id = await _candidate(postgres)
        owner = str(uuid.uuid4())
        lease = await _acquire(postgres, candidate_id, owner)
        wrong_owner = str(uuid.uuid4())
        async with postgres.begin() as conn:
            renewed = await renew_refresh_lease(
                conn, candidate_id, owner, lease.generation, 60
            )
        async with postgres.begin() as conn:
            wrong_renewal = await renew_refresh_lease(
                conn, candidate_id, wrong_owner, lease.generation, 60
            )
        async with postgres.begin() as conn:
            wrong_release = await release_refresh_lease(
                conn, candidate_id, wrong_owner, lease.generation
            )
        async with postgres.begin() as conn:
            released = await release_refresh_lease(
                conn, candidate_id, owner, lease.generation
            )
        return renewed, wrong_renewal, wrong_release, released

    renewed, wrong_renewal, wrong_release, released = _run(scenario())
    assert renewed is True
    assert wrong_renewal is False
    assert wrong_release is False
    assert released is True


def test_expiration_reacquisition_increments_generation(postgres):
    async def scenario():
        candidate_id = await _candidate(postgres)
        first_owner = str(uuid.uuid4())
        first = await _acquire(postgres, candidate_id, first_owner, duration=60)
        async with postgres.begin() as conn:
            await conn.execute(
                text("UPDATE candidate_recommendation_refresh_leases "
                     "SET lease_expires_at = now() - interval '1 second' "
                     "WHERE candidate_id = :cid"),
                {"cid": candidate_id},
            )
        second_owner = str(uuid.uuid4())
        second = await _acquire(postgres, candidate_id, second_owner)
        async with postgres.begin() as conn:
            old_renewal = await renew_refresh_lease(
                conn, candidate_id, first_owner, first.generation, 60
            )
        async with postgres.begin() as conn:
            old_release = await release_refresh_lease(
                conn, candidate_id, first_owner, first.generation
            )
        return first, second, old_renewal, old_release

    first, second, old_renewal, old_release = _run(scenario())
    assert first.acquired is True
    assert second.acquired is True
    assert second.generation == first.generation + 1
    assert old_renewal is False
    assert old_release is False


def test_concurrent_workers_exactly_one_acquires(postgres):
    async def worker(candidate_id: str, owner: str, ready, start):
        async with postgres.connect() as conn:
            await ready()
            await start.wait()
            async with conn.begin():
                return await acquire_refresh_lease(conn, candidate_id, owner, 60)

    async def scenario():
        candidate_id = await _candidate(postgres)
        connected = 0
        connected_lock = asyncio.Lock()
        start = asyncio.Event()

        async def ready():
            nonlocal connected
            async with connected_lock:
                connected += 1
                if connected == 2:
                    start.set()

        return await asyncio.gather(
            worker(candidate_id, str(uuid.uuid4()), ready, start),
            worker(candidate_id, str(uuid.uuid4()), ready, start),
        )

    results = _run(scenario())
    assert sum(result.acquired for result in results) == 1
    assert sum(not result.acquired for result in results) == 1


def test_independent_candidates_acquire_concurrently(postgres):
    async def add_candidate():
        candidate_id = str(uuid.uuid4())
        async with postgres.begin() as conn:
            await conn.execute(
                text("INSERT INTO candidates (id) VALUES (:candidate_id)"),
                {"candidate_id": candidate_id},
            )
        return candidate_id

    async def worker(candidate_id: str, ready, start):
        async with postgres.connect() as conn:
            await ready()
            await start.wait()
            async with conn.begin():
                return await acquire_refresh_lease(conn, candidate_id, str(uuid.uuid4()), 60)

    async def scenario():
        first = await _candidate(postgres)
        second = await add_candidate()
        connected = 0
        connected_lock = asyncio.Lock()
        start = asyncio.Event()

        async def ready():
            nonlocal connected
            async with connected_lock:
                connected += 1
                if connected == 2:
                    start.set()

        return await asyncio.gather(
            worker(first, ready, start), worker(second, ready, start)
        )

    results = _run(scenario())
    assert all(result.acquired for result in results)
    assert {result.generation for result in results} == {1}


def test_stale_generation_cannot_pass_production_persistence_fence(postgres):
    async def scenario():
        candidate_id = await _candidate(postgres)
        old_owner = str(uuid.uuid4())
        old = await _acquire(postgres, candidate_id, old_owner, duration=1)
        async with postgres.begin() as conn:
            await conn.execute(
                text("UPDATE candidate_recommendation_refresh_leases "
                     "SET lease_expires_at = now() - interval '1 second' "
                     "WHERE candidate_id = :cid"),
                {"cid": candidate_id},
            )
        new_owner = str(uuid.uuid4())
        new = await _acquire(postgres, candidate_id, new_owner)
        async with postgres.begin() as conn:
            old_allowed = await _refresh_persistence_is_current(
                conn, candidate_id, old_owner, old.generation, 0
            )
            new_allowed = await _refresh_persistence_is_current(
                conn, candidate_id, new_owner, new.generation, 0
            )
        return old_allowed, new_allowed

    old_allowed, new_allowed = _run(scenario())
    assert old_allowed is False
    assert new_allowed is True


def test_profile_revision_change_blocks_persistence_fence(postgres):
    async def scenario():
        candidate_id = await _candidate(postgres)
        owner = str(uuid.uuid4())
        lease = await _acquire(postgres, candidate_id, owner)
        async with postgres.begin() as conn:
            await conn.execute(
                text("UPDATE candidates SET matching_profile_revision = 1 "
                     "WHERE id = :cid"),
                {"cid": candidate_id},
            )
            return await _refresh_persistence_is_current(
                conn, candidate_id, owner, lease.generation, 0
            )

    assert _run(scenario()) is False


def test_postgresql_transaction_rollback_semantics_only(postgres):
    """Generic PostgreSQL rollback semantics; not production persistence coverage."""
    async def scenario():
        candidate_id = await _candidate(postgres)
        recommendation_id = str(uuid.uuid4())

        try:
            async with postgres.begin() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO candidate_job_recommendations
                            (id, candidate_id, job_id, match_score, recommendation_rank)
                        VALUES (:rid, :cid, :jid, :score, :rank)
                    """),
                    {
                        "rid": recommendation_id,
                        "cid": candidate_id,
                        "jid": str(uuid.uuid4()),
                        "score": 0.9,
                        "rank": 1,
                    },
                )
                await conn.execute(
                    text("""
                        UPDATE candidates
                        SET last_full_recommendation_refresh_at = now(),
                            last_full_recommendation_revision = 0
                        WHERE id = :cid
                    """),
                    {"cid": candidate_id},
                )
                raise RuntimeError("force rollback")
        except RuntimeError:
            pass

        async with postgres.connect() as conn:
            row = (
                await conn.execute(
                    text("""
                        SELECT
                            (SELECT COUNT(*) FROM candidate_job_recommendations
                             WHERE id = :rid) AS recommendation_count,
                            last_full_recommendation_refresh_at,
                            last_full_recommendation_revision
                        FROM candidates WHERE id = :cid
                    """),
                    {"rid": recommendation_id, "cid": candidate_id},
                )
            ).mappings().one()
        return row

    row = _run(scenario())
    assert row["recommendation_count"] == 0
    assert row["last_full_recommendation_refresh_at"] is None
    assert row["last_full_recommendation_revision"] is None


def test_postgresql_transaction_commit_persists_recommendation_and_freshness(postgres):
    """Generic PostgreSQL commit semantics; not production persistence coverage."""
    async def scenario():
        candidate_id = await _candidate(postgres)
        recommendation_id = str(uuid.uuid4())
        async with postgres.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO candidate_job_recommendations
                        (id, candidate_id, job_id, match_score, recommendation_rank)
                    VALUES (:rid, :cid, :jid, :score, :rank)
                """),
                {
                    "rid": recommendation_id,
                    "cid": candidate_id,
                    "jid": str(uuid.uuid4()),
                    "score": 0.9,
                    "rank": 1,
                },
            )
            await conn.execute(
                text("""
                    UPDATE candidates
                    SET last_full_recommendation_refresh_at = now(),
                        last_full_recommendation_revision = 0
                    WHERE id = :cid
                """),
                {"cid": candidate_id},
            )

        async with postgres.connect() as conn:
            return (
                await conn.execute(
                    text("""
                        SELECT
                            (SELECT COUNT(*) FROM candidate_job_recommendations
                             WHERE id = :rid) AS recommendation_count,
                            last_full_recommendation_refresh_at,
                            last_full_recommendation_revision
                        FROM candidates WHERE id = :cid
                    """),
                    {"rid": recommendation_id, "cid": candidate_id},
                )
            ).mappings().one()

    row = _run(scenario())
    assert row["recommendation_count"] == 1
    assert row["last_full_recommendation_refresh_at"] is not None
    assert row["last_full_recommendation_revision"] == 0
