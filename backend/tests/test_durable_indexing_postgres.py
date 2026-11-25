"""Opt-in PostgreSQL integration coverage for the durable indexing queue."""

import asyncio
import logging
import os
import re
import uuid
from pathlib import Path

import pytest
from app.job_ingestion import durable_indexing
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

TEST_DATABASE_URL = os.environ.get("DURABLE_INDEXING_TEST_DATABASE_URL")
RUN_POSTGRES_INTEGRATION = os.environ.get("RUN_POSTGRES_INTEGRATION", "").lower() in {
    "1",
    "true",
}
BACKEND = Path(__file__).parents[1]
MIGRATION = BACKEND / "migrations" / "20261121_job_indexing_work.sql"
JOB_DESCRIPTIONS_DEPENDENCY = """
CREATE TABLE job_descriptions (
    id UUID PRIMARY KEY
)
"""
logger = logging.getLogger(__name__)


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
        url.drivername.startswith("postgresql")
        and host in {"localhost", "127.0.0.1", "::1"}
        and re.fullmatch(r"durable_indexing_[a-z0-9_]*_test", database) is not None
    )


def _configuration_error(run_integration: bool, value: str | None) -> str | None:
    if not run_integration:
        return None
    if not value:
        return (
            "DURABLE_INDEXING_TEST_DATABASE_URL is required when "
            "RUN_POSTGRES_INTEGRATION=true; provide a disposable, dedicated local "
            "PostgreSQL database named durable_indexing_<name>_test"
        )
    if not _safe_test_url(value):
        return (
            "DURABLE_INDEXING_TEST_DATABASE_URL must be a PostgreSQL URL pointing "
            "to localhost, 127.0.0.1, or ::1, with a database name matching "
            "durable_indexing_<name>_test; the database must be disposable and "
            "dedicated to these tests"
        )
    return None


configuration_error = _configuration_error(RUN_POSTGRES_INTEGRATION, TEST_DATABASE_URL)
if configuration_error:
    raise RuntimeError(configuration_error)
assert TEST_DATABASE_URL is not None


async def _cleanup_resources(bootstrap, engine, schema, schema_created):
    """Best-effort cleanup that never raises over the active test exception."""
    drop_engine = engine or bootstrap
    if schema_created and drop_engine is not None:
        try:
            async with drop_engine.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        except Exception:
            logger.exception("Failed to drop durable indexing test schema")

    for name, resource in (("bootstrap", bootstrap), ("test", engine)):
        if resource is None:
            continue
        try:
            await resource.dispose()
        except Exception:
            logger.exception("Failed to dispose %s durable indexing test engine", name)


@pytest.fixture
def postgres(monkeypatch):
    if not RUN_POSTGRES_INTEGRATION:
        pytest.skip(
            "Set RUN_POSTGRES_INTEGRATION=true to run durable indexing PostgreSQL tests"
        )
    schema = "job_indexing_it_" + uuid.uuid4().hex
    bootstrap = None
    engine = None
    schema_created = False

    async def setup():
        nonlocal bootstrap, engine, schema_created
        assert TEST_DATABASE_URL is not None
        bootstrap = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
        async with bootstrap.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        schema_created = True
        await bootstrap.dispose()
        bootstrap = None
        engine = create_async_engine(
            TEST_DATABASE_URL,
            poolclass=NullPool,
            connect_args={"server_settings": {"search_path": schema}},
        )
        async with engine.begin() as conn:
            await conn.execute(text(JOB_DESCRIPTIONS_DEPENDENCY))
            for statement in MIGRATION.read_text(encoding="utf-8").split(";"):
                if statement.strip():
                    await conn.execute(text(statement))
        return engine

    try:
        engine = asyncio.run(setup())
        import server

        monkeypatch.setattr(server, "engine", engine)
        yield engine, async_sessionmaker(engine, expire_on_commit=False)
    finally:
        asyncio.run(_cleanup_resources(bootstrap, engine, schema, schema_created))


def test_configuration_skips_when_integration_is_disabled():
    assert _configuration_error(False, None) is None


def test_configuration_requires_dedicated_url_when_enabled():
    error = _configuration_error(True, None)
    assert error is not None
    assert "DURABLE_INDEXING_TEST_DATABASE_URL" in error
    assert "disposable" in error


def test_configuration_rejects_unsafe_url_when_enabled():
    error = _configuration_error(True, "postgresql+asyncpg://localhost/shared")
    assert error is not None
    assert "durable_indexing_<name>_test" in error


@pytest.mark.asyncio
async def test_cleanup_logs_failures_and_attempts_all_resources(monkeypatch, caplog):
    class FailingEngine:
        def begin(self):
            class Begin:
                async def __aenter__(self):
                    raise RuntimeError("drop failed")

                async def __aexit__(self, *_args):
                    return False

            return Begin()

        async def dispose(self):
            raise RuntimeError("dispose failed")

    class DisposableEngine:
        async def dispose(self):
            self.disposed = True

    bootstrap = FailingEngine()
    engine = DisposableEngine()
    engine.disposed = False

    with caplog.at_level(logging.ERROR):
        await _cleanup_resources(bootstrap, engine, "test_schema", True)

    assert engine.disposed is True
    assert "Failed to drop durable indexing test schema" in caplog.text
    assert "Failed to dispose bootstrap durable indexing test engine" in caplog.text


@pytest.mark.asyncio
async def test_cleanup_does_not_replace_active_exception():
    class FailingEngine:
        async def dispose(self):
            raise RuntimeError("cleanup failed")

    original = RuntimeError("setup failed")
    try:
        try:
            raise original
        finally:
            await _cleanup_resources(FailingEngine(), None, "test_schema", False)
    except RuntimeError as exc:
        assert exc is original


async def _insert_work(SessionLocal, job_id, *, status="pending", attempt_count=0):
    async with SessionLocal() as db:
        await db.execute(
            text("""
                INSERT INTO job_indexing_work
                    (target_job_id, job_id, operation, desired_revision, status,
                     attempt_count, next_retry_at)
                VALUES
                    (CAST(:job_id AS UUID), NULL, 'delete', :revision, 'pending',
                     :attempt_count, NOW())
            """),
            {
                "job_id": job_id,
                "revision": f"delete:{job_id}",
                "attempt_count": attempt_count,
            },
        )
        if status == "processing":
            await db.execute(
                text("""
                    UPDATE job_indexing_work
                    SET status='processing', claim_token=CAST(:token AS UUID),
                        claimed_at=NOW() - INTERVAL '2 minutes',
                        claim_expires_at=NOW() - INTERVAL '1 minute'
                    WHERE target_job_id=CAST(:job_id AS UUID)
                """),
                {"job_id": job_id, "token": str(uuid.uuid4())},
            )
        await db.commit()


async def _get_work(SessionLocal, job_id):
    async with SessionLocal() as db:
        return (
            await db.execute(
                text("""
                    SELECT status, attempt_count, next_retry_at, claim_token,
                           last_error
                    FROM job_indexing_work
                    WHERE target_job_id = CAST(:job_id AS UUID)
                """),
                {"job_id": job_id},
            )
        ).mappings().one()


@pytest.mark.asyncio
async def test_claim_retry_backoff_and_due_reclaim(postgres, monkeypatch):
    _engine, SessionLocal = postgres
    job_id = str(uuid.uuid4())
    await _insert_work(SessionLocal, job_id)
    monkeypatch.setattr(durable_indexing, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(durable_indexing.random, "uniform", lambda _low, _high: 1.0)
    monkeypatch.setattr(
        "app.job_ingestion.qdrant_service.delete_job_embedding",
        lambda _job_id: (_ for _ in ()).throw(RuntimeError("temporary failure")),
    )

    assert await durable_indexing.process_one_work(SessionLocal) is True
    state = await _get_work(SessionLocal, job_id)
    assert state["status"] == "failed"
    assert state["attempt_count"] == 1
    assert state["last_error"] == "temporary failure"
    assert state["claim_token"] is None

    async with SessionLocal() as db:
        retry_seconds = (
            await db.execute(
                text("""
                    SELECT EXTRACT(EPOCH FROM (next_retry_at - NOW()))
                    FROM job_indexing_work
                    WHERE target_job_id=CAST(:job_id AS UUID)
                """),
                {"job_id": job_id},
            )
        ).scalar_one()
    assert 29 <= retry_seconds <= 31

    async with SessionLocal() as db:
        await db.execute(
            text("""
                UPDATE job_indexing_work
                SET next_retry_at=NOW()
                WHERE target_job_id=CAST(:job_id AS UUID)
            """),
            {"job_id": job_id},
        )
        await db.commit()
    async with SessionLocal() as db:
        reclaimed = await durable_indexing._claim(db)
    assert reclaimed is not None
    assert str(reclaimed["target_job_id"]) == job_id
    assert (await _get_work(SessionLocal, job_id))["status"] == "processing"


@pytest.mark.asyncio
async def test_competing_claims_cannot_claim_locked_work(postgres):
    _engine, SessionLocal = postgres
    job_id = str(uuid.uuid4())
    await _insert_work(SessionLocal, job_id)
    lock_acquired = asyncio.Event()
    release_lock = asyncio.Event()

    async def hold_row_lock():
        async with SessionLocal() as db:
            async with db.begin():
                await db.execute(
                    text("""
                        SELECT target_job_id
                        FROM job_indexing_work
                        WHERE target_job_id=CAST(:job_id AS UUID)
                        FOR UPDATE
                    """),
                    {"job_id": job_id},
                )
                lock_acquired.set()
                await release_lock.wait()

    locker = asyncio.create_task(hold_row_lock())
    await lock_acquired.wait()
    try:
        async with SessionLocal() as contender_db:
            contender = await durable_indexing._claim(contender_db)
        assert contender is None
    finally:
        release_lock.set()
        await locker

    async with SessionLocal() as db:
        winner = await durable_indexing._claim(db)
    assert winner is not None
    assert str(winner["target_job_id"]) == job_id
    assert (await _get_work(SessionLocal, job_id))["status"] == "processing"


@pytest.mark.asyncio
async def test_qdrant_failure_then_success_finalizes_work(postgres, monkeypatch):
    _engine, SessionLocal = postgres
    job_id = str(uuid.uuid4())
    await _insert_work(SessionLocal, job_id)
    calls = []

    def delete_embedding(_job_id):
        calls.append(_job_id)
        if len(calls) == 1:
            raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(
        "app.job_ingestion.qdrant_service.delete_job_embedding", delete_embedding
    )
    monkeypatch.setattr(durable_indexing, "LOCK_TIMEOUT_SECONDS", 1)

    assert await durable_indexing.process_one_work(SessionLocal) is True
    failed = await _get_work(SessionLocal, job_id)
    assert failed["status"] == "failed"
    assert failed["last_error"] == "qdrant unavailable"
    assert failed["claim_token"] is None

    async with SessionLocal() as db:
        await db.execute(
            text("""
                UPDATE job_indexing_work SET next_retry_at=NOW()
                WHERE target_job_id=CAST(:job_id AS UUID)
            """),
            {"job_id": job_id},
        )
        await db.commit()
    assert await durable_indexing.process_one_work(SessionLocal) is True
    succeeded = await _get_work(SessionLocal, job_id)
    assert succeeded["status"] == "succeeded"
    assert succeeded["claim_token"] is None
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_expired_claim_is_reclaimed_once(postgres):
    _engine, SessionLocal = postgres
    job_id = str(uuid.uuid4())
    await _insert_work(SessionLocal, job_id, status="processing")
    async with SessionLocal() as db:
        await db.execute(
            text("""
                UPDATE job_indexing_work
                SET claim_token=CAST(:token AS UUID), claimed_at=NOW() - INTERVAL '2 minutes',
                    claim_expires_at=NOW() - INTERVAL '1 minute'
                WHERE target_job_id=CAST(:job_id AS UUID)
            """),
            {"job_id": job_id, "token": str(uuid.uuid4())},
        )
        await db.commit()

    async with SessionLocal() as db:
        first = await durable_indexing._claim(db)
    assert first is not None
    async with SessionLocal() as db:
        assert (await durable_indexing._claim(db)) is None
    state = await _get_work(SessionLocal, job_id)
    assert state["status"] == "processing"
    assert str(state["claim_token"]) == first["claim_token"]


@pytest.mark.asyncio
async def test_dead_letter_replay_only_reopens_requested_work(postgres, monkeypatch):
    _engine, SessionLocal = postgres
    dead_job = str(uuid.uuid4())
    unrelated_job = str(uuid.uuid4())
    await _insert_work(SessionLocal, dead_job)
    await _insert_work(SessionLocal, unrelated_job)
    async with SessionLocal() as db:
        await db.execute(
            text("""
                UPDATE job_indexing_work
                SET next_retry_at=NOW() + INTERVAL '1 hour'
                WHERE target_job_id=CAST(:job_id AS UUID)
            """),
            {"job_id": unrelated_job},
        )
        await db.commit()
    monkeypatch.setattr(durable_indexing, "MAX_ATTEMPTS", 1)
    monkeypatch.setattr(
        "app.job_ingestion.qdrant_service.delete_job_embedding",
        lambda _job_id: (_ for _ in ()).throw(RuntimeError("permanent failure")),
    )

    assert await durable_indexing.process_one_work(SessionLocal) is True
    dead_state = await _get_work(SessionLocal, dead_job)
    other_state = await _get_work(SessionLocal, unrelated_job)
    assert dead_state["status"] == "dead_letter"
    assert other_state["status"] == "pending"

    assert await durable_indexing.replay_dead_letter(dead_job, SessionLocal) is True
    replayed = await _get_work(SessionLocal, dead_job)
    assert replayed["status"] == "pending"
    assert replayed["attempt_count"] == 0
    assert (await _get_work(SessionLocal, unrelated_job))["status"] == other_state[
        "status"
    ]
