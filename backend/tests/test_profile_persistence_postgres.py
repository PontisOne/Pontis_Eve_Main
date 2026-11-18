"""Opt-in PostgreSQL concurrency coverage for profile persistence.

This module never falls back to DATABASE_URL or .env.example.  It requires an
explicit local PostgreSQL TEST_DATABASE_URL whose database name contains
``test``.  Each test uses a disposable schema and never touches shared data.
"""

import asyncio
import json
import os
import re
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import server

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
        "Profile persistence PostgreSQL tests require RUN_POSTGRES_INTEGRATION=1 "
        "and a local TEST_DATABASE_URL using a pontis_*test* database",
        allow_module_level=True,
    )


@pytest.fixture
def postgres(monkeypatch):
    schema = "profile_persistence_it_" + uuid.uuid4().hex
    engine = None

    async def setup():
        bootstrap = create_async_engine(
            TEST_DATABASE_URL, pool_pre_ping=True, poolclass=NullPool
        )
        async with bootstrap.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        await bootstrap.dispose()

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
                    id TEXT PRIMARY KEY,
                    skills JSONB NOT NULL DEFAULT '[]'::jsonb,
                    education JSONB NOT NULL DEFAULT '[]'::jsonb,
                    work_experience JSONB NOT NULL DEFAULT '[]'::jsonb,
                    raw_data JSONB NOT NULL DEFAULT '{}'::jsonb,
                    parsed_resume_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    opportunity_type TEXT,
                    updated_at TIMESTAMPTZ,
                    updated_by_source TEXT
                )
            """)
            )
            await conn.execute(
                text("""
                CREATE TABLE candidate_preferences (
                    id TEXT PRIMARY KEY,
                    candidate_id TEXT UNIQUE NOT NULL,
                    preferred_roles JSONB NOT NULL DEFAULT '[]'::jsonb,
                    preferred_locations JSONB NOT NULL DEFAULT '[]'::jsonb,
                    preferred_industries JSONB NOT NULL DEFAULT '[]'::jsonb,
                    employment_types JSONB NOT NULL DEFAULT '[]'::jsonb,
                    remote_preference TEXT,
                    expected_salary TEXT,
                    willing_to_relocate BOOLEAN,
                    notice_period TEXT,
                    created_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ
                )
            """)
            )
        return engine

    try:
        engine = asyncio.run(setup())
        monkeypatch.setattr(
            server, "SessionLocal", async_sessionmaker(engine, expire_on_commit=False)
        )

        async def verify_schema():
            async with engine.connect() as conn:
                return (await conn.execute(text("SELECT current_schema()"))).scalar()

        assert asyncio.run(verify_schema()) == schema
        yield engine
    finally:
        if engine is not None:

            async def teardown():
                await engine.dispose()
                cleanup = create_async_engine(
                    TEST_DATABASE_URL, pool_pre_ping=True, poolclass=NullPool
                )
                try:
                    async with cleanup.begin() as conn:
                        await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
                finally:
                    await cleanup.dispose()

            asyncio.run(teardown())


def _run(coro):
    return asyncio.run(coro)


async def _seed(engine, candidate_id: str, *, skills=None, raw_data=None):
    async with engine.begin() as conn:
        await conn.execute(
            text("""
                INSERT INTO candidates (id, skills, raw_data, updated_at)
                VALUES (:id, CAST(:skills AS jsonb), CAST(:raw_data AS jsonb), now())
            """),
            {
                "id": candidate_id,
                "skills": json.dumps(skills or []),
                "raw_data": json.dumps(raw_data or {}),
            },
        )


def test_different_concurrent_skills_are_preserved(postgres):
    candidate_id = "candidate-skills"
    _run(_seed(postgres, candidate_id))

    async def update(skill):
        return await server._apply_profile_updates(candidate_id, {"skills": [skill]})

    async def run_both():
        return await asyncio.gather(update("Python"), update("FastAPI"))

    _run(run_both())

    async def read():
        async with postgres.connect() as conn:
            return (
                await conn.execute(
                    text("SELECT skills FROM candidates WHERE id = :id"),
                    {"id": candidate_id},
                )
            ).scalar()

    values = _run(read())
    assert {item for item in values} == {"Python", "FastAPI"}


def test_same_concurrent_skill_is_not_duplicated(postgres):
    candidate_id = "candidate-duplicate-skill"
    _run(_seed(postgres, candidate_id))

    async def update():
        return await server._apply_profile_updates(candidate_id, {"skills": ["Python"]})

    async def run_both():
        return await asyncio.gather(update(), update())

    _run(run_both())

    async def read():
        async with postgres.connect() as conn:
            return (
                await conn.execute(
                    text("SELECT skills FROM candidates WHERE id = :id"),
                    {"id": candidate_id},
                )
            ).scalar()

    assert _run(read()) == ["Python"]


def test_same_concurrent_certification_is_not_duplicated(postgres):
    candidate_id = "candidate-duplicate-cert"
    _run(_seed(postgres, candidate_id, raw_data={"certifications": []}))

    async def update():
        return await server._apply_profile_updates(
            candidate_id, {"certifications": ["AWS Certified Developer"]}
        )

    async def run_both():
        return await asyncio.gather(update(), update())

    _run(run_both())

    async def read():
        async with postgres.connect() as conn:
            return (
                await conn.execute(
                    text("SELECT raw_data FROM candidates WHERE id = :id"),
                    {"id": candidate_id},
                )
            ).scalar()

    assert _run(read())["certifications"] == ["AWS Certified Developer"]


def test_different_concurrent_certifications_are_preserved(postgres):
    candidate_id = "candidate-different-certs"
    _run(_seed(postgres, candidate_id, raw_data={"certifications": []}))

    async def update(certification):
        return await server._apply_profile_updates(
            candidate_id, {"certifications": [certification]}
        )

    async def run_both():
        return await asyncio.gather(update("AWS Certified Developer"), update("PMP"))

    _run(run_both())

    async def read():
        async with postgres.connect() as conn:
            return (
                await conn.execute(
                    text("SELECT raw_data FROM candidates WHERE id = :id"),
                    {"id": candidate_id},
                )
            ).scalar()

    assert set(_run(read())["certifications"]) == {"AWS Certified Developer", "PMP"}


def test_different_candidates_progress_independently(postgres):
    _run(_seed(postgres, "candidate-a"))
    _run(_seed(postgres, "candidate-b"))

    async def update(candidate_id, skill):
        return await server._apply_profile_updates(candidate_id, {"skills": [skill]})

    async def run_both():
        return await asyncio.wait_for(
            asyncio.gather(
                update("candidate-a", "Python"), update("candidate-b", "Go")
            ),
            timeout=2,
        )

    _run(run_both())

    async def read():
        async with postgres.connect() as conn:
            return (
                await conn.execute(
                    text("SELECT id, skills FROM candidates ORDER BY id")
                )
            ).all()

    assert _run(read()) == [("candidate-a", ["Python"]), ("candidate-b", ["Go"])]


def test_preference_failure_rolls_back_candidate_and_preference_changes(
    postgres, monkeypatch
):
    candidate_id = "candidate-rollback"
    _run(_seed(postgres, candidate_id, raw_data={"preferred_roles": ["Backend"]}))

    async def seed_preference():
        async with postgres.begin() as conn:
            await conn.execute(
                text("""
                INSERT INTO candidate_preferences
                    (id, candidate_id, preferred_roles, created_at, updated_at)
                VALUES (:id, :cid, '["Backend"]'::jsonb, now(), now())
            """),
                {"id": str(uuid.uuid4()), "cid": candidate_id},
            )

    _run(seed_preference())
    seen_sessions = []

    async def fail_preferences(candidate, payload, *, db=None):
        seen_sessions.append(db)
        raise RuntimeError("forced preference failure")

    monkeypatch.setattr(server, "_upsert_candidate_preferences", fail_preferences)

    with pytest.raises(RuntimeError, match="forced preference failure"):
        _run(
            server._apply_profile_updates(
                candidate_id,
                {
                    "skills": ["Python"],
                    "preferred_roles": ["Platform Engineer"],
                },
            )
        )

    async def read():
        async with postgres.connect() as conn:
            candidate = (
                await conn.execute(
                    text("SELECT skills FROM candidates WHERE id = :id"),
                    {"id": candidate_id},
                )
            ).scalar()
            preference = (
                await conn.execute(
                    text(
                        "SELECT preferred_roles FROM candidate_preferences "
                        "WHERE candidate_id = :id"
                    ),
                    {"id": candidate_id},
                )
            ).scalar()
            return candidate, preference

    assert _run(read()) == ([], ["Backend"])
    assert len(seen_sessions) == 1
    assert seen_sessions[0] is not None


def test_cache_invalidation_happens_after_commit_only(postgres, monkeypatch):
    candidate_id = "candidate-cache-timing"
    _run(_seed(postgres, candidate_id))
    invalidations = []

    async def invalidate(candidate):
        invalidations.append(candidate)

    monkeypatch.setattr(server.profile_cache, "invalidate", invalidate)
    _run(server._apply_profile_updates(candidate_id, {"skills": ["Python"]}))
    assert invalidations == [candidate_id]

    async def fail_preferences(candidate, payload, *, db=None):
        raise RuntimeError("forced rollback")

    monkeypatch.setattr(server, "_upsert_candidate_preferences", fail_preferences)
    with pytest.raises(RuntimeError, match="forced rollback"):
        _run(
            server._apply_profile_updates(
                candidate_id,
                {"skills": ["FastAPI"], "preferred_roles": ["Platform Engineer"]},
            )
        )
    assert invalidations == [candidate_id]


def test_second_update_waits_for_locked_row_and_reads_latest_state(
    postgres, monkeypatch
):
    candidate_id = "candidate-lock-order"
    _run(_seed(postgres, candidate_id, skills=["Python"]))

    async def scenario():
        started = asyncio.Event()
        original = server._get_candidate_row_for_update

        async def observed_locked_read(db, candidate):
            started.set()
            return await original(db, candidate)

        monkeypatch.setattr(
            server, "_get_candidate_row_for_update", observed_locked_read
        )
        first = server.SessionLocal()
        second_done = asyncio.Event()
        try:
            await first.begin()
            await first.execute(
                text("SELECT id FROM candidates WHERE id = :id FOR UPDATE"),
                {"id": candidate_id},
            )

            async def second_update():
                try:
                    await server._apply_profile_updates(
                        candidate_id, {"skills": ["FastAPI"]}
                    )
                finally:
                    second_done.set()

            task = asyncio.create_task(second_update())
            await asyncio.wait_for(started.wait(), timeout=2)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(second_done.wait(), timeout=0.2)

            await first.commit()
            await asyncio.wait_for(task, timeout=2)
        finally:
            if not first.is_active:
                await first.close()
            else:
                await first.rollback()
                await first.close()

    _run(scenario())

    async def read():
        async with postgres.connect() as conn:
            return (
                await conn.execute(
                    text("SELECT skills FROM candidates WHERE id = :id"),
                    {"id": candidate_id},
                )
            ).scalar()

    assert set(_run(read())) == {"Python", "FastAPI"}
