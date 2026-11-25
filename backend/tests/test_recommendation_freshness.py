import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import server


def test_recent_recommendation_timestamp_is_not_stale():
    now = datetime.now(UTC)

    assert not server._recommendation_timestamp_is_stale(
        now - timedelta(hours=23, minutes=59), now=now
    )


def test_old_or_missing_recommendation_timestamp_is_stale():
    now = datetime.now(UTC)

    assert server._recommendation_timestamp_is_stale(
        now - timedelta(hours=24, seconds=1), now=now
    )
    assert server._recommendation_timestamp_is_stale(None, now=now)


def test_naive_recommendation_timestamp_is_interpreted_as_utc():
    now = datetime.now(UTC)
    naive_recent = (now - timedelta(hours=1)).replace(tzinfo=None)

    assert not server._recommendation_timestamp_is_stale(naive_recent, now=now)


def test_revision_migration_is_idempotent_and_registered_once():
    migration = Path(__file__).parents[1] / "migrations" / (
        "20261118_candidate_recommendation_freshness.sql"
    )
    sql = migration.read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS matching_profile_revision BIGINT NOT NULL DEFAULT 0" in sql
    assert "ADD COLUMN IF NOT EXISTS last_full_recommendation_revision BIGINT NULL" in sql
    assert "last_full_recommendation_refresh_at TIMESTAMPTZ NULL" in sql

    railway = json.loads(
        (Path(__file__).parents[1] / "railway.json").read_text(encoding="utf-8")
    )
    commands = railway["deploy"]["preDeployCommand"]
    target = "psql \"$DATABASE_URL\" -v ON_ERROR_STOP=1 -f migrations/20261118_candidate_recommendation_freshness.sql"
    assert commands.count(target) == 1


@pytest.mark.asyncio
async def test_revision_bump_atomically_invalidates_recommendation_freshness():
    calls = []

    class Database:
        async def execute(self, statement, params):
            calls.append((str(statement), params))

    await server._increment_matching_profile_revision(Database(), "candidate-1")

    assert len(calls) == 1
    statement, params = calls[0]
    assert params == {"cid": "candidate-1"}
    assert "matching_profile_revision = COALESCE(matching_profile_revision, 0) + 1" in statement
    assert "last_full_recommendation_refresh_at = NULL" in statement
    assert "last_full_recommendation_revision = NULL" in statement


class _VoicePersistenceFake:
    def __init__(self, candidate, preferences=None):
        self.candidate = dict(candidate)
        self.preferences = dict(preferences or {})
        self.revision = 0
        self.refresh_at: datetime | None = datetime.now(UTC)
        self.refresh_revision: int | None = 7
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, statement, params=None):
        sql = str(statement).lstrip()
        params = params or {}
        if sql.startswith("SELECT id, preferred_roles"):
            row = dict(self.preferences) if self.preferences else None

            class Result:
                def mappings(self):
                    return self

                def fetchone(inner):
                    return row

            return Result()
        if sql.startswith("UPDATE candidate_preferences"):
            for field in (
                "preferred_roles",
                "preferred_locations",
                "preferred_industries",
                "employment_types",
            ):
                if field in params:
                    self.preferences[field] = json.loads(params[field])
            for field in (
                "remote_preference",
                "expected_salary",
                "willing_to_relocate",
                "notice_period",
            ):
                if field in params:
                    self.preferences[field] = params[field]
            return type("Result", (), {"rowcount": 1})()
        if sql.startswith("INSERT INTO candidate_preferences"):
            self.preferences = {
                field: json.loads(params[field])
                for field in (
                    "preferred_roles",
                    "preferred_locations",
                    "preferred_industries",
                    "employment_types",
                )
            }
            self.preferences.update(
                {
                    field: params[field]
                    for field in (
                        "remote_preference",
                        "expected_salary",
                        "willing_to_relocate",
                        "notice_period",
                    )
                }
            )
            return type("Result", (), {"rowcount": 1})()
        if "matching_profile_revision = COALESCE" in sql:
            self.revision += 1
            self.refresh_at = None
            self.refresh_revision = None
            return type("Result", (), {"rowcount": 1})()
        if sql.startswith("UPDATE candidates SET"):
            for field in (
                "current_role",
                "location",
                "experience_years",
            ):
                if field in params:
                    self.candidate[field] = params[field]
            for field in ("skills", "work_experience", "education", "raw_data"):
                if field in params:
                    self.candidate[field] = json.loads(params[field])
            return type("Result", (), {"rowcount": 1})()
        raise AssertionError(f"Unexpected SQL in fake: {sql[:100]}")

    async def commit(self):
        self.commits += 1


@pytest.fixture
def voice_persistence(monkeypatch):
    fake = _VoicePersistenceFake(
        {
            "current_role": "Analyst",
            "location": "London",
            "experience_years": 3,
            "skills": ["Python"],
            "work_experience": [],
            "education": [],
            "raw_data": {},
        }
    )
    async def invalidate(*_):
        return None

    monkeypatch.setattr(server, "SessionLocal", lambda: fake)
    monkeypatch.setattr(
        server,
        "profile_cache",
        type("Cache", (), {"invalidate": staticmethod(invalidate)})(),
    )
    return fake


@pytest.mark.asyncio
async def test_repeated_voice_persistence_bumps_once_and_invalidates_only_once(
    voice_persistence,
):
    state: dict[str, Any] = {"completed_turns": []}
    voice = {
        "current_role": "Engineer",
        "preferred_locations": ["Berlin"],
    }
    await server._persist_voice_intake_profile_state(
        "candidate-1", voice_persistence.candidate, voice, state
    )
    await server._persist_voice_intake_profile_state(
        "candidate-1", voice_persistence.candidate, voice, state
    )

    assert voice_persistence.revision == 1
    assert voice_persistence.refresh_at is None
    assert voice_persistence.refresh_revision is None
    assert voice_persistence.commits == 2


@pytest.mark.asyncio
async def test_voice_preference_only_change_bumps_once(voice_persistence):
    candidate = dict(voice_persistence.candidate)
    await server._persist_voice_intake_profile_state(
        "candidate-1",
        candidate,
        {"preferred_locations": ["Berlin"]},
        {"completed_turns": []},
    )

    assert voice_persistence.revision == 1
    assert voice_persistence.preferences["preferred_locations"] == ["Berlin"]
    assert voice_persistence.refresh_at is None
    assert voice_persistence.refresh_revision is None


@pytest.mark.asyncio
async def test_voice_profile_and_preferences_change_increment_once(voice_persistence):
    await server._persist_voice_intake_profile_state(
        "candidate-1",
        voice_persistence.candidate,
        {"current_role": "Engineer", "employment_types": ["full-time"]},
        {"completed_turns": []},
    )

    assert voice_persistence.revision == 1
    assert voice_persistence.preferences["employment_types"] == ["Full-time"]
