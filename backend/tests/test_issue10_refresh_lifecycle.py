import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

sys.path.insert(0, str(Path(__file__).parents[1]))

import candidate_job_matching_service as matcher
import server
from qdrant_service import RetrievalDiagnostics, SearchResults


def _empty_results():
    return SearchResults(
        [],
        RetrievalDiagnostics(
            raw_hits=0,
            valid_job_ids=0,
            invalid_job_ids=0,
            unique_job_ids=0,
            duplicate_points_removed=0,
        ),
    )


def _non_empty_results():
    return SearchResults(
        [("job-1", 0.9)],
        RetrievalDiagnostics(
            raw_hits=1,
            valid_job_ids=1,
            invalid_job_ids=0,
            unique_job_ids=1,
            duplicate_points_removed=0,
        ),
    )


def test_healthy_empty_requires_original_diagnostics_and_all_zero_fields():
    assert matcher._healthy_empty_retrieval(_empty_results()) is True
    assert matcher._healthy_empty_retrieval([]) is False
    assert matcher._healthy_empty_retrieval(_non_empty_results()) is False


@pytest.mark.parametrize("field", [
    "raw_hits",
    "valid_job_ids",
    "invalid_job_ids",
    "unique_job_ids",
    "duplicate_points_removed",
])
def test_non_zero_retrieval_diagnostic_is_incomplete(field):
    values = {
        "raw_hits": 0,
        "valid_job_ids": 0,
        "invalid_job_ids": 0,
        "unique_job_ids": 0,
        "duplicate_points_removed": 0,
    }
    values[field] = 1
    result = SearchResults([], RetrievalDiagnostics(**values))
    assert matcher._healthy_empty_retrieval(result) is False


def test_broad_and_target_role_empty_retrievals_require_both_diagnostics():
    assert matcher._healthy_empty_retrieval(_empty_results(), _empty_results()) is True
    assert matcher._healthy_empty_retrieval(_empty_results(), _non_empty_results()) is False


def test_freshness_detects_revision_mismatch():
    class Result:
        def mappings(self):
            return self

        def fetchone(self):
            return {
                "last_full_recommendation_refresh_at": datetime.now(UTC),
                "matching_profile_revision": 8,
                "last_full_recommendation_revision": 7,
            }

    class Database:
        async def execute(self, statement, params):
            return Result()

    assert asyncio.run(server._candidate_recommendations_are_stale(Database(), "c-1"))


def test_freshness_accepts_recent_matching_revision():
    class Result:
        def mappings(self):
            return self

        def fetchone(self):
            return {
                "last_full_recommendation_refresh_at": datetime.now(UTC),
                "matching_profile_revision": 8,
                "last_full_recommendation_revision": 8,
            }

    class Database:
        async def execute(self, statement, params):
            return Result()

    assert not asyncio.run(server._candidate_recommendations_are_stale(Database(), "c-1"))


@pytest.mark.asyncio
async def test_refresh_wrapper_releases_after_success(monkeypatch):
    released = []

    async def release(*args):
        released.append(args)

    monkeypatch.setattr(matcher, "_release_refresh_lease_safely", release)

    @matcher._lease_safe_refresh
    async def refresh(candidate_id, candidate, SessionLocal):
        matcher._refresh_lease_context.set((SessionLocal, candidate_id, "owner", 4))
        return True

    assert await refresh("c-1", {}, object()) is True
    assert len(released) == 1


@pytest.mark.asyncio
async def test_refresh_wrapper_releases_after_exception_and_preserves_exception(monkeypatch):
    released = []

    async def release(*args):
        released.append(args)

    monkeypatch.setattr(matcher, "_release_refresh_lease_safely", release)

    @matcher._lease_safe_refresh
    async def refresh(candidate_id, candidate, SessionLocal):
        matcher._refresh_lease_context.set((SessionLocal, candidate_id, "owner", 4))
        raise RuntimeError("retrieval failed")

    with pytest.raises(RuntimeError, match="retrieval failed"):
        await refresh("c-1", {}, object())
    assert len(released) == 1


@pytest.mark.asyncio
async def test_refresh_wrapper_releases_after_cancellation(monkeypatch):
    released = []

    async def release(*args):
        released.append(args)

    monkeypatch.setattr(matcher, "_release_refresh_lease_safely", release)

    @matcher._lease_safe_refresh
    async def refresh(candidate_id, candidate, SessionLocal):
        matcher._refresh_lease_context.set((SessionLocal, candidate_id, "owner", 4))
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await refresh("c-1", {}, object())
    assert len(released) == 1


@pytest.mark.asyncio
async def test_release_failure_does_not_replace_refresh_exception(monkeypatch):
    class FailingSession:
        async def __aenter__(self):
            raise RuntimeError("release failed")

        async def __aexit__(self, *_):
            return False

    @matcher._lease_safe_refresh
    async def refresh(candidate_id, candidate, SessionLocal):
        matcher._refresh_lease_context.set((SessionLocal, candidate_id, "owner", 4))
        raise ValueError("original failure")

    with pytest.raises(ValueError, match="original failure"):
        await refresh("c-1", {}, lambda: FailingSession())


@pytest.mark.asyncio
async def test_persistence_fence_rejects_wrong_owner_generation_and_revision():
    class Result:
        def __init__(self, row=None, scalar=None):
            self._row = row
            self._scalar = scalar

        def mappings(self):
            return self

        def fetchone(self):
            return self._row

        def scalar(self):
            return self._scalar

    class Database:
        async def execute(self, statement, params):
            sql = str(statement)
            if "FROM candidate_recommendation_refresh_leases" in sql:
                return Result({
                    "owner_token": "other-owner",
                    "generation": 9,
                    "lease_expires_at": datetime.now(UTC) + timedelta(minutes=1),
                })
            if "matching_profile_revision" in sql:
                return Result(scalar=4)
            return Result()

    assert not await matcher._refresh_persistence_is_current(
        Database(), "c-1", "owner", 4, 4
    )


@pytest.mark.asyncio
async def test_persistence_helper_accepts_db_positional_and_keyword_identity():
    class Result:
        def __init__(self, rows=None, scalar=None):
            self.rows = rows or []
            self.scalar_value = scalar

        def mappings(self):
            return self

        def fetchone(self):
            return self.rows[0] if self.rows else None

        def fetchall(self):
            return self.rows

        def scalar(self):
            return self.scalar_value

    class Database:
        async def execute(self, statement, params):
            sql = str(statement)
            if "SELECT id FROM candidates" in sql:
                return Result([{"id": "c-1"}])
            if "FROM candidate_recommendation_refresh_leases" in sql:
                return Result([{
                    "owner_token": "owner",
                    "generation": 4,
                    "lease_expires_at": datetime.now(UTC) + timedelta(minutes=1),
                }])
            if "matching_profile_revision" in sql:
                return Result(scalar=7)
            if "SELECT job_id, id, tracked_at, hidden_at" in sql:
                return Result([])
            return Result()

    assert await matcher._persist_candidate_recommendations(
        Database(),
        candidate_id="c-1",
        owner_token="owner",
        lease_generation=4,
        snapshot_revision=7,
        ranked_jobs=[],
        experience_ineligible_job_ids=set(),
    ) is True


class _MappingResult:
    def __init__(self, row=None, scalar=None):
        self.row = row
        self.scalar_value = scalar

    def mappings(self):
        return self

    def fetchone(self):
        return self.row

    def scalar(self):
        return self.scalar_value


class _RefreshDb:
    def __init__(self, *, candidate=True, lease_row=None, revision=4):
        self.candidate = candidate
        self.lease_row = lease_row
        self.revision = revision
        self.statements = []
        self.commits = 0
        self.rollbacks = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        if "SELECT * FROM candidates" in sql:
            return _MappingResult(
                {
                    "id": "c-1",
                    "matching_profile_revision": self.revision,
                    "current_role": "Engineer",
                    "skills": ["Python"],
                    "work_experience": [],
                    "raw_data": {},
                }
                if self.candidate
                else None
            )
        if "SELECT * FROM candidate_preferences" in sql:
            return _MappingResult(None)
        if "FROM candidate_recommendation_refresh_leases" in sql:
            return _MappingResult(self.lease_row)
        if "matching_profile_revision FROM candidates" in sql:
            return _MappingResult(scalar=self.revision)
        return _MappingResult()

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


@pytest.mark.asyncio
async def test_refresh_missing_authoritative_candidate_does_not_acquire_or_persist(monkeypatch):
    db = _RefreshDb(candidate=False)
    acquired = []

    async def acquire(*args):
        acquired.append(args)
        raise AssertionError("lease acquisition must not run")

    monkeypatch.setattr(matcher, "acquire_refresh_lease", acquire)
    result = await matcher.refresh_candidate_job_matches(
        "c-1", {"current_role": "caller data"}, cast(async_sessionmaker[Any], lambda: db)
    )

    assert result is False
    assert acquired == []
    assert not any("candidate_job_recommendations" in sql for sql in db.statements)
    assert not any("last_full_recommendation" in sql for sql in db.statements)


@pytest.mark.asyncio
async def test_refresh_lease_contention_skips_retrieval_and_persistence(monkeypatch):
    db = _RefreshDb(
        lease_row=None,
    )
    retrieved = []

    async def acquire(*args):
        return type("Lease", (), {"acquired": False, "generation": None})()

    monkeypatch.setattr(matcher, "acquire_refresh_lease", acquire)
    monkeypatch.setattr(
        matcher,
        "search_job_chunks",
        lambda *args, **kwargs: retrieved.append(True),
    )
    result = await matcher.refresh_candidate_job_matches("c-1", {}, cast(async_sessionmaker[Any], lambda: db))

    assert result is False
    assert retrieved == []
    assert not any("candidate_job_recommendations" in sql for sql in db.statements)
    assert not any("last_full_recommendation" in sql for sql in db.statements)


@pytest.mark.asyncio
async def test_refresh_renewal_failure_prevents_retrieval_persistence(monkeypatch):
    db = _RefreshDb(
        lease_row={
            "owner_token": "owner",
            "generation": 1,
            "lease_expires_at": datetime.now(UTC) + timedelta(minutes=1),
        }
    )
    async def acquire(*args):
        db.lease_row["owner_token"] = args[2]
        return type("Lease", (), {"acquired": True, "generation": 1})()

    async def renew(*args):
        return False

    retrieved = []
    monkeypatch.setattr(matcher, "acquire_refresh_lease", acquire)
    monkeypatch.setattr(matcher, "renew_refresh_lease", renew)
    def record_retrieval(*_args, **_kwargs):
        retrieved.append(True)
        return _non_empty_results()

    monkeypatch.setattr(matcher, "search_job_chunks", record_retrieval)
    monkeypatch.setattr(matcher, "generate_embedding", lambda *args: [0.1])
    monkeypatch.setattr(matcher, "build_candidate_text", lambda *_: "Engineer Python")
    monkeypatch.setattr(matcher, "candidate_ready_for_matching", lambda *_: (True, "ok"))

    result = await matcher.refresh_candidate_job_matches("c-1", {}, cast(async_sessionmaker[Any], lambda: db))
    assert result is False
    assert retrieved == [True]
    assert not any("candidate_job_recommendations" in sql for sql in db.statements)
    assert not any("last_full_recommendation" in sql for sql in db.statements)


@pytest.mark.asyncio
async def test_healthy_empty_refresh_persists_freshness_revision(monkeypatch):
    db = _RefreshDb(
        lease_row={
            "owner_token": "owner",
            "generation": 1,
            "lease_expires_at": datetime.now(UTC) + timedelta(minutes=1),
        }
    )

    async def acquire(*args):
        db.lease_row["owner_token"] = args[2]
        return type("Lease", (), {"acquired": True, "generation": 1})()

    async def renew(*args):
        return True

    monkeypatch.setattr(matcher, "acquire_refresh_lease", acquire)
    monkeypatch.setattr(matcher, "renew_refresh_lease", renew)
    async def release(*args):
        return True

    monkeypatch.setattr(matcher, "release_refresh_lease", release)
    monkeypatch.setattr(matcher, "generate_embedding", lambda *args: [0.1])
    monkeypatch.setattr(matcher, "build_candidate_text", lambda *_: "Engineer Python")
    monkeypatch.setattr(matcher, "candidate_ready_for_matching", lambda *_: (True, "ok"))
    monkeypatch.setattr(matcher, "search_job_chunks", lambda *args, **kwargs: _empty_results())

    assert await matcher.refresh_candidate_job_matches("c-1", {}, cast(async_sessionmaker[Any], lambda: db)) is True
    assert any("last_full_recommendation_refresh_at" in sql for sql in db.statements)
    assert any("last_full_recommendation_revision" in sql for sql in db.statements)
    assert db.commits >= 1


@pytest.mark.asyncio
async def test_degraded_empty_refresh_does_not_persist_freshness(monkeypatch):
    db = _RefreshDb(
        lease_row={
            "owner_token": "owner",
            "generation": 1,
            "lease_expires_at": datetime.now(UTC) + timedelta(minutes=1),
        }
    )
    degraded = SearchResults(
        [],
        RetrievalDiagnostics(
            raw_hits=1,
            valid_job_ids=0,
            invalid_job_ids=1,
            unique_job_ids=0,
            duplicate_points_removed=0,
        ),
    )
    async def acquire(*args):
        return type("Lease", (), {"acquired": True, "generation": 1})()

    async def renew(*args):
        return True

    monkeypatch.setattr(matcher, "acquire_refresh_lease", acquire)
    monkeypatch.setattr(matcher, "renew_refresh_lease", renew)
    monkeypatch.setattr(matcher, "generate_embedding", lambda *args: [0.1])
    monkeypatch.setattr(matcher, "build_candidate_text", lambda *_: "Engineer Python")
    monkeypatch.setattr(matcher, "candidate_ready_for_matching", lambda *_: (True, "ok"))
    monkeypatch.setattr(matcher, "search_job_chunks", lambda *args, **kwargs: degraded)

    assert await matcher.refresh_candidate_job_matches("c-1", {}, cast(async_sessionmaker[Any], lambda: db)) is False
    assert not any("last_full_recommendation" in sql for sql in db.statements)
