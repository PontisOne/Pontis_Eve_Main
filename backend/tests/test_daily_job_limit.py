"""Regression coverage for calendar-date based free job access."""

import asyncio
import os
from datetime import UTC, date, datetime
from typing import Any

import pytest

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://unused:unused@localhost/unused"
)
import server


class _Result:
    def __init__(self, rows=(), scalar_value=None):
        self._rows, self._scalar_value = list(rows), scalar_value

    def scalar(self):
        return self._scalar_value

    def fetchall(self):
        return self._rows


class _DailyAccessSession:
    def __init__(self, state):
        self.state = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, statement, params=None):
        query, params = str(statement), params or {}
        if "pg_advisory_xact_lock" in query:
            return _Result()
        if "SELECT COUNT(*) FROM candidate_daily_job_access" in query:
            used = [
                entry
                for entry in self.state["access"]
                if entry[0] == params["cid"] and entry[2] == params["access_date"]
            ]
            return _Result(scalar_value=len(used))
        if "SELECT cjr.id" in query:
            # All historic access is excluded: tomorrow receives new recommendations.
            claimed = {rec_id for _, rec_id, _ in self.state["access"]}
            jobs = self.state.get("jobs", {})
            visible = [
                rec_id
                for rec_id, hidden in self.state["recommendations"]
                if not hidden and rec_id not in claimed
                and rec_id in params["eligible_ids"]
                and (job := jobs.get(rec_id, {
                    "hidden": hidden,
                    "active": True,
                    "status": "active",
                    "job_status": "active",
                    "job_url": "https://example.com/job",
                    "opportunity_type": "jobs",
                }))
                and not job["hidden"]
                and job["active"]
                and job["status"] in {"active", "open", "published"}
                and job["job_status"] in {"active", "open", "published"}
                and job["job_url"].startswith(("http://", "https://"))
                and (job["opportunity_type"] or "jobs") == params["opportunity_type"]
            ]
            assert "eligible_ids" in params
            assert "cjr.id = ANY(CAST(:eligible_ids AS uuid[]))" in query
            assert "COALESCE(jd.opportunity_type, 'jobs') = :opportunity_type" in query
            assert "is_active IS TRUE" in query
            assert "cjr.candidate_id = :cid" in query
            assert "cjr.hidden_at IS NULL" in query
            assert "NOT EXISTS" in query
            assert params["opportunity_type"] == "jobs"
            return _Result([(rec_id,) for rec_id in visible[: params["slots"]]])
        if "INSERT INTO candidate_daily_job_access" in query:
            self.state["access"].add(
                (params["cid"], params["rid"], params["access_date"])
            )
            return _Result()
        raise AssertionError(f"Unexpected query: {query}")

    async def commit(self):
        pass


class _DailyAccessSessionFactory:
    def __init__(self, state):
        self.state = state

    def __call__(self):
        return _DailyAccessSession(self.state)


class _EndpointResult(_Result):
    def mappings(self):
        return self

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _JobsEndpointSession:
    """In-memory SQL seam for the GET /jobs allocation path."""

    def __init__(self, state):
        self.state = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, statement, params=None):
        query, params = str(statement), params or {}
        if "FROM candidates" in query and "last_full_recommendation_refresh_at" in query:
            return _EndpointResult(
                rows=[
                    {
                        "last_full_recommendation_refresh_at": datetime.now(UTC),
                        "matching_profile_revision": 0,
                        "last_full_recommendation_revision": 0,
                    }
                ]
            )
        if (
            "FROM candidate_job_recommendations cjr" in query
            and "SELECT COUNT(*)" in query
        ):
            historical = {rec_id for _, rec_id, _ in self.state["access"]}
            available = [
                rec_id
                for rec_id, hidden in self.state["recommendations"]
                if not hidden and rec_id not in historical
            ]
            return _EndpointResult(scalar_value=len(available))
        if "pg_advisory_xact_lock" in query:
            return _EndpointResult()
        if "SELECT COUNT(*) FROM candidate_daily_job_access" in query:
            used = [
                entry
                for entry in self.state["access"]
                if entry[0] == params["cid"] and entry[2] == params["access_date"]
            ]
            return _EndpointResult(scalar_value=len(used))
        if "SELECT cjr.id" in query and "candidate_job_recommendations cjr" in query:
            if "jd.title, jd.description" in query:
                return _EndpointResult(
                    rows=[
                        {
                            0: rec_id,
                            "id": rec_id,
                            "title": "Job",
                            "description": "A software engineering role",
                            "requirements": "Software engineering experience",
                            "skills": ["Python"],
                            "skills_required": [],
                            "experience_required": None,
                            "location": "Remote",
                            "city": None,
                            "state": None,
                            "country": None,
                            "remote": True,
                            "remote_policy": "remote",
                            "employment_type": "Full-time",
                            "industry": None,
                            "structured_data": {},
                            "responsibilities": None,
                        }
                        for rec_id, hidden in self.state["recommendations"]
                        if not hidden
                    ]
                )
            historical = {rec_id for _, rec_id, _ in self.state["access"]}
            available = [
                rec_id
                for rec_id, hidden in self.state["recommendations"]
                if not hidden and rec_id not in historical
            ]
            return _EndpointResult(
                [(rec_id,) for rec_id in available[: params.get("slots", len(available))]]
            )
        if "SELECT cjr.match_reason, jd.title" in query:
            return _EndpointResult(
                rows=[
                    {
                        "match_reason": None,
                        "title": "Job",
                        "description": "A software engineering role",
                        "requirements": "Software engineering",
                        "skills": [],
                        "skills_required": [],
                        "experience_required": None,
                        "location": "Remote",
                        "employment_type": "Full-time",
                        "remote": True,
                        "remote_policy": "remote",
                        "industry": None,
                        "salary_range": None,
                        "city": None,
                        "state": None,
                        "country": None,
                        "structured_data": {},
                    }
                ]
            )
        if "UPDATE candidate_job_recommendations" in query:
            return _EndpointResult()
        if "INSERT INTO candidate_daily_job_access" in query:
            self.state["access"].add(
                (params["cid"], params["rid"], params["access_date"])
            )
            return _EndpointResult()
        if (
            "SELECT recommendation_id" in query
            and "FROM candidate_daily_job_access" in query
        ):
            claimed = [
                rec_id
                for cid, rec_id, day in self.state["access"]
                if cid == params["cid"] and day == params["access_date"]
            ]
            return _EndpointResult([(rec_id,) for rec_id in claimed])
        if "cjr.id AS rec_id" in query:
            rows = [
                {
                    "rec_id": rec_id,
                    "job_id": f"job-{rec_id}",
                    "match_score": 0.9,
                    "recommendation_rank": index,
                    "match_reason": None,
                    "tracked_at": None,
                    "applied_at": None,
                    "hidden_at": None,
                    "viewed_at": None,
                    "application_status": None,
                    "application_agency_id": None,
                    "application_job_role": None,
                    "title": f"Job {rec_id}",
                    "company_name": "Company",
                    "location": "Remote",
                    "salary_range": None,
                    "employment_type": "Full-time",
                    "remote_policy": "remote",
                    "experience_level": None,
                    "experience_required": None,
                    "created_at": None,
                    "skills_required": [],
                    "description": "A software engineering role building web applications.",
                    "requirements": "Software engineering experience",
                    "skills": ["Python"],
                    "company_logo_url": None,
                    "job_url": "https://example.com/job",
                    "opportunity_type": "jobs",
                }
                for index, (rec_id, hidden) in enumerate(
                    self.state["recommendations"], start=1
                )
                if not hidden
            ]
            return _EndpointResult(rows)
        raise AssertionError(f"Unexpected query: {query}")

    async def commit(self):
        pass


class _JobsEndpointSessionFactory:
    def __init__(self, state):
        self.state = state

    def __call__(self):
        return _JobsEndpointSession(self.state)


def _state():
    return {
        "recommendations": [(f"rec-{i}", False) for i in range(13)],
        "access": set(),
    }


def _claim(
    monkeypatch,
    state,
    day,
    request_more=False,
    candidate=None,
    eligible_ids=None,
    opportunity_type="jobs",
):
    monkeypatch.setattr(server, "_product_current_date", lambda: day)
    monkeypatch.setattr(server, "SessionLocal", _DailyAccessSessionFactory(state))
    return asyncio.run(
        server._claim_daily_job_access(
            "candidate",
            candidate or {},
            request_more,
            eligible_ids=eligible_ids
            if eligible_ids is not None
            else [rec_id for rec_id, _hidden in state["recommendations"]],
            opportunity_type=opportunity_type,
        )
    )


def _day_accesses(state, day):
    return {rec_id for _, rec_id, access_day in state["access"] if access_day == day}


def test_september_15_grants_exactly_three_jobs(monkeypatch):
    state, day = _state(), date(2026, 9, 15)
    assert _claim(monkeypatch, state, day)
    assert _day_accesses(state, day) == {"rec-0", "rec-1", "rec-2"}


def test_fourth_job_on_september_15_returns_existing_limit_response(monkeypatch):
    state, day = _state(), date(2026, 9, 15)
    _claim(monkeypatch, state, day)
    with pytest.raises(server.HTTPException) as exc_info:
        _claim(monkeypatch, state, day, request_more=True)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == {
        "code": "daily_job_limit_reached",
        "message": "Unlock more jobs by subscribing.",
    }


def test_repeated_requests_on_same_date_do_not_reset_usage(monkeypatch):
    state, day = _state(), date(2026, 9, 15)
    _claim(monkeypatch, state, day)
    _claim(monkeypatch, state, day)
    assert len(state["access"]) == 3


def test_next_calendar_date_gets_next_three_previously_unaccessed_jobs(monkeypatch):
    state = _state()
    _claim(monkeypatch, state, date(2026, 9, 15))
    _claim(monkeypatch, state, date(2026, 9, 16))
    assert _day_accesses(state, date(2026, 9, 16)) == {"rec-3", "rec-4", "rec-5"}


def test_calendar_boundary_not_rolling_24_hour_window(monkeypatch):
    state = _state()
    # Adjacent calendar dates work independently; accessed_at is never consulted.
    _claim(monkeypatch, state, date(2026, 9, 15))
    _claim(monkeypatch, state, date(2026, 9, 16))
    assert "accessed_at" not in str(server._claim_daily_job_access.__code__.co_consts)
    assert _day_accesses(state, date(2026, 9, 16)) == {"rec-3", "rec-4", "rec-5"}


def test_subscriber_bypasses_limit_and_does_not_consume_free_access(monkeypatch):
    state = _state()
    assert not _claim(
        monkeypatch, state, date(2026, 9, 15), True, {"subscription_active": True}
    )
    assert state["access"] == set()


def test_hidden_recommendations_are_never_claimed(monkeypatch):
    state = _state()
    state["recommendations"][0] = ("rec-0", True)
    _claim(monkeypatch, state, date(2026, 9, 15))
    assert _day_accesses(state, date(2026, 9, 15)) == {"rec-1", "rec-2", "rec-3"}


def _eligibility_fixture(**job_overrides):
    state = _state()
    state["jobs"] = {
        "rec-0": {
            "hidden": False,
            "active": True,
            "status": "active",
            "job_status": "active",
            "job_url": "https://example.com/job",
            "opportunity_type": "jobs",
            **job_overrides,
        }
    }
    return state


def test_hidden_recommendation_cannot_consume_daily_access(monkeypatch):
    state = _eligibility_fixture(hidden=True)
    _claim(monkeypatch, state, date(2026, 9, 15), eligible_ids=["rec-0"])
    assert _day_accesses(state, date(2026, 9, 15)) == set()


@pytest.mark.parametrize(
    "job_overrides",
    [
        {"active": False},
        {"job_url": "not-a-url"},
    ],
    ids=["inactive", "invalid-url"],
)
def test_inactive_or_invisible_recommendation_cannot_consume_daily_access(
    monkeypatch, job_overrides
):
    state = _eligibility_fixture(**job_overrides)
    _claim(monkeypatch, state, date(2026, 9, 15), eligible_ids=["rec-0"])
    assert _day_accesses(state, date(2026, 9, 15)) == set()


def test_wrong_opportunity_type_cannot_consume_daily_access(monkeypatch):
    state = _eligibility_fixture(opportunity_type="intern")
    _claim(monkeypatch, state, date(2026, 9, 15), eligible_ids=["rec-0"])
    assert _day_accesses(state, date(2026, 9, 15)) == set()


def test_recommendation_excluded_from_eligible_ids_cannot_consume_daily_access(monkeypatch):
    state = _eligibility_fixture()
    _claim(monkeypatch, state, date(2026, 9, 15), eligible_ids=[])
    assert _day_accesses(state, date(2026, 9, 15)) == set()


def test_valid_visible_eligible_recommendation_consumes_daily_access(monkeypatch):
    state = _eligibility_fixture()
    _claim(monkeypatch, state, date(2026, 9, 15), eligible_ids=["rec-0"])
    assert _day_accesses(state, date(2026, 9, 15)) == {"rec-0"}


def test_schema_uses_product_calendar_date():
    assert "access_date" in server.CREATE_DAILY_JOB_ACCESS_TABLE
    assert "CURRENT_DATE" not in str(server._claim_daily_job_access.__code__.co_consts)
    assert server.PRODUCT_TIMEZONE == "Asia/Kolkata"


def _get_jobs(monkeypatch, state, day, refresh, candidate=None, response=None):
    completed_candidate = {
        "current_role": "Software Engineer",
        "raw_data": {"voice_intake": {"status": "completed"}},
    }

    async def candidate_row(_candidate_id):
        return {**completed_candidate, **(candidate or {})}

    async def strength(*_args):
        return 90

    monkeypatch.setattr(server, "_get_candidate_row", candidate_row)
    monkeypatch.setattr(server, "_effective_profile_strength_percent", strength)
    monkeypatch.setattr(server, "_product_current_date", lambda: day)
    monkeypatch.setattr(server, "SessionLocal", _JobsEndpointSessionFactory(state))
    import candidate_job_matching_service as matcher

    async def refresh_stale_scores(*_args, **_kwargs):
        pass

    monkeypatch.setattr(matcher, "refresh_candidate_job_matches", refresh)
    monkeypatch.setattr(matcher, "refresh_stale_candidate_job_scores", refresh_stale_scores)
    monkeypatch.setattr(
        matcher,
        "stored_recommendation_relevance",
        lambda *_args, **_kwargs: {"eligible": True},
    )
    return asyncio.run(
        server.get_candidate_jobs(
            "candidate",
            response=response,
            authorization=f"Bearer {server._issue_candidate_session_token('candidate')}",
        )
    )


def test_jobs_exposes_all_matches_and_redacts_locked_free_jobs(monkeypatch):
    from starlette.responses import Response

    state = _state()

    async def refresh(*_args):
        pass

    response = Response()
    jobs = _get_jobs(monkeypatch, state, date(2026, 9, 17), refresh, response=response)

    assert len(jobs) == 13
    assert [job["locked"] for job in jobs[:3]] == [False, False, False]
    assert all(job["locked"] for job in jobs[3:])
    assert "title" not in jobs[3]
    assert response.headers["x-total-matching-jobs"] == "13"


def test_jobs_refreshes_when_only_visible_recommendation_was_accessed_before_today(
    monkeypatch,
):
    state: dict[str, Any] = {
        "recommendations": [("old", False)],
        "access": {("candidate", "old", date(2026, 9, 16))},
    }
    refreshes = []

    async def refresh(*_args):
        refreshes.append(True)
        state["recommendations"].extend([("new-1", False), ("new-2", False)])

    jobs = _get_jobs(monkeypatch, state, date(2026, 9, 17), refresh)

    assert refreshes == [True]
    assert [job["id"] for job in jobs] == ["old", "new-1", "new-2"]
    assert jobs[0]["locked"] is True
    assert all(not job["locked"] for job in jobs[1:])
    assert _day_accesses(state, date(2026, 9, 17)) == {"new-1", "new-2"}


def test_jobs_does_not_refresh_when_visible_unaccessed_recommendations_exist(
    monkeypatch,
):
    state = {
        "recommendations": [
            ("old", False),
            ("new-1", False),
            ("new-2", False),
            ("new-3", False),
            ("new-4", False),
        ],
        "access": {("candidate", "old", date(2026, 9, 16))},
    }
    refreshes = []

    async def refresh(*_args):
        refreshes.append(True)

    jobs = _get_jobs(monkeypatch, state, date(2026, 9, 17), refresh)

    assert refreshes == []
    assert [job["id"] for job in jobs] == ["old", "new-1", "new-2", "new-3", "new-4"]
    assert jobs[0]["locked"] is True
    assert all(not job["locked"] for job in jobs[1:4])
    assert jobs[4]["locked"] is True
    assert _day_accesses(state, date(2026, 9, 17)) == {"new-1", "new-2", "new-3"}


def test_jobs_refreshes_when_all_visible_recommendations_have_historical_access(
    monkeypatch,
):
    state = {
        "recommendations": [("old-1", False), ("old-2", False), ("hidden", True)],
        "access": {
            ("candidate", "old-1", date(2026, 9, 15)),
            ("candidate", "old-2", date(2026, 9, 16)),
        },
    }
    refreshes = []

    async def refresh(*_args):
        refreshes.append(True)

    _get_jobs(monkeypatch, state, date(2026, 9, 17), refresh)

    assert refreshes == [True]


def test_subscriber_with_historically_accessed_recommendations_bypasses_access_and_refresh(
    monkeypatch,
):
    state = {
        "recommendations": [("old", False)],
        "access": {("candidate", "old", date(2026, 9, 16))},
    }
    refreshes = []

    async def refresh(*_args):
        refreshes.append(True)

    jobs = _get_jobs(
        monkeypatch, state, date(2026, 9, 17), refresh, {"subscription_active": True}
    )

    assert refreshes == []
    assert [job["id"] for job in jobs] == ["old"]
    assert state["access"] == {("candidate", "old", date(2026, 9, 16))}
