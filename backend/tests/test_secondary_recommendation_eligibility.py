"""Mocked route tests for secondary stored-recommendation access paths."""

import pytest
import server
from test_recommendation_route_relevance import _candidate, _jobs


class _Result:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def mappings(self):
        return self

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class _Session:
    def __init__(self, jobs):
        self.jobs = jobs
        self.calls = []
        self.metadata_writes = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def execute(self, statement, params=None):
        sql = str(statement)
        params = params or {}
        self.calls.append((sql, params))
        if "tracked_at IS NOT NULL" in sql:
            assert "cjr.candidate_id = :cid" in sql
            assert "cjr.hidden_at IS NULL" in sql
            assert server.candidate_visible_where("jd").strip() in sql
            assert "COALESCE(jd.opportunity_type, 'jobs') = :opportunity_type" in sql
            assert params["cid"] == "candidate-1"
            assert params["opportunity_type"] == "jobs"
        elif "cjr.id = :rid" in sql:
            assert "cjr.candidate_id = :cid" in sql
            assert "cjr.hidden_at IS NULL" in sql
            assert server.candidate_visible_where("jd").strip() in sql
            assert "COALESCE(jd.opportunity_type, 'jobs') = :opportunity_type" in sql
            assert params["cid"] == "candidate-1"
            assert params["rid"] in {item["id"] for item in self.jobs}
            assert params["opportunity_type"] == "jobs"
        if "INSERT INTO candidate_application_resumes" in sql:
            self.metadata_writes += 1
        job = next(
            (item for item in self.jobs if item["id"] == params.get("rid")),
            None,
        )
        if "tracked_at IS NOT NULL" in sql:
            visible = [
                item
                for item in self.jobs
                if not item["hidden"]
                and item["active"]
                and item["job_url"].startswith(("http://", "https://"))
                and (item["opportunity_type"] or "jobs") == params.get("opportunity_type")
            ]
            return _Result([self._row(item) for item in visible])
        if "cjr.id = :rid" in sql:
            if (
                job
                and not job["hidden"]
                and job["active"]
                and job["job_url"].startswith(("http://", "https://"))
                and (job["opportunity_type"] or "jobs") == params.get("opportunity_type")
            ):
                return _Result([self._row(job)])
            return _Result()
        return _Result()

    def _row(self, job):
        if job is None:
            return None
        row = dict(job)
        row.update(
            {
                "rec_id": job["id"],
                "job_id": f"job-{job['id']}",
                "match_score": job["match_score"],
                "tracked_at": "2026-11-20T00:00:00+00:00",
                "hidden_at": None if not job["hidden"] else "2026-11-19T00:00:00+00:00",
                "company_logo_url": None,
                "id": f"job-{job['id']}",
            }
        )
        return row


def _patch(monkeypatch, jobs):
    async def candidate_row(_candidate_id):
        return _candidate()

    sessions: list[_Session] = []

    def session_factory():
        session = _Session(jobs)
        sessions.append(session)
        return session

    monkeypatch.setattr(server, "SessionLocal", session_factory)
    monkeypatch.setattr(server, "_get_candidate_row", candidate_row)
    return sessions


@pytest.mark.asyncio
async def test_tracked_jobs_revalidate_current_relevance_and_visibility(monkeypatch):
    jobs = _jobs()
    sessions = _patch(monkeypatch, jobs)

    result = await server.get_tracked_jobs(
        "candidate-1",
        authorization=f"Bearer {server._issue_candidate_session_token('candidate-1')}",
    )

    assert [item["job_id"] for item in result] == ["job-relevant"]
    assert sessions[0].calls


@pytest.mark.asyncio
@pytest.mark.parametrize("recommendation_id", ["hidden", "inactive", "invalid-url", "intern"])
async def test_match_improvement_rejects_ineligible_recommendations(
    monkeypatch, recommendation_id
):
    jobs = _jobs()
    _patch(monkeypatch, jobs)

    with pytest.raises(server.HTTPException) as exc_info:
        await server._get_job_match_improvement_row("candidate-1", recommendation_id)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_match_improvement_rejects_currently_irrelevant_recommendation(monkeypatch):
    jobs = _jobs()
    _patch(monkeypatch, jobs)

    with pytest.raises(server.HTTPException) as exc_info:
        await server._get_job_match_improvement_row("candidate-1", "irrelevant")

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_match_improvement_accepts_currently_relevant_recommendation(monkeypatch):
    jobs = _jobs()
    _patch(monkeypatch, jobs)

    row = await server._get_job_match_improvement_row("candidate-1", "relevant")

    assert row["title"] == "Python Backend Engineer"


@pytest.mark.asyncio
async def test_resume_download_rejects_invalid_recommendation_before_side_effects(monkeypatch):
    jobs = _jobs()
    sessions = _patch(monkeypatch, jobs)
    side_effects: list[object] = []

    monkeypatch.setattr(
        server,
        "_application_resume_path",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("resume path must not be reached")
        ),
    )

    with pytest.raises(server.HTTPException) as exc_info:
        await server.download_application_resume(
            "candidate-1",
            "invalid-url",
            authorization=f"Bearer {server._issue_candidate_session_token('candidate-1')}",
        )

    assert exc_info.value.status_code == 404
    assert sessions[0].metadata_writes == 0
    assert side_effects == []
