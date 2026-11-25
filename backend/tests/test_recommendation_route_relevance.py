"""Mocked route-level parity tests for stored recommendation relevance."""

import os
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://unused:unused@localhost/unused"
)

import server


class _Result:
    def __init__(self, rows=(), scalar_value=None, mapping_rows=()):
        self.rows = list(rows)
        self.scalar_value = scalar_value
        self.mapping_rows = list(mapping_rows)

    def scalar(self):
        return self.scalar_value

    def fetchall(self):
        return self.rows

    def mappings(self):
        return self

    def fetchone(self):
        return self.mapping_rows[0] if self.mapping_rows else None


def _candidate():
    return {
        "id": "candidate-1",
        "current_role": "Python Backend Engineer",
        "skills": ["Python", "FastAPI"],
        "experience_years": 3,
        "opportunity_type": "jobs",
        "subscription_active": True,
        "raw_data": {"country": "India"},
    }


def _job(
    recommendation_id,
    title,
    description,
    *,
    skills_required=None,
    hidden=False,
    active=True,
    job_url="https://example.com/job",
    opportunity_type="jobs",
):
    return {
        "id": recommendation_id,
        "title": title,
        "description": description,
        "requirements": "",
        "skills": ["Python", "FastAPI"],
        "skills_required": skills_required or [],
        "experience_required": "2-5 years",
        "location": "India",
        "city": "Bengaluru",
        "state": "KA",
        "country": "India",
        "remote": False,
        "remote_policy": "On-site",
        "employment_type": "Full-time",
        "industry": "Software",
        "structured_data": {},
        "responsibilities": ["Build backend APIs"],
        "company_name": "Example Co",
        "salary_range": "",
        "recommendation_rank": 1,
        "match_score": 0.9,
        "hidden": hidden,
        "active": active,
        "status": "active" if active else "closed",
        "job_status": "active" if active else "closed",
        "job_url": job_url,
        "opportunity_type": opportunity_type,
    }


def _jobs():
    irrelevant = _job(
        "irrelevant", "Product Manager", "Own roadmap and stakeholder planning."
    )
    irrelevant["skills"] = []
    irrelevant["responsibilities"] = ["Own product roadmap"]
    return [
        _job("relevant", "Python Backend Engineer", "Build Python and FastAPI APIs."),
        irrelevant,
        _job(
            "missing-java",
            "Python Backend Engineer",
            "Build Python and FastAPI APIs.",
            skills_required=["Python", "Java"],
        ),
        _job("hidden", "Python Backend Engineer", "Build Python APIs.", hidden=True),
        _job(
            "inactive",
            "Python Backend Engineer",
            "Build Python APIs.",
            active=False,
            job_url="https://example.com/inactive",
        ),
        _job(
            "invalid-url",
            "Python Backend Engineer",
            "Build Python APIs.",
            job_url="not-a-valid-url",
        ),
        _job(
            "intern",
            "Python Backend Engineer Intern",
            "Build Python and FastAPI APIs.",
            opportunity_type="intern",
        ),
    ]


def _validation_tuple(job):
    return (
        job["id"],
        job["title"],
        job["description"],
        job["requirements"],
        job["skills"],
        job["skills_required"],
        job["experience_required"],
        job["location"],
        job["city"],
        job["state"],
        job["country"],
        job["remote"],
        job["remote_policy"],
        job["employment_type"],
        job["industry"],
        job["structured_data"],
        job["responsibilities"],
    )


class _RecommendationSession:
    def __init__(self, jobs, session_holder=None):
        self.jobs = jobs
        self.session_holder = session_holder
        self.final_eligible_ids = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def execute(self, statement, params=None):
        sql = str(statement)
        params = params or {}
        if "eligible_ids" in params:
            self.final_eligible_ids = set(params["eligible_ids"])
        if "last_full_recommendation_refresh_at" in sql:
            return _Result(
                mapping_rows=[
                    {
                        "last_full_recommendation_refresh_at": datetime.now(UTC),
                        "matching_profile_revision": 0,
                        "last_full_recommendation_revision": 0,
                    }
                ]
            )
        if "SELECT COUNT(*)" in sql:
            return _Result(scalar_value=1)
        if "SELECT cjr.id, jd.title" in sql:
            requires_visibility = "jd.is_active IS TRUE" in sql
            rows = [
                _validation_tuple(job)
                for job in self.jobs
                if not job["hidden"]
                and (
                    not requires_visibility
                    or (
                        job["active"]
                        and job["status"] in {"active", "open", "published"}
                        and job["job_status"] in {"active", "open", "published"}
                        and job["job_url"].startswith(("http://", "https://"))
                    )
                )
                and (job["opportunity_type"] or "jobs") == params.get("opportunity_type", "jobs")
            ]
            return _Result(rows=rows)
        if "cjr.id AS rec_id" in sql or "cjr.match_score" in sql or "cjr.id = ANY" in sql:
            eligible_ids = set(params.get("eligible_ids", []))
            self.final_eligible_ids = eligible_ids
            rows = [
                {
                    "rec_id": job["id"],
                    "job_id": f"job-{job['id']}",
                    "match_score": job["match_score"],
                    "recommendation_rank": job["recommendation_rank"],
                    "title": job["title"],
                    "company_name": job["company_name"],
                    "location": job["location"],
                    "salary_range": job["salary_range"],
                    "description": job["description"],
                    "employment_type": job["employment_type"],
                    "remote_policy": job["remote_policy"],
                    "experience_level": None,
                    "experience_required": job["experience_required"],
                    "created_at": None,
                    "skills_required": job["skills_required"],
                    "requirements": job["requirements"],
                    "skills": job["skills"],
                    "company_logo_url": None,
                    "job_url": job["job_url"],
                    "opportunity_type": job["opportunity_type"],
                    "match_reason": None,
                    "tracked_at": None,
                    "applied_at": None,
                    "viewed_at": None,
                    "application_status": None,
                }
                for job in self.jobs
                if job["id"] in eligible_ids
                and not job["hidden"]
                and job["active"]
                and job["job_url"].startswith(("http://", "https://"))
                and (job["opportunity_type"] or "jobs") == params.get("opportunity_type", "jobs")
            ]
            return _Result(rows=rows)
        return _Result()

    async def commit(self):
        return None


def _patch_common(monkeypatch, jobs, session_holder=None):
    async def candidate_row(_cid):
        return _candidate()

    def session_factory():
        session = _RecommendationSession(jobs, session_holder)
        if session_holder is not None:
            session_holder.append(session)
        return session

    monkeypatch.setattr(server, "SessionLocal", session_factory)
    monkeypatch.setattr(server, "_get_candidate_row", candidate_row)
    monkeypatch.setattr(server, "_authorize_candidate", lambda *_args: None)
    monkeypatch.setattr(server, "_voice_intake_completed_for_matching", lambda _row: True)
    monkeypatch.setattr(server, "_effective_profile_strength_percent", _strength)
    monkeypatch.setattr(server, "_claim_daily_job_access", _no_access_claim)
    monkeypatch.setattr(server, "_candidate_recommendations_are_stale", _fresh)


async def _strength(*_args):
    return 100


async def _no_access_claim(*_args, **_kwargs):
    return False


async def _fresh(*_args, **_kwargs):
    return False


@pytest.mark.asyncio
async def test_candidate_jobs_route_uses_real_relevance_and_filters_stored_rows(monkeypatch):
    """Mocked route-level test; not PostgreSQL integration coverage."""
    jobs = _jobs()
    _patch_common(monkeypatch, jobs)

    result = await server.get_candidate_jobs(
        "candidate-1",
        authorization=f"Bearer {server._issue_candidate_session_token('candidate-1')}",
    )

    assert [item["title"] for item in result] == ["Python Backend Engineer"]
    assert result[0]["job_id"] == "job-relevant"
    assert "job-invalid-url" not in {item["job_id"] for item in result}


@pytest.mark.asyncio
async def test_chat_route_uses_real_relevance_and_injects_only_eligible_job(monkeypatch):
    """Mocked route-level test; not PostgreSQL integration coverage."""
    jobs = _jobs()
    sessions: list[_RecommendationSession] = []
    _patch_common(monkeypatch, jobs, sessions)
    prompts = []

    async def empty(*_args, **_kwargs):
        return []

    async def save(*_args, **_kwargs):
        return None

    async def complete(**kwargs):
        prompts.append("\n".join(message["content"] for message in kwargs["messages"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Matches"))]
        )

    monkeypatch.setattr(server, "_load_intake_ledger", empty)
    monkeypatch.setattr(server, "_load_chat_window", empty)
    monkeypatch.setattr(server, "_save_chat_window", save)
    monkeypatch.setattr(server, "_chat_profile_preflight", lambda *_args: None)
    monkeypatch.setattr(server, "_build_authoritative_missing_questions", empty)
    monkeypatch.setattr(server, "_advance_voice_intake_from_chat", save)
    monkeypatch.setattr(server, "_record_demonstrated_skill_usage", save)
    monkeypatch.setattr(
        server,
        "openai_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete))),
    )

    await server.chat(
        server.ChatRequest(
            candidate_id="candidate-1",
            session_id="session-1",
            messages=[server.ChatMessageIn(role="user", content="Find me jobs")],
        )
    )

    chat_session = next(
        session for session in reversed(sessions) if session.final_eligible_ids is not None
    )
    chat_ids = chat_session.final_eligible_ids
    assert chat_ids is not None
    assert "Python Backend Engineer" in prompts[0]
    assert "Product Manager" not in prompts[0]
    assert "Python Backend Engineer Intern" not in prompts[0]
    assert "No matching jobs" not in prompts[0]
    assert "missing-java" not in chat_ids
    assert "invalid-url" not in chat_ids


@pytest.mark.asyncio
async def test_api_and_chat_route_eligibility_ids_match(monkeypatch):
    """The same mocked stored dataset yields the same real-helper eligibility."""
    jobs = _jobs()
    sessions: list[_RecommendationSession] = []
    _patch_common(monkeypatch, jobs, sessions)

    api_result = await server.get_candidate_jobs(
        "candidate-1",
        authorization=f"Bearer {server._issue_candidate_session_token('candidate-1')}",
    )
    api_ids = {item["job_id"] for item in api_result}

    captured = []

    async def complete(**kwargs):
        captured.append("\n".join(message["content"] for message in kwargs["messages"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Matches"))]
        )

    async def empty(*_args, **_kwargs):
        return []

    monkeypatch.setattr(server, "_load_intake_ledger", empty)
    monkeypatch.setattr(server, "_load_chat_window", empty)
    monkeypatch.setattr(server, "_save_chat_window", empty)
    monkeypatch.setattr(server, "_chat_profile_preflight", lambda *_args: None)
    monkeypatch.setattr(server, "_build_authoritative_missing_questions", empty)
    monkeypatch.setattr(server, "_advance_voice_intake_from_chat", empty)
    monkeypatch.setattr(server, "_record_demonstrated_skill_usage", empty)
    monkeypatch.setattr(
        server,
        "openai_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete))),
    )

    await server.chat(
        server.ChatRequest(
            candidate_id="candidate-1",
            session_id="session-1",
            messages=[server.ChatMessageIn(role="user", content="Find me jobs")],
        )
    )

    chat_session = next(
        session for session in reversed(sessions) if session.final_eligible_ids is not None
    )
    assert chat_session.final_eligible_ids is not None
    chat_ids = {f"job-{recommendation_id}" for recommendation_id in chat_session.final_eligible_ids}
    assert api_ids == chat_ids
    assert api_ids == {"job-relevant"}
    assert chat_ids == {"job-relevant"}
    assert "Python Backend Engineer" in captured[0]
