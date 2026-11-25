import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import candidate_job_matching_service as matcher  # noqa: E402
from qdrant_service import RetrievalDiagnostics, SearchResults  # noqa: E402

VOICE_INTAKE_COMPLETED = matcher.voice_intake_completed


def _capture_value(target: dict[str, Any], key: str, value: Any, result: Any) -> Any:
    target[key] = value
    return result


def test_required_skill_extraction_ignores_responsibility_prose_and_headings():
    jd = """
    Essential Key Responsibilities
    Be a trusted advisor to clients, all while making a difference.
    Manage risk across the business.

    Required Skills
    - Python
    - SQL, FastAPI
    - Stakeholder communication
    """

    extracted = matcher._extract_job_required_skills(jd)

    assert extracted == ["Python", "SQL", "FastAPI", "Stakeholder communication"]
    assert "Essential Key Responsibilities" not in extracted
    assert "Be a trusted advisor" not in extracted
    assert "manage risk" not in [skill.lower() for skill in extracted]


def test_fresh_semantic_target_role_survives_final_filtering(monkeypatch):
    state: dict[str, Any] = {
        "jobs": {
            "api-job": {
                "title": "API Engineer",
                "description": "Build Python REST APIs with FastAPI and PostgreSQL.",
                "requirements": "",
                "skills": ["Python", "FastAPI", "PostgreSQL"],
                "skills_required": ["Python"],
            }
        },
        "candidate": {
            "current_role": "Python Developer",
            "skills": ["Python", "FastAPI", "PostgreSQL"],
            "raw_data": {
                "preferred_roles": ["Python Backend Developer"],
                "voice_intake": {"status": "completed"},
            },
        },
    }
    metadata = {
        "api-job": {
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_version": "v2_structured",
        }
    }
    diagnostics = RetrievalDiagnostics(1, 1, 0, 1, 0)
    monkeypatch.setattr(matcher, "build_candidate_text", lambda _: "candidate profile")
    monkeypatch.setattr(
        matcher,
        "generate_embedding",
        lambda text: [0.2] if text.startswith("Target job roles:") else [0.1],
    )
    monkeypatch.setattr(
        matcher,
        "search_job_chunks",
        lambda vector, limit, country_code=None: SearchResults(
            [("api-job", 0.41)] if vector == [0.1] else [("api-job", 0.82)],
            diagnostics,
            metadata,
        ),
    )
    monkeypatch.setattr(matcher, "_get_candidate_intelligence", lambda _: {})

    asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate",
            {
                "skills": ["Python", "FastAPI", "PostgreSQL"],
                "raw_data": {"preferred_roles": ["Python Backend Developer"]},
            },
            _typed_session_factory(state),
        )
    )

    assert state["inserted"] == ["api-job"]
    reason = state["match_reasons"]["api-job"]
    reason = json.loads(reason) if isinstance(reason, str) else reason
    assert reason["target_role_score"] > 0


def test_required_skill_extraction_returns_empty_without_structured_skills():
    jd = "We are a trusted advisor to customers. You will manage risk and build a team."

    assert matcher._extract_job_required_skills(jd) == []


def test_required_skill_extraction_reads_inline_competencies_list():
    jd = "About the role.\nCompetencies: Python, Django, ETL, GCP and Kubernetes"

    assert matcher._extract_job_required_skills(jd) == [
        "Python",
        "Django",
        "ETL",
        "GCP",
        "Kubernetes",
    ]


def test_required_skill_extraction_ignores_generic_requires_prose():
    assert matcher._extract_job_required_skills(
        "The role requires collaboration with backend teams."
    ) == []
    assert matcher._extract_job_required_skills(
        "Requirements: Python, FastAPI, PostgreSQL"
    ) == ["Python", "FastAPI", "PostgreSQL"]
    assert matcher._extract_job_required_skills(
        "Must have experience with Java and Spring Boot."
    ) == ["Java", "Spring Boot"]
    assert matcher._extract_job_required_skills(
        "Nice to have: Docker and Kubernetes."
    ) == []


def test_adding_a_missing_required_skill_increases_skill_match_score():
    intelligence = {"evidence": {"skills": {"evidence_level": 2}}}
    required = ["Python", "Docker", "Kubernetes"]

    before = matcher._evidence_weighted_skills_score(
        ["Python"], "Python Docker Kubernetes", intelligence, required
    )
    after = matcher._evidence_weighted_skills_score(
        ["Python", "Docker"], "Python Docker Kubernetes", intelligence, required
    )

    assert after > before


def test_hybrid_match_increases_when_displayed_inline_jd_skill_is_added():
    job_description = "About the role.\nCompetencies: Python, Django, ETL"
    intelligence = {"evidence": {"skills": {"evidence_level": 2}}}
    before_signals = {
        "target_roles": [],
        "skills": ["Python"],
        "past_roles": [],
        "total_experience_years": 0,
    }
    after_signals = {**before_signals, "skills": ["Python", "Django"]}

    before, _ = matcher._hybrid_score(
        before_signals,
        "Backend Engineer",
        job_description,
        "",
        [],
        0.5,
        intelligence=intelligence,
    )
    after, _ = matcher._hybrid_score(
        after_signals,
        "Backend Engineer",
        job_description,
        "",
        [],
        0.5,
        intelligence=intelligence,
    )

    assert after > before


def test_single_job_refresh_reuses_full_job_metadata_and_recomputes_semantic_score(
    monkeypatch,
):
    selected = {
        "match_reason": {"semantic_score": 0.48},
        "title": "Backend Engineer",
        "description": "Build Python APIs",
        "requirements": "Required skills: Python",
        "skills": ["Python"],
        "skills_required": ["Docker"],
        "experience_required": "3+ years",
        "location": "Hyderabad",
        "employment_type": "Full-time",
        "remote": False,
        "remote_policy": "Hybrid",
        "industry": "Software",
        "salary_range": "15 LPA",
        "city": "Hyderabad",
        "state": "Telangana",
        "country": "India",
        "structured_data": {"company_type": "startup"},
    }
    captured = {}

    class Result:
        def mappings(self):
            return self

        def fetchone(self):
            return selected

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, statement, params=None):
            if "SELECT cjr.match_reason" in str(statement):
                return Result()
            captured["write"] = params
            return Result()

        async def commit(self):
            pass

    def score(signals, title, description, requirements, skills, semantic, **kwargs):
        captured.update(
            {
                "requirements": requirements,
                "skills": skills,
                "semantic": semantic,
                "job_metadata": kwargs["job_metadata"],
            }
        )
        return 0.64, {"final_score": 0.64, "semantic_score": semantic}

    monkeypatch.setattr(matcher, "_hybrid_score", score)
    monkeypatch.setattr(matcher, "_get_candidate_intelligence", lambda candidate: {})
    monkeypatch.setattr(
        matcher,
        "_semantic_similarity_score",
        lambda _candidate_text, job_text: _capture_value(
            captured, "similarity_job_text", job_text, 0.73
        ),
    )
    monkeypatch.setattr(matcher, "build_job_text", lambda job: "CANONICAL JOB TEXT")
    monkeypatch.setattr(
        matcher,
        "_semantic_role_score",
        lambda _roles, job_text, _query: _capture_value(
            captured, "role_job_text", job_text, 0.61
        ),
    )

    asyncio.run(
        matcher.refresh_candidate_job_match(
            "candidate-1",
            "rec-1",
            {
                "skills": ["Python", "Docker"],
                "experience_years": 4,
                "raw_data": {"preferred_roles": ["Backend Engineer"]},
            },
            cast(async_sessionmaker[Any], lambda: Session()),
        )
    )

    assert captured["semantic"] == 0.73
    assert captured["role_job_text"] == "CANONICAL JOB TEXT"
    assert captured["similarity_job_text"] == "CANONICAL JOB TEXT"
    assert captured["skills"] == ["Python", "Docker"]
    assert "3+ years" in captured["requirements"]
    assert captured["job_metadata"]["location"] == "Hyderabad"
    assert captured["job_metadata"]["employment_type"] == "Full-time"


@pytest.mark.asyncio
async def test_stale_job_timestamp_rescores_stored_recommendation(monkeypatch):
    selected = {
        "match_reason": {"semantic_score": 0.48},
        "title": "Updated Backend Engineer",
        "description": "Build Python APIs",
        "requirements": "Required skills: Python",
        "skills": ["Python", "FastAPI"],
        "skills_required": ["FastAPI"],
        "experience_required": "3+ years",
        "location": "Hyderabad",
        "employment_type": "Full-time",
        "remote": False,
        "remote_policy": "Hybrid",
        "industry": "Software",
        "salary_range": "15 LPA",
        "city": "Hyderabad",
        "state": "Telangana",
        "country": "India",
        "structured_data": {},
    }
    captured = {}

    def score(
        _signals, title, _description, _requirements, skills, _semantic, **_kwargs
    ):
        captured["title"] = title
        captured["skills"] = skills
        return 0.64, {"final_score": 0.64}

    monkeypatch.setattr(matcher, "_hybrid_score", score)

    class Result:
        def mappings(self):
            return self

        def fetchone(self):
            return selected

        def fetchall(self):
            return [
                {
                    "id": "rec-1",
                    "generated_at": datetime(2026, 1, 1, tzinfo=UTC),
                    "updated_at": datetime(2026, 1, 2, tzinfo=UTC),
                }
            ]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, *_args, **_kwargs):
            statement = str(_args[0])
            if "SELECT cjr.id, cjr.generated_at, jd.updated_at" in statement:
                return Result()
            if "SELECT cjr.match_reason" in statement:
                return Result()
            captured["write"] = _kwargs.get("params") or (
                _args[1] if len(_args) > 1 else {}
            )
            return Result()

        async def commit(self):
            pass

    await matcher.refresh_stale_candidate_job_scores(
        "candidate-1",
        {
            "skills": ["Python", "FastAPI"],
            "experience_years": 4,
            "raw_data": {"preferred_roles": ["Backend Engineer"]},
        },
        cast(async_sessionmaker[Any], lambda: Session()),
    )

    assert captured["write"]["rid"] == "rec-1"
    assert captured["title"] == "Updated Backend Engineer"
    assert captured["skills"] == ["Python", "FastAPI"]
    assert captured["write"]["score"] == 0.64


@pytest.mark.asyncio
async def test_stale_job_update_hides_recommendation_that_becomes_ineligible():
    selected = {
        "match_reason": {"semantic_score": 0.9},
        "title": "Backend Engineer",
        "description": "Build Python APIs",
        "requirements": "Required skills: Kubernetes",
        "skills": ["Python"],
        "skills_required": ["Kubernetes"],
        "experience_required": "2+ years",
        "location": "Hyderabad",
        "employment_type": "Full-time",
        "remote": False,
        "remote_policy": "Hybrid",
        "industry": "Software",
        "salary_range": "15 LPA",
        "city": "Hyderabad",
        "state": "Telangana",
        "country": "India",
        "structured_data": {},
    }
    writes = []

    class Result:
        def mappings(self):
            return self

        def fetchone(self):
            return selected

        def fetchall(self):
            return [
                {
                    "id": "rec-1",
                    "generated_at": datetime(2026, 1, 1, tzinfo=UTC),
                    "updated_at": datetime(2026, 1, 2, tzinfo=UTC),
                }
            ]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, statement, params=None):
            sql = str(statement)
            if "SELECT cjr.id, cjr.generated_at, jd.updated_at" in sql:
                return Result()
            if "SELECT cjr.match_reason" in sql:
                return Result()
            if "SET hidden_at" in sql:
                writes.append(params)
            return Result()

        async def commit(self):
            pass

    await matcher.refresh_stale_candidate_job_scores(
        "candidate-1",
        {
            "skills": ["Python"],
            "experience_years": 4,
            "raw_data": {"preferred_roles": ["Backend Engineer"]},
        },
        cast(async_sessionmaker[Any], lambda: Session()),
    )

    assert writes == [{"rid": "rec-1", "cid": "candidate-1"}]


@pytest.mark.asyncio
async def test_unchanged_job_timestamp_does_not_rescore_stored_recommendation(
    monkeypatch,
):
    class Result:
        def fetchall(self):
            return []

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, *_args, **_kwargs):
            return Result()

    refreshed = False

    async def refresh(*_args, **_kwargs):
        nonlocal refreshed
        refreshed = True

    monkeypatch.setattr(matcher, "refresh_candidate_job_match", refresh)
    await matcher.refresh_stale_candidate_job_scores(
        "candidate-1",
        {"skills": ["Python"]},
        cast(async_sessionmaker[Any], lambda: Session()),
    )

    assert refreshed is False


@pytest.mark.parametrize(
    ("signals", "job", "reason"),
    [
        (
            {"remote_preference": "remote only", "preferred_locations": []},
            {"remote": False, "remote_policy": "Onsite"},
            "remote_only_job_not_remote",
        ),
        (
            {"remote_preference": "remote only", "preferred_locations": []},
            {"remote": False, "remote_policy": "Hybrid"},
            "remote_only_job_not_remote",
        ),
        (
            {
                "remote_preference": "",
                "preferred_locations": ["Hyderabad"],
                "willing_to_relocate": False,
            },
            {
                "location": "Bengaluru",
                "city": "Bengaluru",
                "state": "Karnataka",
                "country": "India",
                "remote_policy": "Onsite",
            },
            "job_outside_explicit_locations",
        ),
        (
            {
                "remote_preference": "",
                "preferred_locations": ["Hyderabad"],
                "willing_to_relocate": True,
            },
            {
                "location": "Bengaluru",
                "city": "Bengaluru",
                "state": "Karnataka",
                "country": "India",
                "remote_policy": "Onsite",
            },
            None,
        ),
        (
            {
                "remote_preference": "",
                "preferred_locations": [],
                "employment_types": ["full-time"],
            },
            {"employment_type": "Contract"},
            "employment_type_mismatch",
        ),
    ],
)
def test_single_job_refresh_revalidates_current_preferences(
    monkeypatch, signals, job, reason
):
    selected = {
        "match_reason": {"semantic_score": 0.48, "old": "preserved"},
        "title": "Backend Engineer",
        "description": "Build Python APIs",
        "requirements": "",
        "skills": ["Python"],
        "skills_required": [],
        "experience_required": "3+ years",
        "location": job.get("location", "Hyderabad"),
        "employment_type": job.get("employment_type", "Full-time"),
        "remote": job.get("remote", False),
        "remote_policy": job.get("remote_policy", ""),
        "industry": "Software",
        "salary_range": "",
        "city": job.get("city", "Hyderabad"),
        "state": job.get("state", "Telangana"),
        "country": job.get("country", "India"),
        "structured_data": {},
    }
    writes = []

    class Result:
        def mappings(self):
            return self

        def fetchone(self):
            return selected

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, statement, params=None):
            sql = str(statement)
            if "SELECT cjr.match_reason" in sql:
                return Result()
            writes.append((sql, params or {}))
            return Result()

        async def commit(self):
            pass

    monkeypatch.setattr(matcher, "_build_candidate_signals", lambda _: {
        "target_roles": ["Backend Engineer"],
        "skills": ["Python"],
        "past_roles": [],
        "responsibilities": [],
        "experience_reconciliation_status": "consistent",
        **signals,
    })
    monkeypatch.setattr(
        matcher,
        "_hybrid_score",
        lambda *_args, **_kwargs: pytest.fail("ineligible recommendation was rescored"),
    ) if reason else None

    asyncio.run(
        matcher.refresh_candidate_job_match(
            "candidate-1",
            "rec-1",
            {"skills": ["Python"], "experience_years": 4, "raw_data": {}},
            cast(async_sessionmaker[Any], lambda: Session()),
        )
    )

    if reason:
        assert len(writes) == 1
        sql, params = writes[0]
        assert "SET hidden_at = COALESCE(hidden_at, NOW())" in sql
        assert params["rid"] == "rec-1"
    else:
        assert any("SET match_score = :score" in sql for sql, _ in writes)


def test_single_job_refresh_does_not_unhide_existing_hidden_recommendation(monkeypatch):
    selected = {
        "match_reason": {"semantic_score": 0.48},
        "title": "Backend Engineer",
        "description": "Build Python APIs",
        "requirements": "",
        "skills": ["Python"],
        "skills_required": [],
        "experience_required": "3+ years",
        "location": "Hyderabad",
        "employment_type": "Full-time",
        "remote": False,
        "remote_policy": "Onsite",
        "industry": "Software",
        "salary_range": "",
        "city": "Hyderabad",
        "state": "Telangana",
        "country": "India",
        "structured_data": {},
    }
    writes = []

    class Result:
        def mappings(self):
            return self

        def fetchone(self):
            return selected

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, statement, params=None):
            sql = str(statement)
            if "SELECT cjr.match_reason" in sql:
                return Result()
            writes.append((sql, params or {}))
            return Result()

        async def commit(self):
            pass

    monkeypatch.setattr(matcher, "_build_candidate_signals", lambda _: {
        "target_roles": ["Backend Engineer"],
        "skills": ["Python"],
        "past_roles": [],
        "responsibilities": [],
        "experience_reconciliation_status": "consistent",
        "remote_preference": "",
        "preferred_locations": ["Hyderabad"],
        "willing_to_relocate": False,
        "employment_types": ["full-time"],
    })
    monkeypatch.setattr(
        matcher,
        "_hybrid_score",
        lambda *_args, **_kwargs: (0.7, {"final_score": 0.7}),
    )

    asyncio.run(
        matcher.refresh_candidate_job_match(
            "candidate-1",
            "rec-1",
            {"skills": ["Python"], "experience_years": 4, "raw_data": {}},
            cast(async_sessionmaker[Any], lambda: Session()),
        )
    )

    assert any("SET match_score = :score" in sql for sql, _ in writes)
    assert all("hidden_at = NULL" not in sql for sql, _ in writes)


def test_single_job_refresh_hides_stored_recommendation_missing_required_skill(
    monkeypatch,
):
    selected = {
        "match_reason": {"semantic_score": 0.48},
        "title": "Backend Engineer",
        "description": "Build Python APIs",
        "requirements": "",
        "skills": ["Python"],
        "skills_required": ["Python", "FastAPI"],
        "experience_required": "3+ years",
        "location": "Hyderabad",
        "employment_type": "Full-time",
        "remote": False,
        "remote_policy": "Onsite",
        "industry": "Software",
        "salary_range": "",
        "city": "Hyderabad",
        "state": "Telangana",
        "country": "India",
        "structured_data": {},
    }
    writes = []

    class Result:
        def mappings(self):
            return self

        def fetchone(self):
            return selected

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, statement, params=None):
            sql = str(statement)
            if "SELECT cjr.match_reason" in sql:
                return Result()
            writes.append((sql, params or {}))
            return Result()

        async def commit(self):
            pass

    monkeypatch.setattr(
        matcher,
        "_build_candidate_signals",
        lambda _: {
            "target_roles": ["Backend Engineer"],
            "skills": ["Python"],
            "past_roles": [],
            "responsibilities": [],
            "experience_reconciliation_status": "consistent",
            "remote_preference": "",
            "preferred_locations": [],
            "willing_to_relocate": False,
            "employment_types": [],
        },
    )
    monkeypatch.setattr(
        matcher,
        "_hybrid_score",
        lambda *_args, **_kwargs: pytest.fail("ineligible recommendation was rescored"),
    )

    asyncio.run(
        matcher.refresh_candidate_job_match(
            "candidate-1",
            "rec-1",
            {"skills": ["Python"], "experience_years": 4, "raw_data": {}},
            cast(async_sessionmaker[Any], lambda: Session()),
        )
    )

    assert len(writes) == 1
    assert "SET hidden_at = COALESCE(hidden_at, NOW())" in writes[0][0]
    assert writes[0][1] == {"rid": "rec-1", "cid": "candidate-1"}


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2024, 1, 1, tzinfo=tz or UTC)


class FixedDateTime2026(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 8, 25, tzinfo=tz or UTC)


class FakeResult:
    def __init__(self, rows=None, scalar_value=None):
        self._rows = rows or []
        self._scalar_value = scalar_value
        self.rowcount = 1

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def mappings(self):
        return self

    def scalar(self):
        return self._scalar_value


class FakeSession:
    def __init__(self, state):
        self.state = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, statement, params=None):
        query = str(statement)
        params = params or {}

        if "SELECT * FROM candidates" in query:
            candidate = self.state.get("candidate")
            if candidate is None:
                candidate = {
                    "skills": [],
                    "raw_data": {"voice_intake": {"status": "completed"}},
                }
            return FakeResult([candidate])

        if "FROM candidate_preferences" in query:
            return FakeResult([self.state.get("preferences", {})])

        if "SELECT id FROM candidates" in query:
            return FakeResult([{"id": params.get("cid", "candidate")}])

        if "SELECT matching_profile_revision FROM candidates" in query:
            return FakeResult(
                [], scalar_value=self.state.get("matching_profile_revision", 0)
            )

        if "SELECT generation, owner_token, lease_expires_at" in query:
            return FakeResult(
                [
                    {
                        "generation": 1,
                        "owner_token": self.state.get(
                            "lease_owner_token", params.get("owner_token", "owner")
                        ),
                        "lease_expires_at": datetime.now(UTC) + timedelta(seconds=300),
                    }
                ]
            )

        if "INSERT INTO candidate_recommendation_refresh_leases" in query:
            self.state["lease_owner_token"] = params["owner_token"]
            return FakeResult(
                [
                    {
                        "generation": 1,
                        "owner_token": params["owner_token"],
                        "lease_expires_at": datetime.now(UTC) + timedelta(seconds=300),
                    }
                ]
            )

        if "candidate_recommendation_refresh_leases" in query:
            return FakeResult([])

        if "FROM job_descriptions" in query:
            job_ids = [params[key] for key in sorted(params) if key.startswith("jid_")]
            rows = []
            for job_id in job_ids:
                job = self.state["jobs"].get(job_id)
                if job and job.get("visible", True):
                    rows.append(
                        (
                            job_id,
                            job["title"],
                            job["description"],
                            job.get("requirements", ""),
                            job.get("skills", []),
                            job.get("company_name", ""),
                            job.get("department", ""),
                            job.get("location", ""),
                            job.get("employment_type", ""),
                            job.get("experience_required", ""),
                            job.get("salary_range", ""),
                            job.get("city", ""),
                            job.get("state", ""),
                            job.get("country", ""),
                            job.get("remote"),
                            job.get("industry", ""),
                            job.get("skills_required", []),
                            job.get("structured_data", {}),
                            job.get("remote_policy", ""),
                        )
                    )
            return FakeResult(rows)

        if "FROM candidate_job_recommendations" in query:
            rows = self.state.get("existing", [])
            return FakeResult(rows)

        if "SET hidden_at" in query:
            self.state.setdefault("hidden_updates", []).append(
                list(params.get("job_ids") or [])
            )
            return FakeResult([])

        if (
            "UPDATE candidate_job_recommendations" in query
            and "SET match_score" in query
        ):
            self.state.setdefault("updated", []).append(params["rid"])
            self.state.setdefault("upserted", []).append(params["rid"])
            return FakeResult([])

        if "INSERT INTO candidate_job_recommendations" in query:
            self.state.setdefault("inserted", []).append(params["jid"])
            self.state.setdefault("match_scores", {})[params["jid"]] = params["score"]
            self.state.setdefault("match_reasons", {})[params["jid"]] = params[
                "match_reason"
            ]
            return FakeResult([])

        if (
            "UPDATE candidates" in query
            and "last_full_recommendation_refresh_at" in query
        ):
            self.state["last_full_recommendation_refresh_at"] = "updated"
            return FakeResult([])

        raise AssertionError(f"Unexpected query: {query}")

    async def commit(self):
        self.state["commits"] = self.state.get("commits", 0) + 1

    async def rollback(self):
        self.state["rollbacks"] = self.state.get("rollbacks", 0) + 1


class FakeSessionFactory:
    def __init__(self, state):
        self.state = state

    def __call__(self):
        return FakeSession(self.state)


def _typed_session_factory(state: dict[str, Any]) -> async_sessionmaker[Any]:
    return cast(async_sessionmaker[Any], FakeSessionFactory(state))


def test_empty_qdrant_retrieval_commits_candidate_freshness_marker(monkeypatch):
    state: dict[str, Any] = {"jobs": {}}
    state["candidate"] = {
        "skills": ["Java"],
        "raw_data": {"voice_intake": {"status": "completed"}},
    }
    monkeypatch.setattr(matcher, "build_candidate_text", lambda _: "Java engineer")
    monkeypatch.setattr(matcher, "generate_embedding", lambda _: [0.1])
    monkeypatch.setattr(
        matcher,
        "search_job_chunks",
        lambda vector, limit, country_code=None: SearchResults(
            [], RetrievalDiagnostics(0, 0, 0, 0, 0), {}
        ),
    )

    completed = asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate",
            {"skills": ["Java"], "raw_data": {"voice_intake": {"status": "completed"}}},
            _typed_session_factory(state),
        )
    )

    assert completed is True
    assert state["last_full_recommendation_refresh_at"] == "updated"


def test_qdrant_ids_without_usable_database_jobs_do_not_mark_fresh(monkeypatch, caplog):
    state: dict[str, Any] = {"jobs": {}}
    state["candidate"] = {
        "skills": ["Java"],
        "raw_data": {"voice_intake": {"status": "completed"}},
    }
    monkeypatch.setattr(matcher, "build_candidate_text", lambda _: "Java engineer")
    monkeypatch.setattr(matcher, "generate_embedding", lambda _: [0.1])
    monkeypatch.setattr(
        matcher,
        "search_job_chunks",
        lambda vector, limit, country_code=None: [("missing-job", 0.9)],
    )

    with caplog.at_level("WARNING"):
        completed = asyncio.run(
            matcher.refresh_candidate_job_matches(
                "candidate",
                {
                    "skills": ["Java"],
                    "raw_data": {"voice_intake": {"status": "completed"}},
                },
                _typed_session_factory(state),
            )
        )

    assert completed is False
    assert "last_full_recommendation_refresh_at" not in state
    assert "no usable database jobs" in caplog.text


def test_incomplete_refresh_remains_retryable_for_subsequent_access(monkeypatch):
    state: dict[str, Any] = {"jobs": {}}
    state["candidate"] = {
        "skills": ["Java"],
        "raw_data": {"voice_intake": {"status": "completed"}},
    }
    refresh_attempts = []
    monkeypatch.setattr(matcher, "build_candidate_text", lambda _: "Java engineer")
    monkeypatch.setattr(matcher, "generate_embedding", lambda _: [0.1])

    def retrieve(vector, limit, country_code=None):
        refresh_attempts.append(1)
        return [("missing-job", 0.9)]

    monkeypatch.setattr(matcher, "search_job_chunks", retrieve)
    candidate = {
        "skills": ["Java"],
        "raw_data": {"voice_intake": {"status": "completed"}},
    }

    first = asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate", candidate, _typed_session_factory(state)
        )
    )
    second = asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate", candidate, _typed_session_factory(state)
        )
    )

    assert first is False
    assert second is False
    assert len(refresh_attempts) == 2
    assert "last_full_recommendation_refresh_at" not in state


@pytest.fixture(autouse=True)
def fixed_datetime(monkeypatch):
    monkeypatch.setattr(matcher, "datetime", FixedDateTime)
    monkeypatch.setattr(matcher, "timezone", timezone)
    # Existing matching tests exercise completed profiles; completion itself is
    # covered explicitly below.
    monkeypatch.setattr(matcher, "voice_intake_completed", lambda candidate: True)
    # This legacy suite isolates ranking/persistence behavior. Country-boundary
    # behavior is covered independently in test_matching_deterministic_matrix.
    monkeypatch.setattr(matcher, "country_eligible", lambda candidate, job: True)


def test_candidate_total_experience_years_merges_overlaps_and_counts_present():
    candidate: dict[str, Any] = {
        "work_experience": [
            {"start_date": "2020-01-01", "end_date": "2021-01-01"},
            {"start_date": "2020-06-01", "end_date": "2022-01-01"},
            {"start_date": "2022-01-01", "end_date": "Present"},
        ]
    }

    years = matcher._candidate_total_experience_years(candidate)

    assert years == pytest.approx(4.0, rel=0.01)


def test_candidate_total_experience_years_handles_historical_and_present_month_ranges(
    monkeypatch,
):
    monkeypatch.setattr(matcher, "datetime", FixedDateTime2026)
    monkeypatch.setattr(matcher, "timezone", timezone)

    candidate: dict[str, Any] = {
        "experience_years": 0.6,
        "work_experience": [
            {
                "company": "Deepija Telecom",
                "title": "Engineer",
                "dates": "01-11-2023 — 09-10-2024",
            },
            {
                "company": "Viral Bug",
                "title": "Engineer",
                "dates": "Aug 2025 — Present",
            },
        ],
    }

    years = matcher._candidate_total_experience_years(candidate)

    assert years == pytest.approx(2.00, rel=0.02)


def test_candidate_total_experience_years_sums_non_overlapping_jobs(monkeypatch):
    monkeypatch.setattr(matcher, "datetime", FixedDateTime)
    monkeypatch.setattr(matcher, "timezone", timezone)

    candidate: dict[str, Any] = {
        "work_experience": [
            {"start_date": "2018-01-01", "end_date": "2019-01-01"},
            {"start_date": "2020-01-01", "end_date": "2021-01-01"},
            {"start_date": "2022-01-01", "end_date": "2023-01-01"},
        ],
    }

    years = matcher._candidate_total_experience_years(candidate)

    assert years == pytest.approx(3.0, rel=0.01)


def test_candidate_total_experience_years_does_not_double_count_overlaps(monkeypatch):
    monkeypatch.setattr(matcher, "datetime", FixedDateTime)
    monkeypatch.setattr(matcher, "timezone", timezone)

    candidate = {
        "work_experience": [
            {"start_date": "2020-01-01", "end_date": "2021-01-01"},
            {"start_date": "2020-06-01", "end_date": "2022-01-01"},
            {"start_date": "2021-12-01", "end_date": "2023-01-01"},
        ],
    }

    years = matcher._candidate_total_experience_years(candidate)

    assert years == pytest.approx(3.0, rel=0.01)


def test_candidate_total_experience_years_counts_open_ended_employment_to_today(
    monkeypatch,
):
    monkeypatch.setattr(matcher, "datetime", FixedDateTime2026)
    monkeypatch.setattr(matcher, "timezone", timezone)

    candidate = {
        "work_experience": [
            {"start_date": "2025-08-01", "end_date": "Present"},
        ],
    }

    years = matcher._candidate_total_experience_years(candidate)

    assert years == pytest.approx(1.07, rel=0.02)


def test_candidate_total_experience_years_ignores_invalid_dates_safely(monkeypatch):
    monkeypatch.setattr(matcher, "datetime", FixedDateTime)
    monkeypatch.setattr(matcher, "timezone", timezone)

    candidate = {
        "work_experience": [
            {"start_date": "not-a-date", "end_date": "also-bad"},
            {"start_date": "", "end_date": ""},
            {"dates": "??"},
        ],
    }

    years = matcher._candidate_total_experience_years(candidate)

    assert years is None


def test_invalid_nonempty_history_uses_validated_reported_fallback():
    candidate = {
        "experience_years": 2.8,
        "work_experience": [{"title": "Backend Developer", "duration": "2 years"}],
    }
    result = matcher._candidate_experience_reconciliation(candidate)

    assert result == {
        "reported_experience_years": 2.8,
        "calculated_experience_years": None,
        "reconciled_experience_years": 2.8,
        "experience_reconciliation_status": "reported_fallback",
    }


def test_experience_reconciliation_supports_aliases_and_current_employment(monkeypatch):
    monkeypatch.setattr(matcher, "datetime", FixedDateTime2026)
    monkeypatch.setattr(matcher, "timezone", timezone)
    candidate = {
        "work_experience": [
            {"from": "2020-01-01", "to": "2021-01-01"},
            {"start": "2021-01-01", "end": "Present"},
        ]
    }

    result = matcher._candidate_experience_reconciliation(candidate)

    assert result["experience_reconciliation_status"] == "calculated"
    assert result["reconciled_experience_years"] == pytest.approx(6.65, rel=0.01)


def test_experience_reconciliation_preserves_conflicting_reported_value(monkeypatch):
    candidate = {
        "experience_years": 5,
        "work_experience": [
            {"start_date": "2020-01-01", "end_date": "2021-01-01"}
        ],
    }

    result = matcher._candidate_experience_reconciliation(candidate)

    assert result["reported_experience_years"] == 5
    assert result["calculated_experience_years"] == pytest.approx(1.0, rel=0.01)
    assert result["reconciled_experience_years"] == pytest.approx(1.0, rel=0.01)
    assert result["experience_reconciliation_status"] == "conflict"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1, 81])
def test_invalid_reported_experience_is_unknown_without_valid_history(value):
    assert matcher._candidate_total_experience_years(
        {"experience_years": value, "work_experience": []}
    ) is None


def test_matching_signals_expose_reconciliation_provenance():
    signals = matcher._build_candidate_signals(
        {"experience_years": 2.8, "work_experience": [{"duration": "2 years"}]}
    )

    assert signals["total_experience_years"] == 2.8
    assert signals["reported_experience_years"] == 2.8
    assert signals["calculated_experience_years"] is None
    assert signals["experience_reconciliation_status"] == "reported_fallback"


def test_malformed_work_history_does_not_crash_signal_construction():
    signals = matcher._build_candidate_signals(
        {
            "experience_years": 3,
            "work_experience": [
                "invalid",
                42,
                None,
                {
                    "title": "Backend Engineer",
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
            ],
        }
    )

    assert signals["past_roles"] == ["Backend Engineer"]
    assert signals["reconciled_experience_years"] == pytest.approx(1.0, rel=0.01)


def test_dictionary_shaped_work_history_is_unknown_or_reported_fallback():
    signals = matcher._build_candidate_signals(
        {"experience_years": 2.8, "work_experience": {"title": "Engineer"}}
    )

    assert signals["past_roles"] == []
    assert signals["reconciled_experience_years"] == 2.8
    assert signals["experience_reconciliation_status"] == "reported_fallback"


@pytest.mark.parametrize(
    ("start_key", "end_key"),
    [
        ("start_date", "end_date"),
        ("startDate", "endDate"),
        ("start", "end"),
        ("from", "to"),
    ],
)
def test_experience_date_aliases_are_supported(start_key, end_key):
    result = matcher._candidate_experience_reconciliation(
        {"work_experience": [{start_key: "2020-01-01", end_key: "2021-01-01"}]}
    )

    assert result["reconciled_experience_years"] == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("end_value", ["Present", "Current", "Ongoing", "Now"])
def test_open_ended_end_markers_are_supported(monkeypatch, end_value):
    monkeypatch.setattr(matcher, "datetime", FixedDateTime2026)
    monkeypatch.setattr(matcher, "timezone", timezone)
    result = matcher._candidate_experience_reconciliation(
        {"work_experience": [{"start_date": "2025-08-01", "end_date": end_value}]}
    )

    assert result["reconciled_experience_years"] == pytest.approx(1.07, rel=0.02)


@pytest.mark.parametrize("field", ["is_current", "current"])
def test_boolean_current_fields_are_supported(monkeypatch, field):
    monkeypatch.setattr(matcher, "datetime", FixedDateTime2026)
    monkeypatch.setattr(matcher, "timezone", timezone)
    result = matcher._candidate_experience_reconciliation(
        {"work_experience": [{"start_date": "2025-08-01", field: True}]}
    )

    assert result["reconciled_experience_years"] == pytest.approx(1.07, rel=0.02)


def test_total_experience_fallback_and_unknown_malformed_history():
    assert matcher._candidate_total_experience_years(
        {
            "total_experience_years": 4.2,
            "experience_years": None,
            "work_experience": [42],
        }
    ) == 4.2
    assert matcher._candidate_total_experience_years({"work_experience": [42]}) is None


@pytest.mark.parametrize(
    ("candidate_years", "status"),
    [
        (None, "unknown"),
        (2.8, "reported_fallback"),
        (4.0, "calculated"),
        (4.0, "conflict"),
    ],
)
def test_experience_eligibility_preserves_reconciliation_status(
    candidate_years, status
):
    result = matcher.experience_eligibility(
        candidate_years, "", "2+ years", reconciliation_status=status
    )

    assert result["experience_reconciliation_status"] == status
    assert result["eligible"] is True


def test_hybrid_components_and_explanation_preserve_reconciliation_provenance():
    signals = matcher._build_candidate_signals(
        {
            "experience_years": 5,
            "work_experience": [
                {"start_date": "2020-01-01", "end_date": "2021-01-01"}
            ],
        }
    )
    score, components = matcher._hybrid_score(
        signals,
        "Backend Engineer",
        "Build APIs",
        "",
        [],
        0.8,
    )
    diagnostics = matcher._compute_job_specific_confidence(
        signals, "Backend Engineer", "Build APIs", score, components, None
    )

    assert components["experience_reconciliation_status"] == "conflict"
    assert components["reconciled_experience_years"] == pytest.approx(1.0, rel=0.01)
    assert diagnostics["match_explanation"]["experience_reconciliation"][
        "reported_experience_years"
    ] == 5


def test_job_eligibility_filters_experience_and_requires_relevance():
    signals = {
        "skills": ["Java", "Spring Boot"],
        "target_roles": ["Java Backend Engineer"],
    }

    assert matcher._job_is_eligible(
        signals,
        3.0,
        "Java Backend Engineer",
        "Build APIs with Java and Spring Boot",
        "2-3 years of experience",
        ["Java", "Spring Boot"],
    )

    assert not matcher._job_is_eligible(
        signals,
        3.0,
        "Senior Python Engineer",
        "Build ML systems with Python",
        "4+ years of experience",
        ["Python", "ML"],
    )

    assert not matcher._job_is_eligible(
        signals,
        3.0,
        "Content Writer",
        "Create blog content and edit copy",
        "4+ years of experience",
        ["SEO", "Writing"],
    )


def test_explicit_preferred_roles_prioritize_a_transition_over_current_role(
    monkeypatch,
):
    """A Python developer seeking Java must receive Java roles, not Python roles.

    This mirrors candidate f5f8859b-575e-4a68-8b80-100881011bbe: a broad
    Backend Developer preference accompanies the more specific Java target.
    """
    state: dict[str, Any] = {
        "candidate": {
            "current_role": "Python Developer",
            "skills": ["Python", "Django", "Java", "Spring Boot"],
            "raw_data": {
                "preferred_roles": ["Java Backend Developer"],
                "voice_intake": {"status": "completed"},
            },
        },
        "jobs": {
            "python-job": {
                "title": "Python Developer",
                "description": "Build Python APIs with Django.",
                "skills": ["Python", "Django"],
            },
            "java-job": {
                "title": "Java Backend Developer",
                "description": "Build backend APIs with Java and Spring Boot.",
                "skills": ["Java", "Spring Boot"],
            },
        }
    }
    monkeypatch.setattr(
        matcher, "build_candidate_text", lambda candidate: "Java Backend Developer"
    )
    monkeypatch.setattr(matcher, "generate_embedding", lambda text: [0.1])
    monkeypatch.setattr(
        matcher,
        "search_job_chunks",
        lambda vector, limit, country_code=None: [
            ("python-job", 0.99),
            ("java-job", 0.70),
        ],
    )
    monkeypatch.setattr(matcher, "_get_candidate_intelligence", lambda candidate: {})

    asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate",
            {
                "current_role": "Python Developer",
                "skills": ["Python", "Django", "Java", "Spring Boot"],
                "raw_data": {
                    "preferred_roles": [
                        "Java Developer",
                        "Java Backend Developer",
                        "Backend Developer",
                    ]
                },
            },
            _typed_session_factory(state),
        )
    )

    assert state["inserted"] == ["java-job"]


def test_preferred_role_retrieval_adds_java_job_for_python_candidate(monkeypatch):
    """Preferred-role retrieval must reach ranking despite Python-heavy search."""
    state: dict[str, Any] = {
        "candidate": {
            "current_role": "Python Developer",
            "skills": ["Python", "Django", "Java", "Spring Boot"],
            "raw_data": {
                "preferred_roles": ["Java Backend Developer"],
                "voice_intake": {"status": "completed"},
            },
        },
        "jobs": {
            "python-job": {
                "title": "Python Developer",
                "description": "Build Python APIs with Django.",
                "skills": ["Python", "Django"],
            },
            "java-job": {
                "title": "Java Backend Developer",
                "description": "Build backend APIs with Java and Spring Boot.",
                "skills": ["Java", "Spring Boot"],
            },
        }
    }
    embedded_texts = []

    def fake_embedding(text):
        embedded_texts.append(text)
        return [0.2] if text.startswith("Target job roles:") else [0.1]

    def fake_search(vector, limit, country_code=None):
        assert limit == matcher.QDRANT_TOP_K
        return [("java-job", 0.80)] if vector == [0.2] else [("python-job", 0.99)]

    monkeypatch.setattr(
        matcher,
        "build_candidate_text",
        lambda candidate: "Python Developer Python Django",
    )
    monkeypatch.setattr(matcher, "generate_embedding", fake_embedding)
    monkeypatch.setattr(matcher, "search_job_chunks", fake_search)
    monkeypatch.setattr(matcher, "_get_candidate_intelligence", lambda candidate: {})

    asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate",
            {
                "current_role": "Python Developer",
                "skills": ["Python", "Django", "Java", "Spring Boot"],
                "raw_data": {"preferred_roles": ["Java Backend Developer"]},
            },
            _typed_session_factory(state),
        )
    )

    assert embedded_texts == [
        "Python Developer Python Django",
        (
            "Target job roles:\nJava Backend Developer\n\nCurrent professional role:\n"
            "Python Developer\n\nRelevant previous roles:\nPython Developer\n\n"
            "Relevant professional skills:\nPython, Django, Java, Spring Boot"
        ),
    ]
    assert state["inserted"] == ["java-job"]


def test_explicit_target_role_priority_beats_higher_python_match_score(monkeypatch):
    """Selection prioritizes target-role score without changing displayed scores."""
    state: dict[str, Any] = {
        "jobs": {
            "python-job": {
                "title": "Python Developer",
                "description": (
                    "Build Python and Django services with team collaboration."
                ),
                "skills": ["Python", "Django"],
            },
            "java-job": {
                "title": "Java Backend Developer",
                "description": "Build backend APIs with Java.",
                "skills": ["Java"],
            },
        }
    }
    candidate: dict[str, Any] = {
        "current_role": "Python Developer",
        "skills": ["Python", "Django"],
        # Team Lead intentionally causes a weak incidental role score for the
        # Python description, mirroring generic job-description wording.
        "raw_data": {
            "preferred_roles": ["Java Backend Developer", "Team Lead", "relocate"]
        },
    }
    state["candidate"] = {
        **candidate,
        "raw_data": {
            **candidate["raw_data"],
            "voice_intake": {"status": "completed"},
        },
    }
    monkeypatch.setattr(
        matcher,
        "build_candidate_text",
        lambda candidate: "Python Developer Python Django",
    )
    monkeypatch.setattr(
        matcher,
        "generate_embedding",
        lambda text: [0.2] if text.startswith("Target job roles:") else [0.1],
    )
    monkeypatch.setattr(
        matcher,
        "search_job_chunks",
        lambda vector, limit, country_code=None: [("java-job", 0.80)]
        if vector == [0.2]
        else [("python-job", 0.99), ("java-job", 0.80)],
    )
    monkeypatch.setattr(matcher, "_get_candidate_intelligence", lambda candidate: {})

    signals = matcher._build_candidate_signals(candidate)
    assert signals["target_roles"] == ["Java Backend Developer", "Team Lead"]
    python_score, _ = matcher._hybrid_score(
        signals,
        "Python Developer",
        "Build Python and Django services with team collaboration.",
        "",
        ["Python", "Django"],
        0.99,
    )
    java_score, _ = matcher._hybrid_score(
        signals,
        "Java Backend Developer",
        "Build backend APIs with Java.",
        "",
        ["Java"],
        0.80,
    )
    assert python_score > java_score

    asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate", candidate, _typed_session_factory(state)
        )
    )

    # Description-only Team Lead/Python mentions no longer make the unrelated
    # Python job an explicit-target recommendation.
    assert state["inserted"] == ["java-job"]
    assert state["match_scores"] == {"java-job": java_score}


def test_specific_java_target_outranks_python_backend_without_changing_final_scores(
    monkeypatch,
):
    """Technology-specific intent breaks the broad-role tie only at selection time."""
    state: dict[str, Any] = {
        "jobs": {
            "java-job": {
                "title": "Java Developer",
                "description": "Build services using Java.",
                "skills": ["Java"],
            },
            "python-job": {
                "title": "Python Backend Developer",
                "description": "Build backend services using Python and Django.",
                "skills": ["Python", "Django"],
            },
        }
    }
    candidate: dict[str, Any] = {
        "skills": ["Java", "Python", "Django"],
        "raw_data": {"preferred_roles": ["Java Backend Developer"]},
    }
    state["candidate"] = {
        **candidate,
        "raw_data": {
            **candidate["raw_data"],
            "voice_intake": {"status": "completed"},
        },
    }
    monkeypatch.setattr(
        matcher, "build_candidate_text", lambda candidate: "Java Backend Developer"
    )
    monkeypatch.setattr(matcher, "generate_embedding", lambda text: [0.1])
    monkeypatch.setattr(
        matcher,
        "search_job_chunks",
        lambda vector, limit, country_code=None: [
            ("python-job", 0.99),
            ("java-job", 0.80),
        ],
    )
    monkeypatch.setattr(matcher, "_get_candidate_intelligence", lambda candidate: {})

    signals = matcher._build_candidate_signals(candidate)
    java_score, java_components = matcher._hybrid_score(
        signals,
        "Java Developer",
        "Build services using Java.",
        "",
        ["Java"],
        0.80,
        semantic_role_score=0.80,
    )
    python_score, python_components = matcher._hybrid_score(
        signals,
        "Python Backend Developer",
        "Build backend services using Python and Django.",
        "",
        ["Python", "Django"],
        0.99,
    )

    # The target-role retrieval score is authoritative for the Java target;
    # Python remains a lexical-only broad-role match.
    assert (
        java_components["target_role_score"] == 0.8
        and python_components["target_role_score"] == 0.5
    )
    assert java_score > python_score
    assert java_components["final_score"] == round(java_score, 4)
    assert python_components["final_score"] == round(python_score, 4)

    # Specific technology > broad role > explicitly different technology.
    assert (
        matcher._target_role_specificity(signals["target_roles"], "Java Developer") == 2
    )
    assert (
        matcher._target_role_specificity(signals["target_roles"], "Backend Developer")
        == 1
    )
    assert (
        matcher._target_role_specificity(
            signals["target_roles"], "Python Backend Developer"
        )
        == 0
    )

    asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate", candidate, _typed_session_factory(state)
        )
    )

    assert state["inserted"] == ["java-job", "python-job"]
    assert state["match_scores"] == {
        "java-job": java_score,
        "python-job": python_score,
    }


def test_refresh_never_marks_previously_generated_recommendations_hidden(monkeypatch):
    """A refresh result is not a candidate's explicit Not-for-me action."""
    state: dict[str, Any] = {
        "candidate": {
            "skills": ["Java", "Spring Boot"],
            "current_role": "Java Backend Engineer",
            "raw_data": {"voice_intake": {"status": "completed"}},
        },
        "jobs": {
            "new-job": {
                "title": "Java Backend Engineer",
                "description": "Build APIs using Java and Spring Boot.",
                "requirements": "",
                "skills": ["Java", "Spring Boot"],
            }
        },
        # This existing recommendation is absent from the current retrieval result.
        "existing": [("previous-job", "previous-rec", None, None)],
    }
    monkeypatch.setattr(
        matcher, "build_candidate_text", lambda candidate: "Java backend engineer"
    )
    monkeypatch.setattr(matcher, "generate_embedding", lambda text: [0.1])
    monkeypatch.setattr(
        matcher,
        "search_job_chunks",
        lambda vector, limit, country_code=None: [("new-job", 0.9)],
    )
    monkeypatch.setattr(matcher, "_get_candidate_intelligence", lambda candidate: {})

    asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate",
            {
                "skills": ["Java", "Spring Boot"],
                "current_role": "Java Backend Engineer",
            },
            _typed_session_factory(state),
        )
    )

    assert state["inserted"] == ["new-job"]
    assert "hidden_updates" not in state


def test_refresh_can_recommend_more_than_two_distinct_active_eligible_jobs(monkeypatch):
    state = {
        "candidate": {
            "skills": ["Java"],
            "current_role": "Java Backend Engineer",
            "raw_data": {"voice_intake": {"status": "completed"}},
        },
        "jobs": {
            f"job-{i}": {
                "title": "Java Backend Engineer",
                "description": "Build APIs using Java and Spring Boot.",
                "requirements": "",
                "skills": ["Java", "Spring Boot"],
            }
            for i in range(4)
        }
    }
    monkeypatch.setattr(
        matcher, "build_candidate_text", lambda candidate: "Java backend engineer"
    )
    monkeypatch.setattr(matcher, "generate_embedding", lambda text: [0.1])
    monkeypatch.setattr(
        matcher,
        "search_job_chunks",
        lambda vector, limit, country_code=None: [
            (f"job-{i}", 0.9 - i / 100) for i in range(4)
        ],
    )
    monkeypatch.setattr(matcher, "_get_candidate_intelligence", lambda candidate: {})

    asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate",
            {"skills": ["Java"], "current_role": "Java Backend Engineer"},
            _typed_session_factory(state),
        )
    )

    assert state["inserted"] == ["job-0", "job-1", "job-2", "job-3"]


def test_refresh_preserves_active_database_filtering(monkeypatch):
    state = {
        "candidate": {
            "skills": ["Java"],
            "current_role": "Java Engineer",
            "raw_data": {"voice_intake": {"status": "completed"}},
        },
        "jobs": {
            "active": {
                "title": "Java Engineer",
                "description": "Java APIs",
                "skills": ["Java"],
            },
            "inactive": {
                "title": "Java Engineer",
                "description": "Java APIs",
                "skills": ["Java"],
                "visible": False,
            },
        }
    }
    monkeypatch.setattr(
        matcher, "build_candidate_text", lambda candidate: "Java engineer"
    )
    monkeypatch.setattr(matcher, "generate_embedding", lambda text: [0.1])
    monkeypatch.setattr(
        matcher,
        "search_job_chunks",
        lambda vector, limit, country_code=None: [("active", 0.9), ("inactive", 0.8)],
    )
    monkeypatch.setattr(matcher, "_get_candidate_intelligence", lambda candidate: {})

    asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate",
            {"skills": ["Java"], "current_role": "Java Engineer"},
            _typed_session_factory(state),
        )
    )

    assert state["inserted"] == ["active"]


def test_refresh_skips_resume_only_candidate_until_voice_intake_completes(monkeypatch):
    state = {
        "jobs": {
            "java-job": {
                "title": "Java Backend Developer",
                "description": "Build Java APIs.",
                "skills": ["Java"],
            }
        }
        ,
        "candidate": {"skills": ["Java"]},
    }
    monkeypatch.setattr(matcher, "voice_intake_completed", lambda candidate: False)
    monkeypatch.setattr(
        matcher, "build_candidate_text", lambda candidate: "Java Backend Developer"
    )
    monkeypatch.setattr(
        matcher,
        "generate_embedding",
        lambda text: pytest.fail("matching must not embed resume-only profiles"),
    )

    asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate",
            {"skills": ["Java"]},
            _typed_session_factory(state),
        )
    )

    assert "inserted" not in state

    monkeypatch.setattr(matcher, "voice_intake_completed", VOICE_INTAKE_COMPLETED)
    monkeypatch.setattr(matcher, "generate_embedding", lambda text: [0.1])
    monkeypatch.setattr(
        matcher,
        "search_job_chunks",
        lambda vector, limit, country_code=None: [("java-job", 0.9)],
    )
    monkeypatch.setattr(matcher, "_get_candidate_intelligence", lambda candidate: {})
    state["candidate"] = {
        "skills": ["Java"],
        "current_role": "Java Backend Developer",
        "raw_data": {"voice_intake": {"status": "completed"}},
    }
    asyncio.run(
        matcher.refresh_candidate_job_matches(
            "candidate",
            {"skills": ["Java"], "raw_data": {"voice_intake": {"status": "completed"}}},
            _typed_session_factory(state),
        )
    )

    assert state["inserted"] == ["java-job"]


def test_voice_intake_completed_reads_existing_persisted_status():
    assert VOICE_INTAKE_COMPLETED(
        {"raw_data": {"voice_intake": {"status": "completed"}}}
    )
    assert not VOICE_INTAKE_COMPLETED(
        {"raw_data": {"voice_intake": {"status": "in_progress"}}}
    )
    assert not VOICE_INTAKE_COMPLETED({"raw_data": {}})


def _candidate_preferences(**overrides):
    candidate: dict[str, Any] = {
        "experience_years": 2,
        "current_role": "Java Backend Developer",
        "skills": ["Java", "Spring Boot"],
        "raw_data": {
            "preferred_roles": ["Java Backend Developer"],
            "country": "India",
            "country_code": "IN",
        },
    }
    candidate["raw_data"].update(overrides)
    return matcher._build_candidate_signals(candidate)


def _job(**overrides):
    job = {
        "title": "Java Backend Developer",
        "description": "Java Spring Boot APIs",
        "requirements": "",
        "skills": ["Java", "Spring Boot"],
        "country": "India",
    }
    job.update(overrides)
    return job


def test_candidate_first_experience_fit_prefers_1_to_3_over_8_to_10():
    signals = _candidate_preferences()
    near = _job(requirements="1-3 years of experience")
    senior = _job(requirements="8-10 years of experience")
    assert matcher._preference_eligibility(signals, near, 2)[0]
    assert not matcher._preference_eligibility(signals, senior, 2)[0]
    near_score, near_parts = matcher._hybrid_score(
        signals,
        near["title"],
        near["description"],
        near["requirements"],
        near["skills"],
        0.5,
        job_metadata=near,
        candidate_years=2,
    )
    assert near_parts["experience_fit_score"] == 1
    assert near_score > 0


def test_target_role_and_skill_signals_beat_unrelated_role():
    signals = _candidate_preferences()
    matching = _job()
    unrelated = _job(
        title="Content Writer", description="Write editorial copy", skills=["Writing"]
    )
    a, _ = matcher._hybrid_score(
        signals,
        matching["title"],
        matching["description"],
        matching["requirements"],
        matching["skills"],
        0.4,
        job_metadata=matching,
        candidate_years=2,
    )
    b, _ = matcher._hybrid_score(
        signals,
        unrelated["title"],
        unrelated["description"],
        unrelated["requirements"],
        unrelated["skills"],
        0.9,
        job_metadata=unrelated,
        candidate_years=2,
    )
    assert a > b


def test_explicit_preferences_company_location_salary_and_missing_metadata_are_neutral():
    signals = _candidate_preferences(
        company_type_preference=["startup"],
        preferred_locations=["Hyderabad"],
        expected_salary="12 LPA",
    )
    startup = _job(
        location="Hyderabad",
        salary_range="15 LPA",
        structured_data={"company_type": "startup"},
        description="Startup: Java Spring Boot APIs",
    )
    enterprise = _job(
        location="Pune",
        salary_range="8 LPA",
        structured_data={"company_type": "enterprise"},
        description="Enterprise Java Spring Boot APIs",
    )
    startup_pref, _ = matcher._preference_score(signals, startup)
    enterprise_pref, _ = matcher._preference_score(signals, enterprise)
    assert startup_pref > enterprise_pref
    assert not matcher._preference_eligibility(signals, enterprise, 2)[
        0
    ]  # explicit location
    neutral, components = matcher._preference_score(_candidate_preferences(), _job())
    assert neutral == 0.5 and components == {"country_location": 0.5}


def test_remote_only_and_employment_type_are_explicit_eligibility_constraints():
    signals = _candidate_preferences(
        remote_preference="remote only", employment_types=["full-time"]
    )
    assert not matcher._preference_eligibility(
        signals, _job(remote_policy="Onsite", employment_type="full-time"), 2
    )[0]
    assert not matcher._preference_eligibility(
        signals,
        _job(remote=True, remote_policy="Remote", employment_type="contract"),
        2,
    )[0]
    assert matcher._preference_eligibility(
        signals,
        _job(remote=True, remote_policy="Remote", employment_type="full-time"),
        2,
    )[0]


def test_relocation_allows_other_same_country_city_but_unknown_does_not():
    preferred = _candidate_preferences(
        preferred_locations=["Hyderabad"], willing_to_relocate=True
    )
    unknown = _candidate_preferences(preferred_locations=["Hyderabad"])
    bengaluru = _job(location="Bengaluru", city="Bengaluru", country="India")
    assert matcher._preference_eligibility(preferred, bengaluru, 2)[0]
    assert not matcher._preference_eligibility(unknown, bengaluru, 2)[0]


def test_location_aliases_and_exact_city_matching_avoid_substrings():
    signals = _candidate_preferences(preferred_locations=["Bangalore"])
    alias = _job(location="Bengaluru", city="Bengaluru", country="India")
    york = _candidate_preferences(preferred_locations=["York"])
    new_york = _job(location="New York", city="New York", country="India")
    assert matcher._preference_eligibility(signals, alias, 2)[0]
    assert not matcher._preference_eligibility(york, new_york, 2)[0]


def test_explicit_policy_wins_over_conflicting_remote_boolean():
    signals = _candidate_preferences(remote_preference="remote only")
    hybrid = _job(remote=True, remote_policy="Hybrid")
    assert not matcher._preference_eligibility(signals, hybrid, 2)[0]
    assert matcher._job_work_mode(hybrid) == "hybrid"


def test_employment_type_aliases_match_but_contract_does_not():
    signals = _candidate_preferences(employment_types=["full time"])
    assert matcher._preference_eligibility(
        signals, _job(employment_type="permanent"), 2
    )[0]
    assert not matcher._preference_eligibility(
        signals, _job(employment_type="contract"), 2
    )[0]


def test_structured_data_location_is_used_when_top_level_location_is_missing():
    signals = _candidate_preferences(preferred_locations=["Hyderabad"])
    job = _job(
        location="",
        city="",
        state="",
        country="",
        structured_data={"location_details": {"city": "Hyderabad", "country": "India"}},
    )
    assert matcher._preference_eligibility(signals, job, 2)[0]
