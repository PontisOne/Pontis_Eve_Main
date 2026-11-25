"""Chat job results must recheck current job state, even for unhidden recommendations."""

import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://unused:unused@localhost/unused"
)
os.environ.setdefault("GROQ_API_KEY_1", "local-test-placeholder")

import server
from app.job_ingestion.lifecycle import candidate_visible_where


@pytest.mark.parametrize(
    "candidate_type, stored_job_type, expected_title",
    [
        ("jobs", "intern", None),
        ("intern", "jobs", None),
        ("jobs", "jobs", "Jobs match"),
        ("jobs", None, "Jobs match"),
    ],
)
@pytest.mark.asyncio
async def test_chat_filters_recommendations_by_authoritative_opportunity_type(
    monkeypatch, candidate_type, stored_job_type, expected_title
):
    """Chat executes both opportunity-type predicates before injecting results."""
    captured = []
    stale_score_queries = []
    prompts = []

    class Result:
        def __init__(self, rows=(), count=None, mapping_rows=()):
            self.rows = list(rows)
            self.count = count
            self.mapping_rows = list(mapping_rows)

        def scalar(self):
            return self.count

        def fetchall(self):
            return self.rows

        def mappings(self):
            return self

        def fetchone(self):
            return self.mapping_rows[0] if self.mapping_rows else None

        def all(self):
            return self.rows

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, statement, params=None):
            sql = str(statement)
            params = params or {}
            if "SELECT cjr.id, cjr.generated_at, jd.updated_at" in sql:
                stale_score_queries.append(params)
                return Result()
            if "FROM candidates" in sql and "last_full_recommendation_refresh_at" in sql:
                from datetime import UTC, datetime

                return Result(
                    mapping_rows=[
                        {
                            "last_full_recommendation_refresh_at": datetime.now(UTC),
                            "matching_profile_revision": 0,
                            "last_full_recommendation_revision": 0,
                        }
                    ]
                )
            if "SELECT COUNT(*) FROM candidate_job_recommendations" in sql:
                return Result(count=1)
            if "FROM candidate_job_recommendations cjr" in sql:
                assert "COALESCE(jd.opportunity_type, 'jobs') = :opportunity_type" in sql
                captured.append(params["opportunity_type"])
                eligible = (stored_job_type or "jobs") == params["opportunity_type"]
                if not eligible:
                    return Result()
                if "SELECT cjr.id, jd.title" in sql:
                    return Result(
                        rows=[
                            (
                                "rec-1",
                                expected_title,
                                "",
                                "",
                                [],
                                [],
                                "",
                                "",
                                "",
                                "",
                                "",
                                False,
                                "",
                                "",
                                "",
                                {},
                                [],
                            )
                        ]
                    )
                return Result(
                    rows=[
                        {
                            "match_score": 0.9,
                            "title": expected_title,
                            "company_name": "Example",
                            "location": "India",
                            "salary_range": "",
                            "description": "Eligible description",
                        }
                    ]
                )
            return Result()

    async def candidate(_candidate_id):
        return {"id": "candidate-1", "raw_data": {}, "opportunity_type": candidate_type}

    async def noop(*_args, **_kwargs):
        return None

    async def empty(*_args, **_kwargs):
        return []

    async def complete(**kwargs):
        prompts.append("\n".join(message["content"] for message in kwargs["messages"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Matches"))]
        )

    monkeypatch.setattr(server, "SessionLocal", Session)
    monkeypatch.setattr(server, "_authorize_candidate", lambda *_args: None)
    monkeypatch.setattr(server, "_get_candidate_row", candidate)
    monkeypatch.setattr(server, "_normalize_for_frontend", lambda _row: {})
    monkeypatch.setattr(server, "_build_profile_context", lambda _profile: ("Profile", []))
    monkeypatch.setattr(server, "_load_intake_ledger", empty)
    monkeypatch.setattr(server, "_load_chat_window", noop)
    monkeypatch.setattr(server, "_save_chat_window", noop)
    monkeypatch.setattr(server, "_chat_profile_preflight", lambda *_args: None)
    monkeypatch.setattr(server, "_voice_intake_completed_for_matching", lambda _row: True)
    async def strength(*_args):
        return 100

    monkeypatch.setattr(server, "_effective_profile_strength_percent", strength)
    monkeypatch.setattr(server, "_advance_voice_intake_from_chat", noop)
    monkeypatch.setattr(server, "_record_demonstrated_skill_usage", noop)
    monkeypatch.setattr(
        server,
        "openai_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete))),
    )

    import candidate_job_matching_service as matcher

    monkeypatch.setattr(matcher, "stored_recommendation_relevance", lambda *_args: {"eligible": True})

    await server.chat(
        server.ChatRequest(
            candidate_id="candidate-1",
            session_id="session-1",
            messages=[server.ChatMessageIn(role="user", content="Find me jobs")],
        )
    )

    assert captured == [candidate_type, candidate_type]
    assert len(stale_score_queries) == 1
    if expected_title is None:
        assert "Jobs match" not in prompts[0]
        assert "Intern match" not in prompts[0]
    else:
        assert expected_title in prompts[0]


@pytest.mark.parametrize(
    "timestamp, refresh_mode, expect_refresh, expect_jobs",
    [
        (datetime.now(UTC) - timedelta(hours=25), "success", True, True),
        (None, "success", True, True),
        (datetime.now(UTC) - timedelta(hours=1), "success", False, True),
        (datetime.now(UTC) - timedelta(hours=25), "false", True, False),
        (datetime.now(UTC) - timedelta(hours=25), "error", True, False),
    ],
)
@pytest.mark.asyncio
async def test_chat_recommendation_freshness_flow(
    monkeypatch, timestamp, refresh_mode, expect_refresh, expect_jobs
):
    jobs = [
        {
            "id": "closed",
            "title": "Closed role",
            "is_active": False,
            "status": "closed",
            "job_status": "closed",
            "job_url": "https://example.com/closed",
        },
        {
            "id": "active",
            "title": "Active role",
            "is_active": True,
            "status": "active",
            "job_status": "active",
            "job_url": "https://example.com/active",
        },
    ]
    captured_sql = []
    stale_score_sql = []
    prompts = []

    class Result:
        def __init__(self, rows=(), count=None, mapping_rows=None):
            self.rows, self.count = list(rows), count
            self.mapping_rows = list(mapping_rows or [])

        def scalar(self):
            return self.count

        def fetchall(self):
            return self.rows

        def mappings(self):
            return self

        def fetchone(self):
            return self.mapping_rows[0] if self.mapping_rows else None

        def all(self):
            return self.rows

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, statement, params=None):
            sql = str(statement)
            if "SELECT cjr.id, cjr.generated_at, jd.updated_at" in sql:
                stale_score_sql.append(sql)
                return Result()
            if "SELECT * FROM candidates WHERE id = :cid LIMIT 1" in sql:
                return Result(
                    mapping_rows=[
                        {
                            "id": "candidate-1",
                            "raw_data": {},
                            "opportunity_type": "jobs",
                        }
                    ]
                )
            if "voice_intake" in sql and "candidate_id" in sql:
                return Result()
            if "candidate_preferences" in sql:
                return Result()
            if "SELECT COUNT(*) FROM candidate_job_recommendations" in sql:
                return Result(count=2)
            if "last_full_recommendation_refresh_at" in sql:
                return Result(
                    mapping_rows=[
                        {
                            "last_full_recommendation_refresh_at": timestamp,
                            "matching_profile_revision": 0,
                            "last_full_recommendation_revision": 0,
                        }
                    ]
                )
            if "FROM candidate_job_recommendations cjr" in sql:
                captured_sql.append(sql)
                assert "cjr.hidden_at IS NULL" in sql
                assert candidate_visible_where("jd").strip() in sql
                visible = [
                    job
                    for job in jobs
                    if job["is_active"]
                    and job["status"] == "active"
                    and job["job_status"] == "active"
                    and isinstance(job["job_url"], str)
                    and job["job_url"].startswith("https://")
                ]
                if "SELECT cjr.id, jd.title" in sql:
                    return Result(
                        [
                                (
                                    job["id"],
                                    job["title"],
                                    "",
                                    "",
                                    [],
                                    [],
                                    "",
                                    "",
                                    "",
                                    "",
                                    "",
                                    False,
                                    "",
                                    "",
                                    "",
                                    {},
                                    [],
                                )
                            for job in visible
                        ]
                    )
                return Result(
                    [
                        {
                            "match_score": 0.9,
                            "title": job["title"],
                            "company_name": "Example",
                            "location": "India",
                            "salary_range": "",
                            "description": "",
                        }
                        for job in visible
                        if job["id"] in (params or {}).get("eligible_ids", [])
                    ]
                )
            raise AssertionError(f"Unexpected SQL: {sql}")

    async def candidate(_candidate_id):
        return {
            "id": "candidate-1",
            "raw_data": {},
        }

    async def empty(*_args, **_kwargs):
        return []

    async def noop(*_args, **_kwargs):
        return None

    async def complete(**kwargs):
        prompts.append("\n".join(message["content"] for message in kwargs["messages"]))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Here are your matches.")
                )
            ]
        )

    monkeypatch.setattr(server, "SessionLocal", Session)
    monkeypatch.setattr(server, "_authorize_candidate", lambda *_args: None)
    monkeypatch.setattr(server, "_get_candidate_row", candidate)
    monkeypatch.setattr(server, "_normalize_for_frontend", lambda _row: {})
    monkeypatch.setattr(
        server, "_build_profile_context", lambda _profile: ("Profile", [])
    )
    monkeypatch.setattr(server, "_load_intake_ledger", empty)
    monkeypatch.setattr(server, "_load_chat_window", empty)
    monkeypatch.setattr(server, "_save_chat_window", noop)
    monkeypatch.setattr(server, "_chat_profile_preflight", lambda *_args: None)
    monkeypatch.setattr(
        server, "_voice_intake_completed_for_matching", lambda _row: True
    )
    monkeypatch.setattr(
        server, "_effective_profile_strength_percent", lambda *_args: _strength()
    )
    monkeypatch.setattr(server, "_advance_voice_intake_from_chat", noop)
    monkeypatch.setattr(server, "_record_demonstrated_skill_usage", noop)
    refreshes = []

    async def refresh(*_args):
        refreshes.append(True)
        if refresh_mode == "error":
            raise RuntimeError("refresh failed")
        if refresh_mode == "false":
            return False
        jobs.append(
            {
                "id": "refreshed",
                "title": "Refreshed role",
                "is_active": True,
                "status": "active",
                "job_status": "active",
                "job_url": "https://example.com/refreshed",
            }
        )
        return True

    import candidate_job_matching_service as matcher

    monkeypatch.setattr(matcher, "refresh_candidate_job_matches", refresh)
    monkeypatch.setattr(
        matcher, "stored_recommendation_relevance", lambda *_args: {"eligible": True}
    )
    monkeypatch.setattr(
        server,
        "openai_client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=complete))
        ),
    )

    await server.chat(
        server.ChatRequest(
            candidate_id="candidate-1",
            session_id="session-1",
            messages=[server.ChatMessageIn(role="user", content="Find me jobs")],
        )
    )

    assert len(captured_sql) == (2 if expect_jobs else 0)
    assert len(stale_score_sql) == int(expect_jobs or refresh_mode == "success")
    assert refreshes == ([True] if expect_refresh else [])
    assert ("Active role" in prompts[0]) is expect_jobs
    assert ("Refreshed role" in prompts[0]) is (expect_jobs and expect_refresh)
    assert "Closed role" not in prompts[0]


@pytest.mark.parametrize(
    "timestamp, expected_stale",
    [
        (None, True),
        (datetime.now(UTC) - timedelta(hours=25), True),
        ((datetime.now(UTC) - timedelta(hours=1)).replace(tzinfo=None), True),
        (datetime.now(UTC) + timedelta(minutes=1), True),
        (datetime.now(UTC) - timedelta(hours=1), False),
    ],
)
@pytest.mark.asyncio
async def test_chat_candidate_freshness_policy(timestamp, expected_stale):
    class Result:
        def scalar(self):
            return timestamp

        def mappings(self):
            return self

        def fetchone(self):
            return {
                "last_full_recommendation_refresh_at": timestamp,
                "matching_profile_revision": 0,
                "last_full_recommendation_revision": 0,
            }

    class Session:
        async def execute(self, _statement, _params=None):
            return Result()

    assert (
        await server._candidate_recommendations_are_stale(Session(), "candidate-1")
    ) is expected_stale


async def _strength():
    return 95
