import asyncio
import json
import sys
import types
from datetime import UTC, datetime

import httpx
import pytest

import app.job_ingestion.job_ingestion_service as ingestion_service
import app.job_ingestion.scheduler as job_scheduler
from app.job_ingestion.connectors.fantastic import (
    FantasticAPIError,
    FantasticClient,
    FantasticConfig,
)
from app.job_ingestion.job_ingestion_service import _metadata_params, upsert_ats_job
from app.job_ingestion.job_skill_extraction import extract_missing_job_skills
from app.job_ingestion.normalize import normalize_fantastic
from app.job_ingestion.scheduler import _log_fantastic_upsert_error


def test_fantastic_config_defaults_to_two_requests():
    config = FantasticConfig("key")
    assert config.max_requests_per_run == 2
    assert config.limit == 25


class _FakeFantasticSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def rollback(self):
        return None


def _patch_fantastic_scheduler(monkeypatch, fake_client):
    monkeypatch.setattr(job_scheduler, "provider_enabled", lambda provider: True)
    monkeypatch.setattr(
        job_scheduler, "_get_session_local", lambda: _FakeFantasticSession
    )
    monkeypatch.setattr(
        "app.job_ingestion.connectors.fantastic.FantasticClient",
        lambda: fake_client,
    )


@pytest.mark.asyncio
async def test_scheduler_preserves_failed_feed_statistics(monkeypatch):
    class FakeClient:
        config = types.SimpleNamespace(max_requests_per_run=2)
        _requests_used = 2

        def __init__(self):
            self.last_fetch_stats = {}

        async def fetch_active_ats(self):
            self._requests_used = 1
            self.last_fetch_stats = {
                "endpoint": "active-ats",
                "fetched": 0,
                "pages_requested": 1,
                "requests_made": 1,
                "success": True,
                "partial": False,
                "budget_exhausted": False,
                "stop_reason": "empty_response",
                "error": None,
            }
            return []

        async def fetch_active_job_boards(self):
            self._requests_used = 2
            self.last_fetch_stats = {
                "endpoint": "active-jb",
                "fetched": 0,
                "pages_requested": 1,
                "requests_made": 1,
                "success": False,
                "partial": True,
                "budget_exhausted": False,
                "stop_reason": "timeout",
                "error": "timeout",
            }
            raise TimeoutError("timed out")

    _patch_fantastic_scheduler(monkeypatch, FakeClient())

    stats = await job_scheduler.sync_fantastic_jobs()

    assert stats["feeds"]["active-jb"]["requests_made"] == 1
    assert stats["feeds"]["active-jb"]["pages_requested"] == 1
    assert stats["feeds"]["active-jb"]["success"] is False
    assert stats["feeds"]["active-jb"]["partial"] is True
    assert stats["feeds"]["active-jb"]["stop_reason"] == "timeout"
    assert stats["feeds"]["active-jb"]["error"] == "timeout"
    assert stats["requests_made"] == 2
    assert stats["success"] is False
    assert stats["failed"] == 1


@pytest.mark.asyncio
async def test_scheduler_marks_failure_without_connector_statistics(monkeypatch):
    class FakeClient:
        config = types.SimpleNamespace(max_requests_per_run=2)
        _requests_used = 0
        last_fetch_stats = {"endpoint": "active-ats", "success": True}

        async def fetch_active_ats(self):
            return []

        async def fetch_active_job_boards(self):
            self.last_fetch_stats = {"endpoint": "unrelated"}
            raise RuntimeError("failed before statistics")

    _patch_fantastic_scheduler(monkeypatch, FakeClient())

    stats = await job_scheduler.sync_fantastic_jobs()

    failed_feed = stats["feeds"]["active-jb"]
    assert failed_feed["success"] is False
    assert failed_feed["partial"] is True
    assert failed_feed["failed"] == 1
    assert failed_feed["stop_reason"] == "request_error"
    assert failed_feed["error"] == "request_error"
    assert stats["success"] is False


@pytest.mark.asyncio
async def test_scheduler_budget_exhaustion_is_not_failure(monkeypatch):
    class FakeClient:
        config = types.SimpleNamespace(max_requests_per_run=2)
        _requests_used = 2
        last_fetch_stats = {}

        async def fetch_active_ats(self):
            self.last_fetch_stats = {
                "endpoint": "active-ats",
                "fetched": 0,
                "pages_requested": 0,
                "requests_made": 0,
                "success": True,
                "partial": True,
                "budget_exhausted": True,
                "stop_reason": "budget_exhausted",
                "error": None,
            }
            return []

        async def fetch_active_job_boards(self):
            self.last_fetch_stats = {
                "endpoint": "active-jb",
                "fetched": 0,
                "pages_requested": 0,
                "requests_made": 0,
                "success": True,
                "partial": True,
                "budget_exhausted": True,
                "stop_reason": "budget_exhausted",
                "error": None,
            }
            return []

    _patch_fantastic_scheduler(monkeypatch, FakeClient())

    stats = await job_scheduler.sync_fantastic_jobs()

    assert stats["budget_exhausted"] is True
    assert stats["partial"] is True
    assert stats["success"] is True


@pytest.mark.asyncio
async def test_scheduler_marks_persistence_failure_partial_without_failed_feed(
    monkeypatch,
):
    class FakeClient:
        config = types.SimpleNamespace(max_requests_per_run=2)
        _requests_used = 2

        async def fetch_active_ats(self):
            self.last_fetch_stats = {
                "endpoint": "active-ats",
                "fetched": 1,
                "pages_requested": 1,
                "requests_made": 1,
                "success": True,
                "partial": False,
                "budget_exhausted": False,
                "stop_reason": "short_page",
                "error": None,
            }
            return [_record()]

        async def fetch_active_job_boards(self):
            self.last_fetch_stats = {
                "endpoint": "active-jb",
                "fetched": 0,
                "pages_requested": 1,
                "requests_made": 1,
                "success": True,
                "partial": False,
                "budget_exhausted": False,
                "stop_reason": "empty_response",
                "error": None,
            }
            return []

    async def find_existing(*args, **kwargs):
        return None

    async def failing_upsert(*args, **kwargs):
        raise RuntimeError("database write failed")

    _patch_fantastic_scheduler(monkeypatch, FakeClient())
    monkeypatch.setattr(
        ingestion_service, "find_existing_ats_job_by_id_or_url", find_existing
    )
    monkeypatch.setattr(ingestion_service, "upsert_ats_job", failing_upsert)
    monkeypatch.setattr(
        job_scheduler, "_existing_ats_job_needs_extraction", lambda *args: False
    )

    stats = await job_scheduler.sync_fantastic_jobs()

    assert stats["success"] is False
    assert stats["partial"] is True
    assert stats["failed"] == 1
    assert stats["failed_feeds"] == 0


def test_fantastic_config_from_env_defaults_to_two_requests(monkeypatch):
    monkeypatch.delenv("FANTASTIC_MAX_REQUESTS_PER_RUN", raising=False)
    monkeypatch.delenv("FANTASTIC_LIMIT", raising=False)

    config = FantasticConfig.from_env()
    assert config.max_requests_per_run == 2
    assert config.limit == 25


def test_fantastic_config_from_env_respects_explicit_request_budget(monkeypatch):
    monkeypatch.setenv("FANTASTIC_MAX_REQUESTS_PER_RUN", "3")

    assert FantasticConfig.from_env().max_requests_per_run == 3


def _record(**overrides):
    record = {
        "id": "fj-1",
        "organization": "Acme",
        "title": "Backend Engineer",
        "description_text": "Build APIs",
        "url": "https://jobs.example/fj-1",
        "locations_derived": ["Bengaluru", "India"],
        "date_posted": "2020-07-16T22:10:27+05:30",
        "date_created": "2026-07-16T22:10:27+05:30",
        "ai_employment_type": "Full-time",
        "ai_experience_level": "Senior",
        "ai_requirements_summary": "5 years",
        "ai_work_arrangement": "Hybrid",
        "ai_key_skills": ["Python", "FastAPI"],
        "salary_range": "₹20L-₹30L",
        "source": "greenhouse",
        "source_type": "ats",
    }
    record.update(overrides)
    return record


def test_normalization_preserves_fields_and_binds_datetime():
    job = normalize_fantastic(_record())
    assert job["ats_type"] == "fantastic" and job["location"] == "Bengaluru, India"
    assert (
        job["skills_required"] == ["Python", "FastAPI"]
        and job["employment_type"] == "Full-time"
    )
    assert isinstance(job["created_at"], datetime)
    assert isinstance(
        _metadata_params(job)["created_at"], datetime
    )  # asyncpg regression guard
    assert job["created_at"].year == 2020
    assert job["structured_data"]["source"] == "greenhouse"


def test_fantastic_valid_native_skills_skip_llm_and_preserve_both_arrays(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "groq_client",
        types.SimpleNamespace(
            GroqClientPool=lambda: (_ for _ in ()).throw(
                AssertionError("LLM should not be called")
            )
        ),
    )
    job = normalize_fantastic(
        _record(ai_key_skills=["Python", "FastAPI"], ai_keywords=["SQL"])
    )

    result = asyncio.run(extract_missing_job_skills(job))

    assert result is job
    assert result["skills"] == result["skills_required"] == ["Python", "FastAPI", "SQL"]


@pytest.mark.parametrize("required_skills", [None, []])
def test_fantastic_missing_or_empty_required_skills_uses_extraction(
    monkeypatch, required_skills
):
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            message = types.SimpleNamespace(
                content=json.dumps({"skills": ["Postgres"]})
            )
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)]
            )

    class Client:
        def __init__(self):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(
        sys.modules, "groq_client", types.SimpleNamespace(GroqClientPool=Client)
    )
    job = normalize_fantastic(_record())
    job["skills_required"] = required_skills

    result = asyncio.run(extract_missing_job_skills(job))

    assert len(calls) == 1
    assert (
        result["skills"]
        == result["skills_required"]
        == ["Python", "FastAPI", "PostgreSQL"]
    )


def test_normalization_falls_back_locations_and_rejects_bad_url():
    job = normalize_fantastic(
        _record(
            locations_derived=None,
            locations_alt="Remote",
            url="javascript:bad",
            ai_key_skills="Python, SQL",
            date_posted=None,
        )
    )
    assert job["location"] == "Remote" and job["job_url"] is None
    assert job["skills_required"] == ["Python", "SQL"] and job["created_at"] is None


def test_normalization_uses_source_date_posted_not_fantastic_record_created_at():
    job = normalize_fantastic(
        _record(date_posted="2020-01-02T03:04:05Z", date_created="2026-01-02T03:04:05Z")
    )
    assert job["created_at"] == datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_client_counts_timeout_attempt_in_request_budget():
    def handler(_request):
        raise httpx.ReadTimeout("timed out")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(FantasticConfig("key"), client)
            with pytest.raises(FantasticAPIError, match="Fantastic request timed out"):
                await fantastic.fetch_active_ats()
            return fantastic._requests_used

    assert asyncio.run(run()) == 1


def test_client_counts_request_exception_attempt_in_request_budget():
    def handler(_request):
        raise httpx.NetworkError("connection failed")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(FantasticConfig("key"), client)
            with pytest.raises(FantasticAPIError, match="Fantastic request failed"):
                await fantastic.fetch_active_ats()
            return fantastic._requests_used

    assert asyncio.run(run()) == 1


def test_client_counts_successful_request_once():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"jobs": []})
            )
        ) as client:
            fantastic = FantasticClient(FantasticConfig("key"), client)
            await fantastic.fetch_active_ats()
            return fantastic._requests_used

    assert asyncio.run(run()) == 1


def test_last_fetch_stats_exists_before_fetch():
    assert FantasticClient(FantasticConfig("key")).last_fetch_stats == {
        "endpoint": None,
        "fetched": 0,
        "pages_requested": 0,
        "requests_made": 0,
        "success": True,
        "partial": False,
        "budget_exhausted": False,
        "stop_reason": None,
        "error": None,
    }


def test_fetch_stats_for_successful_non_empty_response():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"jobs": [_record()]})
            )
        ) as client:
            fantastic = FantasticClient(FantasticConfig("key"), client)
            await fantastic.fetch_active_ats()
            return fantastic.last_fetch_stats

    stats = asyncio.run(run())
    assert stats == {
        "endpoint": "active-ats",
        "fetched": 1,
        "pages_requested": 1,
        "requests_made": 1,
        "success": True,
        "partial": False,
        "budget_exhausted": False,
        "stop_reason": "short_page",
        "error": None,
    }


def test_fetch_stats_for_successful_empty_response():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"jobs": []})
            )
        ) as client:
            fantastic = FantasticClient(FantasticConfig("key"), client)
            await fantastic.fetch_active_ats()
            return fantastic.last_fetch_stats

    stats = asyncio.run(run())
    assert stats["fetched"] == 0
    assert stats["requests_made"] == stats["pages_requested"] == 1
    assert stats["success"] is True and stats["partial"] is False
    assert stats["stop_reason"] == "empty_response" and stats["error"] is None


def test_fetch_stats_for_exhausted_budget():
    def handler(_request):
        raise AssertionError("budget-exhausted fetch must not send a request")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(FantasticConfig("key"), client)
            fantastic._requests_used = fantastic.config.max_requests_per_run
            await fantastic.fetch_active_ats()
            return fantastic.last_fetch_stats

    stats = asyncio.run(run())
    assert stats["requests_made"] == stats["pages_requested"] == 0
    assert stats["success"] is True and stats["partial"] is True
    assert stats["budget_exhausted"] is True
    assert stats["stop_reason"] == "budget_exhausted"


def test_fetch_stats_for_max_pages_and_max_jobs():
    def handler(request):
        return httpx.Response(
            200, json={"jobs": [_record(id=str(request.url.params["offset"]))]}
        )

    async def run(config):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(config, client)
            await fantastic.fetch_active_ats()
            return fantastic.last_fetch_stats

    max_pages = asyncio.run(
        run(
            FantasticConfig(
                "key", limit=1, max_pages=1, max_jobs_per_run=5, max_requests_per_run=5
            )
        )
    )
    assert max_pages["partial"] is True and max_pages["stop_reason"] == "max_pages"
    max_jobs = asyncio.run(
        run(
            FantasticConfig(
                "key", limit=1, max_pages=5, max_jobs_per_run=1, max_requests_per_run=5
            )
        )
    )
    assert max_jobs["partial"] is True and max_jobs["stop_reason"] == "max_jobs_per_run"


@pytest.mark.parametrize(
    ("exception", "reason"),
    [
        (httpx.ReadTimeout("timed out"), "timeout"),
        (httpx.NetworkError("failed"), "request_error"),
    ],
)
def test_fetch_stats_preserve_failed_attempt(exception, reason):
    def handler(_request):
        raise exception

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(FantasticConfig("key"), client)
            with pytest.raises(FantasticAPIError):
                await fantastic.fetch_active_ats()
            return fantastic.last_fetch_stats

    stats = asyncio.run(run())
    assert stats["requests_made"] == stats["pages_requested"] == 1
    assert stats["success"] is False and stats["partial"] is True
    assert stats["stop_reason"] == stats["error"] == reason


def test_fetch_stats_for_successful_429_retry(monkeypatch):
    calls = []

    async def no_sleep(_delay):
        pass

    def handler(_request):
        calls.append(True)
        return (
            httpx.Response(429, headers={"Retry-After": "1"})
            if len(calls) == 1
            else httpx.Response(200, json={"jobs": []})
        )

    async def run():
        monkeypatch.setattr(
            "app.job_ingestion.connectors.fantastic.asyncio.sleep", no_sleep
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig("key", max_requests_per_run=2), client
            )
            await fantastic.fetch_active_ats()
            return fantastic.last_fetch_stats

    stats = asyncio.run(run())
    assert stats["requests_made"] == 2 and stats["pages_requested"] == 1
    assert stats["success"] is True and stats["stop_reason"] == "empty_response"


def test_fetch_stats_for_second_429_and_malformed_response(monkeypatch):
    async def no_sleep(_delay):
        pass

    def second_429(_request):
        return httpx.Response(429, headers={"Retry-After": "1"})

    def malformed(_request):
        return httpx.Response(200, content=b"not-json")

    async def run(handler):
        monkeypatch.setattr(
            "app.job_ingestion.connectors.fantastic.asyncio.sleep", no_sleep
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig("key", max_requests_per_run=2), client
            )
            with pytest.raises(FantasticAPIError):
                await fantastic.fetch_active_ats()
            return fantastic.last_fetch_stats

    rate = asyncio.run(run(second_429))
    assert (
        rate["requests_made"] == 2
        and rate["stop_reason"] == rate["error"] == "rate_limited"
    )
    malformed_stats = asyncio.run(run(malformed))
    assert (
        malformed_stats["requests_made"] == 1
        and malformed_stats["stop_reason"]
        == malformed_stats["error"]
        == "malformed_response"
    )


def test_failed_second_feed_does_not_retain_first_feed_stats():
    calls = []

    def handler(_request):
        calls.append(True)
        if len(calls) == 1:
            return httpx.Response(200, json={"jobs": [_record()]})
        raise httpx.NetworkError("failed")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig("key", max_requests_per_run=2), client
            )
            await fantastic.fetch_active_ats()
            with pytest.raises(FantasticAPIError):
                await fantastic.fetch_active_job_boards()
            return fantastic.last_fetch_stats

    stats = asyncio.run(run())
    assert stats["endpoint"] == "active-jb"
    assert stats["requests_made"] == stats["pages_requested"] == 1
    assert stats["stop_reason"] == stats["error"] == "request_error"


def test_missing_api_key_resets_stats_for_new_feed():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"jobs": []})
            )
        ) as client:
            fantastic = FantasticClient(FantasticConfig("key"), client)
            await fantastic.fetch_active_ats()
            fantastic.config = FantasticConfig("")
            with pytest.raises(FantasticAPIError, match="API_KEY"):
                await fantastic.fetch_active_job_boards()
            return fantastic.last_fetch_stats

    stats = asyncio.run(run())
    assert stats["endpoint"] == "active-jb"
    assert stats["fetched"] == stats["pages_requested"] == stats["requests_made"] == 0
    assert stats["success"] is False
    assert stats["stop_reason"] == stats["error"] == "authorization_error"


def test_http_400_is_distinguished_from_invalid_json():
    async def run(response):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: response)
        ) as client:
            fantastic = FantasticClient(FantasticConfig("key"), client)
            with pytest.raises(FantasticAPIError):
                await fantastic.fetch_active_ats()
            return fantastic.last_fetch_stats

    http_error = asyncio.run(run(httpx.Response(400)))
    malformed = asyncio.run(run(httpx.Response(200, content=b"not-json")))
    assert http_error["stop_reason"] == http_error["error"] == "http_error"
    assert malformed["stop_reason"] == malformed["error"] == "malformed_response"
    assert http_error["requests_made"] == malformed["requests_made"] == 1


def test_client_sends_no_request_when_budget_is_already_exhausted():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"jobs": []})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig("key", max_requests_per_run=1), client
            )
            fantastic._requests_used = 1
            return await fantastic.fetch_active_ats()

    assert asyncio.run(run()) == []
    assert calls == []


def test_client_checks_budget_again_before_next_page():
    calls = []
    fantastic: FantasticClient | None = None

    def handler(request):
        calls.append(request)
        assert fantastic is not None
        fantastic._requests_used = fantastic.config.max_requests_per_run
        return httpx.Response(200, json={"jobs": [_record(id=str(len(calls)))]})

    async def run():
        nonlocal fantastic
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig(
                    "key",
                    limit=1,
                    max_pages=3,
                    max_jobs_per_run=3,
                    max_requests_per_run=3,
                ),
                client,
            )
            return await fantastic.fetch_active_ats()

    assert len(asyncio.run(run())) == 1
    assert len(calls) == 1


def test_client_counts_retry_before_sending_it(monkeypatch):
    calls = []
    fantastic: FantasticClient | None = None

    async def no_sleep(_delay):
        pass

    def handler(request):
        assert fantastic is not None
        calls.append((request, fantastic._requests_used))
        return (
            httpx.Response(429, headers={"Retry-After": "1"})
            if len(calls) == 1
            else httpx.Response(200, json={"jobs": []})
        )

    async def run():
        nonlocal fantastic
        monkeypatch.setattr(
            "app.job_ingestion.connectors.fantastic.asyncio.sleep", no_sleep
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig("key", max_requests_per_run=2), client
            )
            await fantastic.fetch_active_ats()

    asyncio.run(run())
    assert len(calls) == 2
    assert [request_count for _, request_count in calls] == [1, 2]


def test_client_counts_retry_timeout_in_request_budget(monkeypatch):
    calls = []

    async def no_sleep(_delay):
        pass

    def handler(_request):
        calls.append(True)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        raise httpx.ReadTimeout("retry timed out")

    async def run():
        monkeypatch.setattr(
            "app.job_ingestion.connectors.fantastic.asyncio.sleep", no_sleep
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig("key", max_requests_per_run=2), client
            )
            with pytest.raises(FantasticAPIError, match="Fantastic request timed out"):
                await fantastic.fetch_active_ats()
            return fantastic._requests_used

    assert asyncio.run(run()) == 2
    assert len(calls) == 2


def test_client_counts_retry_network_exception_in_request_budget(monkeypatch):
    calls = []

    async def no_sleep(_delay):
        pass

    def handler(_request):
        calls.append(True)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        raise httpx.NetworkError("retry connection failed")

    async def run():
        monkeypatch.setattr(
            "app.job_ingestion.connectors.fantastic.asyncio.sleep", no_sleep
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig("key", max_requests_per_run=2), client
            )
            with pytest.raises(FantasticAPIError, match="Fantastic request failed"):
                await fantastic.fetch_active_ats()
            return fantastic._requests_used

    assert asyncio.run(run()) == 2
    assert len(calls) == 2


def test_client_does_not_retry_when_retry_budget_is_exhausted(monkeypatch):
    calls = []

    async def no_sleep(_delay):
        pass

    def handler(_request):
        calls.append(True)
        return httpx.Response(429, headers={"Retry-After": "1"})

    async def run():
        monkeypatch.setattr(
            "app.job_ingestion.connectors.fantastic.asyncio.sleep", no_sleep
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig("key", max_requests_per_run=1), client
            )
            with pytest.raises(FantasticAPIError, match="Fantastic rate limit reached"):
                await fantastic.fetch_active_ats()
            return fantastic._requests_used

    assert asyncio.run(run()) == 1
    assert len(calls) == 1


def test_client_does_not_retry_if_budget_exhausts_during_sleep(monkeypatch):
    calls = []
    fantastic: FantasticClient | None = None

    async def exhaust_budget_during_sleep(_delay):
        assert fantastic is not None
        fantastic._requests_used = fantastic.config.max_requests_per_run

    def handler(_request):
        calls.append(True)
        return httpx.Response(429, headers={"Retry-After": "1"})

    async def run():
        nonlocal fantastic
        monkeypatch.setattr(
            "app.job_ingestion.connectors.fantastic.asyncio.sleep",
            exhaust_budget_during_sleep,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig("key", max_requests_per_run=2), client
            )
            with pytest.raises(FantasticAPIError, match="Fantastic rate limit reached"):
                await fantastic.fetch_active_ats()

    asyncio.run(run())
    assert len(calls) == 1


def test_client_successful_retry_consumes_two_requests(monkeypatch):
    calls = []

    async def no_sleep(_delay):
        pass

    def handler(_request):
        calls.append(True)
        return (
            httpx.Response(429, headers={"Retry-After": "1"})
            if len(calls) == 1
            else httpx.Response(200, json={"jobs": []})
        )

    async def run():
        monkeypatch.setattr(
            "app.job_ingestion.connectors.fantastic.asyncio.sleep", no_sleep
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig("key", max_requests_per_run=2), client
            )
            await fantastic.fetch_active_ats()
            return fantastic._requests_used

    assert asyncio.run(run()) == 2
    assert len(calls) == 2


def test_client_raises_rate_limit_after_second_429(monkeypatch):
    calls = []

    async def no_sleep(_delay):
        pass

    def handler(_request):
        calls.append(True)
        return httpx.Response(429, headers={"Retry-After": "1"})

    async def run():
        monkeypatch.setattr(
            "app.job_ingestion.connectors.fantastic.asyncio.sleep", no_sleep
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig("key", max_requests_per_run=2), client
            )
            with pytest.raises(FantasticAPIError, match="Fantastic rate limit reached"):
                await fantastic.fetch_active_ats()
            return fantastic._requests_used

    assert asyncio.run(run()) == 2
    assert len(calls) == 2


def test_shared_client_feeds_cannot_exceed_request_budget():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"jobs": [_record(id=str(len(calls)))]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fantastic = FantasticClient(
                FantasticConfig(
                    "key",
                    limit=1,
                    max_pages=2,
                    max_jobs_per_run=1,
                    max_requests_per_run=2,
                ),
                client,
            )
            await fantastic.fetch_active_ats()
            await fantastic.fetch_active_job_boards()

    asyncio.run(run())
    assert len(calls) == 2


def test_client_paginates_with_caps():
    calls = []

    def handler(request):
        calls.append(request)
        offset = int(request.url.params["offset"])
        return httpx.Response(
            200, json={"jobs": [_record(id=str(offset)), _record(id=str(offset + 1))]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await FantasticClient(
                FantasticConfig(
                    "key",
                    limit=2,
                    max_pages=3,
                    max_jobs_per_run=3,
                    max_requests_per_run=2,
                ),
                client,
            ).fetch_active_ats()

    assert len(asyncio.run(run())) == 3 and len(calls) == 2
    assert [int(call.url.params["offset"]) for call in calls] == [0, 2]
    assert all(call.url.params["location"] == "India" for call in calls)
    assert all("exclude_ats_duplicate" not in call.url.params for call in calls)


def test_client_stops_pagination_on_short_page():
    calls = []

    def handler(request):
        calls.append(request)
        offset = int(request.url.params["offset"])
        count = 2 if offset == 0 else 1
        return httpx.Response(
            200,
            json={
                "jobs": [_record(id=f"job-{offset + index}") for index in range(count)]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await FantasticClient(
                FantasticConfig(
                    "key",
                    limit=2,
                    max_pages=5,
                    max_jobs_per_run=10,
                    max_requests_per_run=5,
                ),
                client,
            ).fetch_active_ats()

    assert len(asyncio.run(run())) == 3
    assert [int(call.url.params["offset"]) for call in calls] == [0, 2]


def test_client_stops_pagination_at_max_pages():
    calls = []

    def handler(request):
        calls.append(request)
        offset = int(request.url.params["offset"])
        return httpx.Response(
            200,
            json={"jobs": [_record(id=f"job-{offset + index}") for index in range(2)]},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await FantasticClient(
                FantasticConfig(
                    "key",
                    limit=2,
                    max_pages=2,
                    max_jobs_per_run=10,
                    max_requests_per_run=5,
                ),
                client,
            ).fetch_active_ats()

    assert len(asyncio.run(run())) == 4
    assert [int(call.url.params["offset"]) for call in calls] == [0, 2]


def test_client_trims_final_page_at_max_jobs_per_run():
    calls = []

    def handler(request):
        calls.append(request)
        offset = int(request.url.params["offset"])
        return httpx.Response(
            200,
            json={"jobs": [_record(id=f"job-{offset + index}") for index in range(2)]},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await FantasticClient(
                FantasticConfig(
                    "key",
                    limit=2,
                    max_pages=5,
                    max_jobs_per_run=3,
                    max_requests_per_run=5,
                ),
                client,
            ).fetch_active_ats()

    assert len(asyncio.run(run())) == 3
    assert [int(call.url.params["offset"]) for call in calls] == [0, 2]


def test_active_job_boards_uses_india_filter_and_shared_pagination():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"jobs": [_record(id=f"jb-{len(calls)}")]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await FantasticClient(
                FantasticConfig(
                    "key",
                    limit=1,
                    max_pages=2,
                    max_jobs_per_run=2,
                    max_requests_per_run=2,
                ),
                client,
            ).fetch_active_job_boards()

    assert len(asyncio.run(run())) == 2
    assert all(call.url.path.endswith("/v1/active-jb") for call in calls)
    assert all(call.url.params["location"] == "India" for call in calls)
    assert all(call.url.params["exclude_ats_duplicate"] == "true" for call in calls)
    assert [int(call.url.params["offset"]) for call in calls] == [0, 1]


def test_fetch_expired_job_boards_requests_daily_ids():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=[123, "456", "  "])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await FantasticClient(
                FantasticConfig("key"), client
            ).fetch_expired_job_boards()

    assert asyncio.run(run()) == ["123", "456"]
    assert calls[0].url.path.endswith("/v1/expired-jb")
    assert calls[0].url.params["time_frame"] == "1d"


def test_fetch_expired_job_boards_handles_empty_response():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))
        ) as client:
            return await FantasticClient(
                FantasticConfig("key"), client
            ).fetch_expired_job_boards()

    assert asyncio.run(run()) == []


def test_expiration_requests_are_counted_on_shared_client():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))
        ) as client:
            fantastic = FantasticClient(FantasticConfig("key"), client)
            await fantastic.fetch_expired_ats_ids()
            await fantastic.fetch_expired_job_boards()
            return fantastic._requests_used

    assert asyncio.run(run()) == 2


@pytest.mark.parametrize("status", [400, 429, 500])
def test_fetch_expired_job_boards_errors(status):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(status))
        ) as client:
            await FantasticClient(
                FantasticConfig("key"), client
            ).fetch_expired_job_boards()

    with pytest.raises(FantasticAPIError):
        asyncio.run(run())


def test_fantastic_country_is_normalized_from_structured_country():
    indian = normalize_fantastic(
        _record(
            countries_derived=["IND"],
            locations_derived=["Bengaluru", "Karnataka", "India"],
        )
    )
    foreign = normalize_fantastic(
        _record(
            countries_derived=["United States"],
            locations_derived=["New York", "United States"],
        )
    )
    unknown = normalize_fantastic(
        _record(countries_derived=None, locations_derived=["Remote"])
    )
    assert indian["country_code"] == "IN" and indian["country"] == "India"
    assert foreign["country_code"] == "US" and foreign["country"] == "United States"
    assert unknown["country_code"] == ""


@pytest.mark.parametrize("status", [401, 429, 500])
def test_client_stops_on_provider_errors(status):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(status))
        ) as client:
            await FantasticClient(FantasticConfig("key"), client).fetch_active_ats()

    with pytest.raises(FantasticAPIError):
        asyncio.run(run())


def test_global_insert_bypasses_company_scoped_ats_validation(monkeypatch):
    class Result:
        def first(self):
            return None

        def scalar_one(self):
            return "new-id"

    class Session:
        def __init__(self):
            self.insert = None

        async def execute(self, statement, params=None):
            sql = str(statement)
            assert "FROM company_registry" not in sql
            if "INSERT" in sql:
                self.insert = params
            return Result()

        async def commit(self):
            pass

    async def company_scoped_agency(*_):
        raise AssertionError(
            "Fantastic must not use company-scoped ATS agency validation"
        )

    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service.get_or_create_ats_agency",
        company_scoped_agency,
    )
    session = Session()
    job = normalize_fantastic(_record())
    job["job_url"] = None
    assert asyncio.run(upsert_ats_job(session, job)) == "new-id"
    assert session.insert is not None
    assert session.insert["company_registry_id"] is None
    assert session.insert["agency_id"] is None
    assert session.insert["ats_type"] == "fantastic"
    assert session.insert["ats_job_id"] == "fj-1"


def test_fantastic_upsert_error_log_preserves_postgres_diagnostics(caplog):
    """Regression guard for Railway logs: do not collapse asyncpg diagnostics."""

    class UniqueViolationError(Exception):
        sqlstate = "23505"
        constraint_name = "uq_job_descriptions_ats_type_ats_job_id"
        column_name = None

        def __str__(self):
            return "duplicate key value violates unique constraint"

    with caplog.at_level("WARNING"):
        _log_fantastic_upsert_error(
            UniqueViolationError(),
            {
                "ats_job_id": "fj-1",
                "title": "Backend Engineer",
                "structured_data": {"secret": "never logged"},
            },
        )

    message = caplog.messages[-1]
    assert "exception_class=UniqueViolationError" in message
    assert "exception_message=duplicate key value violates unique constraint" in message
    assert "sqlstate=23505" in message
    assert "constraint=uq_job_descriptions_ats_type_ats_job_id" in message
    assert "column=None" in message
    assert "id=fj-1" in message and "title='Backend Engineer'" in message
    assert "never logged" not in message
