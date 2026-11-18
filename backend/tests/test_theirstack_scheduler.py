import re

import httpx
import pytest
from app.job_ingestion import scheduler
from app.job_ingestion.connectors import theirstack as theirstack_connector
from app.job_ingestion.connectors.theirstack import (
    TheirStackAPIError,
    TheirStackClient,
    TheirStackConfig,
)


class _Result:
    def first(self):
        return None


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, *_args, **_kwargs):
        return _Result()

    async def rollback(self):
        pass


@pytest.mark.asyncio
async def test_theirstack_fetch_jobs_requests_open_india_jobs_with_pagination():
    class _Response:
        status_code = 200

        def __init__(self, jobs):
            self._data = {"jobs": jobs}

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    class _Client:
        def __init__(self):
            self.payloads = []
            self.requests = []

        async def get(self, _url, *, headers):
            response = _Response([])
            response._data = {"api_credits": 3}
            return response

        async def post(self, _url, *, json, headers):
            self.payloads.append(json)
            self.requests.append((_url, headers))
            return _Response(
                [
                    {"id": f"{len(self.payloads)}-{index}"}
                    for index in range(json["limit"])
                ]
            )

    http_client = _Client()
    client = TheirStackClient(
        TheirStackConfig(
            api_key="test-key",
            limit=2,
            max_pages=2,
            max_jobs_per_run=3,
        ),
        client=http_client,
    )

    jobs = await client.fetch_jobs()

    assert len(jobs) == 3
    assert http_client.payloads == [
        {
            "limit": 2,
            "page": 0,
            "posted_at_max_age_days": 30,
            "job_country_code_or": ["IN"],
            "is_closed": False,
        },
        {
            "limit": 1,
            "page": 1,
            "posted_at_max_age_days": 30,
            "job_country_code_or": ["IN"],
            "is_closed": False,
        },
    ]
    assert all(payload["is_closed"] is False for payload in http_client.payloads)
    assert all(
        url == "https://api.theirstack.com/v1/jobs/search"
        for url, _headers in http_client.requests
    )
    assert all(
        headers
        == {
            "Authorization": "Bearer test-key",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        for _url, headers in http_client.requests
    )


@pytest.mark.asyncio
async def test_theirstack_fetch_jobs_handles_insufficient_api_credits(caplog):
    class _Response:
        def __init__(self, status_code, data=None):
            self.status_code = status_code
            self._data = data

        def raise_for_status(self):
            if self.status_code != 200:
                raise AssertionError(
                    "402 should be handled before generic status handling"
                )

        def json(self):
            return self._data

    class _Client:
        async def get(self, _url, *, headers):
            return _Response(200, {"api_credits": 1})

        async def post(self, _url, *, json, headers):
            return _Response(402)

    client = TheirStackClient(
        TheirStackConfig(api_key="test-key"),
        client=_Client(),
    )

    with caplog.at_level("ERROR"):
        with pytest.raises(TheirStackAPIError, match="insufficient API credits"):
            await client.fetch_jobs()

    assert "insufficient API credits" in caplog.text


@pytest.mark.asyncio
async def test_theirstack_fetch_jobs_retries_429_using_rate_limit_reset(monkeypatch):
    class _Response:
        def __init__(self, status_code, data=None, headers=None):
            self.status_code = status_code
            self._data = data
            self.headers = headers or {}

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    class _Client:
        def __init__(self):
            self.posts = 0

        async def get(self, _url, *, headers):
            return _Response(200, {"api_credits": 1})

        async def post(self, _url, *, json, headers):
            self.posts += 1
            if self.posts == 1:
                return _Response(
                    429,
                    headers={
                        "RateLimit": '"per-second";r=0;t=2;pk="x"',
                        "RateLimit-Policy": '"per-second";q=4;w=1',
                        "RateLimit-Limit": "4",
                        "RateLimit-Remaining": "0",
                        "RateLimit-Reset": "2",
                    },
                )
            return _Response(200, {"jobs": [{"id": "job-1"}]})

    sleeps = []

    async def _sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(theirstack_connector.asyncio, "sleep", _sleep)
    http_client = _Client()
    jobs = await TheirStackClient(
        TheirStackConfig(api_key="test-key"), client=http_client
    ).fetch_jobs()

    assert jobs == [{"id": "job-1"}]
    assert http_client.posts == 2
    assert sleeps == [2.0]


@pytest.mark.asyncio
async def test_theirstack_fetch_jobs_stops_after_bounded_429_retries(monkeypatch):
    class _Response:
        status_code = 429
        headers = {}

        def raise_for_status(self):
            pass

    class _Client:
        def __init__(self):
            self.posts = 0

        async def get(self, _url, *, headers):
            return type(
                "BalanceResponse",
                (),
                {
                    "status_code": 200,
                    "headers": {},
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"api_credits": 1},
                },
            )()

        async def post(self, _url, *, json, headers):
            self.posts += 1
            return _Response()

    sleeps = []

    async def _sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(theirstack_connector.asyncio, "sleep", _sleep)
    http_client = _Client()
    with pytest.raises(TheirStackAPIError, match="rate limit reached"):
        await TheirStackClient(
            TheirStackConfig(api_key="test-key"), client=http_client
        ).fetch_jobs()

    assert http_client.posts == theirstack_connector.MAX_RATE_LIMIT_RETRIES + 1
    assert sleeps == [0.5, 1.0, 2.0]
    assert max(sleeps) <= theirstack_connector.MAX_RATE_LIMIT_WAIT_SECONDS


def test_theirstack_rate_limit_header_parsing_is_bounded_and_has_fallback():
    assert (
        TheirStackClient._rate_limit_wait_seconds({"RateLimit-Reset": "60, 2"}, 0)
        == 2.0
    )
    assert (
        TheirStackClient._rate_limit_wait_seconds(
            {"RateLimit": '"per-second";r=0;t=3;pk="x"'}, 0
        )
        == 3.0
    )
    assert (
        TheirStackClient._rate_limit_wait_seconds(
            {"RateLimit-Reset": "not-a-number"}, 1
        )
        == 1.0
    )
    assert (
        TheirStackClient._rate_limit_wait_seconds({"RateLimit-Reset": "999"}, 0) == 5.0
    )


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (402, "TheirStack has insufficient API credits"),
        (429, "TheirStack rate limit reached"),
        (401, "TheirStack authorization failed (HTTP 401)"),
        (403, "TheirStack authorization failed (HTTP 403)"),
        (500, "TheirStack server error (HTTP 500)"),
        (503, "TheirStack server error (HTTP 503)"),
    ],
)
def test_theirstack_shared_api_status_helper(status, message):
    response = type("Response", (), {"status_code": status})()

    with pytest.raises(TheirStackAPIError, match=re.escape(message)):
        TheirStackClient._raise_for_api_status(response)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"jobs": [{"id": "jobs-1"}]}, [{"id": "jobs-1"}]),
        ({"data": [{"id": "data-1"}]}, [{"id": "data-1"}]),
        ({}, []),
        ({"jobs": []}, []),
        ({"jobs": [{"id": "valid"}, "invalid", None]}, [{"id": "valid"}]),
    ],
)
def test_theirstack_extract_jobs_tolerant_mode(data, expected):
    assert TheirStackClient._extract_jobs(data) == expected


@pytest.mark.parametrize("data", [{"jobs": []}, {"data": []}, {}])
def test_theirstack_extract_jobs_strict_mode_accepts_lists(data):
    assert TheirStackClient._extract_jobs(data, strict=True) == []


def test_theirstack_extract_jobs_strict_mode_filters_non_dict_entries():
    assert TheirStackClient._extract_jobs(
        {"data": [{"id": "valid"}, "invalid", 1]}, strict=True
    ) == [{"id": "valid"}]


def test_theirstack_extract_jobs_strict_mode_rejects_non_list_jobs():
    with pytest.raises(
        TheirStackAPIError, match=re.escape("TheirStack returned a malformed response")
    ):
        TheirStackClient._extract_jobs({"jobs": {"id": "not-a-list"}}, strict=True)

    assert TheirStackClient._extract_jobs({"jobs": {"id": "not-a-list"}}) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 402])
async def test_theirstack_search_does_not_retry_non_retryable_statuses(
    monkeypatch, status
):
    class _Response:
        def __init__(self, status_code, data=None):
            self.status_code = status_code
            self._data = data
            self.headers = {}

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    class _Client:
        def __init__(self):
            self.posts = 0

        async def get(self, _url, *, headers):
            return _Response(200, {"api_credits": 1})

        async def post(self, _url, *, json, headers):
            self.posts += 1
            return _Response(status)

    async def _sleep(_seconds):
        raise AssertionError("non-retryable status must not sleep")

    monkeypatch.setattr(theirstack_connector.asyncio, "sleep", _sleep)
    http_client = _Client()
    with pytest.raises(TheirStackAPIError):
        await TheirStackClient(
            TheirStackConfig(api_key="test-key"), client=http_client
        ).fetch_jobs()
    assert http_client.posts == 1


@pytest.mark.asyncio
async def test_theirstack_fetch_jobs_by_ids_retries_429_consistently(monkeypatch):
    class _Response:
        def __init__(self, status_code, data=None, headers=None):
            self.status_code = status_code
            self._data = data
            self.headers = headers or {}

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    class _Client:
        def __init__(self):
            self.posts = 0

        async def post(self, _url, *, json, headers):
            self.posts += 1
            if self.posts == 1:
                return _Response(429, headers={"RateLimit-Reset": "1"})
            return _Response(200, {"jobs": [{"id": "1", "closed_at": None}]})

    sleeps = []

    async def _sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(theirstack_connector.asyncio, "sleep", _sleep)
    http_client = _Client()
    jobs = await TheirStackClient(
        TheirStackConfig(api_key="test-key"), client=http_client
    ).fetch_jobs_by_ids(["1"])

    assert jobs == [{"id": "1", "closed_at": None}]
    assert http_client.posts == 2
    assert sleeps == [1.0]


@pytest.mark.asyncio
async def test_theirstack_fetch_jobs_by_ids_requests_only_closed_jobs():
    class _Response:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            pass

        def json(self):
            return {"jobs": []}

    class _Client:
        def __init__(self):
            self.payload = None
            self.url = None
            self.headers = None

        async def post(self, _url, *, json, headers):
            self.url = _url
            self.headers = headers
            self.payload = json
            return _Response()

    http_client = _Client()
    await TheirStackClient(
        TheirStackConfig(api_key="test-key"), client=http_client
    ).fetch_jobs_by_ids(["1", "2"])

    assert http_client.payload == {
        "job_id_or": [1, 2],
        "is_closed": True,
        "limit": 2,
        "page": 0,
    }
    assert http_client.url == "https://api.theirstack.com/v1/jobs/search"
    assert http_client.headers == {
        "Authorization": "Bearer test-key",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
async def test_theirstack_fetch_credit_balance_returns_api_credits_only():
    class _Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"api_credits": 7, "used_api_credits": 999}

    class _Client:
        async def get(self, _url, *, headers):
            return _Response()

    assert (
        await TheirStackClient(
            TheirStackConfig(api_key="test-key"), client=_Client()
        ).fetch_credit_balance()
        == 7
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("balance", [2, 0])
async def test_theirstack_fetch_jobs_respects_credit_budget(balance):
    class _Response:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    class _Client:
        def __init__(self):
            self.search_payloads = []

        async def get(self, _url, *, headers):
            return _Response(200, {"api_credits": balance})

        async def post(self, _url, *, json, headers):
            self.search_payloads.append(json)
            return _Response(200, {"jobs": [{"id": "job-1"}]})

    http_client = _Client()
    jobs = await TheirStackClient(
        TheirStackConfig(api_key="test-key", limit=5, max_pages=3, max_jobs_per_run=10),
        client=http_client,
    ).fetch_jobs()

    assert len(jobs) == (1 if balance else 0)
    if balance:
        assert http_client.search_payloads[0]["limit"] == balance
    else:
        assert http_client.search_payloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data", [{}, {"api_credits": "5"}, {"api_credits": -1}, {"api_credits": True}]
)
async def test_theirstack_fetch_jobs_rejects_invalid_credit_balance(data):
    class _Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return data

    class _Client:
        async def get(self, _url, *, headers):
            return _Response()

        async def post(self, *_args, **_kwargs):
            raise AssertionError("invalid balance must prevent job search")

    with pytest.raises(TheirStackAPIError, match="invalid API credit balance"):
        await TheirStackClient(
            TheirStackConfig(api_key="test-key"), client=_Client()
        ).fetch_jobs()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
async def test_theirstack_fetch_credit_balance_handles_billing_errors(status):
    class _Response:
        status_code = status

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "billing error",
                request=httpx.Request(
                    "GET", "https://api.theirstack.com/v0/billing/credit-balance"
                ),
                response=httpx.Response(
                    self.status_code,
                    request=httpx.Request(
                        "GET", "https://api.theirstack.com/v0/billing/credit-balance"
                    ),
                ),
            )

        def json(self):
            return {}

    class _Client:
        async def get(self, _url, *, headers):
            return _Response()

    with pytest.raises(TheirStackAPIError):
        await TheirStackClient(
            TheirStackConfig(api_key="test-key"), client=_Client()
        ).fetch_credit_balance()


@pytest.mark.asyncio
async def test_theirstack_sync_calls_job_skill_extractor(monkeypatch):
    monkeypatch.setenv("THEIRSTACK_ENABLED", "true")

    class _Client:
        async def fetch_jobs(self):
            return [
                {
                    "id": "ts-1",
                    "title": "Engineer",
                    "url": "https://example.test/ts-1",
                    "location": "Bengaluru",
                    "job_country": "India",
                }
            ]

    async def _extract(job):
        job["extracted"] = True
        return job

    persisted = []
    monkeypatch.setattr(
        "app.job_ingestion.connectors.theirstack.TheirStackClient", _Client
    )
    monkeypatch.setattr(
        "app.job_ingestion.job_skill_extraction.extract_missing_job_skills", _extract
    )
    monkeypatch.setattr(scheduler, "_get_session_local", lambda: _Session)
    monkeypatch.setattr(
        scheduler, "upsert_ats_job", lambda *_args, **_kwargs: None, raising=False
    )

    async def _upsert(_db, job, **_kwargs):
        persisted.append(job)

    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service.upsert_ats_job", _upsert
    )

    result = await scheduler.sync_theirstack_jobs()

    assert result["inserted"] == 1
    assert result["failed"] == 0
    assert persisted[0]["extracted"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("existing_type", "existing_id", "expected_updated", "expected_skipped"),
    [
        ("theirstack", "ts-older", 0, 1),  # same-provider URL collision
        (
            "theirstack",
            "ts-1",
            1,
            1,
        ),  # stable-ID match updates, but skips complete extraction
        ("fantastic", "ts-1", 0, 1),  # cross-provider URL collision
    ],
)
async def test_theirstack_url_match_never_overrides_provider_stable_identity(
    monkeypatch, existing_type, existing_id, expected_updated, expected_skipped
):
    monkeypatch.setenv("THEIRSTACK_ENABLED", "true")

    class _Client:
        async def fetch_jobs(self):
            return [
                {
                    "id": "ts-1",
                    "title": "Engineer",
                    "url": "https://example.test/shared",
                    "location": "Bengaluru",
                    "job_country": "India",
                }
            ]

    existing = {
        "id": "existing-row",
        "ats_type": existing_type,
        "ats_job_id": existing_id,
        "job_url": "https://example.test/shared",
        "description": "Old",
        "skills": ["Python"],
        "skills_required": ["Python"],
        "structured_data": {"metadata_extraction_status": "complete"},
    }
    persisted = []

    async def lookup(*_args):
        return existing

    async def upsert(_db, job, **kwargs):
        persisted.append((job, kwargs["existing_job"]))

    monkeypatch.setattr(
        "app.job_ingestion.connectors.theirstack.TheirStackClient", _Client
    )
    monkeypatch.setattr(scheduler, "_get_session_local", lambda: _Session)
    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service.find_existing_ats_job_by_id_or_url",
        lookup,
    )
    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service.upsert_ats_job", upsert
    )

    result = await scheduler.sync_theirstack_jobs()

    assert result["updated"] == expected_updated
    assert result["skipped"] == expected_skipped
    assert len(persisted) == expected_updated
