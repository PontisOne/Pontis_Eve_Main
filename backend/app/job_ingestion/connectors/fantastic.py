"""Async, budget-conscious client for Fantastic.jobs' global active ATS feed."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)
BASE_URL = "https://data.fantastic.jobs/v1"


class FantasticAPIError(RuntimeError):
    """Error raised when Fantastic prevents synchronization from continuing."""


@dataclass(frozen=True)
class FantasticConfig:
    api_key: str
    time_frame: str = "24h"
    limit: int = 25
    max_pages: int = 1
    max_jobs_per_run: int = 25
    max_requests_per_run: int = 2
    timeout: float = 20.0

    @classmethod
    def from_env(cls) -> "FantasticConfig":
        def integer(name: str, default: int, minimum: int = 1) -> int:
            try:
                return max(minimum, int(os.getenv(name, default)))
            except ValueError:
                return default

        try:
            timeout = max(1.0, float(os.getenv("FANTASTIC_TIMEOUT", "20")))
        except ValueError:
            timeout = 20.0
        return cls(
            api_key=os.getenv("FANTASTIC_JOBS_API_KEY", "").strip(),
            time_frame=os.getenv("FANTASTIC_TIME_FRAME", "24h").strip() or "24h",
            limit=integer("FANTASTIC_LIMIT", 25),
            max_pages=integer("FANTASTIC_MAX_PAGES", 1),
            max_jobs_per_run=integer("FANTASTIC_MAX_JOBS_PER_RUN", 25),
            max_requests_per_run=integer("FANTASTIC_MAX_REQUESTS_PER_RUN", 2),
            timeout=timeout,
        )


def _jobs_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Accept documented collection envelopes without accepting malformed jobs."""
    if isinstance(payload, list):
        raw_jobs: Any = payload
    elif isinstance(payload, dict):
        raw_jobs = payload.get("jobs", payload.get("data", payload.get("results", [])))
    else:
        return []
    return (
        [job for job in raw_jobs if isinstance(job, dict)]
        if isinstance(raw_jobs, list)
        else []
    )


class FantasticClient:
    """Fetch active and expired jobs with bounded requests and pagination."""

    @staticmethod
    def _empty_fetch_stats(endpoint: str | None = None) -> dict[str, Any]:
        return {
            "endpoint": endpoint,
            "fetched": 0,
            "pages_requested": 0,
            "requests_made": 0,
            "success": True,
            "partial": False,
            "budget_exhausted": False,
            "stop_reason": None,
            "error": None,
        }

    def __init__(
        self,
        config: FantasticConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        """Initialize the client and track request usage and fetch statistics."""
        self.config = config or FantasticConfig.from_env()
        self.client = client
        self._requests_used = 0
        self.last_fetch_stats = self._empty_fetch_stats()

    def _validate_api_key(self) -> None:
        if not self.config.api_key:
            raise FantasticAPIError("FANTASTIC_JOBS_API_KEY is not configured")

    def _build_auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}

    @staticmethod
    def _classify_http_status(status_code: int) -> str | None:
        if status_code == 429:
            return "rate_limited"
        if status_code in {401, 403}:
            return "authorization_error"
        if status_code >= 500:
            return "server_error"
        return None

    @asynccontextmanager
    async def _managed_client(self):
        if self.client is not None:
            yield self.client
            return

        client = httpx.AsyncClient(timeout=self.config.timeout)
        try:
            yield client
        finally:
            await client.aclose()

    async def _fetch_feed(self, endpoint: str) -> list[dict[str, Any]]:
        """Fetch jobs from an endpoint with bounded pagination and retries."""
        self.last_fetch_stats = self._empty_fetch_stats(endpoint)
        try:
            self._validate_api_key()
        except FantasticAPIError:
            self.last_fetch_stats.update(
                success=False,
                partial=True,
                stop_reason="authorization_error",
                error="authorization_error",
            )
            raise
        jobs: list[dict[str, Any]] = []
        requests_made = 0
        pages_requested = 0

        def update_stats(**values: Any) -> None:
            self.last_fetch_stats.update(
                fetched=len(jobs),
                pages_requested=pages_requested,
                requests_made=requests_made,
                **values,
            )

        async def send_request() -> httpx.Response:
            nonlocal requests_made
            requests_made += 1
            self._requests_used += 1
            try:
                return await client.get(
                    f"{BASE_URL}/{endpoint}",
                    params=params,
                    headers=self._build_auth_headers(),
                )
            except httpx.TimeoutException as exc:
                update_stats(
                    success=False, partial=True, stop_reason="timeout", error="timeout"
                )
                raise FantasticAPIError("Fantastic request timed out") from exc
            except httpx.RequestError as exc:
                update_stats(
                    success=False,
                    partial=True,
                    stop_reason="request_error",
                    error="request_error",
                )
                raise FantasticAPIError(f"Fantastic request failed: {exc}") from exc

        async with self._managed_client() as client:
            for page in range(self.config.max_pages):
                remaining = self.config.max_jobs_per_run - len(jobs)
                if remaining <= 0:
                    update_stats(partial=True, stop_reason="max_jobs_per_run")
                    break
                if self._requests_used >= self.config.max_requests_per_run:
                    update_stats(
                        partial=True,
                        budget_exhausted=True,
                        stop_reason="budget_exhausted",
                    )
                    break
                page_limit = min(self.config.limit, remaining)
                params = {
                    "time_frame": self.config.time_frame,
                    "limit": page_limit,
                    "offset": page * self.config.limit,
                    "description_format": "text",
                    "location": "India",
                }
                if endpoint == "active-jb":
                    params["exclude_ats_duplicate"] = "true"
                logger.info(
                    "[fantastic] requesting %s page=%d request=%d",
                    endpoint,
                    page + 1,
                    page + 1,
                )
                pages_requested += 1
                response = await send_request()
                headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower().startswith(
                        ("x-api-", "x-rate", "rate-limit", "retry-after")
                    )
                }
                logger.info(
                    "[fantastic] response status=%d headers=%s",
                    response.status_code,
                    headers,
                )
                if response.status_code == 429:
                    retry_sent = False
                    # A single bounded retry respects Retry-After.
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = min(float(retry_after or 0), 30.0)
                    except ValueError:
                        delay = 0
                    if delay and page == 0:
                        await asyncio.sleep(delay)
                        if self._requests_used < self.config.max_requests_per_run:
                            retry_sent = True
                            response = await send_request()
                    if response.status_code == 429:
                        if (
                            delay
                            and page == 0
                            and not retry_sent
                            and self._requests_used >= self.config.max_requests_per_run
                        ):
                            update_stats(
                                partial=True,
                                budget_exhausted=True,
                                stop_reason="budget_exhausted",
                            )
                        else:
                            update_stats(
                                success=False,
                                partial=True,
                                stop_reason="rate_limited",
                                error="rate_limited",
                            )
                        raise FantasticAPIError("Fantastic rate limit reached")
                status_category = self._classify_http_status(response.status_code)
                if status_category == "authorization_error":
                    update_stats(
                        success=False,
                        partial=True,
                        stop_reason=status_category,
                        error=status_category,
                    )
                    raise FantasticAPIError(
                        f"Fantastic authorization failed (HTTP {response.status_code})"
                    )
                if status_category == "server_error":
                    update_stats(
                        success=False,
                        partial=True,
                        stop_reason=status_category,
                        error=status_category,
                    )
                    raise FantasticAPIError(
                        f"Fantastic server error (HTTP {response.status_code})"
                    )
                try:
                    response.raise_for_status()
                    batch = _jobs_from_payload(response.json())
                except httpx.HTTPStatusError as exc:
                    update_stats(
                        success=False,
                        partial=True,
                        stop_reason="http_error",
                        error="http_error",
                    )
                    raise FantasticAPIError(
                        "Fantastic returned an unexpected HTTP error"
                    ) from exc
                except ValueError as exc:
                    update_stats(
                        success=False,
                        partial=True,
                        stop_reason="malformed_response",
                        error="malformed_response",
                    )
                    raise FantasticAPIError(
                        "Fantastic returned a malformed response"
                    ) from exc
                jobs.extend(batch)
                logger.info(
                    "[fantastic] fetched %d jobs (total=%d)", len(batch), len(jobs)
                )
                update_stats()
                if not batch:
                    update_stats(stop_reason="empty_response")
                    break
                if len(batch) < page_limit:
                    update_stats(stop_reason="short_page")
                    break
            result = jobs[: self.config.max_jobs_per_run]
            if self.last_fetch_stats["stop_reason"] is None:
                if len(jobs) >= self.config.max_jobs_per_run:
                    update_stats(partial=True, stop_reason="max_jobs_per_run")
                elif pages_requested >= self.config.max_pages:
                    update_stats(partial=True, stop_reason="max_pages")
            self.last_fetch_stats["fetched"] = len(result)
            return result

    async def fetch_active_ats(self) -> list[dict[str, Any]]:
        """Fetch active ATS jobs from Fantastic.jobs."""
        return await self._fetch_feed("active-ats")

    async def fetch_active_job_boards(self) -> list[dict[str, Any]]:
        """Fetch active job-board jobs from Fantastic.jobs."""
        return await self._fetch_feed("active-jb")

    async def _fetch_expired_ids(self, endpoint: str, time_frame: str) -> list[str]:
        self._validate_api_key()
        async with self._managed_client() as client:
            try:
                self._requests_used += 1
                response = await client.get(
                    f"{BASE_URL}/{endpoint}",
                    params={"time_frame": time_frame},
                    headers=self._build_auth_headers(),
                )
                status_category = self._classify_http_status(response.status_code)
                if status_category == "rate_limited":
                    raise FantasticAPIError("Fantastic rate limit reached")
                if status_category == "authorization_error":
                    raise FantasticAPIError(
                        f"Fantastic authorization failed (HTTP {response.status_code})"
                    )
                if status_category == "server_error":
                    raise FantasticAPIError(
                        f"Fantastic server error (HTTP {response.status_code})"
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise FantasticAPIError("Fantastic expired response was malformed")
                return [str(value).strip() for value in payload if str(value).strip()]
            except FantasticAPIError:
                raise
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                raise FantasticAPIError("Fantastic expiration request failed") from exc
            except (httpx.HTTPStatusError, ValueError) as exc:
                raise FantasticAPIError(
                    "Fantastic expired response was malformed"
                ) from exc

    async def fetch_expired_ats_ids(self, time_frame: str = "1d") -> list[str]:
        """Return Fantastic-confirmed expired ATS IDs for the requested window."""
        return await self._fetch_expired_ids("expired-ats", time_frame)

    async def fetch_expired_job_boards(self, time_frame: str = "1d") -> list[str]:
        """Fetch cleaned IDs of job-board listings marked expired by Fantastic."""
        return await self._fetch_expired_ids("expired-jb", time_frame)
