"""Bounded TheirStack jobs API client."""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)
BASE_URL = "https://api.theirstack.com"


class TheirStackAPIError(RuntimeError):
    pass


MAX_RATE_LIMIT_RETRIES = 3
MAX_RATE_LIMIT_WAIT_SECONDS = 5.0
RATE_LIMIT_FALLBACK_WAIT_SECONDS = 0.5


@dataclass(frozen=True)
class TheirStackConfig:
    api_key: str
    limit: int = 25
    max_pages: int = 1
    max_jobs_per_run: int = 25
    timeout: float = 20.0

    @classmethod
    def from_env(cls):
        def integer(name, default):
            try:
                return max(1, int(os.getenv(name, default)))
            except (TypeError, ValueError):
                return default

        try:
            timeout = max(1.0, float(os.getenv("THEIRSTACK_TIMEOUT", "20")))
        except ValueError:
            timeout = 20.0
        return cls(
            os.getenv("THEIRSTACK_API_KEY", "").strip(),
            integer("THEIRSTACK_LIMIT", 25),
            integer("THEIRSTACK_MAX_PAGES", 1),
            integer("THEIRSTACK_MAX_JOBS_PER_RUN", 25),
            timeout,
        )


class TheirStackClient:
    def __init__(self, config=None, client=None):
        self.config = config or TheirStackConfig.from_env()
        self.client = client

    @staticmethod
    def _raise_for_api_status(response):
        if response.status_code == 402:
            logger.error("TheirStack API request rejected: insufficient API credits")
            raise TheirStackAPIError("TheirStack has insufficient API credits")
        if response.status_code == 429:
            raise TheirStackAPIError("TheirStack rate limit reached")
        if response.status_code in (401, 403):
            raise TheirStackAPIError(
                f"TheirStack authorization failed (HTTP {response.status_code})"
            )
        if response.status_code >= 500:
            raise TheirStackAPIError(
                f"TheirStack server error (HTTP {response.status_code})"
            )

    async def _search_jobs(self, client, payload: dict):
        return await self._post_with_rate_limit_retry(
            client,
            f"{BASE_URL}/v1/jobs/search",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

    @staticmethod
    def _extract_jobs(
        data, strict: bool = False, *, include_source_length: bool = False
    ):
        jobs = data.get("jobs", data.get("data", [])) if isinstance(data, dict) else []
        if not isinstance(jobs, list):
            if strict:
                raise TheirStackAPIError("TheirStack returned a malformed response")
            extracted = []
        else:
            extracted = [job for job in jobs if isinstance(job, dict)]
        if include_source_length:
            return extracted, len(jobs) if isinstance(jobs, list) else 0
        return extracted

    @staticmethod
    def _rate_limit_wait_seconds(headers, retry_number: int) -> float:
        """Parse documented rate-limit headers and return a bounded retry wait."""
        headers = headers or {}
        rate_limit = headers.get("RateLimit", "")
        policy = headers.get("RateLimit-Policy", "")
        limit_values = headers.get("RateLimit-Limit", "")
        remaining_values = headers.get("RateLimit-Remaining", "")
        reset_values = headers.get("RateLimit-Reset", "")
        logger.debug(
            "TheirStack rate-limit headers rate_limit=%r policy=%r "
            "limit=%r remaining=%r reset=%r",
            rate_limit,
            policy,
            limit_values,
            remaining_values,
            reset_values,
        )

        def numbers(value: str) -> list[float]:
            return [
                float(item)
                for item in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", str(value))
            ]

        reset_seconds = numbers(reset_values)
        if not reset_seconds:
            reset_seconds = [
                float(value)
                for value in re.findall(
                    r"(?:^|;)\s*t\s*=\s*(\d+(?:\.\d+)?)", str(rate_limit)
                )
            ]
        wait = min((value for value in reset_seconds if value >= 0), default=None)
        if wait is None:
            wait = RATE_LIMIT_FALLBACK_WAIT_SECONDS * (2**retry_number)
        return min(max(wait, 0.0), MAX_RATE_LIMIT_WAIT_SECONDS)

    async def _post_with_rate_limit_retry(
        self, client, url: str, *, json: dict, headers: dict
    ):
        for retry_number in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = await client.post(url, json=json, headers=headers)
            except httpx.TimeoutException as exc:
                raise TheirStackAPIError("TheirStack request timed out") from exc
            except httpx.RequestError as exc:
                raise TheirStackAPIError("TheirStack request failed") from exc
            if response.status_code != 429 or retry_number >= MAX_RATE_LIMIT_RETRIES:
                return response
            wait = self._rate_limit_wait_seconds(
                getattr(response, "headers", None), retry_number
            )
            logger.warning(
                "TheirStack rate limit reached; retrying attempt=%d/%d "
                "wait_seconds=%.3f",
                retry_number + 1,
                MAX_RATE_LIMIT_RETRIES,
                wait,
            )
            await asyncio.sleep(wait)
        raise AssertionError("rate-limit retry loop did not return")

    async def fetch_credit_balance(self, *, client=None) -> int:
        """Return the available API-credit balance without making a job search."""
        if not self.config.api_key:
            raise TheirStackAPIError("THEIRSTACK_API_KEY is not configured")
        client = client or self.client
        owned = client is None
        client = client or httpx.AsyncClient(timeout=self.config.timeout)
        try:
            try:
                response = await client.get(
                    f"{BASE_URL}/v0/billing/credit-balance",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Accept": "application/json",
                    },
                )
            except httpx.TimeoutException as exc:
                raise TheirStackAPIError(
                    "TheirStack credit balance request timed out"
                ) from exc
            except httpx.RequestError as exc:
                raise TheirStackAPIError(
                    "TheirStack credit balance request failed"
                ) from exc
            self._raise_for_api_status(response)
            try:
                response.raise_for_status()
                data = response.json()
            except (httpx.HTTPStatusError, ValueError) as exc:
                raise TheirStackAPIError(
                    "TheirStack returned an invalid credit balance response"
                ) from exc
            credits = data.get("api_credits") if isinstance(data, dict) else None
            if (
                isinstance(credits, bool)
                or not isinstance(credits, (int, float))
                or credits < 0
            ):
                raise TheirStackAPIError(
                    "TheirStack returned an invalid API credit balance"
                )
            return int(credits)
        finally:
            if owned:
                await client.aclose()

    async def fetch_jobs(self) -> list[dict[str, Any]]:
        if not self.config.api_key:
            raise TheirStackAPIError("THEIRSTACK_API_KEY is not configured")
        owned = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.config.timeout)
        jobs: list[dict[str, Any]] = []
        try:
            available_credits = await self.fetch_credit_balance(client=client)
            credit_budget = min(available_credits, self.config.max_jobs_per_run)
            if credit_budget <= 0:
                logger.info("TheirStack job search skipped: no API credits available")
                return []
            for page in range(self.config.max_pages):
                remaining = credit_budget - len(jobs)
                if remaining <= 0:
                    break
                payload = {
                    "limit": min(self.config.limit, remaining),
                    "page": page,
                    "posted_at_max_age_days": 30,
                    "job_country_code_or": ["IN"],
                    "is_closed": False,
                }
                response = await self._search_jobs(client, payload)
                self._raise_for_api_status(response)
                try:
                    response.raise_for_status()
                    data = response.json()
                except (httpx.HTTPStatusError, ValueError) as exc:
                    raise TheirStackAPIError(
                        "TheirStack returned a malformed response"
                    ) from exc
                batch, source_batch_length = self._extract_jobs(
                    data, include_source_length=True
                )
                jobs.extend(batch)
                if source_batch_length < payload["limit"]:
                    break
            return jobs[: self.config.max_jobs_per_run]
        finally:
            if owned:
                await client.aclose()

    async def fetch_jobs_by_ids(self, job_ids: list[str]) -> list[dict[str, Any]]:
        """Look up stored jobs in one exact-ID batch."""
        if not self.config.api_key:
            raise TheirStackAPIError("THEIRSTACK_API_KEY is not configured")
        if not job_ids:
            return []
        owned = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.config.timeout)
        try:
            payload = {
                "job_id_or": [int(job_id) for job_id in job_ids],
                "is_closed": True,
                "limit": len(job_ids),
                "page": 0,
            }
            response = await self._search_jobs(client, payload)
            self._raise_for_api_status(response)
            response.raise_for_status()
            data = response.json()
            return self._extract_jobs(data, strict=True)
        except TheirStackAPIError:
            raise
        except (httpx.HTTPStatusError, ValueError) as exc:
            raise TheirStackAPIError(
                "TheirStack returned a malformed response"
            ) from exc
        finally:
            if owned:
                await client.aclose()
