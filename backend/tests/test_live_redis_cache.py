"""Opt-in live Redis coverage for the Eve caches."""

import json
import os

import pytest
from redis.asyncio import Redis

import profile_cache
import profile_extraction_cache
from profile_extraction_schemas import ExtractionResult

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL")


def _safe_test_url(value: str | None) -> bool:
    return bool(value and value.startswith("redis://127.0.0.1:"))


if not _safe_test_url(TEST_REDIS_URL):
    pytest.skip(
        "Live Redis cache tests require an explicit local TEST_REDIS_URL",
        allow_module_level=True,
    )


@pytest.mark.asyncio
async def test_live_profile_and_extraction_caches() -> None:
    client = Redis.from_url(TEST_REDIS_URL, decode_responses=True)
    original_client = profile_cache._redis_client
    original_attempted = profile_cache._redis_attempted
    profile_cache._redis_client = client
    profile_cache._redis_attempted = True
    profile_id = "live-redis-cache-test"
    key = profile_extraction_cache.cache_key(
        target_field="skills",
        candidate_response="I use Python",
        suggested_question=None,
        relevant_context={},
        model="test",
    )
    assert key is not None
    try:
        await client.delete(profile_cache._key(profile_id), key)
        await profile_cache.set(profile_id, {"skills": ["Python"]})
        assert await profile_cache.get(profile_id) == {"skills": ["Python"]}
        assert await client.ttl(profile_cache._key(profile_id)) > 0

        result = ExtractionResult(
            status="ok",
            target_field="skills",
            confidence=0.9,
            value=["Python"],
        )
        await profile_extraction_cache.set(key, result)
        cached = await profile_extraction_cache.get(key)
        assert cached is not None
        assert cached.value == ["Python"]
        assert await client.ttl(key) > 0

        await client.set(key, json.dumps({"status": "ok", "value": "invalid"}))
        assert await profile_extraction_cache.get(key) is None
        await profile_cache.invalidate(profile_id)
        assert await profile_cache.get(profile_id) is None
    finally:
        await client.delete(profile_cache._key(profile_id), key)
        await client.aclose()
        profile_cache._redis_client = original_client
        profile_cache._redis_attempted = original_attempted
