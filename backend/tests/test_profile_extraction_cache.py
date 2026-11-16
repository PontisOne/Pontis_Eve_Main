"""Tests for the separate Redis-backed profile extraction cache."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import profile_extraction_cache as cache
from profile_extraction_schemas import ExtractionResult
from profile_extraction_service import extract_profile_field


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.set_calls = []

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value
        self.set_calls.append((key, value, ex))


def response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def groq_client(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=response(content))
            )
        )
    )


def result(field="skills"):
    return ExtractionResult(
        status="ok",
        target_field=field,
        value=["Python"] if field == "skills" else {"name": "AWS course"},
        normalized=True,
        confidence=0.9,
    )


def test_keys_change_for_field_answer_question_context_and_versions():
    base = {
        "candidate_response": "I know Python.",
        "suggested_question": "Which skills do you have?",
        "relevant_context": {"existing_skills": []},
        "model": "test-model",
    }
    keys = {
        cache.cache_key(target_field="skills", **base),
        cache.cache_key(target_field="education", **base),
        cache.cache_key(
            target_field="skills", **{**base, "candidate_response": "I know FastAPI."}
        ),
        cache.cache_key(
            target_field="skills",
            **{**base, "suggested_question": "Tell me your tools."},
        ),
        cache.cache_key(
            target_field="skills",
            **{**base, "relevant_context": {"existing_skills": ["Python"]}},
        ),
        cache.cache_key(target_field="skills", **{**base, "schema_version": "v2"}),
        cache.cache_key(target_field="skills", **{**base, "prompt_version": "v2"}),
        cache.cache_key(target_field="skills", **{**base, "model": "other-model"}),
    }
    assert len(keys) == 8
    assert all(key.startswith("eve:profile-extraction:v1:") for key in keys)


def test_unsupported_context_bypasses_cache_key_generation():
    key = cache.cache_key(
        target_field="skills",
        candidate_response="I know Python.",
        suggested_question=None,
        relevant_context={"unsupported": object()},
        model="test-model",
    )
    assert key is None


@pytest.mark.parametrize("configured_value", ["not-an-integer", "0", "-5"])
def test_invalid_cache_ttl_falls_back_to_default(monkeypatch, configured_value):
    monkeypatch.setenv("PROFILE_EXTRACTION_CACHE_TTL_SECONDS", configured_value)
    assert cache._configured_ttl_seconds() == cache.DEFAULT_TTL_SECONDS


@pytest.mark.asyncio
async def test_cache_miss_calls_groq_and_stores_success(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    client = groq_client('{"status":"ok","skills":["Python"],"confidence":0.9}')

    first = await extract_profile_field(
        target_field="skills",
        candidate_response="I know Python.",
        suggested_question="Which skills do you have?",
        client=client,
        model="test-model",
    )

    assert first.status == "ok"
    client.chat.completions.create.assert_awaited_once()
    assert len(redis.set_calls) == 1
    assert redis.set_calls[0][2] == cache._CACHE_TTL_SECONDS


@pytest.mark.asyncio
async def test_cache_ttl_is_configurable(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    monkeypatch.setattr(cache, "_CACHE_TTL_SECONDS", 42)
    await cache.set("eve:profile-extraction:v1:test", result())
    assert redis.set_calls[0][2] == 42


@pytest.mark.asyncio
async def test_cache_hit_skips_groq(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    cached = result("skills")
    key = cache.cache_key(
        target_field="skills",
        candidate_response="I know Python.",
        suggested_question=None,
        relevant_context=None,
        model="test-model",
    )
    redis.values[key] = cached.model_dump_json()
    client = groq_client("{}").chat.completions.create

    value = await extract_profile_field(
        target_field="skills",
        candidate_response="I know Python.",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=client))
        ),
        model="test-model",
    )

    assert value == cached
    client.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        '{"status":"empty","skills":[],"confidence":0.0}',
        '{"status":"ambiguous","skills":[],"confidence":0.3}',
        '{"status":"ok","skills":[],"confidence":0.3}',
    ],
)
async def test_non_success_results_are_not_cached(monkeypatch, payload):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    client = groq_client(payload)

    value = await extract_profile_field(
        target_field="skills",
        candidate_response="I have some experience.",
        client=client,
        model="test-model",
    )

    assert value.status in {"empty", "ambiguous", "invalid"}
    assert redis.set_calls == []


@pytest.mark.asyncio
async def test_provider_failure_is_not_cached(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    create = AsyncMock(side_effect=RuntimeError("Groq unavailable"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    value = await extract_profile_field(
        target_field="education",
        candidate_response="I studied at JNTU.",
        client=client,
        model="test-model",
    )

    assert value.status == "invalid"
    assert redis.set_calls == []


@pytest.mark.asyncio
async def test_malformed_cached_payload_is_discarded(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    key = cache.cache_key(
        target_field="certifications",
        candidate_response="I completed AWS course.",
        suggested_question=None,
        relevant_context=None,
        model="test-model",
    )
    redis.values[key] = json.dumps({"status": "ok", "target_field": "not-a-field"})
    client = groq_client(
        '{"status":"ok","certification":{"name":"AWS course",'
        '"issuer":null,"credential_type":"course"},"confidence":0.9}'
    )

    value = await extract_profile_field(
        target_field="certifications",
        candidate_response="I completed AWS course.",
        client=client,
        model="test-model",
    )

    assert value.status == "ok"
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_redis_failure_falls_back_to_groq(monkeypatch):
    monkeypatch.setattr(
        cache, "_client", AsyncMock(side_effect=TimeoutError("Redis timeout"))
    )
    client = groq_client('{"status":"ok","skills":["Python"],"confidence":0.9}')

    value = await extract_profile_field(
        target_field="skills",
        candidate_response="I know Python.",
        client=client,
        model="test-model",
    )

    assert value.status == "ok"
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "payload"),
    [
        (
            "certifications",
            (
                '{"status":"ok","certification":{"name":"AWS course",'
                '"issuer":null,"credential_type":"course"},"confidence":0.9}'
            ),
        ),
        ("skills", '{"status":"ok","skills":["Python"],"confidence":0.9}'),
        (
            "education",
            '{"status":"ok","education":[{"degree":"MBA","institution":"OU"}],"confidence":0.9}',
        ),
    ],
)
async def test_all_supported_fields_use_the_same_cache(monkeypatch, field, payload):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    client = groq_client(payload)
    answer = "I completed this information."

    await extract_profile_field(
        target_field=field,
        candidate_response=answer,
        client=client,
        model="test-model",
    )
    await extract_profile_field(
        target_field=field,
        candidate_response=answer,
        client=client,
        model="test-model",
    )

    assert client.chat.completions.create.await_count == 1
    assert len(redis.set_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("certifications", ["not-a-certification-object"]),
        ("skills", {"skill": "Python"}),
        ("education", [{"degree": "MBA"}]),
    ],
)
async def test_malformed_field_specific_cache_values_are_misses(
    monkeypatch, field, value
):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    key = cache.cache_key(
        target_field=field,
        candidate_response="Candidate answer",
        suggested_question=None,
        relevant_context=None,
        model="test-model",
    )
    redis.values[key] = json.dumps(
        {
            "status": "ok",
            "target_field": field,
            "value": value,
            "normalized": True,
            "confidence": 0.9,
        }
    )
    assert await cache.get(key) is None


@pytest.mark.asyncio
async def test_client_initialization_failure_returns_invalid_and_is_not_cached(
    monkeypatch,
):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    import profile_extraction_service as service

    def fail_initialization(*args, **kwargs):
        raise RuntimeError("missing Groq config")

    monkeypatch.setattr(service, "_client_and_model", fail_initialization)
    value = await extract_profile_field(
        target_field="skills",
        candidate_response="I know Python.",
        client=groq_client("{}").chat,
        model="test-model",
    )
    assert value.status == "invalid"
    assert redis.set_calls == []


@pytest.mark.asyncio
async def test_low_confidence_valid_result_is_informational(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    client = groq_client('{"status":"ok","skills":["Python"],"confidence":0.0}')
    value = await extract_profile_field(
        target_field="skills",
        candidate_response="I know Python.",
        client=client,
        model="test-model",
    )
    assert value.status == "ok"
    assert value.confidence == 0.0
    assert len(redis.set_calls) == 1


@pytest.mark.asyncio
async def test_cache_hit_preserves_field_result_for_downstream_duplicate_decision(
    monkeypatch,
):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    client = groq_client(
        '{"status":"ok","certification":{"name":"AWS course",'
        '"issuer":"Teks Academy","credential_type":"course"},"confidence":0.9}'
    )
    kwargs = {
        "target_field": "certifications",
        "candidate_response": "I completed an AWS course from Teks Academy.",
        "suggested_question": "Have you completed any certifications?",
        "relevant_context": {"existing_certifications": []},
        "client": client,
        "model": "test-model",
    }
    first = await extract_profile_field(**kwargs)
    second = await extract_profile_field(**kwargs)

    assert first == second
    assert first.value["name"] == "AWS course"
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_question_and_context_changes_do_not_reuse_extraction_cache(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_client", AsyncMock(return_value=redis))
    client = groq_client('{"status":"ok","skills":["Python"],"confidence":0.9}')
    base = {
        "target_field": "skills",
        "candidate_response": "I know Python.",
        "client": client,
        "model": "test-model",
    }
    await extract_profile_field(
        **base,
        suggested_question="Which skills do you have?",
        relevant_context={"existing_skills": []},
    )
    await extract_profile_field(
        **base,
        suggested_question="Which tools do you use?",
        relevant_context={"existing_skills": ["Python"]},
    )

    assert client.chat.completions.create.await_count == 2
