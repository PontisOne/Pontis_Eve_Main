"""Endpoint-handler integration tests for chat profile extraction."""

from types import SimpleNamespace

import pytest

import profile_extraction_service
import server
from profile_extraction_schemas import ExtractionResult


class _EmptyMappings:
    def fetchone(self):
        return None


class _EmptyResult:
    def mappings(self):
        return _EmptyMappings()


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def execute(self, *_args, **_kwargs):
        return _EmptyResult()


def _install_chat_fixture(monkeypatch, candidate, *, completion="I saved that."):
    persisted = []
    invalidated = []

    async def get_candidate(_candidate_id):
        return candidate

    async def persist(candidate_id, updates):
        persisted.append((candidate_id, updates))

    async def invalidate(candidate_id):
        invalidated.append(candidate_id)

    async def empty(*_args, **_kwargs):
        return None

    async def load_empty(*_args, **_kwargs):
        return []

    async def complete(**_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=completion))]
        )

    monkeypatch.setattr(server, "SessionLocal", lambda: _Session())
    monkeypatch.setattr(server, "_authorize_candidate", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_get_candidate_row", get_candidate)
    monkeypatch.setattr(server, "_load_chat_window", load_empty)
    monkeypatch.setattr(server, "_load_intake_ledger", load_empty)
    monkeypatch.setattr(server, "_save_chat_window", empty)
    monkeypatch.setattr(server, "_advance_voice_intake_from_chat", empty)
    monkeypatch.setattr(server, "_record_demonstrated_skill_usage", empty)
    monkeypatch.setattr(server, "_trigger_matching", empty)
    monkeypatch.setattr(server, "_persist_chat_profile_update_background", persist)
    monkeypatch.setattr(server.profile_cache, "invalidate", invalidate)
    monkeypatch.setattr(
        server,
        "_normalize_for_frontend",
        lambda row: {
            "skills": row.get("skills", []),
            "certifications": row.get("certifications", []),
        },
    )
    monkeypatch.setattr(server, "_build_profile_context", lambda _p: ("profile", []))
    monkeypatch.setattr(server, "_missing_canonical_preference_fields", lambda *_a: [])

    async def no_missing_questions(*_args):
        return []

    monkeypatch.setattr(
        server, "_build_authoritative_missing_questions", no_missing_questions
    )
    monkeypatch.setattr(
        server,
        "openai_client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=complete))
        ),
    )
    return persisted, invalidated


@pytest.mark.asyncio
async def test_chat_endpoint_new_certification_extracts_and_persists(monkeypatch):
    candidate = {"id": "candidate-1", "raw_data": {"certifications": []}}
    persisted, invalidated = _install_chat_fixture(monkeypatch, candidate)
    extraction_calls = []

    async def extract(**kwargs):
        extraction_calls.append(kwargs)
        return SimpleNamespace(
            status="ok",
            target_field="certifications",
            value={
                "name": "AWS course",
                "issuer": "Teks Academy",
                "credential_type": "course",
            },
        )

    monkeypatch.setattr(server, "extract_profile_field", extract)
    response = await server.chat(
        server.ChatRequest(
            candidate_id="candidate-1",
            session_id="session-1",
            messages=[
                server.ChatMessageIn(
                    role="assistant", content="Have you completed any certifications?"
                ),
                server.ChatMessageIn(
                    role="user", content="I completed an AWS course from Teks Academy."
                ),
            ],
        )
    )

    assert response.profile_updates == {"certifications": ["AWS course"]}
    assert extraction_calls[0]["target_field"] == "certifications"
    assert extraction_calls[0]["candidate_response"].startswith("I completed")
    assert persisted == [("candidate-1", {"certifications": ["AWS course"]})]
    # Persistence is mocked; its production helper owns invalidation.
    assert invalidated == []


@pytest.mark.asyncio
async def test_chat_endpoint_existing_certification_skips_extraction(monkeypatch):
    candidate = {
        "id": "candidate-1",
        "raw_data": {"certifications": ["AWS course"]},
    }
    persisted, _ = _install_chat_fixture(monkeypatch, candidate)
    extraction_calls = []

    async def extract(**kwargs):
        extraction_calls.append(kwargs)
        raise AssertionError("duplicate certification must bypass extraction")

    monkeypatch.setattr(server, "extract_profile_field", extract)
    response = await server.chat(
        server.ChatRequest(
            candidate_id="candidate-1",
            session_id="session-1",
            messages=[
                server.ChatMessageIn(
                    role="assistant", content="Have you completed any certifications?"
                ),
                server.ChatMessageIn(
                    role="user", content="I completed an AWS course from Teks Academy."
                ),
            ],
        )
    )

    assert not extraction_calls
    assert persisted == []
    assert "AWS course" in response.reply


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "message", "target_field", "extracted", "expected"),
    [
        (
            "What skills do you have?",
            "I have experience working with FastAPI and PostgreSQL.",
            "skills",
            ["FastAPI", "PostgreSQL"],
            {"skills": ["FastAPI", "PostgreSQL"]},
        ),
        (
            "What degree or education have you completed?",
            "I completed a B.Tech in Computer Science at ABC University in 2022.",
            "education",
            [
                {
                    "degree": "B.Tech",
                    "institution": "ABC University",
                    "field_of_study": "Computer Science",
                    "end_date": "2022",
                }
            ],
            {
                "education": [
                    {
                        "degree": "B.Tech",
                        "institution": "ABC University",
                        "end_date": "2022",
                    }
                ]
            },
        ),
    ],
)
async def test_chat_endpoint_routes_specialized_fields_without_fallback_contamination(
    monkeypatch, question, message, target_field, extracted, expected
):
    candidate = {
        "id": "candidate-specialized",
        "raw_data": {"skills": ["Python"], "education": []},
    }
    persisted, _ = _install_chat_fixture(monkeypatch, candidate)

    async def extract(**kwargs):
        assert kwargs["target_field"] == target_field
        return SimpleNamespace(status="ok", target_field=target_field, value=extracted)

    monkeypatch.setattr(server, "extract_profile_field", extract)
    response = await server.chat(
        server.ChatRequest(
            candidate_id="candidate-specialized",
            session_id="session-specialized",
            messages=[
                server.ChatMessageIn(role="assistant", content=question),
                server.ChatMessageIn(role="user", content=message),
            ],
        )
    )

    assert response.profile_updates == expected
    assert persisted == [("candidate-specialized", expected)]


@pytest.mark.asyncio
async def test_chat_endpoint_rejects_ambiguous_specialized_extraction(monkeypatch):
    candidate = {"id": "candidate-ambiguous", "raw_data": {"skills": []}}
    persisted, _ = _install_chat_fixture(monkeypatch, candidate)

    async def extract(**_kwargs):
        return SimpleNamespace(status="ambiguous", target_field="skills", value=None)

    monkeypatch.setattr(server, "extract_profile_field", extract)
    response = await server.chat(
        server.ChatRequest(
            candidate_id="candidate-ambiguous",
            session_id="session-ambiguous",
            messages=[
                server.ChatMessageIn(
                    role="assistant", content="What skills do you have?"
                ),
                server.ChatMessageIn(
                    role="user", content="I use some tools sometimes."
                ),
            ],
        )
    )

    assert response.profile_updates is None
    assert persisted == []


def test_generic_profile_merge_remains_available_for_non_specialized_flows():
    assert server._merge_profile_updates(
        {"location": "Teks Academy", "certifications": ["raw sentence"]},
        {"certifications": ["AWS course"]},
    ) == {
        "location": "Teks Academy",
        "certifications": ["raw sentence", "AWS course"],
    }


@pytest.mark.asyncio
async def test_chat_endpoint_extraction_cache_hit_skips_extraction_llm(monkeypatch):
    candidate = {"id": "candidate-cache-hit", "raw_data": {"skills": []}}
    persisted, _ = _install_chat_fixture(monkeypatch, candidate)
    cached = ExtractionResult(
        status="ok",
        target_field="skills",
        value=["FastAPI"],
        normalized=True,
        confidence=1.0,
    )
    cache_calls = []

    async def cache_get(key):
        cache_calls.append(("get", key))
        return cached

    async def fail_cache_set(*_args):
        raise AssertionError("cache hit must not write")

    async def fail_extraction_llm(**_kwargs):
        raise AssertionError("cache hit must skip extraction LLM")

    monkeypatch.setattr(
        profile_extraction_service.profile_extraction_cache, "get", cache_get
    )
    monkeypatch.setattr(
        profile_extraction_service.profile_extraction_cache, "set", fail_cache_set
    )
    monkeypatch.setattr(
        profile_extraction_service,
        "_client_and_model",
        lambda **_kwargs: (
            SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=fail_extraction_llm)
                )
            ),
            "test-model",
        ),
    )
    response = await server.chat(
        server.ChatRequest(
            candidate_id="candidate-cache-hit",
            session_id="session-cache-hit",
            messages=[
                server.ChatMessageIn(
                    role="assistant", content="What skills do you have?"
                ),
                server.ChatMessageIn(role="user", content="I use FastAPI."),
            ],
        )
    )

    assert cache_calls and cache_calls[0][0] == "get"
    assert response.profile_updates == {"skills": ["FastAPI"]}
    assert persisted == [("candidate-cache-hit", {"skills": ["FastAPI"]})]


@pytest.mark.asyncio
async def test_chat_endpoint_extraction_cache_miss_calls_llm_and_caches(monkeypatch):
    candidate = {"id": "candidate-cache-miss", "raw_data": {"skills": []}}
    persisted, _ = _install_chat_fixture(monkeypatch, candidate)
    cache_writes = []
    llm_calls = []

    async def cache_miss(_key):
        return None

    async def cache_set(key, result):
        cache_writes.append((key, result))

    async def extraction_llm(**kwargs):
        llm_calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"status":"ok","skills":["FastAPI"],"confidence":1.0}'
                    )
                )
            ]
        )

    monkeypatch.setattr(
        profile_extraction_service.profile_extraction_cache, "get", cache_miss
    )
    monkeypatch.setattr(
        profile_extraction_service.profile_extraction_cache, "set", cache_set
    )
    monkeypatch.setattr(
        profile_extraction_service,
        "_client_and_model",
        lambda client=None, model=None: (
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=extraction_llm))
            ),
            model or "test-model",
        ),
    )
    response = await server.chat(
        server.ChatRequest(
            candidate_id="candidate-cache-miss",
            session_id="session-cache-miss",
            messages=[
                server.ChatMessageIn(
                    role="assistant", content="What skills do you have?"
                ),
                server.ChatMessageIn(role="user", content="I use FastAPI."),
            ],
        )
    )

    assert len(llm_calls) == 1
    assert len(cache_writes) == 1
    assert cache_writes[0][1].status == "ok"
    assert response.profile_updates == {"skills": ["FastAPI"]}
    assert persisted == [("candidate-cache-miss", {"skills": ["FastAPI"]})]


@pytest.mark.asyncio
async def test_chat_endpoint_persistence_failure_does_not_confirm_or_invalidate(
    monkeypatch,
):
    candidate = {"id": "candidate-failure", "raw_data": {"skills": []}}
    _, invalidated = _install_chat_fixture(monkeypatch, candidate)

    async def extract(**_kwargs):
        return SimpleNamespace(status="ok", target_field="skills", value=["FastAPI"])

    async def fail_persist(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(server, "extract_profile_field", extract)
    monkeypatch.setattr(server, "_persist_chat_profile_update_background", fail_persist)
    response = await server.chat(
        server.ChatRequest(
            candidate_id="candidate-failure",
            session_id="session-failure",
            messages=[
                server.ChatMessageIn(
                    role="assistant", content="What skills do you have?"
                ),
                server.ChatMessageIn(role="user", content="I use FastAPI."),
            ],
        )
    )

    assert "couldn’t save" in response.reply
    assert "saved that update" not in response.reply
    assert invalidated == []
    assert candidate["raw_data"]["skills"] == []


@pytest.mark.asyncio
async def test_subsequent_chat_request_uses_updated_profile_without_reextracting(
    monkeypatch,
):
    candidate = {"id": "candidate-follow-up", "raw_data": {"skills": []}}
    persisted, _ = _install_chat_fixture(monkeypatch, candidate)
    extraction_calls = []

    async def extract(**kwargs):
        extraction_calls.append(kwargs)
        return SimpleNamespace(status="ok", target_field="skills", value=["FastAPI"])

    async def persist(_candidate_id, updates):
        persisted.append((_candidate_id, updates))
        candidate["raw_data"]["skills"].extend(updates["skills"])

    monkeypatch.setattr(server, "extract_profile_field", extract)
    monkeypatch.setattr(server, "_persist_chat_profile_update_background", persist)
    first = await server.chat(
        server.ChatRequest(
            candidate_id="candidate-follow-up",
            session_id="session-follow-up",
            messages=[
                server.ChatMessageIn(
                    role="assistant", content="What skills do you have?"
                ),
                server.ChatMessageIn(role="user", content="I use FastAPI."),
            ],
        )
    )
    second = await server.chat(
        server.ChatRequest(
            candidate_id="candidate-follow-up",
            session_id="session-follow-up",
            messages=[
                server.ChatMessageIn(
                    role="assistant", content="What skills do you have?"
                ),
                server.ChatMessageIn(role="user", content="FastAPI is my main tool."),
            ],
        )
    )

    assert first.profile_updates == {"skills": ["FastAPI"]}
    assert len(extraction_calls) == 1
    assert "FastAPI" in second.reply
    assert persisted[0][1] == {"skills": ["FastAPI"]}
