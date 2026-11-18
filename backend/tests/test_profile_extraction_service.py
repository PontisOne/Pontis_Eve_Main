"""Unit tests for the standalone field-specific profile extraction service."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from profile_extraction_service import extract_profile_field


def response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def client_for(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=response(content))
            )
        )
    )


@pytest.mark.asyncio
async def test_certification_removes_noise_and_preserves_issuer():
    client = client_for(
        '{"status":"ok","certification":{"name":"AWS course",'
        '"issuer":"Teks Academy","credential_type":"course"},'
        '"confidence":0.96}'
    )
    result = await extract_profile_field(
        target_field="certifications",
        suggested_question="Have you completed any certifications?",
        candidate_response="I have done AWS course from Teks Academy.",
        client=client,
        model="test-model",
    )
    assert result.status == "ok"
    assert result.value == {
        "name": "AWS course",
        "issuer": "Teks Academy",
        "credential_type": "course",
    }
    assert result.normalized is True


@pytest.mark.asyncio
async def test_skill_extraction_returns_only_named_skills():
    client = client_for(
        '{"status":"ok","skills":["Python","FastAPI"],"confidence":0.9}'
    )
    result = await extract_profile_field(
        target_field="skills",
        candidate_response=(
            "I have worked with Python and FastAPI in my previous company."
        ),
        client=client,
        model="test-model",
    )
    assert result.value == ["Python", "FastAPI"]


@pytest.mark.asyncio
async def test_education_extraction_preserves_degree_institution_and_dates():
    client = client_for(
        '{"status":"ok","education":[{"degree":"B.Tech",'
        '"institution":"ABC University","start_date":"2018",'
        '"end_date":"2022"}],"confidence":0.95}'
    )
    result = await extract_profile_field(
        target_field="education",
        candidate_response="I completed my B.Tech at ABC University from 2018 to 2022.",
        client=client,
        model="test-model",
    )
    assert result.status == "ok"
    assert result.value[0]["institution"] == "ABC University"


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["", "   "])
async def test_empty_answer_does_not_call_llm(answer):
    client = client_for("{}").chat.completions.create
    result = await extract_profile_field(
        target_field="skills",
        candidate_response=answer,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=client))
        ),
        model="test-model",
    )
    assert result.status == "empty"
    client.assert_not_awaited()


@pytest.mark.asyncio
async def test_ambiguous_result_is_explicit():
    client = client_for(
        '{"status":"ambiguous","skills":[],"confidence":0.35,'
        '"ambiguity_reason":"No specific skill was named"}'
    )
    result = await extract_profile_field(
        target_field="skills",
        candidate_response="I have some experience.",
        client=client,
        model="test-model",
    )
    assert result.status == "ambiguous"
    assert result.value is None
    assert result.ambiguity_reason == "No specific skill was named"


@pytest.mark.asyncio
async def test_invalid_json_returns_invalid_result():
    client = client_for("not json")
    result = await extract_profile_field(
        target_field="certifications",
        candidate_response="I completed AWS training.",
        client=client,
        model="test-model",
    )
    assert result.status == "invalid"
    assert result.value is None


@pytest.mark.asyncio
async def test_missing_required_fields_returns_invalid_result():
    client = client_for('{"status":"ok","confidence":0.9}')
    result = await extract_profile_field(
        target_field="skills",
        candidate_response="Python is one of my skills.",
        client=client,
        model="test-model",
    )
    assert result.status == "invalid"


@pytest.mark.asyncio
async def test_provider_error_returns_invalid_result():
    create = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = await extract_profile_field(
        target_field="education",
        candidate_response="I studied at ABC University.",
        client=client,
        model="test-model",
    )
    assert result.status == "invalid"


@pytest.mark.asyncio
async def test_confidence_is_validated():
    client = client_for('{"status":"ok","skills":["Python"],"confidence":1.5}')
    result = await extract_profile_field(
        target_field="skills",
        candidate_response="I know Python.",
        client=client,
        model="test-model",
    )
    assert result.status == "invalid"


@pytest.mark.asyncio
async def test_request_contains_target_question_and_original_response():
    client = client_for('{"status":"ok","skills":["Python"],"confidence":0.9}')
    await extract_profile_field(
        target_field="skills",
        suggested_question="Which technologies do you use?",
        candidate_response="I have used Python.",
        relevant_context={"existing_values": []},
        client=client,
        model="test-model",
    )
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"][1]["content"].find('"target_field": "skills"') >= 0
    assert "Which technologies do you use?" in kwargs["messages"][1]["content"]
    assert "I have used Python." in kwargs["messages"][1]["content"]
    assert kwargs["_telemetry"]["workflow"] == "profile_field_extraction"
