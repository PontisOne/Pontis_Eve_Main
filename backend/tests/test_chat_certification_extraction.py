"""Focused tests for certifications-only chat extraction integration."""

from types import SimpleNamespace

import pytest

import server


def _result(value, status="ok"):
    return SimpleNamespace(status=status, target_field="certifications", value=value)


def test_certification_extraction_maps_only_name_to_storage_shape():
    result = _result(
        {"name": "AWS course", "issuer": "Teks Academy", "credential_type": "course"}
    )
    assert server._certification_extraction_updates(result) == {
        "certifications": ["AWS course"]
    }


def test_empty_ambiguous_and_invalid_results_do_not_create_updates():
    assert server._certification_extraction_updates(_result(None, "empty")) is None
    assert server._certification_extraction_updates(_result(None, "ambiguous")) is None
    assert server._certification_extraction_updates(_result(None, "invalid")) is None


def test_existing_certification_is_detected_before_extraction():
    candidate = {"raw_data": {"certifications": ["AWS course"]}}
    assert (
        server._obvious_certification_duplicate(
            candidate, "I have completed an AWS course from Teks Academy."
        )
        == "AWS course"
    )


def test_unrelated_answer_is_not_classified_as_certification():
    candidate = {"raw_data": {"certifications": ["AWS course"]}}
    assert (
        server._obvious_certification_duplicate(
            candidate, "I completed my B.Tech at ABC University."
        )
        is None
    )


def test_certification_question_is_selected_from_active_history():
    history = [
        {"role": "assistant", "content": "Have you completed any certifications?"},
        {"role": "user", "content": "I have done AWS course from Teks Academy."},
    ]
    assert server._certification_question(history) == history[0]["content"]


def test_normalized_result_overrides_raw_authoritative_value():
    normalized = {"certifications": ["AWS course"]}
    raw_authoritative = {"certifications": ["I have done AWS course from Teks Academy"]}
    merged = server._merge_profile_updates(raw_authoritative, normalized)
    merged["certifications"] = normalized["certifications"]
    assert merged["certifications"] == ["AWS course"]


@pytest.mark.asyncio
async def test_extraction_service_receives_certification_context(monkeypatch):
    captured = {}

    async def fake_extract(**kwargs):
        captured.update(kwargs)
        return _result(
            {
                "name": "AWS course",
                "issuer": "Teks Academy",
                "credential_type": "course",
            }
        )

    monkeypatch.setattr(server, "extract_profile_field", fake_extract)
    extraction = await server.extract_profile_field(
        target_field="certifications",
        candidate_response="I have done AWS course from Teks Academy.",
        suggested_question="Have you completed any certifications?",
        relevant_context={"existing_certifications": []},
        candidate_id="candidate-1",
        session_id="session-1",
    )
    assert server._certification_extraction_updates(extraction) == {
        "certifications": ["AWS course"]
    }
    assert captured["target_field"] == "certifications"
    assert captured["candidate_response"].startswith("I have done AWS")
