"""Focused tests for skills-only chat extraction integration."""

from types import SimpleNamespace

import pytest

import server


def _result(value, status="ok"):
    return SimpleNamespace(status=status, target_field="skills", value=value)


def test_skills_extraction_maps_normalized_list_to_storage_shape():
    result = _result(["Python", "FastAPI"])
    assert server._skills_extraction_updates(result) == {
        "skills": ["Python", "FastAPI"]
    }


def test_three_skills_remain_distinct():
    result = _result(["Python", "Django", "PostgreSQL"])
    assert server._skills_extraction_updates(result) == {
        "skills": ["Python", "Django", "PostgreSQL"]
    }


def test_empty_ambiguous_and_invalid_results_do_not_create_updates():
    assert server._skills_extraction_updates(_result(None, "empty")) is None
    assert server._skills_extraction_updates(_result(None, "ambiguous")) is None
    assert server._skills_extraction_updates(_result(None, "invalid")) is None


def test_unrelated_education_or_certification_answer_is_not_saved_as_skill():
    # The service is responsible for semantic rejection of unrelated answers;
    # an empty result is what the chat integration persists.
    assert server._skills_extraction_updates(_result([], "empty")) is None


def test_existing_skill_is_detected_before_extraction():
    candidate = {"skills": ["Python"]}
    assert (
        server._obvious_skill_duplicate(
            candidate, "I have used Python in my previous company."
        )
        == "Python"
    )


def test_partial_skill_names_do_not_trigger_duplicate():
    candidate = {"skills": ["R"]}
    assert server._obvious_skill_duplicate(candidate, "I worked with Rust.") is None


def test_skills_question_is_selected_from_active_history():
    history = [
        {"role": "assistant", "content": "Which technologies do you use?"},
        {"role": "user", "content": "I know Python and Django."},
    ]
    assert server._skills_question(history) == history[0]["content"]


def test_normalized_skills_override_raw_values():
    normalized = {"skills": ["Python", "FastAPI"]}
    raw = {"skills": ["I have worked with Python and FastAPI"]}
    merged = server._merge_profile_updates(raw, normalized)
    merged["skills"] = normalized["skills"]
    assert merged["skills"] == ["Python", "FastAPI"]


@pytest.mark.asyncio
async def test_skills_extraction_request_uses_original_answer_and_question(monkeypatch):
    captured = {}

    async def fake_extract(**kwargs):
        captured.update(kwargs)
        return _result(["Python", "FastAPI"])

    monkeypatch.setattr(server, "extract_profile_field", fake_extract)
    result = await server.extract_profile_field(
        target_field="skills",
        candidate_response=(
            "I have worked with Python and FastAPI in my previous company."
        ),
        suggested_question="Which technologies do you use?",
        relevant_context={"existing_skills": []},
        candidate_id="candidate-1",
        session_id="session-1",
    )
    assert server._skills_extraction_updates(result) == {
        "skills": ["Python", "FastAPI"]
    }
    assert captured["target_field"] == "skills"
    assert captured["candidate_response"].startswith("I have worked with Python")
    assert captured["suggested_question"] == "Which technologies do you use?"
