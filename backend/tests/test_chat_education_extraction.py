"""Focused tests for education-only chat extraction integration."""

from types import SimpleNamespace

import pytest

import server


def _result(value, status="ok"):
    return SimpleNamespace(status=status, target_field="education", value=value)


def test_completed_degree_maps_specialization_and_year_to_existing_shape():
    result = _result(
        [
            {
                "degree": "B.Tech",
                "specialization": "Computer Science",
                "institution": "JNTU Hyderabad",
                "graduation_year": 2022,
                "status": "completed",
            }
        ]
    )
    assert server._education_extraction_updates(result) == {
        "education": [
            {
                "degree": "B.Tech",
                "institution": "JNTU Hyderabad",
                "field_of_study": "Computer Science",
                "end_date": "2022",
            }
        ]
    }


def test_pursuing_degree_does_not_invent_specialization_or_year():
    result = _result(
        [
            {
                "degree": "MBA",
                "institution": "Osmania University",
                "status": "pursuing",
            }
        ]
    )
    assert server._education_extraction_updates(result) == {
        "education": [{"degree": "MBA", "institution": "Osmania University"}]
    }


def test_diploma_preserves_specialization_and_completion_year():
    result = _result(
        [
            {
                "degree": "Diploma",
                "specialization": "Mechanical Engineering",
                "institution": "ABC Institute",
                "graduation_year": 2020,
                "status": "completed",
            }
        ]
    )
    updates = server._education_extraction_updates(result)
    assert updates["education"][0]["field_of_study"] == "Mechanical Engineering"
    assert updates["education"][0]["end_date"] == "2020"


def test_institution_only_answer_is_not_persisted_as_incomplete_record():
    result = _result([{"institution": "JNTU Hyderabad"}])
    assert server._education_extraction_updates(result) is None


def test_empty_ambiguous_and_invalid_results_do_not_create_updates():
    assert server._education_extraction_updates(_result(None, "empty")) is None
    assert server._education_extraction_updates(_result(None, "ambiguous")) is None
    assert server._education_extraction_updates(_result(None, "invalid")) is None


def test_same_degree_and_institution_is_duplicate():
    candidate = {"education": [{"degree": "B.Tech", "institution": "JNTU Hyderabad"}]}
    assert (
        server._education_duplicate(
            candidate, {"degree": "B.Tech", "institution": "JNTU Hyderabad"}
        )
        == candidate["education"][0]
    )


def test_same_institution_different_degree_is_not_duplicate():
    candidate = {"education": [{"degree": "B.Tech", "institution": "JNTU Hyderabad"}]}
    assert (
        server._education_duplicate(
            candidate, {"degree": "MBA", "institution": "JNTU Hyderabad"}
        )
        is None
    )


def test_obvious_duplicate_can_be_detected_before_extraction():
    candidate = {"education": [{"degree": "B.Tech", "institution": "JNTU Hyderabad"}]}
    assert (
        server._obvious_education_duplicate(
            candidate, "I completed B.Tech from JNTU Hyderabad in 2022."
        )
        == "B.Tech"
    )


def test_education_question_is_selected_from_active_history():
    history = [
        {"role": "assistant", "content": "What is your highest level of education?"},
        {"role": "user", "content": "I completed B.Tech from JNTU Hyderabad in 2022."},
    ]
    assert server._education_question(history) == history[0]["content"]


def test_normalized_education_overrides_raw_value():
    normalized = {"education": [{"degree": "B.Tech", "institution": "JNTU Hyderabad"}]}
    raw = {
        "education": [
            {"degree": "I completed B.Tech from JNTU Hyderabad", "institution": ""}
        ]
    }
    merged = server._merge_profile_updates(raw, normalized)
    merged["education"] = normalized["education"]
    assert merged["education"] == normalized["education"]


@pytest.mark.asyncio
async def test_education_extraction_receives_original_answer_question_and_context(
    monkeypatch,
):
    captured = {}

    async def fake_extract(**kwargs):
        captured.update(kwargs)
        return _result(
            [
                {
                    "degree": "B.Tech",
                    "specialization": "Computer Science",
                    "institution": "JNTU Hyderabad",
                    "graduation_year": 2022,
                    "status": "completed",
                }
            ]
        )

    monkeypatch.setattr(server, "extract_profile_field", fake_extract)
    result = await server.extract_profile_field(
        target_field="education",
        candidate_response=(
            "I completed B.Tech in Computer Science from JNTU Hyderabad in 2022."
        ),
        suggested_question="What is your highest level of education?",
        relevant_context={"existing_education": []},
        candidate_id="candidate-1",
        session_id="session-1",
    )
    assert (
        server._education_extraction_updates(result)["education"][0]["degree"]
        == "B.Tech"
    )
    assert captured["target_field"] == "education"
    assert "JNTU Hyderabad" in captured["candidate_response"]
