"""Regression coverage for compound candidate answers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from server import _infer_profile_updates_from_message, _merge_voice_into_profile


def test_compound_preference_answer_is_split_into_canonical_fields():
    updates = _infer_profile_updates_from_message(
        "Looking for a full-time Java backend role in Hyderabad, expecting 7-10 LPA and can join immediately"
    )
    assert updates["preferred_roles"] == ["Java backend"]
    assert updates["employment_types"] == ["Full-time"]
    assert updates["preferred_locations"] == ["Hyderabad"]
    assert updates["salary_expectation"] == "7-10 LPA"
    assert updates["availability"] == "immediately"


def test_compound_values_remain_in_their_profile_sections_when_merged():
    updates = _infer_profile_updates_from_message(
        "Looking for a full-time Java backend role in Hyderabad, expecting 7-10 LPA and can join immediately"
    )
    raw = _merge_voice_into_profile({"raw_data": {}, "skills": []}, updates)["raw_data"]
    assert raw["preferred_roles"] == ["Java backend"]
    assert raw["employment_types"] == ["Full-time"]
    assert raw["preferred_locations"] == ["Hyderabad"]
    assert raw["salary_expectation"] == "7-10 LPA"
    assert raw["availability"] == "immediately"
    assert "Hyderabad" not in raw["preferred_roles"]
    assert "LPA" not in raw["preferred_locations"]


def test_existing_preferred_roles_are_preserved_when_no_new_role_is_supplied():
    merged = _merge_voice_into_profile(
        {"raw_data": {"preferred_roles": ["Backend Engineer"]}, "skills": []},
        {"availability": "2 weeks"},
    )
    assert merged["raw_data"]["preferred_roles"] == ["Backend Engineer"]


def test_duplicate_preferred_roles_are_deduplicated_case_insensitively():
    merged = _merge_voice_into_profile(
        {"raw_data": {"preferred_roles": ["Backend Engineer"]}, "skills": []},
        {"preferred_roles": ["backend engineer"]},
    )
    assert merged["raw_data"]["preferred_roles"] == ["Backend Engineer"]
