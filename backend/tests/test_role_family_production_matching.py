"""Production matching adapter tests for advisory role-family ranking."""

import pytest
from candidate_job_matching_service import (
    _best_role_family_alignment,
    _job_passes_skills_or_role,
    _sort_scored_recommendations,
)


@pytest.mark.parametrize(
    ("candidate_roles", "job_title", "relationship", "priority"),
    [
        (["Backend Developer"], "Backend Engineer", "equivalent", 3),
        (["Python Backend Developer"], "FastAPI Developer", "related", 2),
        (["Python Backend Developer"], "DevOps Engineer", "transition", 1),
        (["Cloud Security Engineer"], "Security Engineer", "unknown", 0),
        (
            ["Python Backend Developer"],
            "Java Backend Developer",
            "different_technology",
            0,
        ),
        (["Python Backend Developer"], "Product Manager", "unrelated", 0),
    ],
)
def test_production_role_family_adapter_is_conservative(
    candidate_roles, job_title, relationship, priority
):
    result = _best_role_family_alignment(candidate_roles, job_title)

    assert result == {"relationship": relationship, "priority": priority}


def test_multiple_preferred_roles_use_strongest_safe_relationship():
    result = _best_role_family_alignment(
        ["Product Manager", "Backend Developer"], "Backend Engineer"
    )

    assert result == {"relationship": "equivalent", "priority": 3}


def test_missing_titles_remain_neutral():
    assert _best_role_family_alignment(["Backend Developer"], "") == {
        "relationship": "unknown",
        "priority": 0,
    }
    assert _best_role_family_alignment([], "Backend Engineer") == {
        "relationship": "unknown",
        "priority": 0,
    }


def test_role_family_signal_does_not_override_mandatory_skill_failure():
    signals = {"target_roles": ["Backend Developer"], "skills": ["Python"]}

    assert (
        _best_role_family_alignment(signals["target_roles"], "Backend Engineer")[
            "priority"
        ]
        == 3
    )
    assert not _job_passes_skills_or_role(
        signals,
        "Backend Engineer",
        "Backend Engineer Python Java",
        required_skills=["Python", "Java"],
    )


def test_role_family_priority_breaks_a_score_tie_without_changing_scores():
    scored = [
        ("unrelated", 0.80, {"target_role_score": 0.5, "role_family_priority": 0}),
        ("equivalent", 0.80, {"target_role_score": 0.5, "role_family_priority": 3}),
    ]
    signals = {
        "has_explicit_target_roles": False,
        "target_roles": ["Backend Developer"],
    }
    job_details = {
        "unrelated": {"title": "Product Manager"},
        "equivalent": {"title": "Backend Engineer"},
    }

    _sort_scored_recommendations(scored, signals, job_details)

    assert [item[0] for item in scored] == ["equivalent", "unrelated"]
    assert [item[1] for item in scored] == [0.80, 0.80]


def test_explicit_target_roles_keep_hybrid_score_before_family_priority():
    scored = [
        ("lower-priority", 0.90, {"target_role_score": 0.5, "role_family_priority": 0}),
        (
            "higher-priority",
            0.80,
            {"target_role_score": 0.5, "role_family_priority": 3},
        ),
    ]
    signals = {"has_explicit_target_roles": True, "target_roles": ["Backend Developer"]}
    job_details = {
        "lower-priority": {"title": "Backend Engineer"},
        "higher-priority": {"title": "Backend Developer"},
    }

    _sort_scored_recommendations(scored, signals, job_details)

    assert [item[0] for item in scored] == ["lower-priority", "higher-priority"]


def test_explicit_target_specificity_and_score_still_precede_hybrid_score():
    scored = [
        ("broad", 0.99, {"target_role_score": 0.99, "role_family_priority": 3}),
        ("specific", 0.10, {"target_role_score": 0.10, "role_family_priority": 0}),
    ]
    signals = {
        "has_explicit_target_roles": True,
        "target_roles": ["Python Backend Developer"],
    }
    job_details = {
        "broad": {"title": "Backend Engineer"},
        "specific": {"title": "Python Backend Engineer"},
    }

    _sort_scored_recommendations(scored, signals, job_details)

    assert [item[0] for item in scored] == ["specific", "broad"]

    scored = [
        (
            "lower-role-score",
            0.99,
            {"target_role_score": 0.40, "role_family_priority": 3},
        ),
        (
            "higher-role-score",
            0.10,
            {"target_role_score": 0.80, "role_family_priority": 0},
        ),
    ]
    job_details = {
        "lower-role-score": {"title": "Backend Engineer"},
        "higher-role-score": {"title": "Backend Engineer"},
    }

    _sort_scored_recommendations(scored, signals, job_details)

    assert [item[0] for item in scored] == ["higher-role-score", "lower-role-score"]
