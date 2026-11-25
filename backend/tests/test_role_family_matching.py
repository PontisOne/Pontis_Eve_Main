import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from role_family_matching import (
    RoleAnalysis,
    analyze_role_title,
    classify_role_relationship,
)


def test_unicode_and_title_punctuation_normalize_consistently():
    for title, expected in (
        ("Software Engineer – Backend", "software engineer backend"),
        ("Software Engineer — Backend", "software engineer backend"),
        ("Backend–Engineer", "backend engineer"),
        ("Backend/Engineer", "backend engineer"),
        ("Backend: Engineer", "backend engineer"),
    ):
        assert analyze_role_title(title).normalized_title == expected


@pytest.mark.parametrize(
    ("title", "family", "language", "runtime", "framework"),
    [
        ("Python Backend Developer", "backend", ("Python",), (), ()),
        ("Java Spring Boot Engineer", None, ("Java",), (), ("Spring Boot",)),
        ("Python FastAPI Backend Developer", "backend", ("Python",), (), ("FastAPI",)),
        ("JavaScript Node.js Developer", None, ("JavaScript",), ("Node.js",), ()),
        ("C# Developer", None, ("C#",), (), ()),
        ("C++ Developer", None, ("C++",), (), ()),
    ],
)
def test_technology_categories_are_complete_and_deterministic(
    title, family, language, runtime, framework
):
    result = analyze_role_title(title)
    assert result.role_family == family
    assert result.languages == language
    assert result.runtimes == runtime
    assert result.frameworks == framework


def test_technology_primary_field_is_backward_compatible_without_hiding_all_values():
    result = analyze_role_title("Python FastAPI Backend Developer")
    assert result.technology == "Python"
    assert result.languages + result.runtimes + result.frameworks == (
        "Python",
        "FastAPI",
    )


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Senior Backend Developer", "senior"),
        ("Sr Backend Developer", "senior"),
        ("Jr Backend Developer", "junior"),
        ("VP Engineering", None),
        ("Senior Lead Backend Engineer", None),
        ("Junior Principal Engineer", None),
        ("Lead Senior Engineer", None),
        ("Staff Principal Engineer", None),
    ],
)
def test_seniority_is_safe_for_abbreviations_and_conflicts(title, expected):
    assert analyze_role_title(title).seniority == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Backend Developer", "individual_contributor"),
        ("Senior Backend Architect", "architecture"),
        ("Engineering Manager", "management"),
        ("Engineering Manager Architect", "ambiguous"),
        ("Director of Architecture", "ambiguous"),
        ("Technical Lead Manager", "management"),
        ("Head of Engineering", "management"),
        ("Unknown Specialist", "unknown"),
    ],
)
def test_professional_track_distinguishes_and_surfaces_conflicts(title, expected):
    assert analyze_role_title(title).professional_track == expected


@pytest.mark.parametrize(
    "title",
    [
        "Cloud Security Engineer",
        "Mobile Security Engineer",
        "Product Designer",
        "Data Security Analyst",
        "Full Stack DevOps Engineer",
    ],
)
def test_multiple_role_families_are_not_resolved_by_arbitrary_precedence(title):
    result = analyze_role_title(title)
    assert result.role_family is None
    assert len(result.role_families) >= 2


def test_role_family_examples_and_safe_unknowns():
    assert analyze_role_title("Backend Engineer").role_family == "backend"
    assert analyze_role_title("Frontend Developer").role_family == "frontend"
    assert analyze_role_title("DevOps Engineer").role_family == "devops"
    assert analyze_role_title("API Developer").role_family is None
    assert analyze_role_title("Unknown Specialist").role_family is None


def test_technology_boundaries_do_not_cross_match():
    assert analyze_role_title("Java Developer").languages == ("Java",)
    assert analyze_role_title("JavaScript Developer").languages == ("JavaScript",)
    assert analyze_role_title("Go Developer").languages == ("Go",)
    assert analyze_role_title("Google Cloud Engineer").languages == ()
    assert analyze_role_title("C# Developer").languages == ("C#",)
    assert analyze_role_title("C++ Developer").languages == ("C++",)
    assert analyze_role_title("Node Developer").runtimes == ("Node.js",)
    assert analyze_role_title("Node.js Developer").runtimes == ("Node.js",)


def test_aliases_are_equivalent_alternatives_and_exclude_original_title():
    assert analyze_role_title("Backend Engineer").recognized_aliases == (
        "backend developer",
    )
    assert analyze_role_title("Backend Developer").recognized_aliases == (
        "backend engineer",
    )
    assert analyze_role_title("Software Engineer – Backend").recognized_aliases == (
        "backend engineer",
        "backend developer",
    )
    for title in (
        "API Developer",
        "API Engineer",
        "Backend Architect",
        "Backend Manager",
        "Java Backend Engineer",
        "Platform Engineer",
    ):
        assert analyze_role_title(title).recognized_aliases == ()


@pytest.mark.parametrize("value", [None, 123, [], {}, ""])
def test_unsupported_inputs_fail_closed(value):
    assert analyze_role_title(value) == RoleAnalysis(
        "", None, (), None, (), (), (), None, (), "unknown", ()
    )


def test_serialization_is_explicit_and_deterministic():
    result = analyze_role_title("Python FastAPI Backend Developer")
    assert result.as_dict() == {
        "normalized_title": "python fastapi backend developer",
        "role_family": "backend",
        "role_families": ["backend"],
        "technology": "Python",
        "languages": ["Python"],
        "runtimes": [],
        "frameworks": ["FastAPI"],
        "seniority": None,
        "seniority_indicators": [],
        "professional_track": "individual_contributor",
        "recognized_aliases": [],
    }


@pytest.mark.parametrize(
    ("candidate", "job", "relationship"),
    [
        ("Python Backend Developer", "Python Backend Developer", "exact"),
        ("Backend Developer", "Backend Engineer", "equivalent"),
        ("Software Engineer – Backend", "Backend Engineer", "equivalent"),
        ("Python Backend Developer", "Python Developer", "related"),
        ("Python Backend Developer", "FastAPI Developer", "related"),
        ("Python Backend Developer", "API Developer", "adjacent"),
        ("Python Backend Developer", "Java Backend Developer", "different_technology"),
        ("Python Backend Developer", "Frontend Developer", "unrelated"),
        ("Python Backend Developer", "DevOps Engineer", "transition"),
        ("Python Backend Developer", "Senior Backend Architect", "seniority_mismatch"),
    ],
)
def test_role_relationship_matrix(candidate, job, relationship):
    assert classify_role_relationship(candidate, job).relationship == relationship


def test_relationship_contains_structured_compatibility_and_reason_codes():
    result = classify_role_relationship(
        "Python Backend Developer", "Java Backend Developer"
    )
    assert result.role_family_match == "same"
    assert result.technology_compatibility == "conflicting"
    assert "conflicting_primary_technology" in result.reason_codes
    assert result.as_dict()["reason_codes"] == list(result.reason_codes)


@pytest.mark.parametrize(
    ("candidate", "job"),
    [
        ("Backend Developer", "Backend Engineer"),
        ("Backend Engineer", "Backend Developer"),
    ],
)
def test_high_confidence_alias_relationship_is_symmetric(candidate, job):
    assert classify_role_relationship(candidate, job).relationship == "equivalent"


@pytest.mark.parametrize(
    "title",
    [
        "API Developer",
        "API Engineer",
        "Backend Architect",
        "Backend Manager",
        "Java Backend Engineer",
        "Platform Engineer",
    ],
)
def test_unsafe_aliases_are_not_equivalent(title):
    result = classify_role_relationship("Backend Developer", title)
    assert result.relationship != "equivalent"


@pytest.mark.parametrize(
    "candidate,job", [(None, "Backend Engineer"), ("Backend Engineer", None), ("", "")]
)
def test_missing_roles_return_unknown(candidate, job):
    result = classify_role_relationship(candidate, job)
    assert result.relationship == "unknown"
    assert "insufficient_role_evidence" in result.reason_codes


def test_ambiguous_roles_do_not_receive_confident_relationships():
    result = classify_role_relationship("Cloud Security Engineer", "Security Engineer")
    assert result.relationship == "unknown"
    assert "ambiguous_role_family" in result.reason_codes


def test_relationship_classifier_is_deterministic():
    first = classify_role_relationship(
        "Python Backend Developer", "Java Backend Developer"
    )
    assert first == classify_role_relationship(
        "Python Backend Developer", "Java Backend Developer"
    )
