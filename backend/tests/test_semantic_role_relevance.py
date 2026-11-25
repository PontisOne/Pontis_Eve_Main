from candidate_job_matching_service import (
    _build_candidate_signals,
    _count_skill_matches,
    _job_passes_skills_or_role,
    _retrieval_score_is_compatible,
    _target_role_score,
)
from qdrant_service import RetrievalDiagnostics, SearchResults


def test_semantic_role_context_recovers_different_but_related_titles():
    roles = ["Python Backend Developer"]
    assert (
        _target_role_score(
            roles,
            "API Engineer",
            "Build Python REST APIs with FastAPI and PostgreSQL.",
            semantic_role_score=0.78,
        )
        == 0.78
    )


def test_semantic_role_context_does_not_override_conflicting_technology():
    roles = ["Python Backend Developer"]
    assert (
        _target_role_score(
            roles,
            "Java Backend Developer",
            "Build Spring Boot services with Java.",
            semantic_role_score=0.95,
        )
        == 0.4
    )


def test_technology_only_preference_is_not_a_target_role():
    signals = _build_candidate_signals(
        {"skills": ["Python"], "raw_data": {"preferred_roles": ["Python"]}}
    )
    assert signals["target_roles"] == []


def test_technology_plus_role_remains_a_target_role():
    signals = _build_candidate_signals(
        {
            "skills": ["Python"],
            "raw_data": {"preferred_roles": ["Python Backend Developer"]},
        }
    )
    assert signals["target_roles"] == ["Python Backend Developer"]


def test_high_semantic_similarity_does_not_qualify_unrelated_role():
    assert (
        _target_role_score(
            ["Python Backend Developer"],
            "Marketing Manager",
            "Lead campaigns and manage brand communications.",
            semantic_role_score=0.98,
        )
        == 0.0
    )


def test_generic_skills_do_not_satisfy_two_skill_fallback():
    assert (
        _count_skill_matches(
            ["Communication", "Teamwork"],
            "Communication and teamwork are important for this marketing role.",
        )
        == 0
    )


def test_technical_skills_still_satisfy_two_skill_fallback():
    assert (
        _count_skill_matches(
            ["Python", "PostgreSQL"],
            "Build Python services backed by PostgreSQL database",
        )
        == 2
    )


def test_combined_stored_job_skills_are_used_for_relevance():
    from candidate_job_matching_service import _combined_job_skills

    assert _combined_job_skills(
        {"skills": ["Python"], "skills_required": ["PostgreSQL"]}
    ) == ["Python", "PostgreSQL"]


def test_mandatory_skills_still_gate_semantic_role_relevance():
    signals = {"target_roles": ["Python Backend Developer"], "skills": ["Python"]}
    assert not _job_passes_skills_or_role(
        signals,
        "API Engineer",
        "Build APIs with Java and Spring Boot.",
        required_skills=["Java"],
        semantic_role_score=0.95,
    )


def test_mixed_retrieval_metadata_is_checked_per_job():
    results = SearchResults(
        [("compatible", 0.9), ("old", 0.8), ("unknown", 0.7)],
        RetrievalDiagnostics(3, 3, 0, 3, 0),
        {
            "compatible": {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_version": "v2_structured",
            },
            "old": {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_version": "v1_legacy",
            },
            "unknown": {"embedding_model": None, "embedding_version": None},
        },
    )
    assert _retrieval_score_is_compatible(results, "compatible")
    assert not _retrieval_score_is_compatible(results, "old")
    assert not _retrieval_score_is_compatible(results, "unknown")
