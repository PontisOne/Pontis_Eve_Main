"""Deterministic calibration benchmark for the existing hybrid score."""

import candidate_job_matching_service as matcher


def _signals(**overrides):
    result = {
        "target_roles": ["Python Backend Developer"],
        "has_explicit_target_roles": True,
        "skills": ["Python", "FastAPI", "PostgreSQL"],
        "past_roles": ["Python Backend Developer"],
        "responsibilities": ["Built REST APIs and backend services"],
        "total_experience_years": 4,
        "preferred_locations": [],
        "preferred_industries": [],
        "preferred_companies": [],
        "employment_types": [],
        "remote_preference": "",
        "willing_to_relocate": None,
        "_candidate": None,
    }
    result.update(overrides)
    return result


def _score(signals, title, description, skills, semantic):
    return matcher._hybrid_score(
        signals,
        title,
        description,
        "",
        skills,
        semantic,
        job_metadata={
            "title": title,
            "description": description,
            "responsibilities": [description] if description else [],
            "skills": skills,
        },
        candidate_years=signals.get("total_experience_years"),
        intelligence={"evidence": {"skills": {"evidence_level": 3}}},
    )


def test_calibration_strong_match_outscores_partial_and_unrelated_jobs():
    signals = _signals()
    strong, _ = _score(
        signals,
        "Python Backend Developer",
        "Build Python REST APIs and backend services with PostgreSQL.",
        ["Python", "FastAPI", "PostgreSQL"],
        0.9,
    )
    partial, _ = _score(signals, "API Engineer", "Build Python APIs.", ["Python"], 0.6)
    unrelated, _ = _score(
        signals,
        "Python Backend Developer",
        "Manage office administration and facilities.",
        ["Python"],
        0.7,
    )
    assert strong > partial
    assert strong > unrelated
    assert unrelated > partial


def test_relevant_responsibility_evidence_is_monotonic():
    signals = _signals()
    without, _ = _score(
        {**signals, "responsibilities": []},
        "Software Engineer",
        "Build Python REST APIs.",
        ["Python"],
        0.7,
    )
    with_evidence, _ = _score(
        signals,
        "Software Engineer",
        "Build Python REST APIs and backend services.",
        ["Python"],
        0.7,
    )
    assert with_evidence >= without


def test_missing_metadata_does_not_create_strong_evidence():
    score, components = _score(
        {
            **_signals(),
            "total_experience_years": None,
            "responsibilities": [],
            "skills": [],
        },
        "Software Engineer",
        "",
        [],
        0.0,
    )
    assert score < 0.5
    assert components["evidence_availability"]["experience_fit"] is False
    assert components["evidence_availability"]["responsibility"] is False
    assert components["score_semantics"] == "normalized_hybrid_ranking_score"


def test_missing_mandatory_skills_does_not_increase_score():
    signals = _signals()
    before, _ = _score(
        signals, "Backend Developer", "Build Python APIs.", ["Python", "FastAPI"], 0.6
    )
    after, _ = _score(
        signals,
        "Backend Developer",
        "Build Python APIs.",
        ["Python", "FastAPI", "PostgreSQL"],
        0.6,
    )
    assert after >= before


def test_high_semantic_similarity_without_lexical_evidence_is_bounded():
    score, components = _score(
        _signals(
            target_roles=["Python Backend Developer"],
            skills=[],
            responsibilities=[],
            past_roles=[],
        ),
        "Software Engineer",
        "Unrelated administrative duties.",
        [],
        0.99,
    )
    assert score <= 0.5
    assert components["ranking_policy"] == "target_role_priority_then_hybrid_score"


def test_all_hybrid_weights_remain_normalized():
    weights = (
        matcher.W_TARGET_ROLE,
        matcher.W_SKILLS,
        matcher.W_EXPERIENCE_FIT,
        matcher.W_RELEVANT_HISTORY,
        matcher.W_PREFERENCES,
        matcher.W_SEMANTIC,
    )
    assert weights == (0.24, 0.20, 0.22, 0.12, 0.12, 0.10)
    assert sum(weights) == 1.0
