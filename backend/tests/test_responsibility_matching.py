import inspect

import candidate_job_matching_service as matcher
import server


def _candidate(work_experience):
    return {
        "current_role": "Python Backend Developer",
        "skills": ["Python", "SQL"],
        "work_experience": work_experience,
        "raw_data": {},
    }


def _responsibility_score(candidate, job):
    signals = matcher._build_candidate_signals(candidate)
    _, components = matcher._hybrid_score(
        signals,
        job["title"],
        job.get("description", ""),
        job.get("requirements", ""),
        job.get("skills", []),
        0.0,
        job_metadata=job,
    )
    return components["responsibility_relevance"]


def test_different_title_with_aligned_responsibilities_is_relevant():
    score = _responsibility_score(
        _candidate(
            [
                {
                    "title": "Python Backend Developer",
                    "description": (
                        "Built REST APIs, backend services, and database integrations."
                    ),
                }
            ]
        ),
        {
            "title": "Software Engineer",
            "responsibilities": [
                "Develop Python REST APIs and backend services",
                "Integrate PostgreSQL databases",
            ],
        },
    )
    assert score is not None and score >= 0.35


def test_similar_title_with_unrelated_responsibilities_is_not_strong():
    score = _responsibility_score(
        _candidate(
            [
                {
                    "title": "Python Backend Developer",
                    "description": "Built REST APIs and backend services.",
                }
            ]
        ),
        {
            "title": "Backend Developer",
            "responsibilities": [
                "Administer infrastructure",
                "Monitor servers and operational alerts",
            ],
        },
    )
    assert score == 0.0


def test_incidental_skill_overlap_does_not_create_responsibility_relevance():
    score = _responsibility_score(
        _candidate(
            [{"title": "Python Backend Developer", "description": "Built REST APIs."}]
        ),
        {
            "title": "Operations Specialist",
            "description": "Python and SQL appear in the environment.",
            "responsibilities": ["Coordinate sales operations and customer outreach"],
        },
    )
    assert score == 0.0


def test_previous_role_responsibilities_contribute():
    score = _responsibility_score(
        _candidate(
            [
                {
                    "title": "Support Specialist",
                    "description": "Resolved customer tickets.",
                },
                {
                    "title": "Platform Engineer",
                    "description": (
                        "Designed and maintained backend services and REST APIs."
                    ),
                },
            ]
        ),
        {
            "title": "Software Engineer",
            "responsibilities": ["Maintain backend services", "Develop REST APIs"],
        },
    )
    assert score is not None and score > 0.0


def test_missing_responsibilities_are_neutral():
    assert (
        _responsibility_score(
            _candidate([{"title": "Python Backend Developer"}]),
            {"title": "Backend Developer"},
        )
        is None
    )


def test_provider_responsibilities_and_html_bullets_are_normalized():
    job = {
        "title": "Software Engineer",
        "structured_data": {
            "responsibilities": (
                "<ul><li>Build REST APIs</li><li>Build REST APIs</li></ul>"
            )
        },
    }
    extracted = matcher._job_responsibilities(job)
    assert extracted == ["Build REST APIs"]


def test_inline_responsibility_headings_extract_only_the_duty_sentence():
    job = {
        "title": "Software Engineer",
        "description": (
            "About the company. Responsibilities include building REST APIs "
            "and backend services. "
            "Requirements: Python and FastAPI. Benefits: generous leave."
        ),
    }
    assert matcher._job_responsibilities(job) == [
        "building REST APIs and backend services."
    ]


def test_role_involves_inline_heading_is_supported():
    assert matcher._job_responsibilities(
        {
            "description": (
                "The role involves maintaining backend services. Requirements: Python."
            )
        }
    ) == ["maintaining backend services."]


def test_generic_verbs_without_shared_duty_objects_are_neutral():
    assert (
        matcher._responsibility_score(
            ["Manage team coordination and support operations"],
            ["Manage server access and support monitoring"],
        )
        == 0.0
    )


def test_backend_duty_objects_survive_generic_verb_guard():
    score = matcher._responsibility_score(
        ["Develop REST APIs and maintain backend services"],
        ["Develop REST APIs and maintain backend services"],
    )
    assert score is not None
    assert score > 0.0


def test_mismatched_responsibilities_preserve_strong_role_history_score():
    assert (
        matcher._experience_score(
            ["Backend Engineer"],
            "Backend Engineer",
            "Backend Engineer",
            ["Manage finance operations"],
            ["Design mobile interfaces"],
        )
        == 1.0
    )


def test_matching_responsibilities_can_improve_history_score():
    assert (
        matcher._experience_score(
            ["Software Engineer"],
            "Backend Engineer",
            "Backend Engineer",
            ["Develop REST APIs and maintain backend services"],
            ["Develop REST APIs and maintain backend services"],
        )
        > 0.0
    )


def test_missing_responsibility_evidence_preserves_history_score():
    assert (
        matcher._experience_score(
            ["Backend Engineer"], "Backend Engineer", "Backend Engineer", None, None
        )
        == 1.0
    )


def test_responsibility_mismatch_does_not_reject_strong_role_and_skill_match():
    signals = {"skills": ["Python", "FastAPI"], "target_roles": ["Backend Engineer"]}
    assert matcher._job_passes_skills_or_role(
        signals,
        "Backend Engineer",
        "Backend Engineer Python FastAPI",
        responsibility_relevance=0.0,
    )


def test_responsibility_evidence_is_optional_for_role_and_skill_eligibility():
    signals = {"skills": ["Python", "FastAPI"], "target_roles": ["Backend Engineer"]}
    assert matcher._job_passes_skills_or_role(
        signals, "Backend Engineer", "Backend Engineer Python FastAPI"
    )
    assert matcher._job_passes_skills_or_role(
        signals,
        "Backend Engineer",
        "Backend Engineer Python FastAPI",
        responsibility_relevance=None,
    )


def test_mandatory_skill_mismatch_still_rejects_despite_role_and_responsibility():
    signals = {"skills": ["Python", "FastAPI"], "target_roles": ["Backend Engineer"]}
    assert not matcher._job_passes_skills_or_role(
        signals,
        "Backend Engineer",
        "Backend Engineer Python FastAPI Java",
        required_skills=["Java"],
        responsibility_relevance=0.0,
    )


def test_explicit_experience_requirement_failure_still_rejects():
    result = matcher.experience_eligibility(2.0, "Backend Engineer", "5+ years")
    assert result["eligible"] is False


def test_recommendation_paths_propagate_responsibility_fields():
    matching_source = inspect.getsource(matcher.refresh_candidate_job_matches)
    assert "opportunity_type, responsibilities" in matching_source
    row = tuple([None] * 15 + [{"responsibilities": ["build APIs"]}, ["build APIs"]])
    converted = server._stored_recommendation_validation_job(row)
    assert converted["structured_data"] == {"responsibilities": ["build APIs"]}
    assert converted["responsibilities"] == ["build APIs"]


def test_stored_validation_uses_structured_responsibilities_when_available():
    candidate = _candidate([{"description": "Built REST APIs and backend services."}])
    candidate["location"] = "India"
    job = {
        "title": "Backend Developer",
        "country": "India",
        "structured_data": {
            "responsibilities": ["Develop REST APIs and backend services"]
        },
    }
    result = matcher.stored_recommendation_relevance(candidate, job)
    assert result["eligible"] is True


def test_malformed_responsibility_values_are_ignored():
    assert matcher._responsibility_phrases([None, 42, {"text": "bad"}, ""]) == []
