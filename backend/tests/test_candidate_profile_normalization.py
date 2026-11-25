import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from candidate_profile_normalization import (  # noqa: E402
    clean_text,
    normalize_candidate_for_matching,
    normalize_experience_value,
    normalize_profile_fields,
    valid_location,
    valid_role,
)


def test_module_imports_and_clean_text_preserves_falsy_values():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("  hello\nworld  ") == "hello world"
    assert clean_text(0) == "0"
    assert clean_text(False) == "False"


def test_rejects_technology_as_role_but_keeps_real_title():
    assert not valid_role("postgres")
    assert valid_role("Python Backend Developer")
    assert valid_role("QA Engineer")


def test_accepts_common_and_unusual_professional_titles():
    for title in (
        "Product Manager",
        "Data Scientist",
        "Software Tester",
        "UX Researcher",
        "Chief Technology Officer",
    ):
        assert valid_role(title)


def test_accepts_scoped_role_abbreviations_and_compound_titles():
    for title in (
        "CEO",
        "CTO",
        "CFO",
        "VP",
        "SRE",
        "Principal Engineer",
        "Staff Software Engineer",
        "Security Engineer",
        "Senior DevRel Engineer",
        "Product PM",
        "General GM",
    ):
        assert valid_role(title)


def test_rejects_ambiguous_or_technology_only_role_values():
    for title in ("PM", "GM", "Python", "PostgreSQL", "React", "I worked as an engineer"):
        assert not valid_role(title)


def test_rejects_placeholder_location_and_preserves_real_location():
    assert not valid_location("my resume")
    assert valid_location("Hyderabad, Telangana, India")


def test_rejects_sentence_like_and_employer_like_locations():
    assert not valid_location("Worked remotely from Hyderabad")
    assert not valid_location("Backend engineering experience")
    assert not valid_location("Acme Corporation")
    assert valid_location("London, United Kingdom")
    assert valid_location("Toronto, Ontario, Canada")


def test_accepts_abbreviated_locations_but_preserves_remote_rejection():
    assert valid_location("St. Louis")
    assert valid_location("Washington D.C.")
    assert not valid_location("Remote")


def test_accepts_scoped_common_location_abbreviations():
    for location in ("NYC", "SF", "DC"):
        assert valid_location(location)


def test_normalization_deduplicates_skills_and_preserves_missing_experience():
    result = normalize_profile_fields(
        {"skills": ["postgres", "PostgreSQL", "Python"], "experience_years": None}
    )
    assert result["skills"] == ["PostgreSQL", "Python"]
    assert "experience_years" not in result


def test_rejects_non_finite_boolean_and_out_of_range_experience():
    for value in (float("nan"), float("inf"), float("-inf"), True, -1, 81):
        assert "experience_years" not in normalize_profile_fields(
            {"experience_years": value}
        )


def test_invalid_incoming_voice_scalars_do_not_replace_valid_profile():
    from server import _merge_voice_into_profile

    merged = _merge_voice_into_profile(
        {
            "current_role": "Backend Engineer",
            "current_company": "Example",
            "location": "Hyderabad",
            "experience_years": 4.0,
            "skills": [],
            "work_experience": [],
            "education": [],
            "raw_data": {},
        },
        {
            "current_role": "PostgreSQL",
            "current_company": "New Example",
            "location": "Worked remotely from Hyderabad",
            "experience_years": float("nan"),
        },
    )

    assert merged["current_role"] == "Backend Engineer"
    assert merged["location"] == "Hyderabad"
    assert merged["experience_years"] == 4.0


@pytest.mark.parametrize("value", [0, 80])
def test_voice_experience_preserves_valid_boundaries(value):
    from server import _merge_voice_into_profile

    merged = _merge_voice_into_profile(
        {
            "current_role": "Backend Engineer",
            "skills": [],
            "work_experience": [],
            "education": [],
            "raw_data": {},
        },
        {"experience_years": value},
    )

    assert merged["experience_years"] == float(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        (4, 4.0),
        (4.5, 4.5),
        ("4", 4.0),
        (None, None),
        (False, None),
        ("invalid", None),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        (-1, None),
        (80, 80.0),
        (81, None),
    ],
)
def test_experience_paths_have_consistent_results(value, expected):
    assert normalize_experience_value(value) == expected
    normalized = normalize_profile_fields({"experience_years": value})
    if expected is None:
        assert "experience_years" not in normalized
    else:
        assert normalized["experience_years"] == expected


def test_valid_headline_replaces_invalid_current_role():
    result = normalize_profile_fields(
        {"current_role": "postgres", "headline": "Backend Engineer"}
    )
    assert result["current_role"] == "Backend Engineer"


def test_matching_copy_does_not_expose_invalid_scalars():
    result = normalize_candidate_for_matching(
        {"current_role": "postgres", "location": "resume", "skills": ["Python"]}
    )
    assert "current_role" not in result
    assert "location" not in result
    assert result["skills"] == ["Python"]


def test_profile_projection_hides_legacy_invalid_scalars_and_uses_valid_fallbacks():
    import server

    profile = server._normalize_for_frontend(
        {
            "id": "candidate-1",
            "current_role": "postgres",
            "headline": "Backend Engineer",
            "location": "my resume",
            "raw_data": {"location": "Hyderabad, Telangana, India"},
            "skills": ["Python", "python"],
            "work_experience": [],
            "education": [],
        }
    )
    assert profile["headline"] == "Backend Engineer"
    assert profile["location"] == "Hyderabad, Telangana, India"
    assert profile["keySkills"] == ["Python"]


def test_embedding_text_excludes_invalid_scalars_and_keeps_valid_data():
    from candidate_text import build_candidate_text

    text = build_candidate_text(
        {
            "current_role": "postgres",
            "headline": "Backend Engineer",
            "location": "my resume",
            "skills": ["postgres", "Python"],
        }
    )
    assert "Candidate Role / Experience:\npostgres" not in text
    assert "Location:\nmy resume" not in text
    assert "Backend Engineer" in text
    assert "PostgreSQL" in text
    assert "Python" in text


@pytest.mark.parametrize(
    "work_experience",
    [
        {"title": "Backend Engineer"},
        ["invalid", 42, None],
        ["invalid", {"title": "Backend Engineer", "company": "Acme"}, None],
    ],
)
def test_embedding_text_ignores_malformed_work_history(work_experience):
    from candidate_text import build_candidate_text

    text = build_candidate_text(
        {"current_role": "Backend Engineer", "work_experience": work_experience}
    )

    assert "Backend Engineer" in text
    if isinstance(work_experience, list) and isinstance(work_experience[1], dict):
        assert " at Acme" in text


def test_matching_signals_use_valid_headline_and_ignore_invalid_location():
    from candidate_job_matching_service import _build_candidate_signals

    signals = _build_candidate_signals(
        {
            "current_role": "postgres",
            "headline": "Backend Engineer",
            "location": "my resume",
            "skills": ["Python"],
            "raw_data": {},
        }
    )
    assert "postgres" not in {role.casefold() for role in signals["past_roles"]}
    assert "Backend Engineer" in signals["past_roles"]
    assert signals["_candidate"].get("location") in (None, "")


def test_valid_manual_role_survives_invalid_resume_and_voice_data():
    from server import _merge_resume_into_existing_profile, _merge_voice_into_profile

    existing = {
        "current_role": "Backend Engineer",
        "location": "Hyderabad",
        "skills": [],
        "work_experience": [],
        "education": [],
        "raw_data": {},
    }
    resume_merged = _merge_resume_into_existing_profile(
        existing, {"current_role": "postgres", "location": "my resume"}
    )
    voice_merged = _merge_voice_into_profile(
        resume_merged, {"current_role": "postgres", "location": "resume"}
    )
    assert voice_merged["current_role"] == "Backend Engineer"
    assert voice_merged["location"] == "Hyderabad"


def test_valid_explicit_voice_role_replaces_existing_role_by_product_rule():
    from server import _merge_voice_into_profile

    merged = _merge_voice_into_profile(
        {
            "current_role": "Backend Engineer",
            "location": "Hyderabad",
            "skills": [],
            "work_experience": [],
            "education": [],
            "raw_data": {},
        },
        {
            "current_role": "Senior Java Engineer",
            "current_company": "Example Corp",
            "skills": [],
        },
    )
    assert merged["current_role"] == "Senior Java Engineer"


@pytest.mark.asyncio
async def test_manual_profile_update_writes_validated_fallback_parameters(monkeypatch):
    import server

    class Result:
        rowcount = 1

    class Session:
        def __init__(self):
            self.params: dict[str, Any] = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _statement, params=None):
            if params and "headline" in params:
                assert params is not None
                self.params = params
            return Result()

        async def commit(self):
            return None

    session = Session()
    existing = {
        "id": "candidate-1",
        "name": "Candidate",
        "current_role": "postgres",
        "location": "my resume",
        "experience_years": 6.0,
        "raw_data": {},
        "parsed_resume_json": {},
    }
    monkeypatch.setattr(server, "_authorize_candidate", lambda *_: None)
    monkeypatch.setattr(server, "_get_candidate_row", _async_return(existing))
    monkeypatch.setattr(server, "SessionLocal", lambda: session)
    monkeypatch.setattr(
        server,
        "_get_candidate_profile_payload",
        _async_return({"ok": True}),
    )

    result = await server.update_candidate_profile(
        "candidate-1",
        server.ManualProfileUpdateRequest(
            name="Candidate", headline="", location="", experience_years=None
        ),
        authorization="token",
    )
    assert result == {"ok": True}
    assert session.params["headline"] == ""
    assert session.params["location"] == ""
    assert session.params["experience_years"] == 6.0


@pytest.mark.asyncio
async def test_manual_matching_profile_change_schedules_one_full_refresh(monkeypatch):
    import server

    class Result:
        rowcount = 1

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _statement, params=None):
            return Result()

        async def commit(self):
            return None

    existing = {
        "id": "candidate-1",
        "name": "Candidate",
        "current_role": "Backend Engineer",
        "location": "Hyderabad",
        "experience_years": 4.0,
        "skills": ["Python"],
        "work_experience": [],
        "raw_data": {"preferred_roles": ["Backend Engineer"]},
        "parsed_resume_json": {},
    }
    scheduled = []

    def schedule(coro):
        scheduled.append(coro)
        coro.close()
        return Mock()

    monkeypatch.setattr(server, "_authorize_candidate", lambda *_: None)
    monkeypatch.setattr(server, "_get_candidate_row", _async_return(existing))
    monkeypatch.setattr(server, "SessionLocal", Session)
    monkeypatch.setattr(server, "profile_cache", type("Cache", (), {"invalidate": _async_return(None)})())
    monkeypatch.setattr(server.asyncio, "ensure_future", schedule)
    monkeypatch.setattr(
        server,
        "_get_candidate_profile_payload",
        _async_return({"ok": True}),
    )

    await server.update_candidate_profile(
        "candidate-1",
        server.ManualProfileUpdateRequest(
            name="Candidate",
            headline="Backend Engineer",
            location="Bengaluru",
            experience_years=4.0,
            skills=["Python"],
        ),
        authorization="token",
    )

    assert len(scheduled) == 1


@pytest.mark.asyncio
async def test_manual_profile_update_preserves_omitted_lists_and_detects_preferred_role_change(
    monkeypatch,
):
    import server

    class Result:
        rowcount = 1

    class Session:
        def __init__(self):
            self.params: dict[str, Any] = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _statement, params=None):
            if params and "skills" in params:
                assert params is not None
                self.params = params
            return Result()

        async def commit(self):
            return None

    existing = {
        "id": "candidate-1",
        "name": "Candidate",
        "current_role": "Backend Engineer",
        "location": "Hyderabad",
        "experience_years": 4.0,
        "skills": ["Python"],
        "work_experience": [{"title": "Engineer", "company": "Example"}],
        "raw_data": {"preferred_roles": ["Backend Engineer"]},
        "parsed_resume_json": {},
    }
    session = Session()
    scheduled = []

    def schedule(coro):
        scheduled.append(coro)
        coro.close()
        return Mock()

    monkeypatch.setattr(server, "_authorize_candidate", lambda *_: None)
    monkeypatch.setattr(server, "_get_candidate_row", _async_return(existing))
    monkeypatch.setattr(server, "SessionLocal", lambda: session)
    monkeypatch.setattr(server.asyncio, "ensure_future", schedule)
    monkeypatch.setattr(
        server,
        "profile_cache",
        type("Cache", (), {"invalidate": _async_return(None)})(),
    )
    monkeypatch.setattr(server, "_get_candidate_profile_payload", _async_return({}))

    await server.update_candidate_profile(
        "candidate-1",
        server.ManualProfileUpdateRequest(preferred_roles=["Data Engineer"]),
        authorization="token",
    )

    assert len(scheduled) == 1
    assert json.loads(session.params["skills"]) == ["Python"]
    assert json.loads(session.params["work_experience"]) == existing["work_experience"]
    assert json.loads(session.params["raw_data"])["preferred_roles"] == [
        "Data Engineer"
    ]


@pytest.mark.asyncio
async def test_manual_profile_update_explicit_empty_lists_clear_and_trigger(monkeypatch):
    import server

    class Result:
        rowcount = 1

    class Session:
        def __init__(self):
            self.params: dict[str, Any] = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _statement, params=None):
            if params and "skills" in params:
                assert params is not None
                self.params = params
            return Result()

        async def commit(self):
            return None

    existing = {
        "id": "candidate-1",
        "name": "Candidate",
        "current_role": "Backend Engineer",
        "location": "Hyderabad",
        "experience_years": 4.0,
        "skills": ["Python"],
        "work_experience": [{"title": "Engineer", "company": "Example"}],
        "raw_data": {},
        "parsed_resume_json": {},
    }
    session = Session()
    scheduled = []

    def schedule(coro):
        scheduled.append(coro)
        coro.close()
        return Mock()

    monkeypatch.setattr(server, "_authorize_candidate", lambda *_: None)
    monkeypatch.setattr(server, "_get_candidate_row", _async_return(existing))
    monkeypatch.setattr(server, "SessionLocal", lambda: session)
    monkeypatch.setattr(server.asyncio, "ensure_future", schedule)
    monkeypatch.setattr(
        server,
        "profile_cache",
        type("Cache", (), {"invalidate": _async_return(None)})(),
    )
    monkeypatch.setattr(server, "_get_candidate_profile_payload", _async_return({}))

    await server.update_candidate_profile(
        "candidate-1",
        server.ManualProfileUpdateRequest(skills=[], work_experience=[]),
        authorization="token",
    )

    assert len(scheduled) == 1
    assert json.loads(session.params["skills"]) == []
    assert json.loads(session.params["work_experience"]) == []


@pytest.mark.asyncio
async def test_manual_profile_update_failure_does_not_schedule_matching(monkeypatch):
    import server

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _statement, params=None):
            class Result:
                rowcount = 1

            return Result()

        async def commit(self):
            raise RuntimeError("commit failed")

    existing = {
        "id": "candidate-1",
        "name": "Candidate",
        "current_role": "Backend Engineer",
        "location": "Hyderabad",
        "experience_years": 4.0,
        "skills": ["Python"],
        "work_experience": [],
        "raw_data": {},
        "parsed_resume_json": {},
    }
    scheduled = []

    def schedule(coro):
        scheduled.append(coro)
        coro.close()

    monkeypatch.setattr(server, "_authorize_candidate", lambda *_: None)
    monkeypatch.setattr(server, "_get_candidate_row", _async_return(existing))
    monkeypatch.setattr(server, "SessionLocal", Session)
    monkeypatch.setattr(server.asyncio, "ensure_future", schedule)

    with pytest.raises(RuntimeError, match="commit failed"):
        await server.update_candidate_profile(
            "candidate-1",
            server.ManualProfileUpdateRequest(location="Bengaluru"),
            authorization="token",
        )

    assert scheduled == []


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"name": "Renamed"}, {"preferred_roles": ["Backend Engineer"]}])
async def test_manual_profile_update_preserves_omitted_or_unchanged_preferred_roles(
    monkeypatch, payload
):
    import server

    class Result:
        rowcount = 1

    class Session:
        def __init__(self):
            self.params: dict[str, Any] = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _statement, params=None):
            assert params is not None
            self.params = params
            return Result()

        async def commit(self):
            return None

    existing = {
        "id": "candidate-1",
        "name": "Candidate",
        "current_role": "Backend Engineer",
        "location": "Hyderabad",
        "experience_years": 4.0,
        "skills": ["Python"],
        "work_experience": [],
        "raw_data": {"preferred_roles": ["Backend Engineer"]},
        "parsed_resume_json": {},
    }
    session = Session()
    scheduled = []

    def schedule(coro):
        scheduled.append(coro)
        coro.close()

    monkeypatch.setattr(server, "_authorize_candidate", lambda *_: None)
    monkeypatch.setattr(server, "_get_candidate_row", _async_return(existing))
    monkeypatch.setattr(server, "SessionLocal", lambda: session)
    monkeypatch.setattr(server.asyncio, "ensure_future", schedule)
    monkeypatch.setattr(
        server,
        "profile_cache",
        type("Cache", (), {"invalidate": _async_return(None)})(),
    )
    monkeypatch.setattr(server, "_get_candidate_profile_payload", _async_return({}))

    await server.update_candidate_profile(
        "candidate-1",
        server.ManualProfileUpdateRequest(**payload),
        authorization="token",
    )

    persisted_raw = json.loads(session.params["raw_data"])
    assert persisted_raw["preferred_roles"] == ["Backend Engineer"]
    assert scheduled == []


def _async_return(value):
    async def _return(*_args, **_kwargs):
        return value

    return _return
