"""Focused integration checks for normalized candidate location matching."""

from types import SimpleNamespace
from typing import Any, cast

import candidate_job_matching_service as matcher
from candidate_profile_normalization import normalize_candidate_for_matching
from location_matching import candidate_location, country_code
from sqlalchemy.ext.asyncio import async_sessionmaker


def test_candidate_location_components_are_normalized_consistently():
    normalized = normalize_candidate_for_matching(
        {
            "country": "India",
            "country_code": "IN",
            "city": "Bangalore",
            "state": " Telangana ",
            "location": "Bangalore, Telangana, India",
        }
    )

    assert normalized["country"] == "India"
    assert normalized["country_code"] == "IN"
    assert normalized["city"] == "bengaluru"
    assert normalized["state"] == "telangana"
    assert candidate_location(normalized)["country_code"] == "IN"
    assert country_code("United States") == "US"
    assert country_code("us") == "US"


def test_placeholder_country_does_not_override_valid_location_data():
    normalized = normalize_candidate_for_matching(
        {
            "country": "my resume",
            "country_code": "N/A",
            "raw_data": {"location": "Hyderabad, Telangana, India"},
        }
    )

    assert "country" not in normalized
    assert "country_code" not in normalized
    assert candidate_location(normalized) == {
        "country_code": "IN",
        "city": "hyderabad",
        "state": "telangana",
        "preferred": "",
    }


def test_refresh_does_not_filter_qdrant_by_country_before_location_validation(
    monkeypatch,
):
    captured = {}

    candidate = {
        "id": "candidate-1",
        "current_role": "Backend Engineer",
        "country": "my resume",
        "raw_data": {
            "location": "Hyderabad, Telangana, India",
            "voice_intake": {"status": "completed"},
        },
        "skills": ["Python"],
        "matching_profile_revision": 0,
    }

    class Result:
        def __init__(self, mapping=None):
            self.mapping = mapping

        def mappings(self):
            return self

        def fetchone(self):
            return self.mapping

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, statement, params=None):
            sql = str(statement)
            if "SELECT * FROM candidates" in sql:
                return Result(candidate)
            if "FROM candidate_preferences" in sql:
                return Result(None)
            raise AssertionError(f"Unexpected SQL: {sql}")

        async def commit(self):
            return None

        async def rollback(self):
            return None

    async def acquire(*_args, **_kwargs):
        return SimpleNamespace(acquired=True, generation=1)

    async def release(*_args, **_kwargs):
        return None

    async def renew(*_args, **_kwargs):
        return True

    def search(_vector, limit=None, country_code=None):
        captured["country_code"] = country_code
        return []

    monkeypatch.setattr(matcher, "acquire_refresh_lease", acquire)
    monkeypatch.setattr(matcher, "_release_refresh_lease_safely", release)
    monkeypatch.setattr(matcher, "_renew_refresh_lease_if_needed", renew)
    monkeypatch.setattr(
        matcher, "build_candidate_text", lambda _candidate: "Backend Engineer"
    )
    monkeypatch.setattr(matcher, "generate_embedding", lambda _text: [0.1])
    monkeypatch.setattr(matcher, "search_job_chunks", search)

    result = __import__("asyncio").run(
        matcher.refresh_candidate_job_matches(
            "candidate-1", candidate, cast(async_sessionmaker[Any], lambda: Session())
        )
    )

    assert result is False
    assert captured["country_code"] is None
