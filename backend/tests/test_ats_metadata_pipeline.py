"""Regression coverage for public ATS payload -> normalized job -> API fields."""

import asyncio
import json
import os
import sys
from datetime import UTC, date, datetime
from types import ModuleType
from typing import Any

import pytest

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://unused:unused@localhost/unused"
)

import candidate_job_matching_service
import server
from app.job_ingestion.embedding_service import build_canonical_job_representation
from app.job_ingestion.job_ingestion_service import (
    _metadata_params,
    _persistence_safe_job,
    is_job_indexable,
    load_persisted_job_for_embedding,
    upsert_ats_job,
)


def _record_embedding(target, job):
    target.append(job)
    return [0.0] * 384
from app.job_ingestion.normalize import (
    normalize_ashby,
    normalize_greenhouse,
    normalize_lever,
    normalize_theirstack,
    normalize_workable,
    parse_ats_datetime,
)


def test_theirstack_company_string_and_company_metadata_are_preserved():
    job = normalize_theirstack(
        {
            "id": "ts-company",
            "title": "Engineer",
            "company": "Larsen & Toubro",
            "company_domain": "larsentoubro.com",
            "company_object": {
                "name": "Larsen & Toubro",
                "domain": "larsentoubro.com",
                "country": "India",
                "country_code": "IN",
            },
        }
    )

    assert job["company_name"] == "Larsen & Toubro"
    structured = job["structured_data"]
    assert structured["company"] == "Larsen & Toubro"
    assert structured["company_domain"] == "larsentoubro.com"
    assert structured["company_object"]["name"] == "Larsen & Toubro"
    assert structured["company_object"]["country"] == "India"
    assert structured["company_object"]["country_code"] == "IN"


def test_theirstack_company_object_name_is_fallback():
    job = normalize_theirstack(
        {
            "id": "ts-company-object",
            "title": "Engineer",
            "company_object": {"name": "Larsen & Toubro"},
        }
    )

    assert job["company_name"] == "Larsen & Toubro"


def test_theirstack_technology_slugs_map_to_canonical_skills_and_preserve_payload():
    job = normalize_theirstack(
        {
            "id": "ts-skills",
            "job_title": "Backend Engineer",
            "company": "Acme",
            "technology_slugs": [
                "python",
                "spring-boot",
                "POSTGRESQL",
                "node-js",
                "python",
            ],
            "keyword_slugs": ["backend", "urgent-hiring"],
        }
    )

    assert job["skills_required"] == ["Python", "Spring Boot", "PostgreSQL", "Node.js"]
    assert job["skills"] == job["skills_required"]
    assert job["structured_data"]["theirstack"]["technology_slugs"] == [
        "python",
        "spring-boot",
        "POSTGRESQL",
        "node-js",
        "python",
    ]
    assert job["structured_data"]["theirstack"]["keyword_slugs"] == [
        "backend",
        "urgent-hiring",
    ]
    assert "backend" not in job["skills_required"]


@pytest.mark.parametrize(
    "technology_slugs", [None, [], [None, {}, "  ", "not a slug!"], {"name": "python"}]
)
def test_theirstack_invalid_technology_slugs_do_not_create_skills(technology_slugs):
    job = normalize_theirstack(
        {
            "id": "ts-invalid-skills",
            "title": "Engineer",
            "technology_slugs": technology_slugs,
        }
    )

    assert job["skills_required"] == []
    assert job["skills"] == []


def test_explicit_undisclosed_marker_is_preserved_in_persistence_metadata():
    job = normalize_greenhouse(
        {
            "id": "g-undisclosed-persist",
            "title": "Engineer",
            "content": "Salary Range: Not disclosed",
        },
        "Acme",
    )

    assert json.loads(_metadata_params(job)["structured_data"])[
        "explicitly_undisclosed"
    ] == {"salary_range": True}


def test_ashby_concrete_compensation_summary_overrides_undisclosed_salary_range():
    job = normalize_ashby(
        {
            "id": "a-salary-conflict",
            "title": "Engineer",
            "salaryRange": "Not disclosed",
            "compensation": {"summary": "$120k-$150k"},
        },
        "Acme",
    )

    assert job["salary_range"] == "$120k-$150k"
    assert "explicitly_undisclosed" not in job["structured_data"]


def test_ashby_concrete_salary_range_overrides_undisclosed_compensation_summary():
    job = normalize_ashby(
        {
            "id": "a-reverse-salary-conflict",
            "title": "Engineer",
            "salaryRange": "$120k-$150k",
            "compensation": {"summary": "Not disclosed"},
        },
        "Acme",
    )

    assert job["salary_range"] == "$120k-$150k"
    assert "explicitly_undisclosed" not in job["structured_data"]


def test_ashby_all_explicitly_undisclosed_salary_sources_keep_marker():
    job = normalize_ashby(
        {
            "id": "a-undisclosed-salary",
            "title": "Engineer",
            "salaryRange": "Not disclosed",
            "compensation": {"summary": "Not provided"},
        },
        "Acme",
    )

    assert job["salary_range"] is None
    assert job["structured_data"]["explicitly_undisclosed"] == {"salary_range": True}


def test_representative_public_ats_payloads_preserve_explicit_metadata():
    ashby = normalize_ashby(
        {
            "id": "a",
            "title": "Engineer",
            "descriptionHtml": "<p>Experience Level: Senior</p><p>Required Skills: Python, SQL</p>",
            "department": {"name": "Engineering"},
            "location": "Remote",
            "employmentType": "Full-time",
            "isRemote": True,
            "publishedAt": "2026-01-02T12:00:00Z",
            "compensation": {
                "currency": "USD",
                "minValue": 120000,
                "maxValue": 160000,
                "interval": "year",
            },
            "jobUrl": "https://jobs.example/a",
        },
        "Acme",
    )
    lever = normalize_lever(
        {
            "id": "l",
            "text": "Developer",
            "descriptionPlain": "Experience Required: 5 years",
            "categories": {
                "team": "Platform",
                "location": "Bengaluru",
                "commitment": "Full-time",
                "allLocations": ["Bengaluru", "Remote"],
            },
            "workplaceType": "hybrid",
            "createdAt": 1760000000000,
            "hostedUrl": "https://jobs.example/l",
        },
        "Acme",
    )
    greenhouse = normalize_greenhouse(
        {
            "id": 1,
            "title": "Analyst",
            "content": "<p>Remote Policy: Remote</p><p>Salary Range: $90k-$110k</p>",
            "departments": [{"name": "Data"}],
            "location": {"name": "New York"},
            "updated_at": "2026-01-03T12:00:00Z",
            "absolute_url": "https://jobs.example/g",
        },
        "Acme",
    )
    workable = normalize_workable(
        {
            "id": "w",
            "title": "Designer",
            "description": "Skills Required: Figma, CSS",
            "department": "Design",
            "employment_type": "contract",
            "location": {"city": "London", "telecommuting": True},
            "published_on": "2026-01-04T12:00:00Z",
            "salary_range": "£50k-£70k",
            "url": "https://jobs.example/w",
        },
        "Acme",
    )
    assert ashby["salary_range"] == "USD 120000 - 160000 year" and ashby[
        "skills_required"
    ] == ["Python", "SQL"]
    assert (
        lever["employment_type"] == "Full-time" and lever["remote_policy"] == "hybrid"
    )
    assert (
        greenhouse["remote_policy"] == "Remote"
        and greenhouse["salary_range"] == "$90k-$110k"
    )
    assert workable["remote_policy"] == "Remote" and workable["skills_required"] == [
        "Figma",
        "CSS",
    ]


def test_ashby_derives_skills_from_labelled_jd_when_structured_skills_are_absent():
    job = normalize_ashby(
        {
            "id": "ashby-derived-skills",
            "title": "Staff Software Engineer",
            "descriptionHtml": "<h2>Requirements</h2><p>Required Skills: Python, PostgreSQL, AWS</p>",
            "jobUrl": "https://jobs.example/ashby-derived-skills",
        },
        "Supa Health",
    )

    assert job["skills_required"] == ["Python", "PostgreSQL", "AWS"]
    assert job["skills"] == job["skills_required"]


@pytest.mark.parametrize(
    ("normalizer", "payload"),
    [
        (normalize_ashby, {"id": "a-no-skills", "title": "Engineer"}),
        (normalize_lever, {"id": "l-no-skills", "text": "Engineer"}),
        (normalize_greenhouse, {"id": "g-no-skills", "title": "Engineer"}),
        (normalize_workable, {"id": "w-no-skills", "title": "Engineer"}),
    ],
)
def test_eligible_normalized_ats_jobs_never_have_null_skills_required(
    normalizer, payload
):
    job = normalizer(payload, "Acme")

    assert job["skills_required"] == []
    assert _metadata_params(job)["skills_required"] == "[]"


def test_ats_dates_are_bind_safe_datetimes_and_preserve_offsets():
    jobs = [
        normalize_ashby(
            {"id": "a", "title": "A", "publishedAt": "2026-07-16T22:10:27.434Z"}, "Acme"
        ),
        normalize_lever({"id": "l", "text": "L", "createdAt": 1784248827434}, "Acme"),
        normalize_greenhouse(
            {"id": "g", "title": "G", "updated_at": "2026-07-16T22:10:27.434000+05:30"},
            "Acme",
        ),
        normalize_workable(
            {"id": "w", "title": "W", "published_on": "2026-07-16"}, "Acme"
        ),
    ]
    for job in (jobs[0], jobs[1], jobs[3]):
        assert isinstance(job["created_at"], datetime)
        assert job["created_at"].tzinfo is not None
        assert isinstance(_metadata_params(job)["created_at"], datetime)
    assert jobs[2]["created_at"] is None
    assert jobs[3]["created_at"] == datetime(2026, 7, 16, tzinfo=UTC)


def test_legacy_normalized_iso_date_is_coerced_before_database_binding():
    params = _metadata_params({"created_at": "2026-07-16T22:10:27.434000+00:00"})
    assert params["created_at"] == datetime(2026, 7, 16, 22, 10, 27, 434000, tzinfo=UTC)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "2026-07-16T22:10:27.434000+00:00",
            datetime(2026, 7, 16, 22, 10, 27, 434000, tzinfo=UTC),
        ),
        (
            "2026-07-16T22:10:27.434000-04:00",
            datetime.fromisoformat("2026-07-16T22:10:27.434000-04:00"),
        ),
        (
            "2026-07-16T22:10:27.434Z",
            datetime(2026, 7, 16, 22, 10, 27, 434000, tzinfo=UTC),
        ),
        ("2026-07-16", datetime(2026, 7, 16, tzinfo=UTC)),
        (date(2026, 7, 16), datetime(2026, 7, 16, tzinfo=UTC)),
        (datetime(2026, 7, 16, 12, tzinfo=UTC), datetime(2026, 7, 16, 12, tzinfo=UTC)),
        ("not-a-timestamp", None),
        (None, None),
    ],
)
def test_ats_timestamp_parser_always_returns_a_bind_safe_datetime_or_none(
    value, expected
):
    parsed = parse_ats_datetime(value)
    assert parsed == expected
    assert parsed is None or isinstance(parsed, datetime)
    assert _metadata_params({"created_at": value})["created_at"] == expected


@pytest.mark.parametrize(
    ("normalizer", "payload"),
    [
        (normalize_ashby, {"id": "a", "title": "A", "publishedAt": "not-a-timestamp"}),
        (normalize_lever, {"id": "l", "text": "L", "createdAt": "not-a-timestamp"}),
        (
            normalize_greenhouse,
            {"id": "g", "title": "G", "updated_at": "2026-01-02T12:00:00Z"},
        ),
        (
            normalize_workable,
            {"id": "w", "title": "W", "published_on": "not-a-timestamp"},
        ),
    ],
)
def test_each_ats_source_drops_invalid_timestamps_before_persistence(
    normalizer, payload
):
    job = normalizer(payload, "Acme")
    assert job["created_at"] is None
    assert _metadata_params(job)["created_at"] is None


@pytest.mark.parametrize(
    ("normalizer", "payload", "expected"),
    [
        (
            normalize_ashby,
            {
                "id": "a-original",
                "title": "A",
                "publishedAt": "2020-01-02T03:04:05Z",
                "createdAt": "2026-01-02T03:04:05Z",
            },
            datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC),
        ),
        (
            normalize_lever,
            {"id": "l-original", "text": "L", "createdAt": 1577934245000},
            datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC),
        ),
        (
            normalize_greenhouse,
            {"id": "g-original", "title": "G", "updated_at": "2026-01-02T03:04:05Z"},
            None,
        ),
    ],
)
def test_public_ats_original_posting_timestamps(normalizer, payload, expected):
    """Only fields representing the job's original posting time reach persistence."""
    job = normalizer(payload, "Acme")
    assert job["created_at"] == expected
    assert _metadata_params(job)["created_at"] == expected


class _Result:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def first(self):
        return self.rows[0] if self.rows else None


class _Session:
    def __init__(self):
        self.params = None

    async def execute(self, statement, params=None):
        if "SELECT id, job_url, description" in str(statement):
            return _Result(
                [
                    (
                        "db-job",
                        None,
                        "old",
                        None,
                        [],
                        [],
                        None,
                        None,
                        None,
                        "jobs",
                        {},
                        "unknown",
                        "ashby",
                        "a",
                    )
                ]
            )
        self.params = params
        return _Result()

    async def commit(self):
        pass


def test_existing_sync_writes_normalized_metadata(monkeypatch):
    job = normalize_ashby(
        {
            "id": "a",
            "title": "Engineer",
            "descriptionHtml": "Required Skills: Python, SQL",
            "employmentType": "Full-time",
            "isRemote": True,
            "jobUrl": "https://jobs.example/a",
        },
        "Acme",
    )
    job["job_url"] = None  # Keep this storage-unit test independent of Qdrant.
    session = _Session()
    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service._valid_http_url", lambda value: value
    )
    asyncio.run(upsert_ats_job(session, job))
    assert session.params is not None
    assert session.params["employment_type"] == "Full-time"
    assert session.params["remote_policy"] == "Remote"
    assert session.params["skills_required"] == '["Python", "SQL"]'


class _ExistingUpdateSession:
    def __init__(self):
        self.params: dict[str, Any] = {}

    async def execute(self, _statement, params=None):
        assert params is not None
        self.params = params
        return _Result()

    async def commit(self):
        pass


def test_final_persisted_job_loader_maps_database_values_without_changes():
    final_row = {
        "id": "persisted-job",
        "ats_job_id": "provider-job",
        "ats_type": "greenhouse",
        "title": "Engineer",
        "company_name": "Acme",
        "department": "Platform",
        "industry": "Software",
        "responsibilities": "Build APIs",
        "requirements": "Python",
        "skills_required": ["Python"],
        "skills": ["FastAPI"],
        "experience_required": "5 years",
        "experience_level": "Senior",
        "employment_type": "Full-time",
        "remote_policy": "Remote",
        "location": "Remote",
        "city": None,
        "state": None,
        "country": "India",
        "remote": True,
        "salary_range": "$100k",
        "description": "Complete description",
        "structured_data": {
            "source": "postgres",
            "summary": "Build reliable platform services",
            "preferred_skills": ["AWS"],
            "technologies": ["Kubernetes"],
            "seniority": "Staff",
        },
        "job_url": "https://jobs.example/provider-job",
    }

    class Result:
        def first(self):
            return final_row

    class Session:
        async def execute(self, statement, params):
            assert "FROM job_descriptions" in str(statement)
            assert params == {"id": "persisted-job"}
            return Result()

    loaded = asyncio.run(load_persisted_job_for_embedding(Session(), "persisted-job"))
    assert loaded == final_row
    representation = build_canonical_job_representation(loaded)
    assert representation["summary"] == "Build reliable platform services"
    assert representation["preferred_skills"] == ["AWS"]
    assert representation["technologies"] == ["Kubernetes"]
    assert representation["seniority"] == "Staff"


@pytest.mark.parametrize(
    "overrides",
    [
        {"description": "Build resilient APIs."},
        {"description": "", "requirements": "Python and SQL."},
        {"description": "", "responsibilities": "Operate production services."},
        {"description": "", "skills_required": ["Python"]},
        {"description": "", "skills": ["Python"]},
        {
            "description": "",
            "structured_data": {"summary": "Build reliable platform services"},
        },
    ],
)
def test_indexability_accepts_any_substantive_candidate_content(overrides):
    job = {
        "title": "Backend Engineer",
        "company_name": "Acme",
        "job_url": "https://jobs.example/backend",
        "description": "",
        "requirements": "",
        "responsibilities": "",
        "skills_required": [],
        "structured_data": {},
    }
    job.update(overrides)
    assert is_job_indexable(job)


@pytest.mark.parametrize(
    "overrides",
    [
        {"description": ""},
        {"description": "   ", "requirements": "N/A"},
        {"description": "Job description", "responsibilities": ""},
        {"description": "", "skills_required": [], "structured_data": {}},
        {"description": "", "skills": []},
        {"description": "Apply now", "requirements": ""},
        {"description": "We are hiring", "requirements": ""},
        {"description": "Equal opportunity employer", "requirements": ""},
        {"description": "Competitive compensation", "requirements": ""},
        {"title": "Untitled ATS job.", "description": "Build APIs."},
        {"company_name": "Unknown company.", "description": "Build APIs."},
        {"company_name": "Unknown employer.", "description": "Build APIs."},
        {"title": "Untitled.", "description": "Build APIs."},
        {"company_name": "", "description": "Build APIs."},
        {"description": "", "location": "Remote", "salary_range": "$100k"},
    ],
)
def test_indexability_rejects_placeholder_or_non_content_jobs(overrides):
    job = {
        "title": "Backend Engineer",
        "company_name": "Acme",
        "job_url": "https://jobs.example/backend",
        "description": "",
        "requirements": "",
        "responsibilities": "",
        "skills_required": [],
        "structured_data": {},
    }
    job.update(overrides)
    assert not is_job_indexable(job)


def test_indexability_allows_meaningful_confidential_employer_label():
    assert is_job_indexable(
        {
            "title": "Backend Engineer",
            "company_name": "Confidential fintech employer",
            "job_url": "https://jobs.example/backend",
            "description": "Build resilient APIs.",
        }
    )


@pytest.mark.parametrize(
    "structured_data",
    [
        {"seniority": "Senior"},
        {"experience_required": "5 years"},
    ],
)
def test_structured_supporting_metadata_alone_does_not_qualify_job(structured_data):
    assert not is_job_indexable(
        {
            "title": "Backend Engineer",
            "company_name": "Acme",
            "job_url": "https://jobs.example/backend",
            "structured_data": structured_data,
        }
    )


@pytest.mark.parametrize(
    "structured_data",
    [
        {"description": "Build resilient APIs."},
        {"skills": ["Python", "FastAPI"]},
        {"technologies": ["Kubernetes"]},
    ],
)
def test_structured_substantive_content_qualifies_job(structured_data):
    assert is_job_indexable(
        {
            "title": "Backend Engineer",
            "company_name": "Acme",
            "job_url": "https://jobs.example/backend",
            "structured_data": structured_data,
        }
    )


def test_existing_update_uses_one_final_persisted_record_for_embedding_and_qdrant(
    monkeypatch,
):
    final_job = {
        "id": "existing-marker-job",
        "ats_job_id": "marker-job",
        "ats_type": "greenhouse",
        "title": "Engineer",
        "company_name": "Acme",
        "description": "Complete persisted description",
        "requirements": "Persisted requirements",
        "skills_required": ["Python"],
        "skills": ["FastAPI"],
        "experience_level": "Senior",
        "employment_type": "Full-time",
        "remote_policy": "Remote",
        "location": "Remote",
        "country": "India",
        "structured_data": {"source": "postgres"},
        "job_url": "https://jobs.example/marker-job",
    }
    generated_from: list[dict[str, Any]] = []
    upserted = []
    embedding = ModuleType("app.job_ingestion.embedding_service")
    qdrant = ModuleType("app.job_ingestion.qdrant_service")
    embedding.__dict__["generate_job_embedding"] = lambda job: _record_embedding(generated_from, job)
    qdrant.__dict__["ensure_collection"] = lambda: None
    qdrant.__dict__["upsert_job_embedding"] = lambda job_id, vector, job: upserted.append((job_id, vector, job))
    monkeypatch.setitem(sys.modules, "app.job_ingestion.embedding_service", embedding)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.qdrant_service", qdrant)
    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service.load_persisted_job_for_embedding",
        lambda *_args: _async_value(final_job),
    )
    session = _ExistingUpdateSession()
    existing = _existing_for_marker_update(
        job_url="https://jobs.example/marker-job",
        description="Complete persisted description",
    )

    asyncio.run(
        upsert_ats_job(
            session,
            _incoming_for_marker_update(
                description="short provider description",
                job_url="https://jobs.example/marker-job",
            ),
            existing_job=existing,
        )
    )

    assert generated_from == [final_job]
    assert upserted == [("existing-marker-job", [0.0] * 384, final_job)]
    assert generated_from[0] is upserted[0][2]


def test_sparse_incoming_url_preserves_valid_final_url_and_upserts_qdrant(monkeypatch):
    final_job = {
        "id": "existing-marker-job",
        "ats_job_id": "marker-job",
        "ats_type": "greenhouse",
        "title": "Engineer",
        "company_name": "Acme",
        "description": "Complete persisted description",
        "skills_required": ["Python"],
        "structured_data": {},
        "job_url": "https://jobs.example/marker-job",
    }
    generated_from: list[dict[str, Any]] = []
    upserted, deleted = [], []
    embedding = ModuleType("app.job_ingestion.embedding_service")
    qdrant = ModuleType("app.job_ingestion.qdrant_service")
    embedding.__dict__["generate_job_embedding"] = lambda job: _record_embedding(generated_from, job)
    qdrant.__dict__["ensure_collection"] = lambda: None
    qdrant.__dict__["upsert_job_embedding"] = lambda job_id, vector, job: upserted.append((job_id, vector, job))
    qdrant.__dict__["delete_job_embedding"] = lambda job_id: deleted.append(job_id)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.embedding_service", embedding)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.qdrant_service", qdrant)
    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service.load_persisted_job_for_embedding",
        lambda *_args: _async_value(final_job),
    )

    asyncio.run(
        upsert_ats_job(
            _ExistingUpdateSession(),
            _incoming_for_marker_update(job_url=None),
            existing_job=_existing_for_marker_update(
                job_url="https://jobs.example/marker-job"
            ),
        )
    )

    assert generated_from == [final_job]
    assert upserted[0][0] == "existing-marker-job"
    assert upserted[0][2] is final_job
    assert deleted == []


def test_existing_ineligible_final_job_deletes_its_qdrant_point(monkeypatch):
    deleted = []
    final_job = {
        "id": "existing-marker-job",
        "ats_job_id": "marker-job",
        "ats_type": "greenhouse",
        "title": "Backend Engineer",
        "company_name": "Acme",
        "job_url": "not-a-url",
        "description": "",
        "requirements": "",
        "responsibilities": "",
        "skills_required": [],
        "structured_data": {},
    }
    qdrant = ModuleType("app.job_ingestion.qdrant_service")
    qdrant.__dict__["delete_job_embedding"] = lambda job_id: deleted.append(job_id)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.qdrant_service", qdrant)
    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service.load_persisted_job_for_embedding",
        lambda *_args: _async_value(final_job),
    )

    asyncio.run(
        upsert_ats_job(
            _ExistingUpdateSession(),
            _incoming_for_marker_update(job_url=None),
            existing_job=_existing_for_marker_update(
                job_url="https://jobs.example/marker-job"
            ),
        )
    )

    assert deleted == ["existing-marker-job"]


def test_new_insert_uses_final_persisted_record_for_embedding(monkeypatch):
    final_job = {
        "id": "new-job",
        "ats_job_id": "new-provider-job",
        "ats_type": "fantastic",
        "title": "Engineer",
        "company_name": "Acme",
        "description": "Persisted description",
        "requirements": "Persisted requirements",
        "skills_required": ["Python"],
        "skills": ["Python"],
        "experience_level": "Not specified",
        "opportunity_type": "jobs",
        "remote_policy": "unknown",
        "structured_data": {"source": "postgres"},
        "job_url": "https://jobs.example/new-provider-job",
    }
    generated_from: list[dict[str, Any]] = []
    upserted = []
    embedding = ModuleType("app.job_ingestion.embedding_service")
    qdrant = ModuleType("app.job_ingestion.qdrant_service")
    embedding.__dict__["generate_job_embedding"] = lambda job: _record_embedding(generated_from, job)
    qdrant.__dict__["ensure_collection"] = lambda: None
    qdrant.__dict__["upsert_job_embedding"] = lambda job_id, vector, job: upserted.append((job_id, vector, job))
    monkeypatch.setitem(sys.modules, "app.job_ingestion.embedding_service", embedding)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.qdrant_service", qdrant)
    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service.load_persisted_job_for_embedding",
        lambda *_args: _async_value(final_job),
    )

    class Result:
        def scalar_one(self):
            return "new-job"

    class Session:
        async def execute(self, statement, params=None):
            sql = str(statement)
            if "INSERT INTO job_descriptions" in sql:
                return Result()
            if "FROM company_registry" in sql:
                return _Result([("company-id",)])
            return _Result()

        async def commit(self):
            pass

    async def agency_id(*_args):
        return "agency-id"

    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service.get_or_create_ats_agency", agency_id
    )
    job = normalize_ashby(
        {
            "id": "new-provider-job",
            "title": "Engineer",
            "jobUrl": "https://jobs.example/new-provider-job",
        },
        "Acme",
    )

    assert asyncio.run(upsert_ats_job(Session(), job)) == "new-job"
    assert generated_from == [final_job]
    assert upserted[0][2] is generated_from[0]
    assert final_job["opportunity_type"] == "jobs"
    assert final_job["remote_policy"] == "unknown"


@pytest.mark.parametrize("failure_stage", ["embedding", "qdrant"])
def test_postgres_commit_survives_post_commit_embedding_or_qdrant_failure(
    monkeypatch, failure_stage
):
    committed = []
    session = _ExistingUpdateSession()

    async def commit():
        committed.append(True)

    monkeypatch.setattr(session, "commit", commit)
    final_job = {
        "id": "existing-marker-job",
        "ats_job_id": "marker-job",
        "ats_type": "greenhouse",
        "title": "Engineer",
        "company_name": "Acme",
        "description": "Complete description",
        "skills_required": ["Python"],
        "skills": ["Python"],
        "structured_data": {"source": "postgres"},
        "job_url": "https://jobs.example/marker-job",
    }
    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service.load_persisted_job_for_embedding",
        lambda *_args: _async_value(final_job),
    )
    embedding = ModuleType("app.job_ingestion.embedding_service")
    qdrant = ModuleType("app.job_ingestion.qdrant_service")
    embedding.__dict__["generate_job_embedding"] = lambda _job: (_ for _ in ()).throw(RuntimeError("embedding failed")) if failure_stage == "embedding" else [0.0] * 384
    qdrant.__dict__["ensure_collection"] = lambda: None
    qdrant.__dict__["upsert_job_embedding"] = lambda *_args: (_ for _ in ()).throw(RuntimeError("qdrant failed")) if failure_stage == "qdrant" else None
    monkeypatch.setitem(sys.modules, "app.job_ingestion.embedding_service", embedding)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.qdrant_service", qdrant)

    asyncio.run(
        upsert_ats_job(
            session,
            _incoming_for_marker_update(job_url="https://jobs.example/marker-job"),
            existing_job=_existing_for_marker_update(
                job_url="https://jobs.example/marker-job"
            ),
        )
    )

    assert committed == [True]


async def _async_value(value):
    return value


def _existing_for_marker_update(**overrides):
    existing = {
        "id": "existing-marker-job",
        "job_url": None,
        "description": "Existing description",
        "ats_type": "greenhouse",
        "ats_job_id": "marker-job",
        "requirements": "Existing requirements",
        "skills": ["Python"],
        "skills_required": ["Python"],
        "experience_required": None,
        "salary_range": None,
        "employment_type": "Full-time",
        "structured_data": {"source": "existing"},
    }
    existing.update(overrides)
    return existing


def _incoming_for_marker_update(**overrides):
    job = {
        "ats_type": "greenhouse",
        "ats_job_id": "marker-job",
        "company_name": "Acme",
        "title": "Engineer",
        "description": "Incoming",
        "skills": ["Docker"],
        "skills_required": ["Docker"],
        "structured_data": {"source": "incoming"},
    }
    job.update(overrides)
    return job


@pytest.mark.parametrize(
    ("field", "existing_marker", "incoming_value", "incoming_marker", "expected"),
    [
        ("salary_range", True, None, None, True),
        ("experience_required", True, None, None, True),
        ("salary_range", True, "$120k-$150k", None, None),
        ("experience_required", True, "5 years", None, None),
        ("salary_range", "true", None, None, None),
        ("salary_range", None, None, True, True),
    ],
)
def test_existing_undisclosed_markers_merge_only_when_canonical_values_are_absent(
    field,
    existing_marker,
    incoming_value,
    incoming_marker,
    expected,
):
    existing_markers = {} if existing_marker is None else {field: existing_marker}
    incoming_markers = {} if incoming_marker is None else {field: incoming_marker}
    existing = _existing_for_marker_update(
        structured_data={
            "source": "existing",
            "explicitly_undisclosed": existing_markers,
        },
    )
    incoming = _incoming_for_marker_update(
        structured_data={
            "source": "incoming",
            "explicitly_undisclosed": incoming_markers,
        },
        **{field: incoming_value},
    )
    session = _ExistingUpdateSession()

    assert (
        asyncio.run(upsert_ats_job(session, incoming, existing_job=existing))
        == "existing-marker-job"
    )
    markers = json.loads(session.params["structured_data"]).get(
        "explicitly_undisclosed", {}
    )
    assert markers.get(field) is expected


def test_existing_marker_merge_keeps_existing_values_skills_and_stable_id():
    existing = _existing_for_marker_update(
        salary_range="$90k-$110k",
        structured_data={"explicitly_undisclosed": {"salary_range": True}},
    )
    session = _ExistingUpdateSession()

    assert (
        asyncio.run(
            upsert_ats_job(
                session, _incoming_for_marker_update(), existing_job=existing
            )
        )
        == "existing-marker-job"
    )
    assert (
        session.params["salary_range"] is None
    )  # SQL COALESCE retains the canonical existing value.
    assert (
        json.loads(session.params["structured_data"]).get("explicitly_undisclosed")
        is None
    )
    assert json.loads(session.params["skills"]) == ["Python", "Docker"]
    assert json.loads(session.params["skills_required"]) == ["Python", "Docker"]


@pytest.mark.parametrize(
    "incoming",
    [
        {},
        {"structured_data": None},
        {"structured_data": {}},
        {"structured_data": {"source": None}},
        {"structured_data": {"source": ""}},
        {"structured_data": {"metadata": {}}},
        {"structured_data": [None]},
        {"structured_data": [""]},
    ],
)
def test_existing_structured_data_is_preserved_for_sparse_provider_update(incoming):
    existing = _existing_for_marker_update(
        structured_data={"source": "existing", "provider_metadata": {"level": 1}}
    )
    session = _ExistingUpdateSession()
    job = _incoming_for_marker_update(**incoming)
    if not incoming:
        job.pop("structured_data")

    asyncio.run(upsert_ats_job(session, job, existing_job=existing))

    persisted = json.loads(session.params["structured_data"])
    assert persisted["source"] == "existing"
    assert persisted["provider_metadata"] == {"level": 1}


def test_existing_structured_data_marker_is_meaningful_and_preserved():
    existing = _existing_for_marker_update(
        structured_data={"source": "existing", "provider_metadata": {"level": 1}}
    )
    session = _ExistingUpdateSession()

    asyncio.run(
        upsert_ats_job(
            session,
            _incoming_for_marker_update(
                structured_data={"explicitly_undisclosed": {"salary_range": True}}
            ),
            existing_job=existing,
        )
    )

    persisted = json.loads(session.params["structured_data"])
    assert persisted["explicitly_undisclosed"] == {"salary_range": True}


def test_existing_structured_data_is_replaced_by_meaningful_provider_update():
    existing = _existing_for_marker_update(
        structured_data={"source": "existing", "stale": True}
    )
    session = _ExistingUpdateSession()

    asyncio.run(
        upsert_ats_job(
            session,
            _incoming_for_marker_update(structured_data={"source": "incoming"}),
            existing_job=existing,
        )
    )

    assert json.loads(session.params["structured_data"])["source"] == "incoming"
    assert "stale" not in json.loads(session.params["structured_data"])


@pytest.mark.parametrize("existing_value", ["intern", "jobs"])
def test_existing_opportunity_type_is_preserved_when_provider_omits_it(existing_value):
    existing = _existing_for_marker_update(opportunity_type=existing_value)
    session = _ExistingUpdateSession()

    asyncio.run(
        upsert_ats_job(session, _incoming_for_marker_update(), existing_job=existing)
    )

    assert session.params["opportunity_type"] is None


def test_explicit_opportunity_type_updates_existing_job():
    existing = _existing_for_marker_update(opportunity_type="jobs")
    session = _ExistingUpdateSession()

    asyncio.run(
        upsert_ats_job(
            session,
            _incoming_for_marker_update(opportunity_type="intern"),
            existing_job=existing,
        )
    )

    assert session.params["opportunity_type"] == "intern"


@pytest.mark.parametrize("incoming_value", [None, "", "unknown", "not-a-type"])
def test_invalid_opportunity_type_does_not_update_existing_job(incoming_value):
    existing = _existing_for_marker_update(opportunity_type="intern")
    session = _ExistingUpdateSession()

    asyncio.run(
        upsert_ats_job(
            session,
            _incoming_for_marker_update(opportunity_type=incoming_value),
            existing_job=existing,
        )
    )

    assert session.params["opportunity_type"] is None


@pytest.mark.parametrize("existing_value", ["remote", "hybrid"])
def test_existing_remote_policy_is_preserved_when_provider_omits_it(existing_value):
    existing = _existing_for_marker_update(remote_policy=existing_value)
    session = _ExistingUpdateSession()

    asyncio.run(
        upsert_ats_job(session, _incoming_for_marker_update(), existing_job=existing)
    )

    assert session.params["remote_policy"] is None


def test_explicit_remote_policy_updates_existing_job():
    existing = _existing_for_marker_update(remote_policy="remote")
    session = _ExistingUpdateSession()

    asyncio.run(
        upsert_ats_job(
            session,
            _incoming_for_marker_update(remote_policy="hybrid"),
            existing_job=existing,
        )
    )

    assert session.params["remote_policy"] == "hybrid"


@pytest.mark.parametrize(
    "incoming_value",
    [None, "", "unknown", "flexible", "work from anywhere"],
)
def test_invalid_remote_policy_does_not_update_existing_job(incoming_value):
    existing = _existing_for_marker_update(remote_policy="remote")
    session = _ExistingUpdateSession()

    asyncio.run(
        upsert_ats_job(
            session,
            _incoming_for_marker_update(remote_policy=incoming_value),
            existing_job=existing,
        )
    )

    assert session.params["remote_policy"] is None


@pytest.mark.parametrize("incoming_value", ["Remote", "hybrid", "on-site", "onsite"])
def test_supported_remote_policy_updates_existing_job(incoming_value):
    existing = _existing_for_marker_update(remote_policy="remote")
    session = _ExistingUpdateSession()

    asyncio.run(
        upsert_ats_job(
            session,
            _incoming_for_marker_update(remote_policy=incoming_value),
            existing_job=existing,
        )
    )

    assert session.params["remote_policy"] == incoming_value


def test_persistence_guard_never_binds_null_skills_required_for_legacy_normalized_jobs():
    assert _metadata_params({"skills_required": None})["skills_required"] == "[]"


@pytest.mark.parametrize(
    ("normalizer", "payload", "expected"),
    [
        (
            normalize_ashby,
            {"id": "a-exp", "title": "A", "experienceLevel": "Senior"},
            "Senior",
        ),
        (
            normalize_lever,
            {"id": "l-exp", "text": "L", "experience_required": "5 years"},
            "5 years",
        ),
        (
            normalize_greenhouse,
            {"id": "g-exp", "title": "G", "content": "Experience Level: Mid-level"},
            "Mid-level",
        ),
        (normalize_workable, {"id": "w-exp", "title": "W"}, "Not specified"),
    ],
)
def test_all_providers_supply_non_null_experience_level(normalizer, payload, expected):
    job = normalizer(payload, "Acme")
    assert job["experience_level"] == expected
    assert _metadata_params(job)["experience_level"] == expected


@pytest.mark.parametrize(
    ("normalizer", "payload"),
    [
        (normalize_ashby, {"id": "a-final", "title": None}),
        (normalize_lever, {"id": "l-final", "text": None}),
        (normalize_greenhouse, {"id": "g-final", "title": None}),
        (normalize_workable, {"id": "w-final", "title": None}),
    ],
)
def test_each_provider_final_insert_values_are_non_null_for_required_ats_contract(
    normalizer, payload
):
    # This is the final pre-SQL boundary, intentionally not merely normalizer output.
    safe = _persistence_safe_job(normalizer(payload, "Acme"))
    params = _metadata_params(safe)
    assert all(
        safe[key] is not None
        for key in ("ats_type", "ats_job_id", "title", "company_name", "description")
    )
    assert all(
        params[key] is not None
        for key in ("experience_level", "skills_required", "skills", "structured_data")
    )
    assert params["remote_policy"] == "unknown"
    assert params["opportunity_type"] == "jobs"


@pytest.mark.parametrize(
    "normalizer",
    [normalize_ashby, normalize_lever, normalize_greenhouse, normalize_workable],
)
def test_missing_provider_ids_are_rejected_instead_of_colliding_on_the_string_none(
    normalizer,
):
    job = normalizer({}, "Acme")
    assert job["ats_job_id"] == ""
    with pytest.raises(ValueError, match="ats_job_id"):
        asyncio.run(upsert_ats_job(_Session(), job))


def test_final_persistence_boundary_coerces_missing_and_malformed_optional_data():
    safe = _persistence_safe_job(
        {
            "ats_type": "ASHBY",
            "ats_job_id": 7,
            "company_name": "Acme",
            "title": None,
            "description": None,
            "experience_level": None,
            "skills_required": None,
            "skills": {"not": "a list"},
            "structured_data": None,
            "created_at": "invalid",
        }
    )
    assert safe["title"] == "Untitled ATS job"
    assert safe["description"] == ""
    assert safe["experience_level"] == "Not specified"
    assert safe["skills_required"] == safe["skills"] == []
    assert safe["structured_data"] == {} and safe["created_at"] is None


def test_metadata_params_rejects_non_json_provider_metadata_before_database_binding():
    with pytest.raises(ValueError, match="JSON-compatible"):
        _metadata_params({"structured_data": {"bad": object()}})


def test_eligible_normalized_job_insert_binds_non_null_skills_required(monkeypatch):
    job = normalize_ashby({"id": "insert-safe", "title": "Engineer"}, "Acme")

    class InsertResult:
        def scalar_one(self):
            return "new-job"

    class NewJobSession:
        def __init__(self):
            self.insert_params: dict[str, Any] = {}

        async def execute(self, statement, params=None):
            sql = str(statement)
            if "SELECT id, job_url, description" in sql:
                return _Result()
            if "FROM company_registry" in sql:
                return _Result([("company-id",)])
            if "INSERT INTO job_descriptions" in sql:
                assert params is not None
                self.insert_params = params
                return InsertResult()
            raise AssertionError(f"Unexpected SQL: {sql}")

        async def commit(self):
            pass

    session = NewJobSession()

    async def agency_id(*_):
        return "agency-id"

    monkeypatch.setattr(
        "app.job_ingestion.job_ingestion_service.get_or_create_ats_agency", agency_id
    )

    assert asyncio.run(upsert_ats_job(session, job)) == "new-job"
    assert session.insert_params["skills_required"] == "[]"
    assert session.insert_params["experience_level"] == "Not specified"
    assert session.insert_params["title"] == "Engineer"
    assert session.insert_params["description"] == ""
    structured_data = json.loads(session.insert_params["structured_data"])
    assert structured_data["source"] == "ashby"
    assert structured_data["raw_source"]["provider"] == "ashby"
    assert structured_data["raw_source"]["schema_version"] == 1
    assert structured_data["raw_source"]["payload"]["id"] == "insert-safe"
    assert structured_data["raw_source"]["payload"]["title"] == "Engineer"
    assert "experience_requirements" in structured_data


def test_candidate_jobs_serializes_normalized_metadata_without_changing_matching(
    monkeypatch,
):
    row = {
        "rec_id": "r",
        "job_id": "j",
        "match_score": 0.8,
        "recommendation_rank": 1,
        "match_reason": None,
        "tracked_at": None,
        "applied_at": None,
        "hidden_at": None,
        "viewed_at": None,
        "application_status": None,
        "application_agency_id": None,
        "application_job_role": None,
        "title": "Engineer",
        "company_name": "Acme",
        "location": "Remote",
        "salary_range": "$100k-$120k",
        "employment_type": "Full-time",
        "remote_policy": "Remote",
        "experience_level": "Senior",
        "experience_required": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "skills_required": ["Python"],
        "description": "",
        "requirements": "",
        "skills": [],
        "company_logo_url": None,
        "job_url": "https://jobs.example/a",
        "opportunity_type": "jobs",
    }

    class Row(dict):
        _positional = (
            "job_id",
            "title",
            "description",
            "requirements",
            "skills",
            "skills_required",
            "experience_required",
        )

        def __getitem__(self, key):
            if isinstance(key, int):
                return super().__getitem__(self._positional[key])
            return super().__getitem__(key)

    row = Row(row)

    class Result:
        def __init__(self, scalar=None, mapping=None, rows=None):
            self.scalar_value = scalar
            self.mapping_value = mapping
            self.rows = [row] if rows is None else rows

        def scalar(self):
            return self.scalar_value

        def mappings(self):
            return self

        def fetchone(self):
            return self.mapping_value

        def fetchall(self):
            return self.rows

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, statement, params=None):
            if "last_full_recommendation_refresh_at" in str(statement):
                from datetime import UTC, datetime

                now = datetime.now(UTC)
                return Result(
                    mapping={
                        "last_full_recommendation_refresh_at": now,
                        "matching_profile_revision": 0,
                        "last_full_recommendation_revision": 0,
                    }
                )
            return Result(1) if "COUNT(*)" in str(statement) else Result()

    async def candidate(_):
        return {"subscription_active": True}

    async def strength(*_):
        return 90

    monkeypatch.setattr(server, "_get_candidate_row", candidate)
    monkeypatch.setattr(server, "_effective_profile_strength_percent", strength)
    monkeypatch.setattr(server, "SessionLocal", lambda: Session())
    monkeypatch.setattr(server, "_voice_intake_completed_for_matching", lambda _: True)
    monkeypatch.setattr(
        candidate_job_matching_service,
        "stored_recommendation_relevance",
        lambda *_: {"eligible": True},
    )
    candidate_id = "c"
    token = server._issue_candidate_session_token(candidate_id)
    response = asyncio.run(
        server.get_candidate_jobs(candidate_id, authorization=f"Bearer {token}")
    )
    assert (
        response[0]["employment_type"] == "Full-time"
        and response[0]["skills_required"] == ["Python"]
        and response[0]["posted_at"] == "2026-01-01T00:00:00+00:00"
    )


def test_stored_recommendation_validation_job_converts_mapping_row():
    class MappingRow:
        _mapping = {
            "title": "Engineer",
            "description": "Build systems",
            "requirements": "Python",
            "skills": ["Python"],
            "skills_required": ["Python"],
            "experience_required": "3 years",
            "location": "Remote",
            "city": "",
            "state": "",
            "country": "IN",
            "remote": True,
            "remote_policy": "Remote",
            "employment_type": "Full-time",
            "industry": "Software",
            "structured_data": {"source": "ats"},
            "responsibilities": ["Build APIs"],
        }

    payload = server._stored_recommendation_validation_job(MappingRow())

    assert payload["title"] == "Engineer"
    assert payload["skills_required"] == ["Python"]
    assert payload["structured_data"] == {"source": "ats"}
    assert payload["responsibilities"] == ["Build APIs"]


def test_stored_recommendation_validation_job_converts_tuple_row():
    row = (
        "rec-1",
        "Analyst",
        "Analyze data",
        "SQL",
        ["SQL"],
        ["SQL"],
        "2 years",
        "Bengaluru",
        "Bengaluru",
        "KA",
        "IN",
        False,
        "On-site",
        "Full-time",
        "Finance",
        {"team": "data"},
        ["Analyze reports"],
    )

    payload = server._stored_recommendation_validation_job(row)

    assert payload["title"] == "Analyst"
    assert payload["experience_required"] == "2 years"
    assert payload["remote"] is False
    assert payload["responsibilities"] == ["Analyze reports"]


def test_stored_recommendation_validation_job_preserves_missing_optional_values():
    class PartialRow(dict):
        _mapping = {"title": "Coordinator", "description": None}

    payload = server._stored_recommendation_validation_job(PartialRow())

    assert payload["title"] == "Coordinator"
    assert payload["description"] is None
    assert payload["requirements"] is None
    assert payload["remote"] is None
    assert payload["responsibilities"] is None
