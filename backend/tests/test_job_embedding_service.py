import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from app.job_ingestion.embedding_service import (
    build_canonical_job_representation,
    build_canonical_job_text,
    build_job_text,
)


class _FakeTokenizer:
    def encode(self, text, **_kwargs):
        return text.split()


class _FakeModel:
    max_seq_length = 256
    tokenizer = _FakeTokenizer()

    def __init__(self):
        self.encode_calls = []

    def encode(self, text, normalize_embeddings=False):
        self.encode_calls.append((text, normalize_embeddings))
        return _Vector([0.1, 0.2, 0.3])


class _Vector(list):
    def tolist(self):
        return list(self)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    from app.job_ingestion import embedding_service

    model = _FakeModel()
    monkeypatch.setattr(embedding_service, "_model", model)
    return model


def test_import_does_not_initialize_sentence_transformer():
    backend_dir = Path(__file__).parents[1]
    script = """
import sys
import app.job_ingestion.embedding_service as embedding_service
assert embedding_service._model is None
assert "sentence_transformers" not in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(backend_dir)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=backend_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_model_initializes_on_embedding_request_and_is_reused(monkeypatch):
    from app.job_ingestion import embedding_service

    models = []

    def load_model():
        model = _FakeModel()
        models.append(model)
        return model

    monkeypatch.setattr(embedding_service, "_model", None)
    monkeypatch.setattr(embedding_service, "_load_model", load_model)

    first = embedding_service.generate_job_embedding({"title": "Engineer"})
    second = embedding_service.generate_job_embedding({"title": "Engineer"})

    assert len(models) == 1
    assert first == second == [0.1, 0.2, 0.3]


def test_concurrent_embedding_calls_initialize_model_once(monkeypatch):
    from app.job_ingestion import embedding_service

    models = []

    def load_model():
        model = _FakeModel()
        models.append(model)
        return model

    monkeypatch.setattr(embedding_service, "_model", None)
    monkeypatch.setattr(embedding_service, "_load_model", load_model)

    jobs = [{"title": f"Engineer {index}"} for index in range(16)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(embedding_service.generate_job_embedding, jobs))

    assert len(models) == 1
    assert results == [[0.1, 0.2, 0.3]] * len(jobs)


def test_model_initialization_failure_is_propagated_and_retried(monkeypatch):
    from app.job_ingestion import embedding_service

    attempts = []

    def load_model():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return _FakeModel()

    monkeypatch.setattr(embedding_service, "_model", None)
    monkeypatch.setattr(embedding_service, "_load_model", load_model)

    with pytest.raises(RuntimeError, match="model unavailable"):
        embedding_service.generate_job_embedding({"title": "Engineer"})
    assert embedding_service._model is None
    assert embedding_service.generate_job_embedding({"title": "Engineer"}) == [0.1, 0.2, 0.3]
    assert len(attempts) == 2


def test_canonical_representation_separates_required_preferred_and_responsibilities():
    job = {
        "title": "Python Backend Engineer",
        "company_name": "Acme",
        "responsibilities": ["Build APIs", "Operate services"],
        "skills_required": ["Python", "PostgreSQL", "python"],
        "preferred_skills": ["AWS", "PostgreSQL"],
        "technologies": ["Docker", "Docker"],
        "experience_required": "3+ years",
        "remote_policy": "Remote",
    }
    representation = build_canonical_job_representation(job)
    assert representation["required_skills"] == ["Python", "PostgreSQL"]
    assert representation["preferred_skills"] == ["AWS"]
    assert representation["technologies"] == ["Docker"]
    assert "Build APIs" in representation["responsibilities"]

    text = build_job_text(job)
    assert "Required Skills: Python, PostgreSQL" in text
    assert "Preferred Skills: AWS" in text
    assert "Responsibilities: Build APIs Operate services" in text
    assert "Description:" not in text


def test_legacy_and_malformed_payloads_are_safe_and_deterministic():
    job = {
        "title": "API Engineer",
        "skills_required": None,
        "preferred_skills": {"unexpected": "object"},
        "structured_data": {"raw_source": {"payload": {"technologies": ["REST"]}}},
        "description": "Build resilient APIs.",
    }
    first = build_job_text(job)
    assert first == build_job_text(job)
    assert "Technologies: REST" in first
    assert "Job Description: Build resilient APIs." in first


def test_long_description_is_bounded_without_dropping_requirements():
    job = {
        "title": "Java Backend Engineer",
        "skills_required": ["Java", "Spring Boot"],
        "responsibilities": "Build payment services.",
        "description": "x" * 20000,
    }
    text = build_job_text(job)
    assert "Required Skills: Java, Spring Boot" in text
    assert "Responsibilities: Build payment services." in text
    assert len(text) <= 12000


def test_canonical_text_truncation_includes_marker_within_exact_limit():
    text = build_canonical_job_text({"title": "Engineer", "description": "x" * 20000})
    assert len(text) == 12000
    assert text.endswith(" [truncated]")


def test_nested_legacy_scalar_and_sections_are_used():
    job = {
        "structured_data": {
            "raw_source": {
                "payload": {
                    "title": "Legacy API Engineer",
                    "company": "Legacy Co",
                    "department": "Platform",
                    "industry": "Fintech",
                    "experience_required": "5 years",
                    "seniority": "Senior",
                    "employment_type": "Full-time",
                    "location": "Remote",
                    "salary_range": "$100k",
                    "remote_policy": "Remote",
                    "responsibilities": "Own API reliability.",
                    "requirements": "Python and SQL.",
                    "preferred_skills": ["AWS"],
                }
            }
        }
    }
    representation = build_canonical_job_representation(job)
    assert representation["title"] == "Legacy API Engineer"
    assert representation["company"] == "Legacy Co"
    assert representation["responsibilities"] == "Own API reliability."
    assert representation["requirements"] == "Python and SQL."
    assert representation["preferred_skills"] == ["AWS"]


def test_partial_structured_sections_preserve_full_description():
    text = build_canonical_job_text({
        "responsibilities": "Build APIs.",
        "description": "Build APIs. Lead incident response and mentor engineers.",
    })
    assert "Lead incident response" in text


def test_token_budget_is_enforced_and_required_skills_survive_long_prose():
    job = {
        "title": "Python Backend Engineer",
        "skills_required": ["Python", "PostgreSQL", "Kubernetes"],
        "responsibilities": "Important responsibility " * 2000,
        "description": "Additional context " * 2000,
    }
    text = build_job_text(job)
    from app.job_ingestion import embedding_service
    model = embedding_service._model
    assert model is not None
    assert embedding_service._token_count(text) <= model.max_seq_length
    assert "Required Skills: Python, PostgreSQL, Kubernetes" in text
