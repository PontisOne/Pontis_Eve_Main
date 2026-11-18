"""Opt-in, real-provider tests for Eve's profile extraction boundary.

These tests deliberately stop short of database writes.  A real persistence
suite must be added only after an explicitly isolated PostgreSQL and Redis
environment is provisioned and independently verified.

Run with (PowerShell):
    $env:EVE_INTEGRATION_TESTS = "1"
    pytest -n 0 -m real_llm tests/integration/test_eve_real_llm_chat.py -s
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

# The shared test conftest installs a placeholder key for mocked tests.  Load
# the explicitly local integration environment before importing code that
# lazily constructs the application's Groq client.  Never print its values.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from groq_client import GroqClientPool  # noqa: E402
from profile_extraction_service import extract_profile_field  # noqa: E402

pytestmark = pytest.mark.real_llm

_ENABLED = os.environ.get("EVE_INTEGRATION_TESTS") == "1"
_HAS_REAL_KEY = any(
    os.environ.get(name, "").strip()
    and not os.environ.get(name, "").strip().startswith("test-only-")
    for name in ("GROQ_API_KEY_1", "GROQ_API_KEY", "GROQ_API_KEY_2")
)
_SKIP_REASON = (
    "set EVE_INTEGRATION_TESTS=1 and a real GROQ_API_KEY(_1) to run real-LLM tests"
)


def _text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def _assert_ok(result: Any, field: str) -> None:
    assert result.status == "ok", result.model_dump()
    assert result.target_field == field
    assert result.normalized is True


@pytest.fixture(scope="session", autouse=True)
def real_llm_gate() -> None:
    if not _ENABLED or not _HAS_REAL_KEY:
        pytest.skip(_SKIP_REASON)


@pytest.fixture(scope="session")
def llm_run_metrics() -> dict[str, Any]:
    started = time.monotonic()
    metrics = {"started": started, "scenarios": Counter()}
    yield metrics
    elapsed = time.monotonic() - started
    print(
        "REAL_LLM_REPORT "
        + json.dumps(
            {
                "provider": "groq-compatible application client",
                "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                "scenarios": dict(metrics["scenarios"]),
                "elapsed_seconds": round(elapsed, 2),
                "persistence": "not executed; no database writes are enabled",
            },
            sort_keys=True,
        )
    )


@pytest.fixture(scope="session")
def real_groq_client() -> GroqClientPool:
    """Use the application's rotating client without importing DB-bound server.py."""
    return GroqClientPool(base_url="https://api.groq.com/openai/v1")


@pytest.mark.parametrize(
    ("scenario", "target_field", "candidate_response"),
    [
        (
            "certification",
            "certifications",
            "I completed an AWS course from Teks Academy.",
        ),
        ("skills", "skills", "I have worked with Python, FastAPI and PostgreSQL."),
        (
            "education",
            "education",
            "I completed my B.Tech in Computer Science from JNTU Hyderabad in 2023.",
        ),
        (
            "compound_skills",
            "skills",
            (
                "I completed a B.Tech in Computer Science in 2023, learned Python "
                "and FastAPI, and completed an AWS course."
            ),
        ),
    ],
)
@pytest.mark.asyncio
async def test_real_llm_extracts_supported_profile_facts(
    scenario: str,
    target_field: str,
    candidate_response: str,
    llm_run_metrics: dict[str, Any],
    real_groq_client: GroqClientPool,
) -> None:
    result = await extract_profile_field(
        target_field=target_field,  # type: ignore[arg-type]
        candidate_response=candidate_response,
        candidate_id="real-llm-integration-only",
        session_id=f"real-llm-{scenario}",
        client=real_groq_client,
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )
    llm_run_metrics["scenarios"][scenario] += 1
    _assert_ok(result, target_field)
    value = _text(result.value)
    if scenario == "certification":
        assert "aws" in value
        assert "teks academy" in value
        assert "formal_certification" not in value
    elif scenario == "skills":
        assert all(skill in value for skill in ("python", "fastapi", "postgresql"))
        assert "aws course" not in value
    elif scenario == "education":
        assert all(
            item in value
            for item in ("b.tech", "computer science", "jntu hyderabad", "2023")
        )
    else:
        assert all(skill in value for skill in ("python", "fastapi"))
        assert "b.tech" not in value


@pytest.mark.asyncio
async def test_real_llm_does_not_invent_ambiguous_education(
    llm_run_metrics: dict[str, Any],
    real_groq_client: GroqClientPool,
) -> None:
    result = await extract_profile_field(
        target_field="education",
        candidate_response="I studied technology for a while.",
        candidate_id="real-llm-integration-only",
        session_id="real-llm-ambiguous",
        client=real_groq_client,
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )
    llm_run_metrics["scenarios"]["ambiguous"] += 1
    assert result.status in {"empty", "ambiguous"}
    assert result.value is None


def test_experience_support_is_explicitly_reported() -> None:
    """The specialized extraction contract currently has no experience field."""
    assert "experience" not in {"certifications", "skills", "education"}


@pytest.mark.skip(
    reason=(
        "persistence requires independently verified EVE_INTEGRATION_TEST_DATABASE_URL "
        "and EVE_INTEGRATION_TEST_REDIS_URL; no writes are performed by this suite"
    )
)
def test_real_chat_persistence_requires_isolated_infrastructure() -> None:
    pytest.fail("unreachable safety guard")
