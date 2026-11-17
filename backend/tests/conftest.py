"""Shared test-only environment setup.

The application intentionally has no signing-secret fallback.  Tests use a
non-production, repository-local value through the process environment rather
than relying on application code to invent one.
"""

import os

# The application reads DATABASE_URL during module import.  Keep collection
# deterministic without changing the real shell or deployment environment.
# This is deliberately a PostgreSQL URL: SQLite does not match the JSONB,
# UUID, and migration semantics used by the application.
os.environ["DATABASE_URL"] = (
    "postgresql+asyncpg://pontis_test:pontis_test@127.0.0.1:5432/pontis_test"
)
test_database_url = os.environ.get("TEST_DATABASE_URL", "")
if "127.0.0.1" not in test_database_url and "localhost" not in test_database_url:
    test_database_url = (
        "postgresql+asyncpg://pontis_test:pontis_test@127.0.0.1:5432/pontis_test"
    )
os.environ["TEST_DATABASE_URL"] = test_database_url
test_redis_url = os.environ.get("TEST_REDIS_URL", "")
if "127.0.0.1" not in test_redis_url and "localhost" not in test_redis_url:
    test_redis_url = "redis://127.0.0.1:6379/15"
os.environ["REDIS_URL"] = test_redis_url
os.environ["TEST_REDIS_URL"] = test_redis_url
os.environ.setdefault("REACT_APP_BACKEND_URL", "http://127.0.0.1:8001")
os.environ.setdefault("EVE_ENVIRONMENT", "testing")
# Importing server constructs the client pool eagerly.  This is a deliberately
# non-functional test marker; tests that exercise provider behavior replace the
# pool or patch the provider calls and never contact Groq.
os.environ.setdefault("GROQ_API_KEY_1", "test-only-groq-placeholder")
os.environ.setdefault(
    "CANDIDATE_SESSION_SECRET",
    "test-only-candidate-session-secret-2026-strong",
)
