"""Safety gates for Issue 12 real-service integration tests.

This module performs validation only. It never loads the application
DATABASE_URL and never creates or mutates a database or Qdrant collection.
"""

import os
import re
import weakref
from importlib import import_module
from urllib.parse import urlparse

_VALIDATED_CLIENTS: weakref.WeakKeyDictionary[object, tuple[str, str]] = weakref.WeakKeyDictionary()


def require_issue12_qdrant_config() -> tuple[str, str]:
    url = os.environ.get("ISSUE12_QDRANT_URL", "")
    collection = os.environ.get("ISSUE12_QDRANT_COLLECTION", "")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise RuntimeError("Issue 12 Qdrant tests require a local HTTP Qdrant URL")
    if not re.fullmatch(r"issue12_[a-z0-9_]+_test", collection):
        raise RuntimeError(
            "Issue 12 Qdrant tests require a dedicated collection named "
            "issue12_<name>_test"
        )
    if collection == "job_chunks":
        raise RuntimeError("Issue 12 tests may not use the production Qdrant collection")
    return url, collection


def create_issue12_qdrant_client():
    """Construct and register a client only after validating Issue 12 settings."""
    url, collection = require_issue12_qdrant_config()
    qdrant_service = import_module("app.job_ingestion.qdrant_service")

    client = qdrant_service.create_qdrant_client(
        qdrant_service.QdrantConfig(
            url=url, api_key=None, collection_name=collection
        )
    )
    _VALIDATED_CLIENTS[client] = (url, collection)
    return client


def require_issue12_database_url() -> str:
    value = os.environ.get("ISSUE12_TEST_DATABASE_URL", "")
    parsed = urlparse(value)
    database = (parsed.path or "").lstrip("/").lower()
    if not parsed.scheme.startswith("postgresql+") or parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise RuntimeError("Issue 12 tests require an explicitly supplied local PostgreSQL URL")
    if not re.fullmatch(r"pontis_[a-z0-9_]*test[a-z0-9_]*", database) or any(
        token in database for token in ("prod", "production", "live", "railway")
    ):
        raise RuntimeError("Issue 12 tests require a dedicated pontis_*test* database")
    return value


def assert_safe_cleanup_collection(collection: str) -> None:
    expected = os.environ.get("ISSUE12_QDRANT_COLLECTION", "")
    if not expected or collection != expected or collection == "job_chunks":
        raise RuntimeError("Qdrant cleanup is restricted to the dedicated Issue 12 collection")


def delete_issue12_collection(client) -> None:
    """Delete only the validated Issue 12 collection after a real test."""
    url, collection = require_issue12_qdrant_config()
    assert_safe_cleanup_collection(collection)
    try:
        registered = _VALIDATED_CLIENTS.get(client)
    except TypeError:
        registered = None
    if registered != (url, collection):
        raise RuntimeError("Qdrant cleanup requires a client created by the validated Issue 12 factory")
    client.delete_collection(collection_name=collection)
