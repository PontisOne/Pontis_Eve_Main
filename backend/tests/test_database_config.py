from __future__ import annotations

import pytest

from app.database_config import (
    DatabaseConfigurationError,
    create_async_database_runtime,
    validate_async_database_url,
)


@pytest.mark.parametrize(
    "value",
    [
        "postgresql+asyncpg://user:password@localhost:5432/pontis",
        "sqlite+aiosqlite:///./pontis.db",
    ],
)
def test_validate_async_database_url_accepts_supported_drivers(value):
    assert validate_async_database_url(value) == value


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "postgresql+psycopg2://user:password@localhost/pontis",
        "postgresql://user:password@localhost/pontis",
        "mysql+aiomysql://user:password@localhost/pontis",
    ],
)
def test_validate_async_database_url_rejects_missing_or_sync_drivers(value):
    with pytest.raises(DatabaseConfigurationError):
        validate_async_database_url(value)


def test_validation_preserves_credentials_and_query_parameters():
    value = "postgresql+asyncpg://user:p%40ss@localhost/pontis?sslmode=require"
    assert validate_async_database_url(value) == value


def test_invalid_url_is_rejected_before_engine_creation():
    created = []

    def engine_factory(*args, **kwargs):
        created.append((args, kwargs))
        return object()

    with pytest.raises(DatabaseConfigurationError):
        create_async_database_runtime(
            "postgresql+psycopg2://user:password@localhost/pontis",
            engine_factory=engine_factory,
        )
    assert created == []


def test_runtime_construction_uses_validated_url_and_expected_options():
    engine = object()
    calls = []

    def engine_factory(url, **kwargs):
        calls.append(("engine", url, kwargs))
        return engine

    def session_factory(**kwargs):
        calls.append(("session", kwargs))
        return "sessions"

    runtime = create_async_database_runtime(
        "sqlite+aiosqlite:///./pontis.db",
        engine_factory=engine_factory,
        session_factory=session_factory,
    )

    assert runtime.engine is engine
    assert runtime.session_local == "sessions"
    assert calls == [
        ("engine", "sqlite+aiosqlite:///./pontis.db", {"pool_pre_ping": True}),
        ("session", {"bind": engine, "expire_on_commit": False}),
    ]
