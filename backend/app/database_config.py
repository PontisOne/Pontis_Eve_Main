"""Validated async database configuration and runtime construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


SUPPORTED_ASYNC_DRIVERS = frozenset({"postgresql+asyncpg", "sqlite+aiosqlite"})


class DatabaseConfigurationError(ValueError):
    """Raised when an application database URL is missing or not async-capable."""


def validate_async_database_url(
    value: str | None,
    *,
    variable_name: str = "DATABASE_URL",
    supported_drivers: frozenset[str] = SUPPORTED_ASYNC_DRIVERS,
) -> str:
    """Validate an async SQLAlchemy URL without exposing credentials."""
    if not value or not value.strip():
        raise DatabaseConfigurationError(
            f"{variable_name} is required and must use an async SQLAlchemy driver "
            f"({', '.join(sorted(supported_drivers))})."
        )
    try:
        parsed = make_url(value)
    except Exception as error:
        raise DatabaseConfigurationError(
            f"{variable_name} is not a valid SQLAlchemy database URL."
        ) from error
    if parsed.drivername not in supported_drivers:
        raise DatabaseConfigurationError(
            f"{variable_name} uses '{parsed.drivername}', but the async application "
            f"requires one of: {', '.join(sorted(supported_drivers))}."
        )
    return value


@dataclass(frozen=True)
class AsyncDatabaseRuntime:
    engine: Any
    session_local: Any


def create_async_database_runtime(
    database_url: str | None,
    *,
    engine_factory: Callable[..., Any] = create_async_engine,
    session_factory: Callable[..., Any] = async_sessionmaker,
    engine_kwargs: dict[str, Any] | None = None,
    session_kwargs: dict[str, Any] | None = None,
) -> AsyncDatabaseRuntime:
    """Validate and construct the application engine and session factory once."""
    validated_url = validate_async_database_url(database_url)
    engine = engine_factory(validated_url, **(engine_kwargs or {"pool_pre_ping": True}))
    sessions = session_factory(
        bind=engine,
        **(session_kwargs or {"expire_on_commit": False}),
    )
    return AsyncDatabaseRuntime(engine=engine, session_local=sessions)
