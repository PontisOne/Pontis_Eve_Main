"""Explicit runtime bindings shared by application subsystems.

The application bootstrap configures these bindings once after creating the
database engine.  Modules may still override them in focused tests without
importing the HTTP application module.
"""

from __future__ import annotations

from typing import Any

_session_local: Any = None
_engine: Any = None


def configure_database(session_local: Any, engine: Any) -> None:
    global _session_local, _engine
    _session_local = session_local
    _engine = engine


def session_local() -> Any:
    if _session_local is None:
        raise RuntimeError("Database runtime dependencies are not configured")
    return _session_local


def engine() -> Any:
    if _engine is None:
        raise RuntimeError("Database runtime dependencies are not configured")
    return _engine
