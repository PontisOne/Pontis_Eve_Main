import asyncio

import pytest
import server
from fastapi import HTTPException


class _FakeResult:
    def fetchone(self):
        return ("candidate-1",)


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, *_args, **_kwargs):
        return _FakeResult()


def test_local_development_login_uses_allowlist_and_normal_session(monkeypatch):
    monkeypatch.setattr(server, "_runtime_environment", lambda: "development")
    monkeypatch.setenv(
        "LOCAL_DEV_AUTH_EMAILS", "developer@example.com, other@example.com"
    )
    monkeypatch.setattr(server, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(
        server,
        "_get_candidate_profile_payload",
        lambda _candidate_id: asyncio.sleep(0, result={"candidate_id": "candidate-1"}),
    )
    monkeypatch.setattr(
        server, "_issue_candidate_session_token", lambda _candidate_id: "signed-token"
    )

    result = asyncio.run(server.local_development_login(" DEVELOPER@example.com "))

    assert result == {
        "candidate_id": "candidate-1",
        "candidate_token": "signed-token",
        "profile": {"candidate_id": "candidate-1"},
        "needs_onboarding": False,
    }


def test_local_development_login_is_rejected_in_production(monkeypatch):
    monkeypatch.setattr(server, "_runtime_environment", lambda: "production")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.local_development_login("developer@example.com"))

    assert exc_info.value.status_code == 404


def test_local_development_login_is_rejected_outside_development(monkeypatch):
    monkeypatch.setattr(server, "_runtime_environment", lambda: "staging")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.local_development_login("developer@example.com"))

    assert exc_info.value.status_code == 404
