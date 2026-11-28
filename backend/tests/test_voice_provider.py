from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.integrations.voice_provider import (
    VoiceProviderDependencies,
    create_voice_provider_router,
)


class _Result:
    rowcount = 1


class _Db:
    def __init__(self) -> None:
        self.execute = AsyncMock(return_value=_Result())
        self.commit = AsyncMock()


def _session_factory(db: _Db):
    @asynccontextmanager
    async def factory():
        yield db

    return factory


def _provider(db: _Db, intake_handler) -> VoiceProviderDependencies:
    return VoiceProviderDependencies(
        configuration_status=lambda: {"VAPI_PUBLIC_KEY": True, "EVE_VAPI_ASSISTANT_ID": True},
        session_local=_session_factory(db),
        intake_handler=intake_handler,
        issue_candidate_token=lambda candidate_id: f"token-{candidate_id}",
        get_bearer_token=lambda _authorization: "secret",
        webhook_secret="secret",
    )


def _app(dependencies: VoiceProviderDependencies) -> FastAPI:
    app = FastAPI()
    app.include_router(create_voice_provider_router(dependencies))
    return app


def _payload() -> dict:
    return {
        "type": "end-of-call-report",
        "call": {"id": "call-1", "metadata": {"candidateId": "9ef2e4f9-90a6-43f8-ba94-98af39c86fc4"}},
        "artifact": {"transcript": "hello"},
    }


def test_provider_router_registers_expected_vapi_routes():
    app = _app(_provider(_Db(), AsyncMock(return_value={"status": "completed"})))
    routes = {(route.path, method) for route in app.routes for method in getattr(route, "methods", set())}
    assert ("/webhooks/vapi", "POST") in routes
    assert ("/config/vapi", "GET") in routes


def test_provider_accepts_valid_signature_and_invokes_intake():
    db = _Db()
    intake = AsyncMock(return_value={"status": "completed"})
    client = TestClient(_app(_provider(db, intake)))

    response = client.post("/webhooks/vapi", headers={"x-vapi-secret": "secret"}, json=_payload())

    assert response.status_code == 200
    assert response.json()["status"] == "processed"
    intake.assert_awaited_once()
    assert intake.await_args.kwargs["candidate_id"] == "9ef2e4f9-90a6-43f8-ba94-98af39c86fc4"


def test_provider_rejects_invalid_signature():
    client = TestClient(_app(_provider(_Db(), AsyncMock())))

    response = client.post("/webhooks/vapi", headers={"x-vapi-secret": "wrong"}, json=_payload())

    assert response.status_code == 401


def test_provider_acknowledges_unknown_event_and_malformed_payload():
    client = TestClient(_app(_provider(_Db(), AsyncMock())))

    unknown = client.post(
        "/webhooks/vapi",
        headers={"x-vapi-secret": "secret"},
        json={"type": "future-event"},
    )
    malformed = client.post(
        "/webhooks/vapi",
        headers={"x-vapi-secret": "secret", "content-type": "application/json"},
        content="[not-json-object]",
    )

    assert unknown.status_code == 200
    assert unknown.json() == {"status": "ok"}
    assert malformed.status_code == 200


def test_provider_reports_missing_secret_configuration(monkeypatch):
    monkeypatch.delenv("VAPI_WEBHOOK_SECRET", raising=False)
    dependencies = _provider(_Db(), AsyncMock())
    dependencies = VoiceProviderDependencies(
        **{**dependencies.__dict__, "webhook_secret": ""}
    )
    client = TestClient(_app(dependencies))

    response = client.post("/webhooks/vapi", json=_payload())

    assert response.status_code == 503
