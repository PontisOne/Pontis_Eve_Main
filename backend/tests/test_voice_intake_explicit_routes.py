from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.api.voice_intake_routes import (
    create_voice_intake_router,
    handle_candidate_voice_intake_progress,
)
from test_voice_dependency_contracts import (
    _contract,
)


class FinalRequest(BaseModel):
    candidate_id: str
    transcript: str | None = None
    voice_notes: list[dict[str, object]] | None = None
    vapi_call_id: str | None = None
    provider_event_id: str | None = None
    termination_reason: str | None = None
    transcript_revision: str | None = None


class ProgressRequestModel(BaseModel):
    candidate_id: str
    transcript: str | None = None
    voice_notes: list[dict[str, object]] | None = None
    vapi_call_id: str | None = None
    transcript_revision: str | None = None
    termination_reason: str | None = None


def _app_with_overridable_contract(current: dict[str, object]) -> FastAPI:
    app = FastAPI()
    router = create_voice_intake_router(
        FinalRequest,
        ProgressRequestModel,
        lambda: current["contract"],
    )
    app.include_router(router, prefix="/api")
    return app


def _http_contract():
    contract = _contract()
    return replace(
        contract,
        http=replace(contract.http, http_exception=HTTPException),
    )


def test_final_route_observes_override_after_router_construction():
    original_calls: list[str] = []
    replacement_calls: list[str] = []
    original = _http_contract()
    replacement = replace(
        original,
        http=replace(
            original.http,
            authorize_candidate=lambda *_args: original_calls.append("original"),
        ),
    )
    current: dict[str, object] = {"contract": original}
    app = _app_with_overridable_contract(current)
    current["contract"] = replace(
        replacement,
        http=replace(
            replacement.http,
            authorize_candidate=_replacement_authorizer(replacement_calls),
        ),
    )

    response = TestClient(app).post(
        "/api/voice/candidate-intake",
        json={"candidate_id": "candidate", "transcript": "hello"},
    )

    assert response.status_code == 401
    assert replacement_calls == ["replacement"]
    assert original_calls == []


def test_progress_route_observes_override_after_router_construction():
    original_calls: list[str] = []
    replacement_calls: list[str] = []
    original = _http_contract()
    replacement = replace(
        original,
        http=replace(
            original.http,
            authorize_candidate=lambda *_args: original_calls.append("original"),
        ),
    )
    current: dict[str, object] = {"contract": original}
    app = _app_with_overridable_contract(current)
    current["contract"] = replace(
        replacement,
        http=replace(
            replacement.http,
            authorize_candidate=lambda *_args: replacement_calls.append("replacement"),
        ),
    )

    response = TestClient(app).post(
        "/api/voice/candidate-intake/progress",
        json={"candidate_id": "candidate", "transcript": ""},
    )

    assert response.status_code == 400
    assert replacement_calls == ["replacement"]
    assert original_calls == []


def test_both_voice_routes_are_registered_once_and_have_openapi_models():
    app = _app_with_overridable_contract({"contract": _http_contract()})
    routes = [
        route
        for route in app.routes
        if route.path.startswith("/api/voice/candidate-intake")
    ]
    assert [(route.path, route.methods) for route in routes] == [
        ("/api/voice/candidate-intake", {"POST"}),
        ("/api/voice/candidate-intake/progress", {"POST"}),
    ]
    schemas = app.openapi()["paths"]
    assert "FinalRequest" in schemas["/api/voice/candidate-intake"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert "ProgressRequestModel" in schemas["/api/voice/candidate-intake/progress"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]


@pytest.mark.asyncio
async def test_progress_handler_direct_success_persists_and_returns_resume():
    contract = _http_contract()
    saved: list[dict[str, object]] = []
    persisted: list[dict[str, object]] = []
    contract = replace(
        contract,
        http=replace(
            contract.http,
            get_candidate_row=lambda _candidate_id: _candidate(),
        ),
        persistence=replace(
            contract.persistence,
            save_voice_intake_resume=lambda _candidate_id, resume: _record(saved, resume),
            persist_profile_state=lambda _candidate_id, _candidate, _voice_data, resume: _record(persisted, resume),
        ),
    )

    response = await handle_candidate_voice_intake_progress(
        ProgressRequestModel(candidate_id="candidate", transcript="I build APIs."),
        dependencies=contract,
    )

    assert response["status"] == "saved"
    assert response["accepted"] is True
    assert response["candidate_id"] == "candidate"
    assert saved and persisted
    assert "voice_intake_resume" in response


@pytest.mark.asyncio
async def test_progress_handler_directly_rejects_empty_submission():
    with pytest.raises(Exception) as error:
        await handle_candidate_voice_intake_progress(
            ProgressRequestModel(candidate_id="candidate"),
            dependencies=_http_contract(),
        )
    assert getattr(error.value, "status_code", None) == 400


async def _candidate() -> dict[str, object]:
    return {"id": "candidate", "raw_data": {}}


async def _record(target: list[dict[str, object]], resume: dict[str, object]) -> None:
    target.append(resume)


def _replacement_authorizer(calls: list[str]):
    def authorize(*_args: object) -> None:
        calls.append("replacement")
        raise HTTPException(status_code=401, detail="unauthorized")

    return authorize
