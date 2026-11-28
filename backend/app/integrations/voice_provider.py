"""Vapi-specific configuration and webhook integration.

Voice-intake state and persistence remain owned by their respective services;
this module only authenticates and normalizes provider traffic.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from starlette.requests import Request


@dataclass(frozen=True)
class VoiceProviderDependencies:
    configuration_status: Callable[[], dict[str, bool]]
    session_local: Callable[..., Any]
    intake_handler: Callable[..., Awaitable[dict]]
    issue_candidate_token: Callable[[str], str]
    get_bearer_token: Callable[[str | None], str]
    logger: Any = logging.getLogger(__name__)
    public_key: str = ""
    assistant_id: str = ""
    webhook_secret: str = ""


def browser_config(dependencies: VoiceProviderDependencies) -> dict[str, str]:
    """Return browser-safe Vapi configuration without exposing secrets."""
    status = dependencies.configuration_status()
    if not status.get("VAPI_PUBLIC_KEY") or not status.get("EVE_VAPI_ASSISTANT_ID"):
        missing = [name for name, configured in status.items() if not configured]
        dependencies.logger.error(
            "Vapi browser configuration incomplete; missing=%s", ",".join(missing)
        )
        raise HTTPException(status_code=503, detail="Voice intake is not configured.")
    return {"publicKey": dependencies.public_key, "assistantId": dependencies.assistant_id}


async def handle_webhook(request: Request, dependencies: VoiceProviderDependencies) -> dict:
    """Authenticate and process Vapi's authoritative terminal report."""
    webhook_secret = (dependencies.webhook_secret or os.environ.get("VAPI_WEBHOOK_SECRET", "")).strip()
    if not webhook_secret:
        raise HTTPException(status_code=503, detail="Vapi webhook is not configured.")
    supplied = request.headers.get("x-vapi-secret", "")
    if not supplied:
        supplied = dependencies.get_bearer_token(request.headers.get("authorization"))
    if not hmac.compare_digest(supplied, webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid Vapi webhook signature.")
    try:
        payload = await request.json()
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    message = payload.get("message")
    message_type = message.get("type") if isinstance(message, dict) else None
    event_type = payload.get("type") or message_type
    dependencies.logger.info("Received authenticated VAPI webhook event: %s", event_type)
    if event_type != "end-of-call-report":
        return {"status": "ok"}

    event = message if isinstance(message, dict) else payload
    call = event.get("call") if isinstance(event.get("call"), dict) else {}
    metadata = call.get("metadata") or event.get("metadata") or {}
    overrides = call.get("assistantOverrides") if isinstance(call.get("assistantOverrides"), dict) else {}
    variable_values = overrides.get("variableValues") if isinstance(overrides.get("variableValues"), dict) else {}
    candidate_id = metadata.get("candidateId") or metadata.get("candidate_id")
    candidate_id = candidate_id or variable_values.get("candidateId") or variable_values.get("candidate_id")
    artifact = event.get("artifact") if isinstance(event.get("artifact"), dict) else {}
    transcript = artifact.get("transcript") or event.get("transcript")
    call_id = str(call.get("id") or event.get("callId") or "").strip()
    transcript = transcript if isinstance(transcript, str) else ""
    transcript_hash = hashlib.sha256(transcript.encode()).hexdigest()
    provider_event_id = str(event.get("id") or payload.get("id") or f"vapi:{call_id or 'unknown'}:{event_type}:{transcript_hash}")
    try:
        uuid.UUID(str(candidate_id))
    except (ValueError, AttributeError, TypeError):
        candidate_id = None
    if candidate_id and transcript.strip():
        async with dependencies.session_local() as db:
            result = await db.execute(
                text(
                    "INSERT INTO candidate_voice_provider_events "
                    "(provider_event_id, candidate_id, vapi_call_id, transcript_hash, event_type, payload) "
                    "VALUES (:event_id, :candidate_id, NULLIF(:call_id, ''), :hash, :event_type, CAST(:payload AS jsonb)) "
                    "ON CONFLICT (provider_event_id) DO NOTHING RETURNING provider_event_id"
                ),
                {"event_id": provider_event_id, "candidate_id": candidate_id, "call_id": call_id, "hash": transcript_hash, "event_type": event_type, "payload": json.dumps({"type": event_type, "call_id": call_id})},
            )
            await db.commit()
            inserted = (
                result.fetchone()
                if hasattr(result, "fetchone")
                else (object() if getattr(result, "rowcount", 0) else None)
            )
        if not inserted:
            return {"status": "duplicate", "provider_event_id": provider_event_id}
        try:
            outcome = await dependencies.intake_handler(
                transcript=transcript,
                candidate_id=candidate_id,
                vapi_call_id=call_id or None,
                provider_event_id=provider_event_id,
                authorization=f"Bearer {dependencies.issue_candidate_token(candidate_id)}",
            )
            async with dependencies.session_local() as db:
                await db.execute(
                    text(
                        "UPDATE candidate_voice_provider_events SET status = 'processed', processed_at = now() WHERE provider_event_id = :event_id"
                    ), {"event_id": provider_event_id}
                )
                await db.commit()
            return {"status": "processed", "provider_event_id": provider_event_id, "intake_status": outcome.get("status")}
        except Exception:
            async with dependencies.session_local() as db:
                await db.execute(
                    text(
                        "UPDATE candidate_voice_provider_events SET status = 'failed' WHERE provider_event_id = :event_id"
                    ), {"event_id": provider_event_id}
                )
                await db.commit()
            raise
    return {"status": "ok"}


def create_voice_provider_router(dependencies: VoiceProviderDependencies) -> APIRouter:
    router = APIRouter()

    @router.get("/config/vapi")
    async def get_vapi_config() -> dict[str, str]:
        return browser_config(dependencies)

    @router.post("/webhooks/vapi")
    async def vapi_webhook(request: Request) -> dict:
        return await handle_webhook(request, dependencies)

    return router
