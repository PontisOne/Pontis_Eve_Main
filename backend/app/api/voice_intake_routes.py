"""Dependency contract for the voice-intake HTTP routes.

The existing handlers remain in ``server.py`` until this contract is wired and
the route bodies are moved in Batch 3C-2.  This module deliberately does not
import ``server.py``.
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from app.services.voice_dependency_contracts import (
    VoiceDependencyContract,
)


async def handle_candidate_voice_intake(
    request: VoiceCandidateIntakeRequest,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    dependencies: VoiceDependencyContract | None = None,
):
    """
    Receive voice intake transcript, extract structured info, merge into candidate profile.
    candidate_id is validated against the DB — never trusted blindly from the browser.
    """
    # Request validation must precede every candidate/database or transcript
    # processing operation.  In particular, do not let an empty submission
    # reach the database-backed candidate lookup.
    if dependencies is None:
        raise RuntimeError("VoiceDependencyContract is required")
    http = dependencies.http
    runtime = dependencies.runtime
    state = dependencies.state
    persistence = dependencies.persistence
    processing = dependencies.processing
    matching = dependencies.matching
    HTTPException = http.http_exception
    SessionLocal = runtime.session_factory
    _authorize_candidate = http.authorize_candidate
    _get_candidate_row = http.get_candidate_row
    _load_candidate_certificates = http.load_candidate_certificates
    _safe_voice_request_id = processing.safe_request_id
    _safe_voice_vapi_call_id = processing.safe_vapi_call_id
    _safe_voice_transcript_revision = processing.safe_transcript_revision
    _voice_intake_timing = processing.record_timing
    _normalize_voice_notes = state.normalize_voice_notes
    _voice_intake_turn_pairs = state.voice_intake_turn_pairs
    _llm_analyze_intake = state.analyze_intake
    _build_voice_intake_resume_from_notes = state.build_resume
    _voice_intake_turns_to_transcript = state.turns_to_transcript
    _clean_str = state.clean_text
    _extract_voice_info = processing.extract_voice_info
    _validate_voice_current_employment = processing.validate_current_employment
    _persist_voice_intake_profile_state = persistence.persist_profile_state
    _sync_voice_ledger = persistence.sync_voice_ledger
    _sync_profile_updates_to_ledger = persistence.sync_profile_updates_to_ledger
    _normalize_for_frontend = processing.normalize_for_frontend
    _schedule_voice_intake_matching = matching.schedule_matching
    _trigger_matching = matching.trigger_matching
    _parse_raw_data = state.parse_raw_data
    logger = runtime.logger_provider()
    text = runtime.sql_text
    uuid_validator = runtime.uuid_validator
    new_intake_id = runtime.new_intake_id
    monotonic = runtime.monotonic
    json_dumps = runtime.json_dumps
    schedule_task = runtime.schedule_task
    request_id = _safe_voice_request_id(x_request_id)
    timing_vapi_call_id = _safe_voice_vapi_call_id(request.vapi_call_id)
    timing_revision = _safe_voice_transcript_revision(request.transcript_revision)
    endpoint = "/api/voice/candidate-intake"
    started = monotonic()
    _voice_intake_timing(
        "request_started",
        request_id=request_id,
        vapi_call_id=timing_vapi_call_id,
        transcript_revision=timing_revision,
        endpoint=endpoint,
        source="provider_webhook" if request.provider_event_id else "browser",
    )
    if not request.transcript or not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")
    transcript = request.transcript.strip()
    _authorize_candidate(request.candidate_id, authorization)

    # Reject malformed UUIDs before they reach the database driver (which may
    # otherwise surface a cast error as an internal server error).
    try:
        uuid_validator(str(request.candidate_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=422, detail="Invalid candidate_id") from None

    # 1. Validate candidate exists only after cheap request validation.
    candidate = await _get_candidate_row(request.candidate_id)

    # 2. Idempotency: check for recent duplicate (same candidate, same transcript hash)
    async with SessionLocal() as db:
        dup = await db.execute(
            text("""
                SELECT id FROM candidate_voice_intakes
                WHERE candidate_id = :cid
                  AND md5(transcript) = md5(:transcript)
                  AND created_at > now() - interval '10 minutes'
                LIMIT 1
            """),
            {"cid": request.candidate_id, "transcript": transcript},
        )
        if hasattr(dup, "fetchone") and dup.fetchone():
            logger.info("Duplicate voice intake ignored")
            _voice_intake_timing(
                "request_finished",
                request_id=request_id,
                vapi_call_id=timing_vapi_call_id,
                transcript_revision=timing_revision,
                endpoint=endpoint,
                duration_ms=(monotonic() - started) * 1000,
                outcome="duplicate",
            )
            return {"status": "duplicate", "candidate_id": request.candidate_id}

    # 3. Persist intake record (status=pending)
    intake_id = new_intake_id()
    voice_notes = _normalize_voice_notes(request.voice_notes, transcript)
    async with SessionLocal() as db:
        await db.execute(
            text(
                "\n                INSERT INTO candidate_voice_intakes\n     "
                "               (id, candidate_id, transcript, voice_notes,"
                " status, created_at)\n                VALUES (:id, :cid, :t"
                "ranscript, CAST(:notes AS jsonb), 'processing', now())\n   "
                "         "
            ),
            {
                "id": intake_id,
                "cid": request.candidate_id,
                "transcript": transcript,
                "notes": json_dumps(voice_notes),
            },
        )
        await db.commit()

    existing_candidate = await _get_candidate_row(request.candidate_id)
    existing_raw = _parse_raw_data(existing_candidate.get("raw_data"))
    existing_vi = _parse_raw_data(existing_raw.get("voice_intake"))
    completed_turns, pending_question = _voice_intake_turn_pairs(
        voice_notes, transcript
    )
    stage_started = monotonic()
    llm_analysis = await _llm_analyze_intake(
        existing_candidate,
        completed_turns,
        pending_question,
        # The idempotency lookup above has already ruled out an exact
        # duplicate.  A new transcript must be evaluated on its own rather
        # than inheriting the previous request's active question/status.
        authoritative_next_question="",
        vapi_call_id=timing_vapi_call_id,
        endpoint=endpoint,
        request_id=request_id,
    )
    _voice_intake_timing(
        "analysis_finished",
        request_id=request_id,
        vapi_call_id=timing_vapi_call_id,
        transcript_revision=timing_revision,
        endpoint=endpoint,
        duration_ms=(monotonic() - stage_started) * 1000,
    )
    voice_intake_state = _build_voice_intake_resume_from_notes(
        voice_notes,
        transcript,
        existing_vi,
        candidate_profile=existing_candidate,
        llm_analysis=llm_analysis,
    )
    # Progress persistence is the canonical state writer.  Final processing
    # may use a different LLM pass for extraction, but must not replace its
    # canonical next question with spoken wording.
    if voice_intake_state.get("status") != "completed" and _clean_str(
        existing_vi.get("next_question")
    ):
        voice_intake_state["next_question"] = existing_vi["next_question"]

    genuinely_new_completed_transcript = (
        bool(voice_intake_state.get("completed_turns"))
        and bool(existing_vi.get("completed_turns"))
        and not pending_question
        and voice_intake_state.get("completed_turns")
        != existing_vi.get("completed_turns")
    )
    if genuinely_new_completed_transcript:
        # Turn merging is intentionally cumulative, but it is not transcript
        # identity.  Once the exact raw-transcript duplicate check above has
        # accepted this request as new, a completed LLM result with changed
        # canonical turns must not be reopened by the previous resume's stale
        # question.
        voice_intake_state["status"] = "completed"
        voice_intake_state["current_question"] = None
        voice_intake_state["next_question"] = None
        voice_intake_state["has_open_question"] = False
    if request.termination_reason and request.termination_reason != "completed":
        # The termination reason records how the call ended, but the backend's
        # independent topic analysis remains authoritative for completion.
        voice_intake_state["termination_reason"] = request.termination_reason
        if request.transcript_revision is not None:
            try:
                voice_intake_state["transcript_revision"] = int(
                    request.transcript_revision
                )
            except (TypeError, ValueError):
                pass
    # 4. Extract structured info via LLM
    voice_data_source = _voice_intake_turns_to_transcript(
        voice_intake_state.get("completed_turns") or []
    )
    stage_started = monotonic()
    voice_data = await _extract_voice_info(
        voice_data_source or transcript,
        request.candidate_id,
        timing_vapi_call_id,
        endpoint,
        request_id,
    )
    _voice_intake_timing(
        "extraction_finished",
        request_id=request_id,
        vapi_call_id=timing_vapi_call_id,
        transcript_revision=timing_revision,
        endpoint=endpoint,
        duration_ms=(monotonic() - stage_started) * 1000,
    )
    voice_data = _validate_voice_current_employment(
        voice_data, voice_data_source or transcript
    )

    # 5. Merge into existing profile, preferences, and resume state.
    stage_started = monotonic()
    merged = await _persist_voice_intake_profile_state(
        request.candidate_id,
        existing_candidate,
        voice_data,
        voice_intake_state,
    )
    _voice_intake_timing(
        "profile_persisted",
        request_id=request_id,
        vapi_call_id=timing_vapi_call_id,
        transcript_revision=timing_revision,
        endpoint=endpoint,
        duration_ms=(monotonic() - stage_started) * 1000,
    )
    await _sync_voice_ledger(
        request.candidate_id, voice_intake_state, request.provider_event_id
    )
    await _sync_profile_updates_to_ledger(request.candidate_id, voice_data, "vapi")

    # 6. Mark intake as completed only when the actual intake questions have all been answered.
    async with SessionLocal() as db:
        await db.execute(
            text(
                "\n                UPDATE candidate_voice_intakes\n          "
                "      SET status = :status,\n                    completed_"
                "at = CASE WHEN :status = 'completed' THEN now() ELSE compl"
                "eted_at END\n                WHERE id = :id\n            "
            ),
            {"id": intake_id, "status": voice_intake_state["status"]},
        )
        await db.commit()

    logger.info("Voice intake saved status=%s", voice_intake_state["status"])
    _voice_intake_timing(
        "request_finished",
        request_id=request_id,
        vapi_call_id=timing_vapi_call_id,
        transcript_revision=timing_revision,
        endpoint=endpoint,
        duration_ms=(monotonic() - started) * 1000,
        intake_status=voice_intake_state["status"],
    )

    if voice_intake_state["status"] == "completed":
        schedule_task(_trigger_matching(request.candidate_id))

    # Return updated profile so dashboard can refresh immediately
    updated_candidate = await _get_candidate_row(request.candidate_id)
    updated_profile_row = (
        dict(updated_candidate) if isinstance(updated_candidate, dict) else {}
    )
    updated_profile_row["candidate_certificates"] = await _load_candidate_certificates(
        request.candidate_id
    )
    updated_profile = _normalize_for_frontend(updated_profile_row)
    updated_profile["voice_intake_resume"] = voice_intake_state

    # The intake record is already committed above.  A sparse legacy profile
    # (or an unexpected empty merge result) must not turn that successful save
    # into a 500 while calculating this response-only metadata.
    candidate_for_comparison = candidate if isinstance(candidate, dict) else {}
    # A non-dict return represents no merge result, so report no field changes
    # rather than treating absent list keys as updates.
    merged_for_comparison = (
        merged if isinstance(merged, dict) else candidate_for_comparison
    )

    return {
        "status": voice_intake_state["status"],
        "intake_id": intake_id,
        "candidate_id": request.candidate_id,
        "fields_updated": [
            *[
                field
                for field in (
                    "summary",
                    "current_role",
                    "current_company",
                    "location",
                    "experience_years",
                    "skills",
                    "work_experience",
                    "education",
                )
                if merged_for_comparison.get(field)
                != candidate_for_comparison.get(field)
            ],
            *(
                ["projects"]
                if _parse_raw_data(merged_for_comparison.get("raw_data")).get(
                    "projects"
                )
                != _parse_raw_data(candidate_for_comparison.get("raw_data")).get(
                    "projects"
                )
                else []
            ),
        ],
        "profile": updated_profile,
        "voice_intake_state": voice_intake_state,
    }


async def handle_candidate_voice_intake_progress(
    request: VoiceCandidateIntakeProgressRequest,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    dependencies: VoiceDependencyContract | None = None,
):
    """Persist an in-progress voice intake snapshot without completing the profile merge."""
    if dependencies is None:
        raise RuntimeError("VoiceDependencyContract is required")
    http = dependencies.http
    runtime = dependencies.runtime
    state = dependencies.state
    persistence = dependencies.persistence
    processing = dependencies.processing
    matching = dependencies.matching
    HTTPException = http.http_exception
    _authorize_candidate = http.authorize_candidate
    _get_candidate_row = http.get_candidate_row
    _safe_voice_request_id = processing.safe_request_id
    _safe_voice_vapi_call_id = processing.safe_vapi_call_id
    _safe_voice_transcript_revision = processing.safe_transcript_revision
    _voice_intake_timing = processing.record_timing
    _normalize_voice_notes = state.normalize_voice_notes
    _voice_intake_turn_pairs = state.voice_intake_turn_pairs
    _llm_analyze_intake = state.analyze_intake
    _build_voice_intake_resume_from_notes = state.build_resume
    _voice_intake_turns_to_transcript = state.turns_to_transcript
    _clean_str = state.clean_text
    _parse_raw_data = state.parse_raw_data
    _extract_voice_info = processing.extract_voice_info
    _validate_voice_current_employment = processing.validate_current_employment
    _save_voice_intake_resume = persistence.save_voice_intake_resume
    _persist_voice_intake_profile_state = persistence.persist_profile_state
    _sync_voice_ledger = persistence.sync_voice_ledger
    _sync_profile_updates_to_ledger = persistence.sync_profile_updates_to_ledger
    _schedule_voice_intake_matching = matching.schedule_matching
    logger = runtime.logger_provider()
    monotonic = runtime.monotonic
    request_id = _safe_voice_request_id(x_request_id)
    timing_vapi_call_id = _safe_voice_vapi_call_id(request.vapi_call_id)
    timing_revision = _safe_voice_transcript_revision(request.transcript_revision)
    endpoint = "/api/voice/candidate-intake/progress"
    started = monotonic()
    _voice_intake_timing(
        "request_started",
        request_id=request_id,
        vapi_call_id=timing_vapi_call_id,
        transcript_revision=timing_revision,
        endpoint=endpoint,
        source="browser",
    )
    transcript = (request.transcript or "").strip()
    _authorize_candidate(request.candidate_id, authorization)
    if not transcript and not request.voice_notes:
        raise HTTPException(status_code=400, detail="Transcript is empty.")
    candidate = await _get_candidate_row(request.candidate_id)
    voice_notes = _normalize_voice_notes(request.voice_notes, transcript)

    existing_raw = _parse_raw_data(candidate.get("raw_data"))
    existing_vi = _parse_raw_data(existing_raw.get("voice_intake"))
    try:
        incoming_revision = (
            int(request.transcript_revision)
            if request.transcript_revision is not None
            else None
        )
    except (TypeError, ValueError):
        incoming_revision = None
    try:
        stored_revision = (
            int(existing_vi.get("transcript_revision"))
            if existing_vi.get("transcript_revision") is not None
            else -1
        )
    except (TypeError, ValueError):
        stored_revision = -1
    stored_vapi_call_id = _clean_str(existing_vi.get("vapi_call_id"))
    incoming_vapi_call_id = _clean_str(request.vapi_call_id)
    same_vapi_call = bool(
        incoming_vapi_call_id
        and stored_vapi_call_id
        and incoming_vapi_call_id == stored_vapi_call_id
    )
    if (
        same_vapi_call
        and incoming_revision is not None
        and incoming_revision < stored_revision
    ):
        logger.warning("[voice-intake] ignored stale progress")
        _voice_intake_timing(
            "request_finished",
            request_id=request_id,
            vapi_call_id=timing_vapi_call_id,
            transcript_revision=timing_revision,
            endpoint=endpoint,
            duration_ms=(monotonic() - started) * 1000,
            outcome="stale",
        )
        return {
            "status": "ignored_stale",
            "accepted": False,
            "candidate_id": request.candidate_id,
            "transcript_revision": stored_revision,
            "voice_intake_resume": existing_vi,
        }
    completed_turns, pending_question = _voice_intake_turn_pairs(
        voice_notes, transcript
    )
    stage_started = monotonic()
    llm_analysis = await _llm_analyze_intake(
        candidate,
        completed_turns,
        pending_question,
        authoritative_next_question=(
            existing_vi.get("current_question")
            or existing_vi.get("next_question")
            or ""
        ),
        vapi_call_id=timing_vapi_call_id,
        endpoint=endpoint,
        request_id=request_id,
    )
    _voice_intake_timing(
        "analysis_finished",
        request_id=request_id,
        vapi_call_id=timing_vapi_call_id,
        transcript_revision=timing_revision,
        endpoint=endpoint,
        duration_ms=(monotonic() - stage_started) * 1000,
    )
    resume = _build_voice_intake_resume_from_notes(
        voice_notes,
        transcript,
        existing_vi,
        candidate_profile=candidate,
        llm_analysis=llm_analysis,
    )
    if not resume.get("status"):
        resume["status"] = "in_progress"
    if incoming_revision is not None:
        resume["transcript_revision"] = incoming_revision
    if incoming_vapi_call_id:
        resume["vapi_call_id"] = incoming_vapi_call_id
    if request.termination_reason:
        resume["termination_reason"] = request.termination_reason

    await _save_voice_intake_resume(request.candidate_id, resume)
    voice_data_source = _voice_intake_turns_to_transcript(
        resume.get("completed_turns") or []
    )
    stage_started = monotonic()
    voice_data = (
        await _extract_voice_info(
            voice_data_source or transcript,
            request.candidate_id,
            timing_vapi_call_id,
            endpoint,
            request_id,
        )
        if (voice_data_source or transcript)
        else {}
    )
    _voice_intake_timing(
        "extraction_finished",
        request_id=request_id,
        vapi_call_id=timing_vapi_call_id,
        transcript_revision=timing_revision,
        endpoint=endpoint,
        duration_ms=(monotonic() - stage_started) * 1000,
    )
    voice_data = _validate_voice_current_employment(
        voice_data, voice_data_source or transcript
    )
    stage_started = monotonic()
    await _persist_voice_intake_profile_state(
        request.candidate_id, candidate, voice_data, resume
    )
    _voice_intake_timing(
        "profile_persisted",
        request_id=request_id,
        vapi_call_id=timing_vapi_call_id,
        transcript_revision=timing_revision,
        endpoint=endpoint,
        duration_ms=(monotonic() - stage_started) * 1000,
    )
    await _sync_voice_ledger(request.candidate_id, resume)
    await _sync_profile_updates_to_ledger(request.candidate_id, voice_data, "vapi")
    resume_status = resume.get("status")
    if resume_status == "completed":
        logger.info(
            "[voice-intake] persisted progress status=%s; scheduling matching refresh",
            resume_status,
        )
    else:
        logger.info(
            "[voice-intake] persisted progress status=%s; matching refresh deferred",
            resume_status,
        )
    _schedule_voice_intake_matching(request.candidate_id, resume_status)
    _voice_intake_timing(
        "request_finished",
        request_id=request_id,
        vapi_call_id=timing_vapi_call_id,
        transcript_revision=timing_revision,
        endpoint=endpoint,
        duration_ms=(monotonic() - started) * 1000,
        intake_status=resume_status,
    )
    return {
        "status": "saved",
        "accepted": True,
        "candidate_id": request.candidate_id,
        "transcript_revision": resume.get("transcript_revision"),
        "voice_intake_resume": resume,
    }




def create_voice_intake_router(
    final_request_model: type[BaseModel],
    progress_request_model: type[BaseModel],
    dependency_factory: Callable[[], VoiceDependencyContract],
) -> APIRouter:
    """Create both voice-intake routes with per-request dependencies."""

    async def final_endpoint(
        request: BaseModel,
        authorization: str | None = Header(default=None),
        x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    ):
        return await handle_candidate_voice_intake(
            request, authorization, x_request_id, dependency_factory()
        )

    async def progress_endpoint(
        request: BaseModel,
        authorization: str | None = Header(default=None),
        x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    ):
        return await handle_candidate_voice_intake_progress(
            request, authorization, x_request_id, dependency_factory()
        )

    final_endpoint.__annotations__["request"] = final_request_model
    progress_endpoint.__annotations__["request"] = progress_request_model
    router = APIRouter()
    router.post("/voice/candidate-intake")(final_endpoint)
    router.post("/voice/candidate-intake/progress")(progress_endpoint)
    return router
