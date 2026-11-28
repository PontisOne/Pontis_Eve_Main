from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from app.services.voice_dependency_contracts import (
    VoiceDependencyContract,
    VoiceHttpDependencies,
    VoiceMatchingDependencies,
    VoicePersistenceDependencies,
    VoiceProcessingDependencies,
    VoiceRuntimeDependencies,
    VoiceStateDependencies,
)


class FinalRequest(BaseModel):
    candidate_id: str


class ProgressRequest(BaseModel):
    candidate_id: str


@asynccontextmanager
async def session_factory():
    yield object()


async def _candidate(candidate_id: str) -> dict[str, object]:
    return {"id": candidate_id}


async def _certificates(candidate_id: str) -> list[dict[str, object]]:
    return []


async def _empty_dict(*args: object, **kwargs: object) -> dict[str, object]:
    return {}


async def _empty_none(*args: object, **kwargs: object) -> None:
    return None


def _contract() -> VoiceDependencyContract:
    return VoiceDependencyContract(
        http=VoiceHttpDependencies(
            final_request_model=FinalRequest,
            progress_request_model=ProgressRequest,
            http_exception=Exception,
            authorize_candidate=lambda candidate_id, authorization: None,
            get_candidate_row=_candidate,
            load_candidate_certificates=_certificates,
        ),
        runtime=VoiceRuntimeDependencies(
            session_factory=session_factory,
            sql_text=lambda value: value,
            new_intake_id=lambda: str(uuid4()),
            uuid_validator=UUID,
            monotonic=lambda: 0.0,
            json_dumps=lambda value: "{}",
            logger_provider=lambda: logging.getLogger("test-voice-contract"),
            schedule_task=lambda awaitable: asyncio.create_task(awaitable),
        ),
        state=VoiceStateDependencies(
            parse_raw_data=lambda value: {},
            normalize_voice_notes=lambda *args: [],
            voice_intake_turn_pairs=lambda *args: ([], None),
            analyze_intake=_empty_dict,
            build_resume=lambda *args, **kwargs: {},
            turns_to_transcript=lambda turns: "",
            clean_text=lambda value: str(value or ""),
        ),
        persistence=VoicePersistenceDependencies(
            persist_profile_state=_empty_dict,
            save_voice_intake_resume=_empty_none,
            sync_voice_ledger=_empty_none,
            sync_profile_updates_to_ledger=_empty_none,
        ),
        processing=VoiceProcessingDependencies(
            safe_request_id=lambda value: "request-id",
            safe_vapi_call_id=lambda value: None,
            safe_transcript_revision=lambda value: None,
            record_timing=lambda *args, **kwargs: None,
            extract_voice_info=_empty_dict,
            validate_current_employment=lambda data, transcript: data,
            normalize_for_frontend=lambda profile: profile,
        ),
        matching=VoiceMatchingDependencies(
            schedule_matching=lambda candidate_id, status: None,
            trigger_matching=_empty_none,
        ),
    )


def test_voice_dependency_contract_has_all_grouped_sections():
    contract = _contract()
    assert contract.http.final_request_model is FinalRequest
    assert contract.runtime.session_factory is session_factory
    assert contract.state.analyze_intake is _empty_dict
    assert contract.persistence.persist_profile_state is _empty_dict
    assert contract.processing.extract_voice_info is _empty_dict
    assert contract.matching.trigger_matching is _empty_none


def test_voice_dependency_contract_requires_each_group():
    with pytest.raises(TypeError):
        VoiceDependencyContract()  # type: ignore[call-arg]
