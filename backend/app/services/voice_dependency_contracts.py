"""Typed dependency contracts for voice-intake application boundaries.

This module contains no route or application imports.  It is intentionally
usable by the API layer, voice services, and the application composition root.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
import logging
from typing import Awaitable, Callable, Protocol, Sequence, TypeVar
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import TextClause


T = TypeVar("T")


class SessionFactory(Protocol):
    def __call__(self) -> AbstractAsyncContextManager[AsyncSession]: ...


class TaskScheduler(Protocol):
    def __call__(self, awaitable: Awaitable[T]) -> object: ...


class CandidateAuthorizer(Protocol):
    def __call__(self, candidate_id: str, authorization: str | None) -> None: ...


class CandidateLoader(Protocol):
    def __call__(self, candidate_id: str) -> Awaitable[dict[str, object]]: ...


class CertificateLoader(Protocol):
    def __call__(self, candidate_id: str) -> Awaitable[list[dict[str, object]]]: ...


class VoiceAnalyzer(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> Awaitable[dict[str, object]]: ...


class VoiceExtractor(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> Awaitable[dict[str, object]]: ...


@dataclass(frozen=True)
class VoiceHttpDependencies:
    final_request_model: type[BaseModel]
    progress_request_model: type[BaseModel]
    http_exception: type[Exception]
    authorize_candidate: CandidateAuthorizer
    get_candidate_row: CandidateLoader
    load_candidate_certificates: CertificateLoader


@dataclass(frozen=True)
class VoiceRuntimeDependencies:
    session_factory: SessionFactory
    sql_text: Callable[[str], TextClause]
    new_intake_id: Callable[[], str]
    uuid_validator: Callable[[str], UUID]
    monotonic: Callable[[], float]
    json_dumps: Callable[[object], str]
    logger_provider: Callable[[], logging.Logger]
    schedule_task: TaskScheduler


@dataclass(frozen=True)
class VoiceStateDependencies:
    parse_raw_data: Callable[[object], dict[str, object]]
    normalize_voice_notes: Callable[..., list[dict[str, object]]]
    voice_intake_turn_pairs: Callable[..., tuple[list[dict[str, object]], str | None]]
    analyze_intake: VoiceAnalyzer
    build_resume: Callable[..., dict[str, object]]
    turns_to_transcript: Callable[[Sequence[dict[str, object]]], str]
    clean_text: Callable[[object], str]


@dataclass(frozen=True)
class VoicePersistenceDependencies:
    persist_profile_state: Callable[..., Awaitable[dict[str, object]]]
    save_voice_intake_resume: Callable[..., Awaitable[None]]
    sync_voice_ledger: Callable[..., Awaitable[None]]
    sync_profile_updates_to_ledger: Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class VoiceProcessingDependencies:
    safe_request_id: Callable[[object], str]
    safe_vapi_call_id: Callable[[object], str | None]
    safe_transcript_revision: Callable[[object], str | None]
    record_timing: Callable[..., None]
    extract_voice_info: VoiceExtractor
    validate_current_employment: Callable[[dict[str, object], str], dict[str, object]]
    normalize_for_frontend: Callable[[dict[str, object]], dict[str, object]]


@dataclass(frozen=True)
class VoiceMatchingDependencies:
    schedule_matching: Callable[[str, str], object]
    trigger_matching: Callable[[str], Awaitable[object]]


@dataclass(frozen=True)
class VoiceDependencyContract:
    """Complete grouped contract for a voice-intake route invocation."""

    http: VoiceHttpDependencies
    runtime: VoiceRuntimeDependencies
    state: VoiceStateDependencies
    persistence: VoicePersistenceDependencies
    processing: VoiceProcessingDependencies
    matching: VoiceMatchingDependencies
