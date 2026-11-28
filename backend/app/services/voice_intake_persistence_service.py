"""Persistence workflows for voice-intake state and evidence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy import text

from app.services.voice_intake_service import (
    _build_voice_intake_resume_from_notes,
    _voice_intake_turn_pairs,
    _voice_intake_turns_to_transcript,
)


@dataclass(frozen=True)
class VoiceIntakePersistenceDependencies:
    session_local: Callable[..., Any]
    parse_raw_data: Callable[[Any], dict]
    voice_intake_turn_pairs: Callable[..., tuple[list[dict], str | None]] = _voice_intake_turn_pairs
    build_voice_intake_resume: Callable[..., dict] = _build_voice_intake_resume_from_notes
    voice_intake_turns_to_transcript: Callable[..., str] = _voice_intake_turns_to_transcript
    logger: Any = None
    sanitize_profile_field_mapping: Callable[[dict], dict] = lambda value: value
    merge_voice_into_profile: Callable[[dict, dict], dict] = lambda candidate, voice: {**candidate, **voice}
    normalize_candidate_opportunity_type: Callable[[Any], str] = lambda value: value or ""
    normalize_profile_text: Callable[[Any], str] = lambda value: str(value or "")
    append_demonstrated_skill_evidence: Callable[..., dict] = lambda raw, *args: raw
    profile_cache: Callable[[], Any] = lambda: None
    upsert_candidate_preferences: Callable[..., Any] = lambda *args, **kwargs: None
    clean_text: Callable[[Any], str] = lambda value: " ".join(str(value).split()) if value is not None else ""
    stable_topic: Callable[[str], str] = lambda value: value
    intake_answer_status: Callable[[str, str], str] = lambda _topic, _answer: "ANSWERED"

async def _save_voice_intake_resume(
    dependencies: VoiceIntakePersistenceDependencies,
    candidate_id: str,
    resume: dict,
) -> None:
    async with dependencies.session_local() as db:
        await db.execute(
            text("""
                UPDATE candidates
                SET raw_data = jsonb_set(
                        COALESCE(raw_data, '{}'::jsonb),
                        '{voice_intake}',
                        CAST(:voice_intake AS jsonb),
                        true
                    ),
                    updated_at = now(),
                    updated_by_source = 'eve_voice'
                WHERE id = :cid
            """),
            {"voice_intake": json.dumps(resume), "cid": candidate_id},
        )
        await db.commit()
    cache = dependencies.profile_cache()
    if cache is not None:
        await cache.invalidate(candidate_id)


INTAKE_TOPIC_PRIORITY = (
    "name",
    "location",
    "bio",
    "current_role",
    "current_company",
    "experience_years",
    "skills",
    "work_experience",
    "preferred_roles",
    "preferred_locations",
    "remote_preference",
    "employment_types",
    "willing_to_relocate",
    "open_to_opportunities",
    "notice_period",
    "expected_salary",
    "education",
    "certifications",
    "projects",
)
_PROFILE_FIELD_TOPICS = {
    "current_role": "current_role",
    "current_company": "current_company",
    "experience_years": "experience_years",
    "skills": "skills",
    "preferred_roles": "preferred_roles",
    "preferred_locations": "preferred_locations",
    "remote_preference": "remote_preference",
    "notice_period": "notice_period",
    "availability": "notice_period",
    "expected_salary": "expected_salary",
    "education": "education",
    "certifications": "certifications",
    "projects": "projects",
}

async def _upsert_intake_ledger(
    dependencies: VoiceIntakePersistenceDependencies,
    candidate_id: str,
    topic_id: str,
    channel: str,
    question_text: str = "",
    answer_text: str = "",
    source_event_id: str | None = None,
    status: str | None = None,
    evidence_reference: str | None = None,
) -> None:
    final_status = status or dependencies.intake_answer_status(topic_id, answer_text)
    question_id = f"{topic_id}:canonical"
    async with dependencies.session_local() as db:
        await db.execute(
            text(
                "\n            INSERT INTO candidate_intake_ledger\n             "
                "   (candidate_id, topic_id, question_id, channel, question_tex"
                "t, answer_text,\n                 status, source_event_id, evid"
                "ence_reference, asked_at, answered_at)\n            VALUES (:ci"
                "d, :topic, :qid, :channel, NULLIF(:question, ''), NULLIF(:answ"
                "er, ''),\n                    :status, :event_id, :evidence,\n  "
                "                  CASE WHEN :question <> '' THEN now() ELSE NU"
                "LL END,\n                    CASE WHEN :answer <> '' THEN now()"
                " ELSE NULL END)\n            ON CONFLICT (candidate_id, topic_i"
                "d) DO UPDATE SET\n                channel = EXCLUDED.channel,\n "
                "               question_text = COALESCE(EXCLUDED.question_text"
                ", candidate_intake_ledger.question_text),\n                answ"
                "er_text = COALESCE(EXCLUDED.answer_text, candidate_intake_ledg"
                "er.answer_text),\n                status = CASE\n               "
                "     WHEN candidate_intake_ledger.status = 'ANSWERED' AND EXCL"
                "UDED.status <> 'INVALIDATED'\n                    THEN 'ANSWERE"
                "D' ELSE EXCLUDED.status END,\n                source_event_id ="
                " COALESCE(EXCLUDED.source_event_id, candidate_intake_ledger.so"
                "urce_event_id),\n                evidence_reference = COALESCE("
                "EXCLUDED.evidence_reference, candidate_intake_ledger.evidence_"
                "reference),\n                asked_at = COALESCE(candidate_inta"
                "ke_ledger.asked_at, EXCLUDED.asked_at),\n                answer"
                "ed_at = COALESCE(EXCLUDED.answered_at, candidate_intake_ledger"
                ".answered_at),\n                updated_at = now()\n        "
            ),
            {
                "cid": candidate_id,
                "topic": topic_id,
                "qid": question_id,
                "channel": channel,
                "question": question_text,
                "answer": answer_text,
                "status": final_status,
                "event_id": source_event_id,
                "evidence": evidence_reference,
            },
        )
        await db.commit()

async def _sync_voice_ledger(
    dependencies: VoiceIntakePersistenceDependencies,
    candidate_id: str, resume: dict, source_event_id: str | None = None
) -> None:
    for turn in resume.get("completed_turns") or []:
        question, answer = (
            dependencies.clean_text(turn.get("question")),
            dependencies.clean_text(turn.get("answer")),
        )
        if question:
            await _upsert_intake_ledger(
                dependencies,
                candidate_id,
                dependencies.stable_topic(question),
                "vapi",
                question,
                answer,
                source_event_id,
                evidence_reference="raw_data.voice_intake.completed_turns",
            )
    pending = dependencies.clean_text(
        resume.get("current_question") or resume.get("next_question")
    )
    if pending:
        await _upsert_intake_ledger(
            dependencies,
            candidate_id,
            dependencies.stable_topic(pending),
            "vapi",
            pending,
            "",
            source_event_id,
            status="ASKED",
        )

async def _sync_profile_updates_to_ledger(
    dependencies: VoiceIntakePersistenceDependencies,
    candidate_id: str, updates: dict, channel: str = "chat"
) -> None:
    for field, value in (updates or {}).items():
        topic = _PROFILE_FIELD_TOPICS.get(field)
        if not topic or value in (None, "", []):
            continue
        answer = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
        await _upsert_intake_ledger(
            dependencies,
            candidate_id,
            topic,
            channel,
            answer_text=answer,
            status="ANSWERED",
            evidence_reference=f"candidate_profile.{field}",
        )

async def _load_intake_ledger(
    dependencies: VoiceIntakePersistenceDependencies, candidate_id: str
) -> list[dict]:
    async with dependencies.session_local() as db:
        result = await db.execute(
            text("""
            SELECT topic_id, question_id, channel, question_text, answer_text, status
            FROM candidate_intake_ledger WHERE candidate_id = :cid ORDER BY updated_at
        """),
            {"cid": candidate_id},
        )
        return [dict(row) for row in result.mappings().fetchall()]


_INTAKE_TOPIC_QUESTIONS = {
    "name": "What name would you like employers to use?",
    "location": "Where are you currently located?",
    "bio": "How would you briefly describe your professional background?",
    "current_role": "What's your current job title and industry?",
    "experience_years": "How many years of professional experience do you have?",
    "skills": "What are your strongest professional and technical skills?",
    "work_experience": (
        "Could you share your work experience, including your roles, "
        "companies, and dates?"
    ),
    "preferred_roles": "What kinds of roles are you looking for?",
    "preferred_locations": "Which locations would you prefer to work in?",
    "remote_preference": "Do you prefer remote, hybrid, on-site, or flexible work?",
    "preferred_industries": "Which industries would you like to work in?",
    "employment_types": (
        "Are you looking for full-time, part-time, contract, or freelance work?"
    ),
    "willing_to_relocate": "Are you open to relocating for the right opportunity?",
    "open_to_opportunities": "Are you currently open to new opportunities?",
    "notice_period": "What is your notice period or when could you start?",
    "expected_salary": "What salary range are you targeting?",
    "education": "What's your highest level of education?",
    "certifications": "Do you have any professional certifications?",
    "projects": "Tell me about a project that best demonstrates your skills.",
}

async def _persist_voice_intake_profile_state(
    dependencies: VoiceIntakePersistenceDependencies,
    candidate_id: str,
    candidate: dict,
    voice_data: dict,
    voice_intake_state: dict,
) -> dict:
    """
    Persist the canonical profile, preferences, and voice intake resume in sync.

    The same helper is used by both the final intake endpoint and the progress
    endpoint so repeated cumulative submissions remain idempotent.
    """
    # A missing profile payload means there is nothing canonical to merge yet.
    # Keep the save path valid and let the voice data establish the profile.
    candidate = candidate if isinstance(candidate, dict) else {}
    voice_data = dependencies.sanitize_profile_field_mapping(voice_data)
    merged = dependencies.merge_voice_into_profile(candidate, voice_data)

    update_params: dict[str, Any] = {"cid": candidate_id}
    set_clauses: list[str] = []

    for field, col in [
        ("summary", "summary"),
        ("current_role", '"current_role"'),
        ("current_company", "current_company"),
        ("location", "location"),
    ]:
        if merged.get(field) != candidate.get(field):
            set_clauses.append(f"{col} = :{field}")
            update_params[field] = merged[field]

    if merged.get("experience_years") != candidate.get("experience_years"):
        set_clauses.append("experience_years = :experience_years")
        update_params["experience_years"] = merged.get("experience_years")

    for field in ("skills", "work_experience", "education"):
        if merged.get(field) != candidate.get(field):
            set_clauses.append(f"{field} = CAST(:{field} AS json)")
            update_params[field] = json.dumps(merged.get(field) or [])

    candidate_opportunity_type = dependencies.normalize_candidate_opportunity_type(
        merged.get("opportunity_type")
    )
    existing_candidate_opportunity_type = dependencies.normalize_candidate_opportunity_type(
        candidate.get("opportunity_type")
    )
    if (
        candidate_opportunity_type
        and candidate_opportunity_type != existing_candidate_opportunity_type
    ):
        set_clauses.append("opportunity_type = :opportunity_type")
        update_params["opportunity_type"] = candidate_opportunity_type

    merged_raw = merged.get("raw_data") or {}
    # The canonical profile intentionally merges resume and intake fields.
    # Save the intake extraction separately so the onboarding recap can remain
    # grounded only in what the candidate said, including after a refresh.
    merged_raw["voice_intake_summary"] = {
        **(voice_data if isinstance(voice_data, dict) else {}),
        "voice_intake_resume": voice_intake_state,
    }
    merged_raw["voice_intake"] = voice_intake_state
    # Store only direct candidate assertions of having used a skill that was
    # already present before this intake.  The extracted voice skills and
    # project technology lists remain claimed/corroborated evidence.
    voice_usage_text = "\n".join(
        dependencies.normalize_profile_text(turn.get("answer"))
        for turn in (voice_intake_state.get("completed_turns") or [])
        if isinstance(turn, dict)
    )
    merged_raw = dependencies.append_demonstrated_skill_evidence(
        merged_raw, candidate.get("skills") or [], voice_usage_text, "eve_voice"
    )
    set_clauses.append("raw_data = CAST(:raw_data AS jsonb)")
    update_params["raw_data"] = json.dumps(merged_raw)

    set_clauses.append("updated_at = now()")
    set_clauses.append("updated_by_source = 'eve_voice'")
    async with dependencies.session_local() as db:
        await db.execute(
            text(f"UPDATE candidates SET {', '.join(set_clauses)} WHERE id = :cid"),
            update_params,
        )
        await db.commit()

    cache = dependencies.profile_cache()
    if cache is not None:
        await cache.invalidate(candidate_id)

    await dependencies.upsert_candidate_preferences(candidate_id, voice_data)
    return merged


@dataclass(frozen=True)
class VoiceIntakePersistenceService:
    """State-free persistence facade bound to one immutable dependency set."""

    dependencies: VoiceIntakePersistenceDependencies

    async def save_voice_intake_resume(self, candidate_id: str, resume: dict) -> None:
        await _save_voice_intake_resume(self.dependencies, candidate_id, resume)

    async def upsert_intake_ledger(
        self,
        candidate_id: str,
        topic_id: str,
        channel: str,
        question_text: str = "",
        answer_text: str = "",
        source_event_id: str | None = None,
        status: str | None = None,
        evidence_reference: str | None = None,
    ) -> None:
        await _upsert_intake_ledger(
            self.dependencies,
            candidate_id,
            topic_id,
            channel,
            question_text,
            answer_text,
            source_event_id,
            status,
            evidence_reference,
        )

    async def sync_voice_ledger(
        self, candidate_id: str, resume: dict, source_event_id: str | None = None
    ) -> None:
        await _sync_voice_ledger(
            self.dependencies, candidate_id, resume, source_event_id
        )

    async def sync_profile_updates_to_ledger(
        self, candidate_id: str, updates: dict, channel: str = "chat"
    ) -> None:
        await _sync_profile_updates_to_ledger(
            self.dependencies, candidate_id, updates, channel
        )

    async def load_intake_ledger(self, candidate_id: str) -> list[dict]:
        return await _load_intake_ledger(self.dependencies, candidate_id)

    async def persist_profile_state(
        self,
        candidate_id: str,
        candidate: dict,
        voice_data: dict,
        voice_intake_state: dict,
    ) -> dict:
        return await _persist_voice_intake_profile_state(
            self.dependencies, candidate_id, candidate, voice_data, voice_intake_state
        )
