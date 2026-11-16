"""Standalone field-specific profile extraction through the existing Groq client."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from pydantic import ValidationError

import profile_extraction_cache
from profile_extraction_schemas import (
    CertificationExtraction,
    EducationExtraction,
    ExtractionResult,
    SkillsExtraction,
    TargetField,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "profile-extraction-v1"


def _client_and_model(client: Any = None, model: str | None = None) -> tuple[Any, str]:
    """Use the application's shared client lazily, avoiding a second LLM client."""
    if client is not None and model:
        return client, model
    if client is not None:
        raise ValueError("model is required when client is supplied")

    # Importing lazily keeps this module independently testable and avoids
    # importing server.py during schema/unit-test collection.
    from server import GROQ_MODEL, openai_client

    return openai_client, model or GROQ_MODEL


def _instructions(target_field: TargetField) -> str:
    common = (
        "Return ONLY valid JSON. Use only facts explicitly present in the candidate "
        "response. Do not infer, embellish, or invent missing values. Remove polite "
        "and conversational wording while preserving meaningful names and details."
    )
    if target_field == "certifications":
        return f"""{common}
Target field: certifications.
Extract the credential/course/certification name only. Keep an issuer separately
when explicitly stated. Distinguish a course from a formal certification: do not
call something a formal certification unless the candidate says certified,
certification, credential, license, or equivalent. Return this exact shape:
{{"status":"ok|empty|ambiguous","certification":{{"name":"...","issuer":null,"credential_type":"course|formal_certification|credential|license|unknown"}},"confidence":0.0,"ambiguity_reason":null}}
Use certification=null for empty or ambiguous results."""
    if target_field == "skills":
        return f"""{common}
Target field: skills.
Return only explicitly named skills, technologies, tools, or methods. Do not turn
job titles, companies, courses, or vague claims into skills. Return this exact shape:
{{"status":"ok|empty|ambiguous","skills":["..."],"confidence":0.0,"ambiguity_reason":null}}"""
    return f"""{common}
Target field: education.
Extract explicitly stated degrees or education records and their institutions.
Do not classify a certification or course as a degree unless the response supports
that. Do not infer dates, specialization, or completion status. Return this exact shape:
{{"status":"ok|empty|ambiguous","education":[{{"degree":"...","specialization":null,"institution":"...","graduation_year":null,"status":"completed|pursuing|discontinued|unknown","start_date":"","end_date":""}}],"confidence":0.0,"ambiguity_reason":null}}"""


def _user_payload(
    target_field: TargetField,
    candidate_response: str,
    suggested_question: str | None,
    relevant_context: dict[str, Any] | None,
) -> str:
    return json.dumps(
        {
            "target_field": target_field,
            "suggested_question": suggested_question,
            "candidate_response": candidate_response,
            "relevant_context": relevant_context or {},
        },
        ensure_ascii=False,
    )


def _invalid_result(target_field: TargetField) -> ExtractionResult:
    return ExtractionResult(
        status="invalid",
        target_field=target_field,
        value=None,
        normalized=False,
        confidence=0.0,
    )


def _result_from_payload(
    target_field: TargetField, payload: dict[str, Any]
) -> ExtractionResult:
    schema = {
        "certifications": CertificationExtraction,
        "skills": SkillsExtraction,
        "education": EducationExtraction,
    }[target_field]
    parsed = schema.model_validate(payload)
    if parsed.status == "ok":
        if target_field == "certifications" and parsed.certification is not None:
            value = parsed.certification.model_dump()
        elif target_field == "skills" and parsed.skills:
            value = parsed.skills
        elif target_field == "education" and parsed.education:
            value = [item.model_dump() for item in parsed.education]
        else:
            return _invalid_result(target_field)
    else:
        value = None
    return ExtractionResult(
        status=parsed.status,
        target_field=target_field,
        value=value,
        normalized=parsed.status == "ok",
        confidence=parsed.confidence,
        ambiguity_reason=parsed.ambiguity_reason,
    )


async def extract_profile_field(
    *,
    target_field: TargetField,
    candidate_response: str,
    suggested_question: str | None = None,
    relevant_context: dict[str, Any] | None = None,
    client: Any = None,
    model: str | None = None,
    candidate_id: str | None = None,
    session_id: str | None = None,
) -> ExtractionResult:
    """Extract one profile field using the application's shared Groq client."""
    if not isinstance(candidate_response, str) or not candidate_response.strip():
        return ExtractionResult(status="empty", target_field=target_field, value=None)

    selected_model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    key = profile_extraction_cache.cache_key(
        target_field=target_field,
        candidate_response=candidate_response.strip(),
        suggested_question=suggested_question,
        relevant_context=relevant_context,
        model=selected_model,
    )
    cached = await profile_extraction_cache.get(key) if key else None
    if cached is not None:
        logger.info(
            "[profile-extraction] cache_hit field=%s",
            target_field,
        )
        return cached

    started = time.monotonic()
    logger.info("[profile-extraction] cache_miss field=%s", target_field)
    try:
        shared_client, selected_model = _client_and_model(client, selected_model)
        response = await shared_client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": _instructions(target_field)},
                {
                    "role": "user",
                    "content": _user_payload(
                        target_field,
                        candidate_response.strip(),
                        suggested_question,
                        relevant_context,
                    ),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
            _telemetry={
                "workflow": "profile_field_extraction",
                "function_name": "extract_profile_field",
                "candidate_id": candidate_id,
                "endpoint": "/api/chat",
                "session_id": session_id,
                "schema_version": _SCHEMA_VERSION,
            },
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return _invalid_result(target_field)
        result = _result_from_payload(target_field, payload)
        if result.status == "ok" and key:
            await profile_extraction_cache.set(key, result)
        logger.info(
            "[profile-extraction] completed field=%s status=%s latency_ms=%.1f",
            target_field,
            result.status,
            (time.monotonic() - started) * 1000,
        )
        return result
    except (
        json.JSONDecodeError,
        IndexError,
        AttributeError,
        TypeError,
        ValidationError,
    ):
        logger.warning(
            "[profile-extraction] invalid response for field=%s", target_field
        )
        return _invalid_result(target_field)
    except Exception:
        logger.exception(
            "[profile-extraction] provider failure for field=%s", target_field
        )
        return ExtractionResult(status="invalid", target_field=target_field, value=None)
