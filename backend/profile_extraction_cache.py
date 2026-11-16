"""Fail-open Redis cache for validated profile extraction results."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from pydantic import ValidationError

from profile_extraction_schemas import (
    CertificationExtraction,
    EducationExtraction,
    ExtractionResult,
    SkillsExtraction,
    TargetField,
)

logger = logging.getLogger(__name__)

CACHE_NAMESPACE = "eve:profile-extraction:v1"
SCHEMA_VERSION = "profile-extraction-v1"
PROMPT_VERSION = "profile-extraction-prompt-v1"
DEFAULT_TTL_SECONDS = 600


def _configured_ttl_seconds() -> int:
    """Return a safe Redis TTL even when runtime configuration is malformed."""
    raw_value = os.getenv("PROFILE_EXTRACTION_CACHE_TTL_SECONDS")
    if raw_value is None:
        return DEFAULT_TTL_SECONDS
    try:
        ttl_seconds = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "[profile-extraction-cache] invalid TTL configuration; using default"
        )
        return DEFAULT_TTL_SECONDS
    if ttl_seconds <= 0:
        logger.warning(
            "[profile-extraction-cache] non-positive TTL configuration; using default"
        )
        return DEFAULT_TTL_SECONDS
    return ttl_seconds


_CACHE_TTL_SECONDS = _configured_ttl_seconds()


def _canonical_context(
    relevant_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return deterministic JSON-compatible context, or None to bypass caching."""
    if not isinstance(relevant_context, dict):
        return {}

    def canonical(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, dict):
            return {str(key): canonical(value[key]) for key in sorted(value, key=str)}
        raise TypeError(f"unsupported cache context value: {type(value).__name__}")

    try:
        normalized = canonical(relevant_context)
        json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return normalized
    except (TypeError, ValueError, OverflowError):
        return None


def cache_key(
    *,
    target_field: TargetField,
    candidate_response: str,
    suggested_question: str | None,
    relevant_context: dict[str, Any] | None,
    model: str,
    schema_version: str = SCHEMA_VERSION,
    prompt_version: str = PROMPT_VERSION,
) -> str | None:
    """Build a stable key without including candidate/session identifiers."""
    payload = {
        "target_field": target_field,
        "candidate_response": candidate_response,
        "suggested_question": suggested_question,
        "relevant_context": _canonical_context(relevant_context),
        "schema_version": schema_version,
        "prompt_version": prompt_version,
        "model": model,
    }
    if payload["relevant_context"] is None:
        return None
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{CACHE_NAMESPACE}:{digest}"


async def _client():
    """Reuse the existing profile-cache Redis connection and fail open."""
    try:
        import profile_cache

        return await profile_cache._client()
    except Exception:
        logger.warning(
            "[profile-extraction-cache] Redis client unavailable", exc_info=True
        )
        return None


async def get(key: str) -> ExtractionResult | None:
    """Return a validated cached result, treating all cache errors as misses."""
    try:
        client = await _client()
        if not client:
            logger.debug("[profile-extraction-cache] miss reason=redis_unavailable")
            return None
        raw = await client.get(key)
        if not raw:
            logger.debug("[profile-extraction-cache] miss")
            return None
        result = ExtractionResult.model_validate(json.loads(raw))
        if result.status != "ok":
            logger.info("[profile-extraction-cache] discarded_non_ok_result")
            return None
        _validate_field_specific_result(result)
        logger.info("[profile-extraction-cache] hit")
        return result
    except (json.JSONDecodeError, TypeError, ValidationError):
        logger.warning("[profile-extraction-cache] invalid_cached_payload")
        return None
    except Exception:
        logger.warning("[profile-extraction-cache] read_failed", exc_info=True)
        return None


def _validate_field_specific_result(result: ExtractionResult) -> None:
    """Validate cached values with the same field schema used on Groq output."""
    validators = {
        "certifications": CertificationExtraction,
        "skills": SkillsExtraction,
        "education": EducationExtraction,
    }
    field = result.target_field
    payload = {"status": "ok", "confidence": result.confidence}
    if field == "certifications":
        payload["certification"] = result.value
    elif field == "skills":
        payload["skills"] = result.value
    else:
        payload["education"] = result.value
    validators[field].model_validate(payload)


async def set(key: str, result: ExtractionResult) -> None:
    """Cache only valid successful extraction results."""
    if result.status != "ok":
        return
    try:
        client = await _client()
        if client:
            await client.set(
                key,
                result.model_dump_json(),
                ex=_CACHE_TTL_SECONDS,
            )
            logger.info("[profile-extraction-cache] stored ttl=%ss", _CACHE_TTL_SECONDS)
    except Exception:
        logger.warning("[profile-extraction-cache] write_failed", exc_info=True)
