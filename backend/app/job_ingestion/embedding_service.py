"""Canonical job representation and embedding generation."""

import html
import logging
import os
import re
import threading
from typing import Any

from dotenv import load_dotenv
from skill_normalization import canonical_skill_key

load_dotenv()
_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
_logger = logging.getLogger(__name__)
_model = None
_model_lock = threading.Lock()
_TEXT_LIMIT = 12000


def _get_model():
    """Load the embedding model once, only when model-backed work is needed."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _logger.info("Loading embedding model: %s", _MODEL_NAME)
                _model = _load_model()
    return _model


def _load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = "\n".join(item for item in value if isinstance(item, str))
    if not isinstance(value, str):
        return ""
    value = html.unescape(value)
    value = re.sub(r"<\s*(?:br|p|div|li|h[1-6])\b[^>]*>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"^\s*[\u2022*]\s*|^\s*[-+]\s+", "", value, flags=re.M)
    lines, seen = [], set()
    for line in re.split(r"\n+", value):
        line = re.sub(r"\s+", " ", line).strip()
        key = line.casefold()
        if line and key not in seen:
            seen.add(key)
            lines.append(line)
    return " ".join(lines)


def _values(value: Any) -> list[str]:
    raw = re.split(r"[,;|\n]", value) if isinstance(value, str) else value
    if not isinstance(raw, (list, tuple, set)):
        return []
    result, seen = [], set()
    for item in raw:
        value_text = _text(item)
        if not value_text or len(value_text) > 120:
            continue
        key = canonical_skill_key(value_text) or value_text.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value_text)
    return result


def _structured(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("structured_data")
    if not isinstance(value, dict):
        return {}
    # Provider payloads are retained by the normalizer for legacy rows. They
    # are a fallback only; top-level normalized fields always win.
    payload = value.get("raw_source", {}).get("payload")
    return {**(payload if isinstance(payload, dict) else {}), **value}


def _first(job: dict[str, Any], *keys: str) -> str:
    nested = _structured(job)
    for key in keys:
        value = _text(job.get(key)) or _text(nested.get(key))
        if value:
            return value
    return ""


def _field_values(job: dict[str, Any], *keys: str) -> list[str]:
    nested = _structured(job)
    result, seen = [], set()
    for key in keys:
        for value in [*_values(job.get(key)), *_values(nested.get(key))]:
            canonical = canonical_skill_key(value) or value.casefold()
            if canonical not in seen:
                seen.add(canonical)
                result.append(value)
    return result


def build_canonical_job_representation(job: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic normalized job fields without invoking the model."""
    if not isinstance(job, dict):
        return {}
    required = _field_values(job, "skills_required", "required_skills")
    required_keys = {canonical_skill_key(value) for value in required}
    preferred = [
        value
        for value in _field_values(
            job, "skills_preferred", "preferred_skills", "nice_to_have"
        )
        if canonical_skill_key(value) not in required_keys
    ]
    description = _first(job, "description")
    for section in (_first(job, "responsibilities"), _first(job, "requirements")):
        if section and section.casefold() != description.casefold():
            description = re.sub(
                re.escape(section), "", description, flags=re.I
            ).strip()
    return {
        "title": _first(job, "title", "job_title"),
        "company": _first(job, "company_name", "company", "organization"),
        "department": _first(job, "department"),
        "industry": _first(job, "industry"),
        "summary": _first(job, "summary", "job_summary"),
        "responsibilities": _first(job, "responsibilities"),
        "requirements": _first(job, "requirements"),
        "required_skills": required,
        "preferred_skills": preferred,
        "technologies": _field_values(job, "technologies", "technology_slugs", "tools"),
        "experience": _first(
            job, "experience_required", "experience_level", "seniority"
        ),
        "seniority": _first(job, "seniority", "experience_level"),
        "employment_type": _first(job, "employment_type"),
        "work_arrangement": _first(
            job, "remote_policy", "work_arrangement", "workplace_type"
        ),
        "location": _first(job, "location", "city"),
        "salary": _first(job, "salary_range", "salary"),
        # Normalized responsibilities/requirements are more useful than
        # repeating the full description. Legacy rows still retain it.
        "description": description,
    }


def _section_lines(rep: dict[str, Any]) -> list[str]:
    lines = []
    for key, label in (
        ("title", "Title"),
        ("company", "Company"),
        ("department", "Department"),
        ("industry", "Industry"),
        ("summary", "Summary"),
        ("required_skills", "Required Skills"),
        ("responsibilities", "Responsibilities"),
        ("requirements", "Requirements"),
        ("preferred_skills", "Preferred Skills"),
        ("technologies", "Technologies"),
        ("experience", "Experience"),
        ("seniority", "Seniority"),
        ("employment_type", "Employment Type"),
        ("work_arrangement", "Work Arrangement"),
        ("location", "Location"),
        ("salary", "Salary"),
        ("description", "Job Description"),
    ):
        value = rep.get(key)
        if value:
            value = ", ".join(value) if isinstance(value, list) else value
            lines.append(f"{label}: {value}")
    return lines


def build_canonical_job_text(job: dict[str, Any]) -> str:
    """Build the complete canonical text, bounded to 12,000 characters."""
    text = "\n".join(_section_lines(build_canonical_job_representation(job)))
    if len(text) <= _TEXT_LIMIT:
        return text
    marker = " [truncated]"
    return text[: _TEXT_LIMIT - len(marker)].rstrip() + marker


def _token_count(text: str) -> int:
    model = _get_model()
    return len(model.tokenizer.encode(text, add_special_tokens=True, truncation=False))


def build_job_text(job: dict[str, Any]) -> str:
    """Build deterministic model input within the model's real token budget."""
    rep = build_canonical_job_representation(job)
    model = _get_model()
    budget = min(getattr(model, "max_seq_length", 256), 256)
    result: list[str] = []
    for line in _section_lines(rep):
        candidate = "\n".join([*result, line])
        if _token_count(candidate) <= budget:
            result.append(line)
            continue
        # Keep the section label and as much content as the remaining token
        # budget allows. Required skills occur before prose by design.
        prefix = line.split(":", 1)[0] + ":"
        words = line[len(prefix) :].strip().split()
        kept = prefix
        for word in words:
            trial = "\n".join([*result, kept + (" " if kept != prefix else " ") + word])
            if _token_count(trial) > budget:
                break
            kept = kept + (" " if kept != prefix else " ") + word
        if kept != prefix:
            result.append(kept)
        break
    text = "\n".join(result)
    if len(text) > _TEXT_LIMIT:
        marker = " [truncated]"
        text = text[: _TEXT_LIMIT - len(marker)].rstrip() + marker
    return text


def generate_job_embedding(job: dict[str, Any]) -> list[float]:
    model = _get_model()
    embedding = model.encode(build_job_text(job), normalize_embeddings=True)
    return embedding.tolist()
