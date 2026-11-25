"""Deterministic normalization for candidate profile values at data boundaries."""

from __future__ import annotations

import math
import re
from typing import Any

from location_matching import country_code, normalize_location_value
from skill_normalization import merge_skills

_PLACEHOLDER = re.compile(
    r"^(?:n/?a|none|null|unknown|not\s+(?:provided|mentioned|specified)|"
    r"my\s+resume|resume|cv|not\s+available|prefer\s+not\s+to\s+say)$",
    re.IGNORECASE,
)
_TIMELINE = re.compile(
    r"^(?:\d{4}|[A-Za-z]{3,9}\s+\d{4})\s*(?:-|–|—|to|through|until)\s*"
    r"(?:\d{4}|[A-Za-z]{3,9}\s+\d{4}|present|current|now)$",
    re.IGNORECASE,
)
ROLE_WORDS = {
    "architect",
    "analyst",
    "associate",
    "consultant",
    "coordinator",
    "designer",
    "developer",
    "director",
    "engineer",
    "executive",
    "founder",
    "head",
    "intern",
    "lead",
    "manager",
    "officer",
    "partner",
    "president",
    "producer",
    "professor",
    "researcher",
    "scientist",
    "specialist",
    "supervisor",
    "technician",
    "tester",
    "qa",
    "recruiter",
    "administrator",
    "writer",
    "accountant",
    "attorney",
}
TECHNOLOGY_ONLY = {
    "postgres",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "sqlite",
    "oracle",
    "sql",
    "python",
    "java",
    "javascript",
    "typescript",
    "fastapi",
    "django",
    "flask",
    "react",
    "angular",
    "node",
    "node.js",
    "spring",
    "spring boot",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "git",
    "linux",
    "html",
    "css",
}
_ROLE_ABBREVIATIONS = {"CEO", "CTO", "CFO", "VP", "SRE"}
_AMBIGUOUS_ROLE_ABBREVIATIONS = {"PM", "GM"}
_AMBIGUOUS_ROLE_CONTEXT = {
    "engineering",
    "general",
    "operations",
    "people",
    "product",
    "program",
    "project",
}
_ROLE_SENTENCE_MARKERS = {
    "based",
    "experience",
    "looking",
    "responsible",
    "seeking",
    "worked",
}
_LOCATION_ABBREVIATIONS = {"dc", "nyc", "sf"}
_EMPLOYER_MARKERS = re.compile(
    r"\b(?:corporation|company|co\.?|inc\.?|llc|ltd\.?|limited|solutions|"
    r"technologies|technology|systems|pvt\.?\s*ltd\.?)\b",
    re.IGNORECASE,
)


def clean_text(value: Any) -> str:
    """Clean text values for candidate profile fields.

    Extra whitespace is removed and values are converted to strings.
    """
    return re.sub(r"\s+", " ", str(value) if value is not None else "").strip()


def valid_role(value: Any) -> bool:
    value = clean_text(value)
    if not value or _PLACEHOLDER.fullmatch(value) or _TIMELINE.fullmatch(value):
        return False
    lowered = value.casefold()
    if lowered in TECHNOLOGY_ONLY:
        return False
    words = re.findall(r"[a-z]+", lowered)
    if not words or len(words) > 8 or _ROLE_SENTENCE_MARKERS.intersection(words):
        return False
    if value.upper() in _ROLE_ABBREVIATIONS:
        return True
    if any(word.upper() in _AMBIGUOUS_ROLE_ABBREVIATIONS for word in words):
        return bool(_AMBIGUOUS_ROLE_CONTEXT.intersection(words))
    return bool(set(words) & ROLE_WORDS)


def valid_location(value: Any) -> bool:
    value = clean_text(value)
    if not value or _PLACEHOLDER.fullmatch(value) or _TIMELINE.fullmatch(value):
        return False
    if value.casefold() in TECHNOLOGY_ONLY:
        return False
    if value.casefold() in _LOCATION_ABBREVIATIONS:
        return True
    abbreviated_periods = re.findall(r"\b([A-Za-z]+)\.", value)
    if (
        re.search(r"[!?;:]", value)
        or any(len(part) > 3 for part in abbreviated_periods)
        or len(value.split()) > 6
        or _EMPLOYER_MARKERS.search(value)
        or re.search(
            r"\b(?:worked|located|based|living|remote|from|experience)\b",
            value,
            re.I,
        )
    ):
        return False
    if len(value) > 160 or re.search(
        r"\b(?:skills?|experience|resume|cv|technolog(?:y|ies))\b", value, re.I
    ):
        return False
    return bool(re.search(r"[A-Za-z]", value))


def normalize_profile_fields(values: Any) -> dict[str, Any]:
    """Return safe candidate fields without inventing missing information."""
    if not isinstance(values, dict):
        return {}
    result = dict(values)
    role = clean_text(result.get("current_role"))
    headline = clean_text(result.get("headline"))
    if role and valid_role(role):
        result["current_role"] = role
    elif headline and valid_role(headline):
        result["current_role"] = headline
    else:
        result.pop("current_role", None)
    if "headline" in result and not valid_role(headline):
        result.pop("headline", None)
    location = clean_text(result.get("location"))
    if location and valid_location(location):
        result["location"] = location
    else:
        result.pop("location", None)
    for field in ("city", "state"):
        value = clean_text(result.get(field))
        if value and valid_location(value):
            result[field] = normalize_location_value(value)
        else:
            result.pop(field, None)
    for field in ("country", "country_code"):
        value = clean_text(result.get(field))
        if value and valid_location(value):
            if field == "country":
                result[field] = value
            else:
                result[field] = country_code(value) or normalize_location_value(value)
        else:
            result.pop(field, None)
    if "skills" in result:
        result["skills"] = merge_skills(result.get("skills"), canonical_display=True)
    if "experience_years" in result:
        years = normalize_experience_value(result["experience_years"])
        if years is None:
            result.pop("experience_years", None)
        else:
            result["experience_years"] = years
    return result


def normalize_experience_value(value: Any) -> float | None:
    """Convert supported experience formats to floats and reject invalid values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        years = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(years) or years < 0 or years > 80:
        return None
    return years


def normalize_candidate_for_matching(candidate: dict[str, Any]) -> dict[str, Any]:
    """Copy a candidate with invalid matching inputs removed."""
    return normalize_profile_fields(candidate)
