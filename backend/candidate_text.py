import json
import re
from typing import Any

_CERTIFICATION_BOILERPLATE_WORDS = {
    "cert",
    "certificate",
    "certificates",
    "certification",
    "certifications",
    "certified",
    "course",
    "courses",
    "credential",
    "credentials",
    "training",
}
_CERTIFICATION_MARKER_RE = re.compile(
    r"\b(?:cert|certificate|certificates|certification|certifications|certified|course|courses|credential|credentials|training|license|licence)\b",
    re.I,
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _format_experience_duration(value: Any) -> str:
    try:
        years = float(value)
    except (TypeError, ValueError):
        return ""
    if years < 0:
        return ""
    total_months = round(years * 12)
    whole_years, months = divmod(total_months, 12)
    parts = []
    if whole_years:
        parts.append(f"{whole_years} year{'s' if whole_years != 1 else ''}")
    if months:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    return " ".join(parts) or "0 months"


def _normalize_key(value: Any) -> str:
    return _normalize_text(value).lower()


def _relaxed_cert_key(value: Any) -> str:
    stripped = re.sub(r"[^\w\s]+", " ", _normalize_key(value))
    tokens = [
        token
        for token in stripped.split()
        if token not in _CERTIFICATION_BOILERPLATE_WORDS
    ]
    return " ".join(tokens) or stripped


def _normalize_certifications(certifications: Any) -> list[str]:
    if not isinstance(certifications, list):
        return []
    normalized: list[str] = []
    seen_strict: set[str] = set()
    seen_relaxed: set[str] = set()
    for cert in certifications:
        cleaned = _normalize_text(cert)
        if not cleaned:
            continue
        strict_key = _normalize_key(cleaned)
        relaxed_key = _relaxed_cert_key(cleaned)
        if strict_key in seen_strict or relaxed_key in seen_relaxed:
            continue
        seen_strict.add(strict_key)
        seen_relaxed.add(relaxed_key)
        normalized.append(cleaned)
    return normalized


def _normalize_skills(skills: Any, certifications: Any = None) -> list[str]:
    if not isinstance(skills, list):
        return []

    normalized_certs = _normalize_certifications(certifications or [])
    normalized: list[str] = []
    seen: set[str] = set()

    for skill in skills:
        cleaned = _normalize_text(
            skill.get("name") if isinstance(skill, dict) else skill
        )
        if not cleaned:
            continue
        key = _normalize_key(cleaned)
        if key in seen:
            continue
        if any(_normalize_key(cert) == key for cert in normalized_certs) and (
            _CERTIFICATION_MARKER_RE.search(cleaned)
            or any(_CERTIFICATION_MARKER_RE.search(cert) for cert in normalized_certs)
        ):
            continue
        seen.add(key)
        normalized.append(cleaned)

    return normalized


def _parse_json_value(value: Any, fallback: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return fallback


def build_candidate_text(candidate: dict[str, Any]) -> str:
    """
    Build structured semantic text from a candidate DB row for embedding.
    Mirrors Adam's candidate text format for compatibility with the shared vector space.
    """
    from candidate_profile_normalization import normalize_candidate_for_matching

    candidate = normalize_candidate_for_matching(candidate)
    raw_data = _parse_json_value(candidate.get("raw_data") or {}, {})

    parts = []

    # Put explicit career intent first so semantic retrieval follows the
    # candidate's desired direction, while their current role remains context.
    try:
        from profile_strength_service import get_canonical_preferences

        preferred_roles = (
            get_canonical_preferences(candidate, candidate.get("_prefs_row")).get(
                "preferred_roles"
            )
            or []
        )
    except Exception:
        preferred_roles = (
            raw_data.get("preferred_roles") or raw_data.get("target_roles") or []
        )
    if isinstance(preferred_roles, str):
        preferred_roles = [preferred_roles]
    if preferred_roles:
        parts.append(
            "Target / Preferred Roles:\n"
            f"{', '.join(str(role) for role in preferred_roles if role)}"
        )

    current_role = (candidate.get("current_role") or "").strip()
    if current_role:
        parts.append(f"Candidate Role / Experience:\n{current_role}")

    current_company = (candidate.get("current_company") or "").strip()
    if current_company:
        parts.append(f"Current Company:\n{current_company}")

    location = (candidate.get("location") or "").strip()
    if location:
        parts.append(f"Location:\n{location}")

    exp_years = candidate.get("experience_years") or candidate.get(
        "total_experience_years"
    )
    if exp_years is not None:
        duration = _format_experience_duration(exp_years)
        if duration:
            parts.append(f"Experience:\n{duration} of professional experience")

    skills_raw = candidate.get("skills") or []
    skills_raw = _parse_json_value(skills_raw, [])
    certifications = _normalize_certifications(raw_data.get("certifications") or [])
    skill_names = _normalize_skills(skills_raw, certifications=certifications)
    if skill_names:
        parts.append(f"Skills:\n{', '.join(skill_names)}")

    work_exp = candidate.get("work_experience") or []
    work_exp = _parse_json_value(work_exp, [])
    if isinstance(work_exp, list) and work_exp:
        lines = []
        for w in work_exp:
            if not isinstance(w, dict):
                continue
            title = w.get("title", "")
            company = w.get("company", "")
            desc = w.get("description", "")
            entry = " at ".join(filter(None, [title, company]))
            if desc:
                entry = f"{entry}: {desc}"
            if entry.strip():
                lines.append(entry.strip())
        if lines:
            parts.append("Work Experience:\n" + "\n".join(lines))

    education = candidate.get("education") or []
    education = _parse_json_value(education, [])
    if education:
        lines = []
        for e in education:
            degree = e.get("degree", "")
            institution = e.get("institution", "")
            entry = " at ".join(filter(None, [degree, institution]))
            if entry.strip():
                lines.append(entry.strip())
        if lines:
            parts.append("Education:\n" + "\n".join(lines))

    if certifications:
        parts.append(f"Certifications:\n{', '.join(certifications)}")

    summary = (candidate.get("summary") or "").strip()
    if summary:
        parts.append(f"Summary:\n{summary}")

    additional = (raw_data.get("additional_information") or "").strip()
    if additional:
        parts.append(f"Additional Information:\n{additional}")

    # Use parsed_resume_text if available, else resume_text
    resume_text = (
        candidate.get("parsed_resume_text") or candidate.get("resume_text") or ""
    ).strip()
    if resume_text:
        parts.append(f"Resume:\n{resume_text[:2000]}")

    return "\n\n".join(parts)
