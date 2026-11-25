import html
import json
import logging
import re
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from typing import Any

from app.job_ingestion.embedding_service import build_job_text
from app.job_ingestion.lifecycle import candidate_visible_where
from candidate_profile_normalization import (
    normalize_candidate_for_matching,
    normalize_experience_value,
)
from candidate_text import build_candidate_text
from embedding_service import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_VERSION,
    generate_embedding,
)
from location_matching import (
    country_eligible,
    job_location,
    location_preference_matches,
    location_preference_score,
)
from qdrant_service import merge_search_results, search_job_chunks
from recommendation_refresh_lease import (
    acquire_refresh_lease,
    release_refresh_lease,
    renew_refresh_lease,
)
from role_family_matching import classify_role_relationship
from skill_normalization import (
    canonical_skill,
    canonical_skill_key,
    skill_values,
    skill_variants,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)

# Backward-compatible test seam retained for callers that patch the legacy name.
timezone = UTC

QDRANT_TOP_K = 150
MAX_RECOMMENDATIONS = 50
REFRESH_LEASE_SECONDS = 120
_refresh_lease_context: ContextVar[tuple[Any, str, str, int] | None] = ContextVar(
    "refresh_lease_context", default=None
)


def _lease_safe_refresh(func):
    @wraps(func)
    async def wrapped(candidate_id, candidate, SessionLocal):
        try:
            return await func(candidate_id, candidate, SessionLocal)
        finally:
            state = _refresh_lease_context.get()
            if state is not None and state[1] == candidate_id:
                _refresh_lease_context.set(None)
                await _release_refresh_lease_safely(
                    state[0], state[1], state[2], state[3]
                )

    return wrapped


def _healthy_empty_retrieval(*results: Any) -> bool:
    """A complete empty search has diagnostics and zero raw/invalid hits."""
    if not results:
        return False
    for result in results:
        diagnostics = getattr(result, "diagnostics", None)
        if diagnostics is None:
            return False
        if any(
            getattr(diagnostics, field, None) != 0
            for field in (
                "raw_hits",
                "valid_job_ids",
                "invalid_job_ids",
                "unique_job_ids",
                "duplicate_points_removed",
            )
        ):
            return False
    return True


# Candidate-first ranking weights (must sum to 1.0).  Qdrant is deliberately
# one signal here, rather than the decision-maker.
W_TARGET_ROLE = 0.24
W_SKILLS = 0.20
W_EXPERIENCE_FIT = 0.22
W_RELEVANT_HISTORY = 0.12
W_PREFERENCES = 0.12
W_SEMANTIC = 0.10

_ROLE_FAMILY_RANK_PRIORITY = {
    "exact": 3,
    "equivalent": 3,
    "related": 2,
    "transition": 1,
}


def _embedding_metadata_compatible(metadata: dict[str, Any] | None) -> bool:
    """Allow semantic reuse only for explicitly compatible job vectors."""
    return bool(
        isinstance(metadata, dict)
        and metadata.get("embedding_model") == EMBEDDING_MODEL_NAME
        and metadata.get("embedding_version") == EMBEDDING_VERSION
    )


def _retrieval_metadata(results: Any) -> dict[str, dict[str, Any]]:
    metadata = getattr(results, "metadata_by_job_id", None)
    return metadata if isinstance(metadata, dict) else {}


def _retrieval_score_is_compatible(
    results: Any, job_id: str, *, allow_legacy_adapter: bool = False
) -> bool:
    """Keep legacy list-based test/adaptor seams lexical-only compatible.

    Production SearchResults always carries metadata; a plain list is an
    older in-process adapter that cannot make a semantic-version claim.
    """
    metadata = getattr(results, "metadata_by_job_id", None)
    if metadata is None:
        return allow_legacy_adapter
    return _embedding_metadata_compatible(metadata.get(job_id))


def voice_intake_completed(candidate: dict[str, Any]) -> bool:
    """Return the persisted Voice Intake completion state for a candidate.

    ``raw_data.voice_intake`` is the canonical profile snapshot written by the
    existing intake flow.  Preferences alone are intentionally insufficient:
    they can be populated by a resume, chat, or an in-progress intake and must
    not start personalized recommendations.
    """
    raw_data = candidate.get("raw_data") or {}
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except Exception:
            return False
    if not isinstance(raw_data, dict):
        return False
    intake = raw_data.get("voice_intake") or {}
    if isinstance(intake, str):
        try:
            intake = json.loads(intake)
        except Exception:
            return False
    return (
        isinstance(intake, dict)
        and str(intake.get("status") or "").lower() == "completed"
    )


def candidate_ready_for_matching(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Use persisted profile signal, rather than resumable intake status."""
    candidate_text = build_candidate_text(candidate)
    if not candidate_text.strip():
        return False, "empty_candidate_profile_text"
    raw = candidate.get("raw_data") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    intake = raw.get("voice_intake") if isinstance(raw, dict) else {}
    if (not isinstance(raw, dict) or not raw) and not candidate.get("current_role"):
        return False, "voice_intake_state_missing"
    status = intake.get("status") if isinstance(intake, dict) else "profile_data"
    status = status or "profile_data"
    return True, f"profile_text_available_voice_intake_{status or 'unknown'}"


# Evidence level weights for skill scoring (Phase 7/8)
_EVIDENCE_WEIGHT = {
    0: 0.3,  # Unknown
    1: 0.6,  # Claimed
    2: 0.8,  # Corroborated
    3: 1.0,  # Demonstrated
    4: 1.0,  # Verified
}


def _get_candidate_intelligence(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """
    Compute candidate intelligence from profile_strength_service.
    Cached in candidate dict under '_intelligence' to avoid recomputation.
    """
    if "_intelligence" in candidate:
        return candidate["_intelligence"]
    try:
        from profile_strength_service import calculate_profile_strength_v2

        result = calculate_profile_strength_v2(candidate)
        candidate["_intelligence"] = result
        return result
    except Exception as exc:
        logger.warning("[matching] Could not compute candidate intelligence: %s", exc)
        return None


def _evidence_weighted_skills_score(
    candidate_skills: list[str],
    job_text: str,
    intelligence: dict[str, Any] | None,
    required_skills: Any = None,
) -> float:
    """
    Score skills with evidence quality weighting.
    Demonstrated/verified skills count fully; claimed skills count at 0.6.
    """
    if not candidate_skills:
        return 0.0
    evidence = (intelligence or {}).get("evidence") or {}
    skill_ev = evidence.get("skills", {})
    ev_level = skill_ev.get("evidence_level", 1)  # default: claimed
    base_weight = _EVIDENCE_WEIGHT.get(ev_level, 0.6)

    # When the job supplies an explicit skill list, score coverage of that
    # list. The previous formula divided by the candidate's skill count, so a
    # candidate with one matching skill could receive the maximum skill score
    # while still missing ten required skills; adding those missing skills then
    # produced no gain. This denominator now matches the gaps shown by Fix My
    # Resume.
    explicit_required: list[str] = []
    if isinstance(required_skills, list):
        for skill in required_skills:
            value = _normalize_text(
                skill.get("name") or skill.get("title") or skill.get("skill")
                if isinstance(skill, dict)
                else skill
            )
            if value and canonical_skill_key(value) not in {
                canonical_skill_key(item) for item in explicit_required
            }:
                explicit_required.append(value)
    if explicit_required:
        candidate_text = " ".join(canonical_skill(skill) for skill in candidate_skills)
        candidate_tokens = _phrase_set(candidate_text)
        hits = sum(
            base_weight
            for required in explicit_required
            if _term_in_text(
                _normalize(canonical_skill(required)), candidate_text, candidate_tokens
            )
        )
        return min(hits / len(explicit_required), 1.0)

    text_tokens = _phrase_set(job_text)
    hits = sum(
        base_weight
        for s in candidate_skills
        if s and _term_in_text(_normalize(canonical_skill(s)), job_text, text_tokens)
    )
    return min(hits / len(candidate_skills), 1.0)


def _check_hard_constraints(
    constraint_profile: dict[str, Any],
    job_text: str,
) -> tuple[float, list[str]]:
    """
    Check candidate hard constraints against job text.
    Returns (penalty_multiplier, incompatibilities).
    1.0 = no penalty, 0.1 = near-disqualifying.
    """
    if not constraint_profile:
        return 1.0, []
    penalty = 1.0
    incompatibilities: list[str] = []
    job_lower = job_text.lower()

    work_mode = constraint_profile.get("work_mode_constraint", "unknown")
    if work_mode == "hard_remote_only":
        onsite_signals = [
            "on-site",
            "onsite",
            "on site",
            "in-office",
            "in office",
            "office based",
            "office-based",
            "must be present",
        ]
        if any(s in job_lower for s in onsite_signals):
            penalty *= 0.1
            incompatibilities.append("candidate_remote_only_job_requires_onsite")

    salary_min = constraint_profile.get("salary_min")
    if salary_min and salary_min > 0:
        nums = re.findall(r"[\d]+(?:\.\d+)?", job_lower.replace(",", ""))
        job_nums = [float(n) for n in nums if n]
        if job_nums:
            job_max = max(job_nums)
            if job_max < salary_min * 0.7:
                penalty *= 0.5
                incompatibilities.append("salary_below_candidate_minimum")

    return penalty, incompatibilities


EXPERIENCE_EPSILON_YEARS = 1e-6
_EXPERIENCE_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_EXPERIENCE_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _normalize(term: str) -> str:
    """Normalize punctuation and whitespace for whole-term comparison."""
    term = term.lower().strip()
    # Collapse internal whitespace
    term = re.sub(r"\s+", " ", term)
    return term


def _phrase_set(text: str) -> set:
    """
    Split text into a set of normalized single-word tokens AND keep the full
    normalized phrase so multi-word terms (e.g. 'react native') can be matched.
    Each token is the full word including punctuation chars like #, +, .
    so that 'c#', 'c++', '.net' are distinct from 'c'.
    """
    normalized = _normalize(text)
    # Word tokens that preserve tech punctuation: c#, c++, .net, node.js, etc.
    tokens = set(re.findall(r"[a-z0-9][a-z0-9#.+]*", normalized))
    return tokens


def _term_in_text(term: str, text: str, text_tokens: set) -> bool:
    """
    Return True only when the normalized term matches as a complete token (single-word)
    or as an exact phrase (multi-word) in the normalized text.
    Prevents 'java' matching 'javascript', 'c' matching 'c++', etc.
    """
    norm = _normalize(term)
    words = norm.split()
    if not words:
        return False
    if len(words) == 1:
        return words[0] in text_tokens
    # Multi-word: require the exact phrase to appear verbatim in the normalized text
    return norm in _normalize(text)


# Words that must not create a match by themselves: seniority and generic
# job-title words.
_GENERIC_ROLE_WORDS = {
    # seniority / level
    "junior",
    "senior",
    "sr",
    "jr",
    "lead",
    "principal",
    "staff",
    "associate",
    "mid",
    "entry",
    "level",
    "i",
    "ii",
    "iii",
    "iv",
    # generic job-title nouns
    "developer",
    "engineer",
    "programmer",
    "specialist",
    "consultant",
    "architect",
    "manager",
    "analyst",
    "designer",
    "administrator",
    "director",
    "officer",
    "head",
    "expert",
    "professional",
}

# These preference values describe availability or work constraints, not a
# job family.  They must never create a target-role match or retrieval query.
_NON_ROLE_PREFERENCE_VALUES = {
    "relocate",
    "relocation",
    "willing to relocate",
    "open to relocate",
    "remote",
    "remote only",
    "hybrid",
    "onsite",
    "on site",
}

# Role-family words say what kind of work a job involves, but not which
# technology it uses.  They are deliberately separate from _GENERIC_ROLE_WORDS:
# the former can make a broad role match, while the latter carry no role intent.
_BROAD_ROLE_TOKENS = {
    "backend",
    "frontend",
    "fullstack",
    "full",
    "stack",
    "software",
    "web",
    "mobile",
    "platform",
    "cloud",
    "data",
    "devops",
}

# Technologies that commonly qualify an otherwise broad engineering title.
# This is used only for ordering equally-scored target-role matches; it does
# not affect the hybrid score, eligibility, or retrieval.
_ROLE_TECHNOLOGY_TOKENS = {
    "java",
    "python",
    "javascript",
    "typescript",
    "node",
    "node.js",
    "c",
    "c#",
    "c++",
    "dotnet",
    ".net",
    "go",
    "golang",
    "ruby",
    "php",
    "scala",
    "kotlin",
    "swift",
    "rust",
    "react",
    "angular",
    "vue",
    "django",
    "flask",
    "spring",
    "springboot",
    "rails",
}

# These skills remain in candidate profiles and continue to be available to
# scoring and mandatory-skill checks.  They must not, by themselves, establish
# job relevance because they occur in almost every job description.
_GENERIC_TRANSFERABLE_SKILL_KEYS = {
    "communication",
    "communications",
    "teamwork",
    "teamworkskills",
    "leadership",
    "problemsolving",
    "criticalthinking",
    "collaboration",
    "adaptability",
    "timemanagement",
    "interpersonalskills",
}


def _role_tokens(role: str) -> set:
    """Return meaningful (non-generic) whole tokens from a role string."""
    return {t for t in _phrase_set(role) if t not in _GENERIC_ROLE_WORDS}


def _is_technology_only_role(role: str) -> bool:
    """Return whether a preference contains technology but no role intent."""
    tokens = _role_tokens(role)
    return bool(tokens) and tokens.issubset(_ROLE_TECHNOLOGY_TOKENS)


def _combined_job_skills(job: dict[str, Any]) -> list[Any]:
    """Combine both persisted skill projections without changing their values."""
    from skill_normalization import merge_skills

    return merge_skills(job.get("skills"), job.get("skills_required"))


def _target_role_score(
    target_roles: list[str],
    job_title: str,
    job_text: str,
    semantic_role_score: float | None = None,
) -> float:
    """
    Token-overlap target-role scoring.
    All meaningful tokens from the candidate's target role must appear as whole
    tokens in the job title/text — preserving exact tech boundaries (java ≠ javascript).
    Score = (matched meaningful tokens / total meaningful tokens) weighted by
    title (0.8) vs full text (0.2).
    """
    if not target_roles:
        return 0.0
    title_tokens = _phrase_set(job_title)
    text_tokens = _phrase_set(job_text)
    best = 0.0
    for role in target_roles:
        role_toks = _role_tokens(role)
        if not role_toks:
            continue
        title_hit = len(role_toks & title_tokens) / len(role_toks)
        # Description-only mentions are not role evidence; the title must
        # share at least one meaningful role token first.
        if title_hit <= 0:
            continue
        text_hit = len(role_toks & text_tokens) / len(role_toks)
        score = title_hit * 0.8 + text_hit * 0.2
        best = max(best, score)
    lexical = min(best, 1.0)
    if semantic_role_score is None:
        return lexical
    # The independent target-role retrieval already compares the role intent
    # against the canonical job embedding (title, duties, requirements and
    # skills). Keep the established lexical signal, but allow that semantic
    # comparison to recover differently titled roles.
    semantic = max(0.0, min(1.0, float(semantic_role_score)))
    target_technology = {
        token
        for role in target_roles
        for token in _role_tokens(role)
        if token in _ROLE_TECHNOLOGY_TOKENS
    }
    job_context_tokens = _phrase_set(f"{job_title} {job_text}")
    job_technology = job_context_tokens & _ROLE_TECHNOLOGY_TOKENS
    if (
        target_technology
        and job_technology
        and not target_technology & job_technology
        and not target_technology & job_context_tokens
    ):
        # A strong generic embedding score must not erase an explicit
        # Python-vs-Java (or similar) technology conflict.
        return lexical
    if semantic <= lexical:
        return lexical

    # Semantic similarity can recover differently titled but related roles;
    # it must not turn an unrelated title into a role match.  Reuse the
    # approved role-family classifier as the conservative evidence gate.
    if any(
        classify_role_relationship(role, job_title).relationship
        in {"exact", "equivalent", "related", "adjacent", "transition"}
        for role in target_roles
    ):
        return semantic
    return lexical


def _semantic_role_score(
    target_roles: list[str],
    job_text: str,
    role_context: str | None = None,
    role_vector: list[float] | None = None,
) -> float | None:
    """Compare role intent with complete job context for stored revalidation."""
    if not target_roles or not job_text:
        return None
    role_vector = role_vector or generate_embedding(
        role_context or _build_target_role_retrieval_text(target_roles)
    )
    job_vector = generate_embedding(job_text)
    denominator = (
        sum(value * value for value in role_vector) ** 0.5
        * sum(value * value for value in job_vector) ** 0.5
    )
    if not denominator:
        return None
    return (
        sum(left * right for left, right in zip(role_vector, job_vector, strict=True))
        / denominator
    )


def _semantic_similarity_score(left_text: str, right_text: str) -> float | None:
    """Compute a fresh compatible semantic score for stored-row rescoring."""
    if not left_text.strip() or not right_text.strip():
        return None
    left_vector = generate_embedding(left_text)
    right_vector = generate_embedding(right_text)
    denominator = (
        sum(value * value for value in left_vector) ** 0.5
        * sum(value * value for value in right_vector) ** 0.5
    )
    if not denominator:
        return None
    return (
        sum(left * right for left, right in zip(left_vector, right_vector, strict=True))
        / denominator
    )


def _target_role_specificity(target_roles: list[str], job_title: str) -> int:
    """Classify title intent without changing the target-role score.

    A preferred technology in the title is a specific match (2); a title that
    only shares the role family is broad (1); a role-family title explicitly
    qualified by another technology is a different-technology match (0).
    """
    title_tokens = _phrase_set(job_title)
    best = 0
    for role in target_roles:
        role_tokens = _role_tokens(role)
        target_technologies = role_tokens & _ROLE_TECHNOLOGY_TOKENS
        broad_tokens = role_tokens & _BROAD_ROLE_TOKENS
        if target_technologies & title_tokens:
            best = max(best, 2)
        elif broad_tokens & title_tokens:
            if (title_tokens & _ROLE_TECHNOLOGY_TOKENS) - target_technologies:
                best = max(best, 0)
            else:
                best = max(best, 1)
    return best


def _best_role_family_alignment(
    target_roles: list[str], job_title: str
) -> dict[str, Any]:
    """Return a conservative role-family advisory signal for one job title.

    This signal is intentionally separate from eligibility and the hybrid score.
    Multiple candidate roles use the strongest relationship; unknown, ambiguous,
    conflicting, and unrelated relationships remain neutral.
    """
    if not target_roles or not str(job_title or "").strip():
        return {"relationship": "unknown", "priority": 0}
    relationships = [
        classify_role_relationship(role, job_title) for role in target_roles
    ]
    best = max(
        relationships,
        key=lambda relationship: _ROLE_FAMILY_RANK_PRIORITY.get(
            relationship.relationship, 0
        ),
    )
    return {
        "relationship": best.relationship,
        "priority": _ROLE_FAMILY_RANK_PRIORITY.get(best.relationship, 0),
    }


def _sort_scored_recommendations(
    scored: list[tuple[str, float, dict]],
    signals: dict[str, Any],
    job_details: dict[str, dict],
) -> None:
    """Apply existing ordering with role-family priority as a final tie-breaker."""
    if signals["has_explicit_target_roles"]:
        scored.sort(
            key=lambda item: (
                _target_role_specificity(
                    signals["target_roles"], job_details[item[0]]["title"]
                ),
                item[2].get("target_role_score", 0.0),
                item[1],
                item[2].get("role_family_priority", 0),
            ),
            reverse=True,
        )
    else:
        scored.sort(
            key=lambda item: (item[1], item[2].get("role_family_priority", 0)),
            reverse=True,
        )


def _skills_score(candidate_skills: list[str], job_text: str) -> float:
    """Fraction of candidate skills found as whole normalized terms."""
    if not candidate_skills:
        return 0.0
    text_tokens = _phrase_set(job_text)
    hits = sum(
        1
        for s in candidate_skills
        if s and _term_in_text(_normalize(canonical_skill(s)), job_text, text_tokens)
    )
    return min(hits / len(candidate_skills), 1.0)


def _experience_score(
    candidate_roles: list[str],
    job_title: str,
    job_text: str,
    candidate_responsibilities: list[str] | None = None,
    job_responsibilities: list[str] | None = None,
) -> float:
    """Score relevant history, preferring duties when both sides provide them."""
    combined = job_title + " " + job_text
    combined_tokens = _phrase_set(combined)
    best = 0.0
    for role in candidate_roles:
        norm_role = _normalize(role)
        if not norm_role:
            continue
        score = 1.0 if _term_in_text(norm_role, combined, combined_tokens) else 0.0
        best = max(best, score)
    history_score = min(best, 1.0)
    responsibility_score = _responsibility_score(
        candidate_responsibilities or [], job_responsibilities or []
    )
    if responsibility_score is None:
        return history_score
    return max(history_score, responsibility_score)


_RESPONSIBILITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "will",
    "you",
    "your",
}
_RESPONSIBILITY_ACTIONS = {
    "administer",
    "analyze",
    "architect",
    "build",
    "collaborate",
    "configure",
    "create",
    "debug",
    "design",
    "develop",
    "ensure",
    "implement",
    "integrate",
    "maintain",
    "manage",
    "monitor",
    "operate",
    "optimize",
    "ship",
    "support",
    "test",
    "troubleshoot",
    "write",
}
_GENERIC_RESPONSIBILITY_ACTIONS = {
    "collaborate",
    "ensure",
    "manage",
    "operate",
    "support",
    "work",
}


def _responsibility_phrases(values: Any) -> list[str]:
    """Normalize responsibility entries without treating titles or skills as duties."""
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = re.sub(r"<\s*(?:br|p|div|li)\b[^>]*>", "\n", value, flags=re.I)
        text = re.sub(r"<[^>]+>", "", html.unescape(text))
        for line in re.split(r"\n+", text):
            line = re.sub(r"^\s*(?:[-*•▪‣]|\d+[.)])\s*", "", line).strip()
            line = re.sub(r"\s+", " ", line)
            if len(line) < 12 or len(line) > 500:
                continue
            key = re.sub(r"[^a-z0-9 ]", "", line.casefold()).strip()
            if key and key not in seen:
                seen.add(key)
                result.append(line)
    return result


def _candidate_responsibilities(candidate: dict[str, Any]) -> list[str]:
    """Collect only explicit duty/project descriptions from professional history."""
    values: list[Any] = []
    work_exp = candidate.get("work_experience") or []
    if isinstance(work_exp, str):
        try:
            work_exp = json.loads(work_exp)
        except (TypeError, ValueError):
            work_exp = []
    if isinstance(work_exp, list):
        for item in work_exp:
            if not isinstance(item, dict):
                continue
            values.extend(
                item.get(key) for key in ("responsibilities", "description", "duties")
            )
            values.extend(item.get(key) for key in ("projects", "project_descriptions"))
    for key in (
        "responsibilities",
        "responsibilities_projects",
        "professional_responsibilities",
    ):
        values.append(candidate.get(key))
    raw_data = candidate.get("raw_data") or {}
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except (TypeError, ValueError):
            raw_data = {}
    if isinstance(raw_data, dict):
        for key in (
            "responsibilities",
            "responsibilities_projects",
            "professional_responsibilities",
        ):
            values.append(raw_data.get(key))
    flattened = [
        item
        for value in values
        for item in (value if isinstance(value, list) else [value])
    ]
    return _responsibility_phrases(flattened)


def _job_responsibilities(job: dict[str, Any]) -> list[str]:
    """Collect normalized responsibilities from normalized and provider job fields."""
    values: list[Any] = [job.get("responsibilities")]
    structured = job.get("structured_data") or {}
    if isinstance(structured, dict):
        for key, value in structured.items():
            if "responsibil" in str(key).casefold() or str(key).casefold() in {
                "duties",
                "job_duties",
            }:
                values.append(value)
        raw_source = structured.get("raw_source")
        if isinstance(raw_source, dict) and isinstance(raw_source.get("payload"), dict):
            payload = raw_source["payload"]
            for key, value in payload.items():
                if "responsibil" in str(key).casefold() or str(key).casefold() in {
                    "duties",
                    "job_duties",
                }:
                    values.append(value)
    description = job.get("description") or ""
    if isinstance(description, str):
        section = re.search(
            r"(?is)(?:^|\n)\s*(?:key\s+)?responsibilities\s*:?[ \t]*\n?"
            r"(.*?)(?=\n\s*(?:requirements?|qualifications?|preferred|"
            r"benefits?|about\s+the\s+company|what\s+we\s+offer)\b|$)",
            description,
        )
        if section:
            values.append(section.group(1))
        inline = re.search(
            r"(?is)\b(?:key\s+)?responsibilities\s+(?:include|are)\s+(?P<content>[^.!?]+[.!?]?)"
            r"|\bthe\s+role\s+involves\s+(?P<role_content>[^.!?]+[.!?]?)",
            description,
        )
        if inline:
            values.append(inline.group("content") or inline.group("role_content"))
    flattened = [
        item
        for value in values
        for item in (value if isinstance(value, list) else [value])
    ]
    return _responsibility_phrases(flattened)


def _responsibility_tokens(value: str) -> set[str]:
    tokens = re.findall(r"[a-z][a-z0-9+#.-]{2,}", value.casefold())
    normalized: set[str] = set()
    for token in tokens:
        if token in _RESPONSIBILITY_STOPWORDS:
            continue
        normalized.add(token)
        if token.endswith("ies") and len(token) > 4:
            normalized.add(token[:-3] + "y")
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 4:
            normalized.add(token[:-1])
        for suffix in ("ed", "ing"):
            if token.endswith(suffix) and len(token) > len(suffix) + 3:
                normalized.add(token[: -len(suffix)])
    return normalized


def _responsibility_score(
    candidate_values: list[str], job_values: list[str]
) -> float | None:
    """Return None for insufficient evidence; otherwise compare duty vocabulary."""
    if not candidate_values or not job_values:
        return None
    candidate_tokens = set().union(
        *(_responsibility_tokens(value) for value in candidate_values)
    )
    job_tokens = set().union(*(_responsibility_tokens(value) for value in job_values))
    if not candidate_tokens or not job_tokens:
        return None
    overlap = len(candidate_tokens & job_tokens) / max(1, len(candidate_tokens))
    action_overlap = {
        token
        for token in candidate_tokens & job_tokens
        if token in _RESPONSIBILITY_ACTIONS
        or token.rstrip("s") in _RESPONSIBILITY_ACTIONS
    }
    meaningful_overlap = {
        token
        for token in candidate_tokens & job_tokens
        if token not in _RESPONSIBILITY_ACTIONS
        and token not in _GENERIC_RESPONSIBILITY_ACTIONS
    }
    # Shared duties must contain more than a bare technology token.
    if not meaningful_overlap:
        return 0.0
    if not action_overlap and overlap < 0.35:
        return 0.0
    return round(min(1.0, overlap + (0.15 if action_overlap else 0.0)), 4)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _split_experience_range(text: str) -> list[str] | None:
    if not text:
        return None
    parts = [
        part.strip()
        for part in re.split(r"\s+(?:[\u2013\u2014-]|—)\s+", text, maxsplit=1)
    ]
    if len(parts) == 2:
        return parts
    return None


def _is_open_ended_experience_value(value: Any) -> bool:
    normalized = _normalize_text(value).lower()
    return bool(
        normalized and re.search(r"\b(?:present|current|ongoing|now)\b", normalized)
    )


def _parse_experience_date(value: Any, role: str = "end") -> datetime | None:
    text = _normalize_text(value)
    if not text or _is_open_ended_experience_value(text):
        return None
    cleaned = text.replace(".", "")

    if m := re.match(r"^(\d{4})$", cleaned):
        year = int(m.group(1))
        # Preserve the existing inclusive year policy: 2020–2022 means
        # 2020-01-01 through 2023-01-01. Month-only end dates likewise use
        # the first day of the following month.
        if role == "end":
            return datetime(year + 1, 1, 1, tzinfo=UTC)
        return datetime(year, 1, 1, tzinfo=UTC)

    if m := re.match(r"^(\d{4})[./](\d{1,2})$", cleaned):
        year = int(m.group(1))
        month = int(m.group(2))
        if not 1 <= month <= 12:
            return None
        if role == "end":
            if month == 12:
                return datetime(year + 1, 1, 1, tzinfo=UTC)
            return datetime(year, month + 1, 1, tzinfo=UTC)
        return datetime(year, month, 1, tzinfo=UTC)

    if m := re.match(r"^([A-Za-z]{3,9})\s+(\d{4})$", cleaned):
        parsed_month = _EXPERIENCE_MONTHS.get(m.group(1).lower())
        if parsed_month is None:
            return None
        year = int(m.group(2))
        if role == "end":
            if parsed_month == 12:
                return datetime(year + 1, 1, 1, tzinfo=UTC)
            return datetime(year, parsed_month + 1, 1, tzinfo=UTC)
        return datetime(year, parsed_month, 1, tzinfo=UTC)

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m-%d-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue

    if m := re.match(r"^(\d{4})[./](\d{1,2})[./](\d{1,2})$", cleaned):
        year, month, day = map(int, m.groups())
        return datetime(year, month, day, tzinfo=UTC)

    return None


def _parse_experience_window(item: Any) -> tuple[datetime | None, datetime | None]:
    if not isinstance(item, dict):
        return None, None

    start_value = next(
        (
            item.get(key)
            for key in ("start_date", "startDate", "start", "from")
            if item.get(key)
        ),
        None,
    )
    end_value = next(
        (
            item.get(key)
            for key in ("end_date", "endDate", "end", "to")
            if item.get(key)
        ),
        None,
    )
    start = _parse_experience_date(start_value, "start")
    end = _parse_experience_date(end_value, "end")
    if item.get("is_current") is True or item.get("current") is True:
        end = None

    dates_text = _normalize_text(item.get("dates") or item.get("duration") or "")
    if dates_text:
        parts = _split_experience_range(dates_text)
        if parts:
            left, right = parts
            if start is None:
                start = _parse_experience_date(left, "start")
            if _is_open_ended_experience_value(right):
                end = None
            elif end is None:
                end = _parse_experience_date(right, "end")
        else:
            if start is None:
                start = _parse_experience_date(dates_text, "start")

    return start, end


def _candidate_experience_reconciliation(candidate: dict[str, Any]) -> dict[str, Any]:
    """Reconcile explicit experience with canonicalized employment intervals."""
    work_exp = candidate.get("work_experience") or []
    if isinstance(work_exp, str):
        try:
            work_exp = json.loads(work_exp)
        except Exception:
            work_exp = []

    reported = normalize_experience_value(candidate.get("experience_years"))
    if reported is None:
        reported = normalize_experience_value(candidate.get("total_experience_years"))

    intervals: list[tuple[datetime, datetime]] = []
    now = datetime.now(UTC)

    for item in work_exp if isinstance(work_exp, list) else []:
        if not isinstance(item, dict):
            continue
        # Personal/self projects are not employer experience, even when an
        # extractor placed them in the work_experience array.
        kind = " ".join(
            str(item.get(key) or "")
            for key in (
                "type",
                "category",
                "employment_type",
                "title",
                "company",
                "project_name",
            )
        ).lower()
        if item.get("is_project") or any(
            marker in kind
            for marker in (
                "self project",
                "personal project",
                "side project",
                "independent project",
            )
        ):
            continue
        start, end = _parse_experience_window(item)
        if start is None:
            continue
        effective_end = end or now
        if effective_end < start:
            continue
        intervals.append((start, effective_end))

    if not intervals:
        return {
            "reported_experience_years": reported,
            "calculated_experience_years": None,
            "reconciled_experience_years": reported,
            "experience_reconciliation_status": (
                "reported_fallback" if reported is not None else "unknown"
            ),
        }

    intervals.sort(key=lambda item: item[0])
    merged_days = 0.0
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_days += (current_end - current_start).total_seconds() / 86400.0
            current_start, current_end = start, end

    merged_days += (current_end - current_start).total_seconds() / 86400.0
    calculated = max(0.0, merged_days / 365.25)
    status = "calculated"
    if reported is not None and abs(reported - calculated) > 0.25:
        status = "conflict"
    return {
        "reported_experience_years": reported,
        "calculated_experience_years": calculated,
        "reconciled_experience_years": calculated,
        "experience_reconciliation_status": status,
    }


def _candidate_total_experience_years(candidate: dict[str, Any]) -> float | None:
    return _candidate_experience_reconciliation(candidate)[
        "reconciled_experience_years"
    ]


def _job_text(
    job_title: str,
    job_description: str,
    job_requirements: Any = "",
    job_skills: Any = None,
) -> str:
    parts = [job_title or "", job_description or "", _normalize_text(job_requirements)]
    if isinstance(job_skills, list):
        skill_parts: list[str] = []
        for skill in job_skills:
            if isinstance(skill, dict):
                skill_parts.append(
                    _normalize_text(
                        skill.get("name") or skill.get("title") or skill.get("skill")
                    )
                )
            else:
                skill_parts.append(_normalize_text(skill))
        skill_parts.extend(canonical_skill(skill) for skill in list(skill_parts))
        parts.append(", ".join(part for part in skill_parts if part))
    elif job_skills:
        parts.append(_normalize_text(job_skills))
    return " ".join(part for part in parts if part)


def normalize_job_experience(
    job_text: Any = "", experience_required: Any = ""
) -> tuple[float | None, float | None]:
    """Normalize explicit experience from any source into common min/max years."""
    return _job_experience_bounds(_job_text("", job_text, experience_required))


def _job_experience_bounds(job_text: str) -> tuple[float | None, float | None]:
    normalized = _normalize_text(job_text).lower().replace("–", "-").replace("—", "-")
    if not normalized:
        return None, None

    for word, number in sorted(
        _EXPERIENCE_NUMBER_WORDS.items(), key=lambda item: -len(item[0])
    ):
        normalized = re.sub(rf"\b{word}\b", str(number), normalized)

    min_years: float | None = None
    max_years: float | None = None

    for pattern in (
        r"(?P<min>\d+(?:\.\d+)?)\s*[-–—]\s*(?P<max>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
        r"(?P<min>\d+(?:\.\d+)?)\s+(?:to)\s+(?P<max>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
    ):
        for match in re.finditer(pattern, normalized):
            low = float(match.group("min"))
            high = float(match.group("max"))
            min_years = low if min_years is None else max(min_years, low)
            max_years = high if max_years is None else min(max_years, high)

    for match in re.finditer(
        (
            r"(?:minimum|min\.?|at least|requires?|requiring)?\s*"
            r"(?:of\s*)?(?P<years>\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)"
        ),
        normalized,
    ):
        years = float(match.group("years"))
        min_years = years if min_years is None else max(min_years, years)

    # Do not re-interpret the upper end of an already parsed ``1-3 years``
    # range as a separate minimum requirement.
    range_free = re.sub(
        r"\d+(?:\.\d+)?\s*(?:[-\u2013\u2014]|to)\s*\d+(?:\.\d+)?\s*(?:years?|yrs?)",
        "",
        normalized,
    )
    for match in re.finditer(
        (
            r"(?:minimum|min\.?|at least|requires?|requiring)?\s*"
            r"(?:of\s*)?(?P<years>\d+(?:\.\d+)?)\s*(?:years?|yrs?)"
        ),
        range_free,
    ):
        years = float(match.group("years"))
        min_years = years if min_years is None else max(min_years, years)
        # Bare "5 years" is treated as an exact stated band; phrases that
        # explicitly establish a floor remain open-ended.
        prefix = (
            match.group(0)[: match.group(0).lower().find(str(int(years)))]
            .strip()
            .lower()
            if str(int(years)) in match.group(0)
            else ""
        )
        if not prefix and max_years is None:
            max_years = years

    if min_years is not None and max_years is not None and max_years < min_years:
        max_years = None

    return min_years, max_years


def _count_skill_matches(candidate_skills: list[str], job_text: str) -> int:
    if not candidate_skills:
        return 0
    text_tokens = _phrase_set(job_text)
    return sum(
        1
        for skill in candidate_skills
        if skill
        and canonical_skill_key(skill) not in _GENERIC_TRANSFERABLE_SKILL_KEYS
        and any(
            _term_in_text(_normalize(variant), job_text, text_tokens)
            for variant in skill_variants(skill)
        )
    )


def _job_is_eligible(
    signals: dict[str, Any],
    candidate_years: float,
    job_title: str,
    job_description: str,
    job_requirements: Any = "",
    job_skills: Any = None,
) -> bool:
    job_text = _job_text(job_title, job_description, job_requirements, job_skills)
    if not _job_passes_experience(candidate_years, job_text, job_requirements):
        return False
    return _job_passes_skills_or_role(signals, job_title, job_text)


def experience_eligibility(
    candidate_years: float | None,
    job_text: Any = "",
    experience_required: Any = "",
    reconciliation_status: str | None = None,
) -> dict[str, Any]:
    """Central hard gate applied before scoring, regardless of job source."""
    min_years, max_years = normalize_job_experience(job_text, experience_required)
    reason = None
    if (
        candidate_years is not None
        and min_years is not None
        and candidate_years + EXPERIENCE_EPSILON_YEARS < min_years
    ):
        reason = "candidate_below_job_minimum"
    elif (
        candidate_years is not None
        and max_years is not None
        and candidate_years - EXPERIENCE_EPSILON_YEARS > max_years
    ):
        reason = "candidate_above_job_maximum"
    result = {
        "candidate_years": candidate_years,
        "experience_reconciliation_status": reconciliation_status
        or ("unknown" if candidate_years is None else "calculated"),
        "job_min_years": min_years,
        "job_max_years": max_years,
        "eligible": reason is None,
        "decision": "eligible" if reason is None else "rejected",
        "rejection_reason": reason,
    }
    logger.info(
        "[experience-eligibility] candidate_years=%s job_min_years=%s "
        "job_max_years=%s decision=%s rejection_reason=%s",
        result["candidate_years"],
        result["job_min_years"],
        result["job_max_years"],
        result["decision"],
        result["rejection_reason"],
    )
    return result


def stored_recommendation_experience_eligibility(
    candidate: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    """Read-only experience gate for an already-persisted recommendation.

    This deliberately evaluates only the hard experience boundary.  It does
    not retrieve, embed, score, refresh, or mutate recommendation data.
    """
    reconciliation = _candidate_experience_reconciliation(candidate)
    candidate_years = reconciliation["reconciled_experience_years"]
    job_text = _job_text(
        job.get("title", ""),
        job.get("description", ""),
        job.get("requirements", ""),
        job.get("skills") or job.get("skills_required") or [],
    )
    # Normalize once here so retrieval uses the same stored-job interpretation
    # as the matcher; experience_eligibility remains the authoritative decision.
    normalize_job_experience(job_text, job.get("experience_required"))
    return experience_eligibility(
        candidate_years,
        job_text,
        job.get("experience_required"),
        reconciliation["experience_reconciliation_status"],
    )


def stored_recommendation_relevance(
    candidate: dict[str, Any],
    job: dict[str, Any],
    role_embedding: list[float] | None = None,
) -> dict[str, Any]:
    """Revalidate a stored recommendation against the current candidate/job."""
    signals = _build_candidate_signals(candidate)
    job_title = _normalize_text(job.get("title"))
    job_description = _normalize_text(job.get("description"))
    job_requirements = job.get("requirements") or ""
    job_skills = _combined_job_skills(job)
    job_text = _job_text(job_title, job_description, job_requirements, job_skills)
    reconciliation = _candidate_experience_reconciliation(candidate)
    experience = experience_eligibility(
        reconciliation["reconciled_experience_years"],
        job_text,
        job.get("experience_required"),
        reconciliation["experience_reconciliation_status"],
    )
    if not experience["eligible"]:
        return {"eligible": False, "reason": "experience_outside_required_range"}
    preferences_ok, preference_reasons = _preference_eligibility(
        signals, job, reconciliation["reconciled_experience_years"]
    )
    if not preferences_ok:
        return {"eligible": False, "reason": preference_reasons[0]}
    raw_required_skills = job.get("skills_required")
    required_skills = _normalize_structured_required_skills(raw_required_skills)
    if raw_required_skills is None or raw_required_skills == []:
        required_skills = _extract_job_required_skills(job_text)
    responsibility_relevance = _responsibility_score(
        signals.get("responsibilities") or [], _job_responsibilities(job)
    )
    role_semantic_score = None
    # Stored rows do not carry the original Qdrant payload. Only opt into a
    # local semantic fallback when the caller supplies explicit compatible
    # provenance; otherwise lexical and mandatory-skill checks remain the gate.
    if _embedding_metadata_compatible(job.get("embedding_metadata")):
        try:
            from app.job_ingestion.embedding_service import build_job_text

            semantic_job_text = build_job_text(job)
        except (ImportError, TypeError, ValueError):
            semantic_job_text = job_text
        role_semantic_score = _semantic_role_score(
            signals["target_roles"],
            semantic_job_text,
            signals.get("_target_role_query"),
            role_embedding,
        )
    if not _job_passes_skills_or_role(
        signals,
        job_title,
        job_text,
        required_skills,
        role_semantic_score,
        responsibility_relevance,
    ):
        return {"eligible": False, "reason": "insufficient_relevance_evidence"}
    return {
        "eligible": True,
        "reason": None,
        "experience_reconciliation_status": reconciliation[
            "experience_reconciliation_status"
        ],
    }


def _job_passes_experience(
    candidate_years: float, job_text: str, experience_required: Any = ""
) -> bool:
    return bool(
        experience_eligibility(candidate_years, job_text, experience_required)[
            "eligible"
        ]
    )


def _job_passes_skills_or_role(
    signals: dict[str, Any],
    job_title: str,
    job_text: str,
    required_skills: list[str] | None = None,
    semantic_role_score: float | None = None,
    responsibility_relevance: float | None = None,
) -> bool:
    """Require meaningful role/skill evidence before ranking a job.

    An explicit mandatory-skill section must be fully covered. Otherwise, a
    strong title match or at least two matching candidate skills is sufficient;
    one incidental skill and description-only role mentions are insufficient.
    """
    skill_hits = _count_skill_matches(signals["skills"], job_text)
    role_score = _target_role_score(
        signals["target_roles"], job_title, job_text, semantic_role_score
    )
    title_skill_hits = _count_skill_matches(signals["skills"], job_title)
    if required_skills:
        candidate_keys = {
            canonical_skill_key(skill) for skill in signals["skills"] if skill
        }
        if any(
            canonical_skill_key(required_skill) not in candidate_keys
            for skill in required_skills
            for required_skill in skill_values(skill)
            if canonical_skill_key(required_skill)
        ):
            return False
        # Fully covered explicit mandatory skills are themselves sufficient
        # relevance evidence, even when the title is generic.
        return True
    return role_score >= 0.5 or skill_hits >= 2 or title_skill_hits >= 1


def _normalize_structured_required_skills(value: Any) -> list[str]:
    """Keep only usable structured mandatory skill names, conservatively."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, str):
            candidates = skill_values(item)
        elif isinstance(item, dict):
            raw_name = item.get("name") or item.get("skill")
            candidates = skill_values(raw_name) if isinstance(raw_name, str) else []
        else:
            candidates = []
        for candidate in candidates:
            cleaned = _normalize_text(candidate)
            key = canonical_skill_key(cleaned)
            if not key or key in seen:
                continue
            seen.add(key)
            # Keep the source display name for structured requirements while
            # using canonical_skill_key above for alias-aware deduplication and
            # matching (for example, AWS and Amazon Web Services).
            normalized.append(cleaned)
    return normalized


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = [value]
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _norm_set(values: Any) -> set:
    return {_normalize(v) for v in _as_list(values) if _normalize(v)}


def _preference_text_match(preferences: Any, value: Any) -> bool:
    """Whole-value preference match; never treats an absent job field as bad."""
    wanted, actual = _norm_set(preferences), _normalize_text(value).lower()
    return bool(
        wanted
        and actual
        and any(p == actual or p in actual or actual in p for p in wanted)
    )


def _remote_only(value: Any) -> bool:
    text = _normalize_text(value).lower()
    return "remote" in text and any(
        word in text for word in ("only", "exclusive", "strict")
    )


def _job_work_mode(job: dict[str, Any]) -> str:
    policy = _normalize_text(job.get("remote_policy")).lower()
    if "hybrid" in policy:
        return "hybrid"
    if any(word in policy for word in ("onsite", "on-site", "on site")):
        return "onsite"
    if "remote" in policy:
        return "remote"
    if job.get("remote") is True:
        return "remote"
    return ""


def _employment_type_key(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", _normalize_text(value).casefold()).strip()
    aliases = {
        "full time": "full-time",
        "permanent": "full-time",
        "part time": "part-time",
        "temporary": "temporary",
        "intern": "internship",
        "internship": "internship",
        "contractor": "contract",
    }
    return aliases.get(text, text)


def _salary_number(value: Any) -> float | None:
    text = _normalize_text(value).lower().replace(",", "")
    if not text:
        return None
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    amount = max(float(n) for n in numbers)
    if "lpa" in text or "lakh" in text:
        amount *= 100000
    elif re.search(r"\b(?:k|thousand)\b", text):
        amount *= 1000
    return amount


def _preference_eligibility(
    signals: dict[str, Any], job: dict[str, Any], candidate_years: float | None
) -> tuple[bool, list[str]]:
    """Apply only explicit constraints. Unknown job metadata remains eligible."""
    reasons: list[str] = []
    # Country is an eligibility boundary, never a ranking signal.
    if signals.get("_candidate") is not None and not country_eligible(
        signals.get("_candidate", {}), job
    ):
        reasons.append("job_country_ineligible")
    mode = _job_work_mode(job)
    if _remote_only(signals.get("remote_preference")) and mode in ("onsite", "hybrid"):
        reasons.append("remote_only_job_not_remote")
    locations = signals.get("preferred_locations") or []
    # Explicit locations are a restriction, except a remote job satisfies an
    # explicit remote-or-location preference.
    normalized_job_location = job_location(job)
    has_location_evidence = bool(
        normalized_job_location.get("locations")
        and any(
            item.get("city") or item.get("state") or item.get("country_code")
            for item in normalized_job_location["locations"]
        )
    )
    relocation = signals.get("willing_to_relocate") is True
    if (
        locations
        and mode != "remote"
        and (
            not has_location_evidence or not location_preference_matches(locations, job)
        )
        and not relocation
    ):
        reasons.append("job_outside_explicit_locations")
    types = signals.get("employment_types") or []
    if (
        types
        and job.get("employment_type")
        and not any(
            _employment_type_key(preferred)
            == _employment_type_key(job.get("employment_type"))
            for preferred in types
        )
    ):
        reasons.append("employment_type_mismatch")
    # The established experience boundary remains an eligibility gate.
    job_text = _job_text(
        job.get("title", ""),
        job.get("description", ""),
        job.get("requirements", ""),
        job.get("skills"),
    )
    experience = experience_eligibility(
        candidate_years,
        job_text,
        job.get("experience_required"),
        signals.get("experience_reconciliation_status"),
    )
    if not experience["eligible"]:
        reasons.append("experience_outside_required_range")
    return not reasons, reasons


def _experience_fit_score(candidate_years: float | None, job_text: str) -> float:
    if candidate_years is None:
        return 0.5
    low, high = _job_experience_bounds(job_text)
    if low is None and high is None:
        return 0.5  # absent ATS metadata is neutral, never a penalty
    if low is not None and candidate_years < low:
        return max(0.0, 1.0 - ((low - candidate_years) / 3.0))
    if high is not None and candidate_years > high:
        return max(0.0, 1.0 - ((candidate_years - high) / 3.0))
    return 1.0


def _preference_score(
    signals: dict[str, Any], job: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    """Average only preferences for which the job supplies comparable metadata."""
    scored: dict[str, float] = {}
    if signals.get("_candidate"):
        scored["country_location"] = location_preference_score(
            signals["_candidate"], job
        )
    mode = _job_work_mode(job)
    if signals.get("remote_preference") and mode:
        desired = _normalize_text(signals["remote_preference"]).lower()
        scored["work_mode"] = 1.0 if mode in desired else 0.0
    location = " ".join(
        str(job.get(k) or "") for k in ("location", "city", "state", "country")
    )
    if signals.get("preferred_locations") and location:
        scored["location"] = (
            1.0
            if location_preference_matches(signals["preferred_locations"], job)
            else 0.0
        )
    if signals.get("employment_types") and job.get("employment_type"):
        scored["employment_type"] = (
            1.0
            if any(
                _employment_type_key(preferred)
                == _employment_type_key(job["employment_type"])
                for preferred in signals["employment_types"]
            )
            else 0.0
        )
    if signals.get("preferred_industries") and job.get("industry"):
        scored["industry"] = (
            1.0
            if _preference_text_match(signals["preferred_industries"], job["industry"])
            else 0.0
        )
    company_type = " ".join(
        str(job.get(k) or "")
        for k in ("company_type", "structured_data", "description")
    )
    if signals.get("preferred_companies") and company_type:
        scored["company_type"] = (
            1.0
            if _preference_text_match(signals["preferred_companies"], company_type)
            else 0.0
        )
    expected, offered = (
        _salary_number(signals.get("expected_salary")),
        _salary_number(job.get("salary_range")),
    )
    if expected and offered:
        scored["salary"] = 1.0 if offered >= expected else max(0.0, offered / expected)
    return (sum(scored.values()) / len(scored) if scored else 0.5), scored


def _build_candidate_signals(candidate: dict[str, Any]) -> dict[str, Any]:
    """Extract matching signals from the candidate profile."""
    candidate = normalize_candidate_for_matching(candidate)
    import json

    raw_data = candidate.get("raw_data") or {}
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except Exception:
            raw_data = {}

    # preferred_roles is the candidate's stated job direction.  A current role
    # describes experience; it must not redirect a candidate who is explicitly
    # seeking a different role or technology.
    preferences: dict[str, Any] = {}
    try:
        from profile_strength_service import get_canonical_preferences

        preferences = get_canonical_preferences(candidate, candidate.get("_prefs_row"))
        preferred_roles = preferences.get("preferred_roles") or []
    except Exception:
        preferred_roles = (
            raw_data.get("preferred_roles") or raw_data.get("target_roles") or []
        )
        preferences = raw_data
    if isinstance(preferred_roles, str):
        preferred_roles = [preferred_roles]
    current_role = (candidate.get("current_role") or "").strip()
    target_roles = list(
        dict.fromkeys(
            str(role).strip()
            for role in preferred_roles
            if str(role).strip()
            and _normalize(str(role)) not in _NON_ROLE_PREFERENCE_VALUES
            and not _is_technology_only_role(str(role))
        )
    )
    # A broad role saved alongside a more specific role is useful profile
    # context, but it must not dilute the candidate's explicit direction.
    # For example, "Backend Developer" is subsumed by "Java Backend
    # Developer"; allowing the former to be a target match makes a Python
    # backend job look like an intent match.  Keep distinct directions (for
    # example, "Java Backend Developer" and "Team Lead") intact.
    target_role_tokens = [_role_tokens(role) for role in target_roles]
    target_roles = [
        role
        for index, role in enumerate(target_roles)
        if not any(
            target_role_tokens[index] < other_tokens
            for other_index, other_tokens in enumerate(target_role_tokens)
            if other_index != index
        )
    ]
    explicit_target_roles = list(target_roles)
    # Only use current role as a direction fallback when no role preference was saved.
    if not target_roles and current_role:
        target_roles = [current_role]

    # Skills: merge DB skills with any voice-extracted skills stored in raw_data
    skills_raw = candidate.get("skills") or []
    if isinstance(skills_raw, str):
        try:
            skills_raw = json.loads(skills_raw)
        except Exception:
            skills_raw = []
    skills = [s["name"] if isinstance(s, dict) else str(s) for s in skills_raw if s]

    # Also include voice-extracted skills from raw_data; they may not yet be
    # merged into the skills column.
    voice_skills = raw_data.get("skills") or []
    if isinstance(voice_skills, list):
        seen = {s.lower() for s in skills}
        for vs in voice_skills:
            name = vs["name"] if isinstance(vs, dict) else str(vs)
            if name and name.lower() not in seen:
                skills.append(name)
                seen.add(name.lower())

    # Past roles from work experience
    work_exp = candidate.get("work_experience") or []
    if isinstance(work_exp, str):
        try:
            work_exp = json.loads(work_exp)
        except Exception:
            work_exp = []
    past_roles = (
        [w.get("title", "") for w in work_exp if isinstance(w, dict) and w.get("title")]
        if isinstance(work_exp, list)
        else []
    )
    if current_role:
        past_roles.append(current_role)

    return {
        "target_roles": target_roles,
        "has_explicit_target_roles": bool(explicit_target_roles),
        "skills": skills,
        "past_roles": past_roles,
        "responsibilities": _candidate_responsibilities(candidate),
        # get_canonical_preferences prioritizes candidate_preferences, then
        # persisted profile/resume/chat input.  This preserves provenance and
        # avoids creating a second preference source of truth.
        "preferred_locations": _as_list(preferences.get("preferred_locations")),
        "preferred_industries": _as_list(preferences.get("preferred_industries")),
        "preferred_companies": _as_list(
            preferences.get("preferred_companies")
            or preferences.get("company_types")
            or raw_data.get("company_types")
            or raw_data.get("company_type_preference")
        ),
        "employment_types": _as_list(preferences.get("employment_types")),
        "remote_preference": preferences.get("remote_preference") or "",
        "expected_salary": preferences.get("expected_salary") or "",
        "willing_to_relocate": preferences.get("willing_to_relocate"),
        "opportunity_type": preferences.get("opportunity_type", "jobs"),
        "_target_role_query": _build_target_role_retrieval_text(
            target_roles,
            current_role=current_role,
            past_roles=past_roles,
            skills=skills,
        ),
        **_candidate_experience_reconciliation(candidate),
        "total_experience_years": _candidate_total_experience_years(candidate),
        "_candidate": candidate,
    }


def _build_target_role_retrieval_text(
    target_roles: list[str],
    current_role: str = "",
    past_roles: list[str] | None = None,
    skills: list[str] | None = None,
) -> str:
    """Build an intent-only Qdrant query, excluding experience and skills.

    Target-role retrieval deliberately contains no current role, resume text, or
    candidate skills.  Those are qualification and ranking signals, whereas the
    candidate's saved preferred roles determine which job families must be
    available to that later stage.
    """
    parts = ["Target job roles:\n" + "\n".join(target_roles)]
    if current_role:
        parts.append(f"Current professional role:\n{current_role}")
    if past_roles:
        parts.append("Relevant previous roles:\n" + "\n".join(past_roles[:5]))
    if skills:
        parts.append("Relevant professional skills:\n" + ", ".join(skills[:20]))
    return "\n\n".join(parts)


def _extract_job_required_skills(job_text: str) -> list[str]:
    """
    Heuristically extract required skills from job text.
    Looks for skills listed after 'required:', 'requirements:', 'must have:',
    or in parenthetical skill lists. Falls back to all tech tokens.
    """
    # Only inspect explicitly structured sections; surrounding JD prose is
    # intentionally ignored because it often contains sentence fragments.
    text = str(job_text or "").replace("\\n", "\n").replace("\\r", "\r")
    text = re.sub(r"<[^>]+>", "\n", text)
    heading_text = (
        r"(?:required\s+skills?|technical\s+skills?|key\s+skills?|"
        r"core\s+skills?|qualifications?|requirements?|must[- ]have|"
        r"what\s+you\s+(?:need|bring)|competencies)"
    )
    section = re.compile(rf"^\s*{heading_text}\s*:?[ \t]*$", re.I)
    inline_section = re.compile(rf"^\s*{heading_text}\s*:\s*(?P<items>.+)$", re.I)
    requires_line = re.compile(
        r"\b(?:requires?|must\s+have)\s+(?:experience\s+with\s+)?(?P<items>[^.!?]+)",
        re.I,
    )
    technical_markers = {
        *_ROLE_TECHNOLOGY_TOKENS,
        "api",
        "apis",
        "etl",
        "sql",
        "rest",
        "graphql",
    }
    stop = re.compile(
        r"^\s*(?:preferred|nice\s+to\s+have|benefits?|responsibilities|about\s+(?:the\s+company|us|the\s+role)|what\s+we\s+offer)\b",
        re.I,
    )
    bullet = re.compile(r"^\s*(?:[-*•▪‣]|\d+[.)])\s+")
    rejected_start = re.compile(
        r"^(?:be|build|communicate|conduct|create|develop|ensure|help|lead|manage|maintain|provide|support|work|all|the|a|an|and|or)\b",
        re.I,
    )
    rejected = re.compile(
        r"\b(?:location|salary|compensation|travel|clearance|citizenship|eligib|authorization|visa|degree|education|bachelor|master|phd|years?\s+of\s+experience)\b",
        re.I,
    )
    skills = []
    collecting = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if stop.match(line):
            collecting = False
            continue
        requires_match = requires_line.search(line)
        inline: re.Match[str] | None
        if requires_match and any(
            re.search(rf"\b{re.escape(marker)}\b", requires_match.group("items"), re.I)
            for marker in technical_markers
        ):
            inline = inline_section.match(line) or requires_match
        else:
            inline = inline_section.match(line)
        if inline:
            collecting = True
            line = inline.group("items").strip()
            is_bullet = True
        elif section.match(line):
            collecting = True
            continue
        else:
            is_bullet = bool(bullet.match(line))
        if not collecting:
            continue
        item_text = bullet.sub("", line).strip(" .:;()")
        items = (
            re.split(r"[,;]|\band\b", item_text, flags=re.I)
            if is_bullet
            else [item_text]
        )
        for item in items:
            item = re.sub(
                (
                    r"^(?:strong|proven|demonstrated|hands[- ]on)\s*"
                    r"(?:experience|proficiency|knowledge|expertise|"
                    r"familiarity)\s+(?:with|in|of)\s+"
                ),
                "",
                item.strip(" .:;()"),
                flags=re.I,
            )
            if (
                2 <= len(item) <= 60
                and len(item.split()) <= 6
                and not rejected_start.match(item)
                and not rejected.search(item)
                and not item.endswith((".", "!", "?"))
            ):
                skills.append(item)
    return list(dict.fromkeys(skills))
    # Look for explicit required section
    for marker in (
        "required:",
        "requirements:",
        "must have:",
        "you must have:",
        "requires ",
        "requiring ",
    ):
        idx = job_text.lower().find(marker)
        if idx != -1:
            snippet = job_text[idx : idx + 300]
            # Extract comma/newline separated items
            items = re.split(r"[,\n;]", snippet)
            skills = []
            for item in items[1:8]:  # skip the marker itself
                item = item.strip().strip(".-•*")
                if 2 <= len(item) <= 40 and not item.lower().startswith(
                    ("and ", "or ", "the ")
                ):
                    skills.append(item)
            if skills:
                return skills
    return []


def _skill_evidence_breakdown(
    candidate_skills: list[str],
    job_text: str,
    intelligence: dict[str, Any] | None,
) -> dict[str, list[str]]:
    """
    Classify each candidate skill that appears in the job text by evidence level.
    Returns {strong, partial, missing_required} lists for explainability.
    """
    text_tokens = _phrase_set(job_text)
    evidence = (intelligence or {}).get("evidence") or {}
    skill_ev = evidence.get("skills", {})
    ev_level = skill_ev.get("evidence_level", 1)

    strong, partial = [], []
    for s in candidate_skills:
        if not s:
            continue
        if _term_in_text(_normalize(s), job_text, text_tokens):
            if ev_level >= 3:  # DEMONSTRATED or VERIFIED
                strong.append(f"{s} — demonstrated")
            elif ev_level >= 2:  # CORROBORATED
                strong.append(f"{s} — corroborated")
            else:
                partial.append(f"{s} — claimed only")
    return {"strong": strong, "partial": partial}


def _compute_job_specific_confidence(
    signals: dict[str, Any],
    job_title: str,
    job_text: str,
    hybrid_score: float,
    components: dict[str, Any],
    intelligence: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Job-Specific Recommendation Confidence.

    Formula (avoids double-counting hybrid_score components):

      base = hybrid_score * 100          # target_role + skills + experience + semantic
      readiness_factor                   # scales base by candidate readiness
      constraint_gate                    # hard override on hard incompatibility
      evidence_gate                      # cap when required skills are unconfirmed
      contradiction_penalty              # reduce for relevant contradictions

      raw_confidence = base * readiness_factor
      raw_confidence = min(raw_confidence, evidence_cap)
      raw_confidence -= contradiction_penalty
      if constraint_gate: raw_confidence = min(raw_confidence, 25)

    Keeps Profile Strength, Recommendation Readiness, and Job-Specific
    Confidence separate.
    """
    rec_readiness = (intelligence or {}).get("recommendation_readiness") or {}
    readiness_level = rec_readiness.get("level", "low")

    # Readiness factor: scales base score — does NOT replace it
    # High readiness gets full weight; low readiness has a 0.6 floor because
    # job-critical information may still be known.
    if readiness_level == "high":
        readiness_factor = 1.0
    elif readiness_level == "medium":
        readiness_factor = 0.85
    else:
        # Low readiness: still allow high confidence if job-critical signals are strong
        # Use hybrid score as the primary signal; readiness only damps slightly.
        readiness_factor = 0.70

    base = hybrid_score * 100.0
    raw = base * readiness_factor

    # Evidence cap: claimed skills without corroboration are capped at 65.
    evidence = (intelligence or {}).get("evidence") or {}
    skill_ev_level = evidence.get("skills", {}).get("evidence_level", 1)
    if skill_ev_level <= 1:  # claimed only
        evidence_cap = 65.0
    elif skill_ev_level == 2:  # corroborated
        evidence_cap = 85.0
    else:  # demonstrated / verified
        evidence_cap = 100.0
    confidence_caps: list[str] = []
    if evidence_cap < raw:
        confidence_caps.append("evidence_quality")
        raw = evidence_cap

    evidence_availability = components.get("evidence_availability") or {}
    unavailable_evidence = sorted(
        dimension
        for dimension, available in evidence_availability.items()
        if available is False
    )

    # Required-skill coverage cap: if job has explicit required skills and candidate
    # is missing some, cap confidence proportionally.
    # This prevents high confidence when semantic similarity is high but skills
    # are absent.
    missing_required = list(components.get("missing_required_skills") or [])
    if missing_required:
        required_skills = [
            skill for skill in _extract_job_required_skills(job_text) if skill
        ]
        coverage = (
            1.0 - (len(missing_required) / len(required_skills))
            if required_skills
            else 0.0
        )
        # 0 coverage → max 40; partial coverage may be useful, but an
        # unresolved required skill must never be labelled high confidence
        # (the high-confidence threshold is 70).
        required_cap = min(40.0 + coverage * 55.0, 69.0)
        if required_cap < raw:
            raw = required_cap
            confidence_caps.append("missing_required_skills")

    # Hard constraint gate: near-disqualify
    incompatibilities = components.get("incompatibilities") or []
    has_hard_constraint = bool(incompatibilities)
    if has_hard_constraint:
        if 25.0 < raw:
            raw = 25.0
            confidence_caps.append("hard_constraint")

    # Apply penalties only when the contradiction is reliably associated with
    # this job. Preserve unknown/unrelated records in diagnostics below.
    inconsistencies = (intelligence or {}).get("inconsistencies") or []
    relevant_inconsistencies: list[dict[str, Any]] = []
    contradiction_diagnostics: list[dict[str, Any]] = []
    job_tokens = _phrase_set(job_text)
    experience_is_explicit = _job_experience_bounds(job_text) != (None, None)
    has_work_mode_conflict = "candidate_remote_only_job_requires_onsite" in (
        components.get("incompatibilities") or []
    )
    for inconsistency in inconsistencies:
        if not isinstance(inconsistency, dict):
            contradiction_diagnostics.append(
                {"record": inconsistency, "relevance": "unknown", "applied": False}
            )
            continue
        field = inconsistency.get("field")
        relevant = False
        if isinstance(field, str) and field.startswith("skill:"):
            skill = field.split(":", 1)[1].strip()
            relevant = bool(skill) and _term_in_text(
                _normalize(canonical_skill(skill)), job_text, job_tokens
            )
        elif field in ("experience_years", "employment_timeline"):
            relevant = experience_is_explicit
        elif field == "work_mode_preference":
            relevant = has_work_mode_conflict
        contradiction_diagnostics.append(
            {
                "field": field,
                "severity": inconsistency.get("severity"),
                "relevance": "relevant" if relevant else "unknown_or_unrelated",
                "applied": bool(
                    relevant and inconsistency.get("severity") in ("high", "medium")
                ),
            }
        )
        if relevant:
            relevant_inconsistencies.append(inconsistency)
    high_sev = sum(1 for i in relevant_inconsistencies if i.get("severity") == "high")
    medium_sev = sum(
        1 for i in relevant_inconsistencies if i.get("severity") == "medium"
    )
    contradiction_penalty = high_sev * 12 + medium_sev * 4
    raw -= contradiction_penalty
    raw = max(raw, 0.0)

    score = int(round(min(raw, 100.0)))

    if has_hard_constraint or score < 30:
        level = "low"
        tier = "near_disqualified" if has_hard_constraint else "limited"
    elif score >= 70:
        level = "high"
        tier = "strong_personalized"
    elif score >= 45:
        level = "medium"
        tier = "broader_matching"
    else:
        level = "low"
        tier = "limited"

    # Build match explanation
    skill_breakdown = _skill_evidence_breakdown(
        signals["skills"], job_text, intelligence
    )
    strong_reasons = list(skill_breakdown["strong"])
    partial_reasons = list(skill_breakdown["partial"])

    if components.get("target_role_score", 0) >= 0.5:
        strong_reasons.insert(0, "Target role matches")
    if components.get("experience_score", 0) >= 0.5:
        strong_reasons.append("Experience requirement satisfied")

    constraint_notes = []
    if has_hard_constraint:
        for inc in incompatibilities:
            if inc == "candidate_remote_only_job_requires_onsite":
                constraint_notes.append("Remote-only candidate — job requires onsite")
            elif inc == "salary_below_candidate_minimum":
                constraint_notes.append("Job salary below candidate minimum")
            else:
                constraint_notes.append(inc.replace("_", " "))
    elif components.get("constraint_penalty", 1.0) >= 1.0:
        if (intelligence or {}).get("constraint_profile", {}).get(
            "work_mode_constraint"
        ) not in (None, "unknown"):
            constraint_notes.append("Work mode compatible")

    concerns = []
    for i in inconsistencies:
        if isinstance(i, dict) and i.get("severity") in ("high", "medium"):
            concerns.append(i.get("description", ""))

    return {
        "recommendation_confidence": {
            "score": score,
            "level": level,
            "tier": tier,
            "evidence_availability": evidence_availability,
            "unavailable_evidence": unavailable_evidence,
            "confidence_caps": confidence_caps,
            "contradiction_penalty": contradiction_penalty,
            "contradiction_diagnostics": contradiction_diagnostics,
        },
        "match_explanation": {
            "strong": strong_reasons,
            "partial": partial_reasons,
            "missing": missing_required,
            "constraints": constraint_notes,
            "concerns": concerns,
            "experience_reconciliation": {
                key: components.get(key)
                for key in (
                    "reported_experience_years",
                    "calculated_experience_years",
                    "reconciled_experience_years",
                    "experience_reconciliation_status",
                )
            },
        },
    }


def _hybrid_score(
    signals: dict[str, Any],
    job_title: str,
    job_description: str,
    job_requirements: Any,
    job_skills: Any,
    semantic_score: float | None,
    intelligence: dict[str, Any] | None = None,
    job_metadata: dict[str, Any] | None = None,
    candidate_years: float | None = None,
    semantic_role_score: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Compute a weighted hybrid ranking score for one job.

    ``final`` is a normalized matching/ranking signal, not a probability of
    hiring or a calibrated probability of relevance. Recommendation rank may
    additionally apply the explicit target-role priority policy.
    Integrates candidate intelligence (evidence quality + hard constraints).
    Returns (final_score, component_scores).
    """
    job_text = _job_text(job_title, job_description, job_requirements, job_skills)

    tr_score = _target_role_score(
        signals["target_roles"], job_title, job_text, semantic_role_score
    )
    score_required_skills = (
        job_skills
        if isinstance(job_skills, list) and job_skills
        else _extract_job_required_skills(job_text)
    )
    sk_score = _evidence_weighted_skills_score(
        signals["skills"], job_text, intelligence, score_required_skills
    )
    job_for_responsibilities = job_metadata or {}
    history_score = _experience_score(
        signals["past_roles"],
        job_title,
        job_text,
        signals.get("responsibilities"),
        _job_responsibilities(job_for_responsibilities),
    )
    years = candidate_years
    if years is None:
        years = signals.get("total_experience_years")
    experience_fit = _experience_fit_score(years, job_text)
    preference_score, preference_components = _preference_score(
        signals, job_metadata or {}
    )
    sem_score = (
        max(0.0, min(1.0, float(semantic_score))) if semantic_score is not None else 0.0
    )
    responsibility_relevance = _responsibility_score(
        signals.get("responsibilities") or [],
        _job_responsibilities(job_metadata or {}),
    )

    final = (
        W_TARGET_ROLE * tr_score
        + W_SKILLS * sk_score
        + W_EXPERIENCE_FIT * experience_fit
        + W_RELEVANT_HISTORY * history_score
        + W_PREFERENCES * preference_score
        + W_SEMANTIC * sem_score
    )

    constraint_profile = (intelligence or {}).get("constraint_profile") or {}
    constraint_penalty, incompatibilities = _check_hard_constraints(
        constraint_profile, job_text
    )
    final *= constraint_penalty

    components: dict[str, Any] = {
        "score_semantics": "normalized_hybrid_ranking_score",
        "ranking_policy": (
            "target_role_priority_then_hybrid_score"
            if signals.get("has_explicit_target_roles")
            else "hybrid_score"
        ),
        "target_role_score": round(tr_score, 4),
        "semantic_role_score": (
            round(max(0.0, min(1.0, float(semantic_role_score))), 4)
            if semantic_role_score is not None
            else None
        ),
        "skills_score": round(sk_score, 4),
        "experience_score": round(history_score, 4),
        "responsibility_relevance": (
            round(responsibility_relevance, 4)
            if responsibility_relevance is not None
            else None
        ),
        "evidence_availability": {
            "experience_fit": years is not None
            and bool(_job_experience_bounds(job_text) != (None, None)),
            "preferences": bool(preference_components),
            "responsibility": responsibility_relevance is not None,
            "semantic": semantic_score is not None,
        },
        "experience_fit_score": round(experience_fit, 4),
        "reported_experience_years": signals.get("reported_experience_years"),
        "calculated_experience_years": signals.get("calculated_experience_years"),
        "reconciled_experience_years": signals.get("reconciled_experience_years"),
        "experience_reconciliation_status": signals.get(
            "experience_reconciliation_status"
        ),
        "preference_score": round(preference_score, 4),
        "preference_components": {
            key: round(value, 4) for key, value in preference_components.items()
        },
        "semantic_score": round(sem_score, 4),
        "constraint_penalty": round(constraint_penalty, 4),
        "final_score": round(final, 4),
    }
    required_skills = _extract_job_required_skills(job_text)
    if not required_skills and isinstance(job_skills, list):
        required_skills = [
            _normalize_text(
                skill.get("name") or skill.get("title") or skill.get("skill")
            )
            if isinstance(skill, dict)
            else _normalize_text(skill)
            for skill in job_skills
        ]
    candidate_skills = signals.get("skills") or []
    components["missing_required_skills"] = [
        skill
        for skill in required_skills
        if skill
        and not any(
            canonical_skill_key(skill) == canonical_skill_key(candidate_skill)
            for candidate_skill in candidate_skills
            if candidate_skill
        )
    ]
    if incompatibilities:
        components["incompatibilities"] = incompatibilities

    # Job-specific confidence layer (built on top of hybrid score, not replacing it)
    job_confidence = _compute_job_specific_confidence(
        signals, job_title, job_text, final, components, intelligence
    )
    components.update(job_confidence)

    return final, components


async def _persist_candidate_recommendations(
    db,
    *,
    candidate_id: str,
    owner_token: str,
    lease_generation: int,
    snapshot_revision: int,
    ranked_jobs: list[tuple[str, float, dict]],
    experience_ineligible_job_ids: set[str],
) -> bool:
    """Persist one fenced refresh using the caller-owned transaction."""
    candidate_result = await db.execute(
        text("SELECT id FROM candidates WHERE id = :cid FOR UPDATE"),
        {"cid": candidate_id},
    )
    if not candidate_result.mappings().fetchone():
        return False

    lease_result = await db.execute(
        text("""
            SELECT generation, owner_token, lease_expires_at
            FROM candidate_recommendation_refresh_leases
            WHERE candidate_id = :cid
            FOR UPDATE
        """),
        {"cid": candidate_id},
    )
    lease_row = lease_result.mappings().fetchone()
    revision_result = await db.execute(
        text("SELECT matching_profile_revision FROM candidates WHERE id = :cid"),
        {"cid": candidate_id},
    )
    current_revision = revision_result.scalar()
    if (
        not lease_row
        or str(lease_row["owner_token"]) != owner_token
        or int(lease_row["generation"]) != lease_generation
        or lease_row["lease_expires_at"] <= datetime.now(UTC)
        or int(current_revision or 0) != int(snapshot_revision)
    ):
        return False

    existing_rows = await db.execute(
        text("""
            SELECT job_id, id, tracked_at, hidden_at
            FROM candidate_job_recommendations
            WHERE candidate_id = :cid
        """),
        {"cid": candidate_id},
    )
    existing = {
        str(row[0]): {
            "id": str(row[1]),
            "tracked_at": row[2],
            "hidden_at": row[3],
        }
        for row in existing_rows.fetchall()
    }

    import json as _json

    for stale_job_id in experience_ineligible_job_ids:
        await db.execute(
            text("""
                UPDATE candidate_job_recommendations
                SET hidden_at = COALESCE(hidden_at, NOW()), generated_at = NOW()
                WHERE candidate_id = :cid AND job_id = :jid
            """),
            {"cid": candidate_id, "jid": stale_job_id},
        )
    for rank, (job_id, score, components) in enumerate(ranked_jobs, start=1):
        existing_row = existing.get(job_id)
        match_reason = _json.dumps({"type": "hybrid_match", **components})
        if existing_row:
            await db.execute(
                text("""
                    UPDATE candidate_job_recommendations
                    SET match_score = :score,
                        recommendation_rank = :rank,
                        match_reason = CAST(:match_reason AS jsonb),
                        generated_at = now()
                    WHERE id = :rid
                """),
                {
                    "score": score,
                    "rank": rank,
                    "rid": existing_row["id"],
                    "match_reason": match_reason,
                },
            )
        else:
            await db.execute(
                text("""
                    INSERT INTO candidate_job_recommendations
                        (id, candidate_id, job_id, match_score, recommendation_rank,
                         match_reason, created_at, generated_at)
                    VALUES
                        (:id, :cid, :jid, :score, :rank,
                         CAST(:match_reason AS jsonb), now(), now())
                """),
                {
                    "id": str(uuid.uuid4()),
                    "cid": candidate_id,
                    "jid": job_id,
                    "score": score,
                    "rank": rank,
                    "match_reason": match_reason,
                },
            )

    await db.execute(
        text("""
            UPDATE candidates
            SET last_full_recommendation_refresh_at = NOW(),
                last_full_recommendation_revision = :revision
            WHERE id = :cid
        """),
        {"cid": candidate_id, "revision": snapshot_revision},
    )
    return True


async def refresh_candidate_job_matches(
    candidate_id: str,
    candidate: dict[str, Any],
    SessionLocal: async_sessionmaker,
) -> bool:
    """
    Build candidate embedding, search Qdrant, re-rank with hybrid scoring,
    and upsert into candidate_job_recommendations.
    """
    owner_token = str(uuid.uuid4())
    lease_generation: int | None = None
    snapshot_revision: int | None = None

    # Read authoritative matching inputs before starting expensive work. The
    # caller-provided dictionary is retained only as a compatibility fallback
    # for test adapters that do not expose a real candidates table.
    async with SessionLocal() as db:
        result = await db.execute(
            text("SELECT * FROM candidates WHERE id = :cid LIMIT 1"),
            {"cid": candidate_id},
        )
        authoritative = result.mappings().fetchone()
        if not authoritative:
            await db.rollback()
            return False
        candidate = dict(authoritative)
        prefs_result = await db.execute(
            text(
                "SELECT * FROM candidate_preferences WHERE candidate_id = :cid LIMIT 1"
            ),
            {"cid": candidate_id},
        )
        prefs = prefs_result.mappings().fetchone()
        candidate["_prefs_row"] = dict(prefs) if prefs else None
        snapshot_revision = int(candidate.get("matching_profile_revision") or 0)
        lease = await acquire_refresh_lease(
            db, candidate_id, owner_token, REFRESH_LEASE_SECONDS
        )
        await db.commit()
    if not lease.acquired:
        logger.info("[matching] Refresh contention for candidate %s", candidate_id)
        return False
    lease_generation = lease.generation
    if lease_generation is None:
        logger.error("[matching] Acquired refresh lease without a generation")
        return False
    _refresh_lease_context.set(
        (SessionLocal, candidate_id, owner_token, lease_generation)
    )

    ready, readiness_reason = candidate_ready_for_matching(candidate)
    logger.info(
        "[matching] candidate=%s readiness=%s reason=%s",
        candidate_id,
        ready,
        readiness_reason,
    )
    if not ready:
        await _release_refresh_lease_safely(
            SessionLocal, candidate_id, owner_token, lease_generation
        )
        return False

    # Read target intent before retrieval; it is also reused by the existing
    # eligibility and hybrid ranking logic below.
    signals = _build_candidate_signals(candidate)
    opportunity_type = signals.get("opportunity_type", "jobs")
    candidate_text = build_candidate_text(candidate)
    if not candidate_text.strip():
        logger.info(
            "[matching] Candidate %s has no profile text — skipping", candidate_id
        )
        await _release_refresh_lease_safely(
            SessionLocal, candidate_id, owner_token, lease_generation
        )
        return False

    # Preserve the existing broad candidate-semantic retrieval.
    query_vector = generate_embedding(candidate_text)
    job_scores = search_job_chunks(query_vector, limit=QDRANT_TOP_K)
    broad_retrieval = job_scores
    broad_retrieval_ids = {job_id for job_id, _ in job_scores}
    broad_job_scores = list(job_scores)
    logger.info(
        "[matching] candidate=%s broad_qdrant_results=%d raw_hits=%d "
        "unique_job_ids=%d duplicate_points_removed=%d",
        candidate_id,
        len(job_scores),
        getattr(getattr(job_scores, "diagnostics", None), "raw_hits", len(job_scores)),
        len(job_scores),
        getattr(
            getattr(job_scores, "diagnostics", None), "duplicate_points_removed", 0
        ),
    )

    # An explicit preferred role receives an independent, intent-only path.
    # Current role, skills, and resume history are intentionally excluded so a
    # career transition cannot be redirected by past experience before ranking.
    target_role_scores: list[tuple[str, float]] = []
    if signals["has_explicit_target_roles"]:
        target_role_query = signals.get(
            "_target_role_query"
        ) or _build_target_role_retrieval_text(signals["target_roles"])
        logger.info(
            "[matching] candidate=%s target-role retrieval query=%r "
            "normalized_preferred_roles=%r",
            candidate_id,
            target_role_query,
            signals["target_roles"],
        )
        target_role_scores = search_job_chunks(
            generate_embedding(target_role_query),
            limit=QDRANT_TOP_K,
        )
        logger.info(
            "[matching] candidate=%s target_role_qdrant_results=%d raw_hits=%d "
            "unique_job_ids=%d duplicate_points_removed=%d",
            candidate_id,
            len(target_role_scores),
            getattr(
                getattr(target_role_scores, "diagnostics", None),
                "raw_hits",
                len(target_role_scores),
            ),
            len(target_role_scores),
            getattr(
                getattr(target_role_scores, "diagnostics", None),
                "duplicate_points_removed",
                0,
            ),
        )
    # Keep the original SearchResults object so its RetrievalDiagnostics remain
    # available after job_scores is replaced by the merged ranked list below.
    target_role_retrieval = target_role_scores
    target_role_retrieval_ids = {job_id for job_id, _ in target_role_scores}
    # Reuse the independent role-intent retrieval scores during eligibility
    # and hybrid scoring; this avoids re-embedding every retrieved job.
    signals["_target_role_semantic_scores"] = {
        job_id: score
        for job_id, score in target_role_scores
        if _retrieval_score_is_compatible(
            target_role_scores,
            job_id,
            allow_legacy_adapter=not hasattr(target_role_scores, "metadata_by_job_id"),
        )
    }

    # Merge both paths by ID and retain the strongest semantic score.  Hybrid
    # scoring itself remains exactly as it was.
    job_scores, merged_retrieval_diagnostics = merge_search_results(
        broad_retrieval, target_role_retrieval
    )
    logger.info(
        "[matching] candidate=%s retrieval_diagnostics "
        "broad_raw_hits=%d broad_valid_job_ids=%d broad_invalid_job_ids=%d "
        "broad_unique_job_ids=%d broad_duplicate_points_removed=%d "
        "target_role_raw_hits=%d target_role_valid_job_ids=%d "
        "target_role_invalid_job_ids=%d target_role_unique_job_ids=%d "
        "target_role_duplicate_points_removed=%d "
        "cross_channel_duplicate_job_ids=%d final_merged_unique_job_ids=%d",
        candidate_id,
        merged_retrieval_diagnostics.broad.raw_hits,
        merged_retrieval_diagnostics.broad.valid_job_ids,
        merged_retrieval_diagnostics.broad.invalid_job_ids,
        merged_retrieval_diagnostics.broad.unique_job_ids,
        merged_retrieval_diagnostics.broad.duplicate_points_removed,
        merged_retrieval_diagnostics.target_role.raw_hits,
        merged_retrieval_diagnostics.target_role.valid_job_ids,
        merged_retrieval_diagnostics.target_role.invalid_job_ids,
        merged_retrieval_diagnostics.target_role.unique_job_ids,
        merged_retrieval_diagnostics.target_role.duplicate_points_removed,
        merged_retrieval_diagnostics.cross_channel_duplicate_job_ids,
        merged_retrieval_diagnostics.final_merged_unique_job_ids,
    )
    if not job_scores:
        logger.info("[matching] No Qdrant results for candidate %s", candidate_id)
        empty_results = [broad_retrieval]
        if signals["has_explicit_target_roles"]:
            empty_results.append(target_role_retrieval)
        if not _healthy_empty_retrieval(*empty_results):
            logger.warning(
                "[matching] Empty retrieval is not complete for candidate %s; "
                "diagnostics are missing or degraded",
                candidate_id,
            )
            return False
        async with SessionLocal() as db:
            valid = await _refresh_persistence_is_current(
                db, candidate_id, owner_token, lease_generation, snapshot_revision
            )
            if not valid:
                await db.rollback()
                await _release_refresh_lease_safely(
                    SessionLocal, candidate_id, owner_token, lease_generation
                )
                return False
            await db.execute(
                text("""
                    UPDATE candidates
                    SET last_full_recommendation_refresh_at = NOW(),
                        last_full_recommendation_revision = :revision
                    WHERE id = :cid
                """),
                {"cid": candidate_id, "revision": snapshot_revision},
            )
            await db.commit()
        await _release_refresh_lease_safely(
            SessionLocal, candidate_id, owner_token, lease_generation
        )
        return True

    # Keep broad candidate-to-job semantic similarity separate from the
    # independent target-role semantic signal to avoid double-counting it.
    semantic_map = {
        jid: score
        for jid, score in broad_job_scores
        if _retrieval_score_is_compatible(
            broad_retrieval,
            jid,
            allow_legacy_adapter=not hasattr(broad_retrieval, "metadata_by_job_id"),
        )
    }
    candidate_job_ids = [jid for jid, _ in job_scores]
    logger.info(
        "[matching] candidate=%s retrieval_diagnostics broad_unique=%d "
        "target_role_unique=%d overlap=%d combined_unique=%d",
        candidate_id,
        len(broad_retrieval_ids),
        len(target_role_retrieval_ids),
        len(broad_retrieval_ids & target_role_retrieval_ids),
        len(candidate_job_ids),
    )
    candidate_years = _candidate_total_experience_years(candidate)

    # 2. Fetch job details for all candidates
    async with SessionLocal() as db:
        if not await _renew_refresh_lease_if_needed(
            db, candidate_id, owner_token, lease_generation
        ):
            await db.rollback()
            await _release_refresh_lease_safely(
                SessionLocal, candidate_id, owner_token, lease_generation
            )
            return False
        await db.commit()
        placeholders = ", ".join(f":jid_{i}" for i in range(len(candidate_job_ids)))
        params = {f"jid_{i}": jid for i, jid in enumerate(candidate_job_ids)}
        rows = await db.execute(
            text(f"""
                SELECT id, title, description, requirements, skills, company_name,
                       department, location, employment_type, experience_required,
                       salary_range, city, state, country, remote, industry,
                       skills_required, structured_data, remote_policy,
                       opportunity_type, responsibilities
                FROM job_descriptions
                WHERE id::text IN ({placeholders})
                  AND {candidate_visible_where("job_descriptions")}
                  AND COALESCE(job_descriptions.opportunity_type, 'jobs') = "
                  ":opportunity_type
            """),
            {**params, "opportunity_type": opportunity_type},
        )
        job_details = {
            str(r[0]): {
                "title": r[1] or "",
                "description": r[2] or "",
                # ATS normalizers may place experience and skills in either the
                # legacy columns or their normalized counterparts.  Feed both to
                # the existing text/evidence routines without changing storage.
                "requirements": " ".join(str(value) for value in (r[3], r[9]) if value),
                "skills": [*(r[4] or []), *(r[16] or [])]
                if isinstance(r[4], list) and isinstance(r[16], list)
                else (r[4] or r[16] or []),
                "company_name": r[5] or "",
                "department": r[6] or "",
                "location": r[7] or "",
                "employment_type": r[8] or "",
                "experience_required": r[9] or "",
                "salary_range": r[10] or "",
                "city": r[11] or "",
                "state": r[12] or "",
                "country": r[13] or "",
                "remote": r[14],
                "industry": r[15] or "",
                "skills_required": r[16] or [],
                "structured_data": r[17] or {},
                "remote_policy": r[18] or "",
                # Legacy test/session adapters and pre-migration rows do not expose
                # the new column; those are ordinary jobs by definition.
                "opportunity_type": (r[19] if len(r) > 19 else None) or "jobs",
                "responsibilities": r[20] if len(r) > 20 else None,
            }
            for r in rows.fetchall()
        }

    if not job_details:
        logger.info(
            "[matching] candidate=%s active DB overlap=0 "
            "discarded_missing_or_inactive=%d",
            candidate_id,
            len(candidate_job_ids),
        )
        logger.warning(
            "[matching] Incomplete refresh for candidate %s: Qdrant returned %d "
            "job IDs but no usable database jobs were found",
            candidate_id,
            len(candidate_job_ids),
        )
        await _release_refresh_lease_safely(
            SessionLocal, candidate_id, owner_token, lease_generation
        )
        return False
    logger.info(
        "[matching] candidate=%s active DB overlap=%d discarded_missing_or_inactive=%d",
        candidate_id,
        len(job_details),
        len(candidate_job_ids) - len(job_details),
    )
    logger.info(
        "[matching] candidate=%s eligibility_input_jobs=%d",
        candidate_id,
        len(job_details),
    )

    # 3. Compute candidate intelligence once (evidence quality + constraints)
    intelligence = _get_candidate_intelligence(candidate)

    # 4. Hybrid re-ranking
    scored: list[tuple[str, float, dict]] = []
    rejected_experience = 0
    retrieved_job_ids_not_returned_by_database_query = len(candidate_job_ids) - len(
        job_details
    )
    experience_ineligible_job_ids: set[str] = set()
    rejected_skills_role = 0
    rejected_preferences = 0
    for index, (job_id, job_data) in enumerate(job_details.items(), start=1):
        if index == 1 or index % 25 == 0:
            async with SessionLocal() as lease_db:
                if not await _renew_refresh_lease_if_needed(
                    lease_db, candidate_id, owner_token, lease_generation
                ):
                    await lease_db.rollback()
                    return False
                await lease_db.commit()
        job_text = _job_text(
            job_data["title"],
            job_data["description"],
            job_data["requirements"],
            job_data["skills"],
        )
        preference_eligible, ineligibility = _preference_eligibility(
            signals, job_data, candidate_years
        )
        if not preference_eligible:
            if "experience_outside_required_range" in ineligibility:
                rejected_experience += 1
                experience_ineligible_job_ids.add(job_id)
            else:
                rejected_preferences += 1
            continue
        raw_required_skills = job_data["skills_required"]
        required_skills = _normalize_structured_required_skills(raw_required_skills)
        if raw_required_skills is None or raw_required_skills == []:
            required_skills = _extract_job_required_skills(job_text)
        role_semantic_score = signals.get("_target_role_semantic_scores", {}).get(
            job_id
        )
        responsibility_relevance = _responsibility_score(
            signals.get("responsibilities") or [], _job_responsibilities(job_data)
        )
        if not _job_passes_skills_or_role(
            signals,
            job_data["title"],
            job_text,
            required_skills,
            role_semantic_score,
            responsibility_relevance,
        ):
            rejected_skills_role += 1
            continue
        sem = semantic_map.get(job_id)
        final, components = _hybrid_score(
            signals,
            job_data["title"],
            job_data["description"],
            job_data["requirements"],
            job_data["skills"],
            sem,
            intelligence=intelligence,
            job_metadata=job_data,
            candidate_years=candidate_years,
            semantic_role_score=role_semantic_score,
        )
        role_family_alignment = _best_role_family_alignment(
            signals["target_roles"], job_data["title"]
        )
        components["role_family_relationship"] = role_family_alignment["relationship"]
        components["role_family_priority"] = role_family_alignment["priority"]
        scored.append((job_id, final, components))
        logger.info(
            "[job-match] job=%r target_role_score=%.4f skills_score=%.4f "
            "experience_score=%.4f semantic_score=%.4f final_score=%.4f",
            job_data["title"],
            components["target_role_score"],
            components["skills_score"],
            components["experience_score"],
            components["semantic_score"],
            components["final_score"],
        )

    # When candidates state a target role, preserve that intent as the job
    # direction. Skills and experience still decide qualification and score,
    # but a current-role/skill match cannot displace an available target-role
    # match. Fall back to the broader eligible set only if Qdrant returned no
    # target-role matches.
    if signals["has_explicit_target_roles"]:
        intent_scored = [
            item for item in scored if item[2].get("target_role_score", 0.0) > 0.0
        ]
        if intent_scored:
            scored = intent_scored

    # Selection priority is deliberately separate from the match score shown
    # to candidates.  When explicit target-role matches exist, choose the
    # strongest target-role matches first, then retain the unchanged hybrid
    # score as the tie-breaker.  Without explicit intent, preserve the legacy
    # final-score ordering.
    _sort_scored_recommendations(scored, signals, job_details)
    ranked_jobs = scored[:MAX_RECOMMENDATIONS]
    logger.info(
        "[matching] candidate=%s passing_eligibility=%d rejected_experience=%d "
        "rejected_skills_or_role=%d rejected_explicit_preferences=%d "
        "retrieved_job_ids_not_returned_by_database_query=%d "
        "rejected_wrong_opportunity_type=unknown_unavailable "
        "final_recommendations=%d",
        candidate_id,
        len(scored),
        rejected_experience,
        rejected_skills_role,
        rejected_preferences,
        retrieved_job_ids_not_returned_by_database_query,
        len(ranked_jobs),
    )

    async with SessionLocal() as db:
        persisted = await _persist_candidate_recommendations(
            db,
            candidate_id=candidate_id,
            owner_token=owner_token,
            lease_generation=lease_generation,
            snapshot_revision=int(snapshot_revision or 0),
            ranked_jobs=ranked_jobs,
            experience_ineligible_job_ids=experience_ineligible_job_ids,
        )
        if not persisted:
            await db.rollback()
            await _release_refresh_lease_safely(
                SessionLocal, candidate_id, owner_token, lease_generation
            )
            return False
        await db.commit()

    await _release_refresh_lease_safely(
        SessionLocal, candidate_id, owner_token, lease_generation
    )
    return True


async def _renew_refresh_lease_if_needed(
    SessionLocal, candidate_id, owner_token, lease_generation
):
    return await renew_refresh_lease(
        SessionLocal, candidate_id, owner_token, lease_generation, REFRESH_LEASE_SECONDS
    )


async def _refresh_persistence_is_current(
    db, candidate_id, owner_token, lease_generation, snapshot_revision
):
    await db.execute(
        text("SELECT id FROM candidates WHERE id = :cid FOR UPDATE"),
        {"cid": candidate_id},
    )
    result = await db.execute(
        text("""
        SELECT generation, owner_token, lease_expires_at
        FROM candidate_recommendation_refresh_leases
        WHERE candidate_id = :cid
        FOR UPDATE
    """),
        {"cid": candidate_id},
    )
    lease = result.mappings().fetchone()
    revision = await db.execute(
        text("SELECT matching_profile_revision FROM candidates WHERE id = :cid"),
        {"cid": candidate_id},
    )
    current_revision = revision.scalar()
    return bool(
        lease
        and str(lease["owner_token"]) == owner_token
        and int(lease["generation"]) == lease_generation
        and lease["lease_expires_at"] > datetime.now(UTC)
        and int(current_revision or 0) == int(snapshot_revision or 0)
    )


async def _release_refresh_lease_safely(
    SessionLocal, candidate_id, owner_token, lease_generation
):
    if lease_generation is None:
        return
    try:
        async with SessionLocal() as db:
            await release_refresh_lease(db, candidate_id, owner_token, lease_generation)
            await db.commit()
    except Exception:
        logger.exception(
            "[matching] Failed to release refresh lease for %s", candidate_id
        )


async def refresh_candidate_job_match(
    candidate_id: str,
    recommendation_id: str,
    candidate: dict[str, Any],
    SessionLocal: async_sessionmaker,
) -> None:
    """Re-score one existing recommendation without retrieval or insertion.

    This intentionally reuses the established hybrid scorer and the semantic
    component recorded on the recommendation. It never changes daily access,
    subscription behaviour, recommendation selection, or Qdrant retrieval.
    """
    import json as _json

    async with SessionLocal() as db:
        row = await db.execute(
            text("""
            SELECT cjr.match_reason, jd.title, jd.description, jd.requirements,
                   jd.skills, jd.skills_required, jd.experience_required,
                   jd.location, jd.employment_type, jd.remote, jd.remote_policy,
                   jd.industry, jd.salary_range, jd.city, jd.state, jd.country,
                   jd.structured_data
            FROM candidate_job_recommendations cjr
            JOIN job_descriptions jd ON jd.id = cjr.job_id
            WHERE cjr.id = :rid AND cjr.candidate_id = :cid
            LIMIT 1
        """),
            {"rid": recommendation_id, "cid": candidate_id},
        )
        selected = row.mappings().fetchone()
    if not selected:
        return
    reason = selected["match_reason"] or {}
    if isinstance(reason, str):
        try:
            reason = _json.loads(reason)
        except Exception:
            reason = {}
    # Persisted semantic scores belong to the previous candidate/job snapshot
    # and have no provenance that can validate freshness. Recompute both
    # semantic signals from the current inputs instead of treating old values
    # as current evidence.
    semantic_score = None
    semantic_role_score = None
    signals = _build_candidate_signals(candidate)
    # Use the same normalized job inputs as the full recommendation refresh.
    # Omitting metadata here reset preference/experience components whenever a
    # resume was saved, which could make a genuinely added required skill look
    # like it lowered the match.
    job_requirements = " ".join(
        str(value)
        for value in (selected["requirements"], selected["experience_required"])
        if value
    )
    legacy_skills = selected["skills"] if isinstance(selected["skills"], list) else []
    required_skills = (
        selected["skills_required"]
        if isinstance(selected["skills_required"], list)
        else []
    )
    job_skills = _combined_job_skills(
        {"skills": legacy_skills, "skills_required": required_skills}
    )
    job_metadata = {
        "title": selected["title"] or "",
        "description": selected["description"] or "",
        "requirements": job_requirements,
        "skills": job_skills,
        "skills_required": selected["skills_required"] or [],
        "experience_required": selected["experience_required"],
        "location": selected["location"] or "",
        "employment_type": selected["employment_type"] or "",
        "remote": selected["remote"],
        "remote_policy": selected["remote_policy"] or "",
        "industry": selected["industry"] or "",
        "salary_range": selected["salary_range"] or "",
        "city": selected["city"] or "",
        "state": selected["state"] or "",
        "country": selected["country"] or "",
        "structured_data": selected["structured_data"] or {},
    }
    if signals.get("target_roles"):
        semantic_role_score = _semantic_role_score(
            signals["target_roles"],
            build_job_text(job_metadata),
            signals.get("_target_role_query"),
        )
    semantic_score = _semantic_similarity_score(
        build_candidate_text(candidate),
        build_job_text(job_metadata),
    )
    reconciliation = _candidate_experience_reconciliation(candidate)
    experience = experience_eligibility(
        reconciliation["reconciled_experience_years"],
        _job_text(
            selected["title"] or "",
            selected["description"] or "",
            job_requirements,
            job_skills,
        ),
        selected["experience_required"],
        reconciliation["experience_reconciliation_status"],
    )
    if not experience["eligible"]:
        async with SessionLocal() as db:
            await db.execute(
                text("""
                UPDATE candidate_job_recommendations
                SET hidden_at = COALESCE(hidden_at, NOW()), generated_at = NOW()
                WHERE id = :rid AND candidate_id = :cid
            """),
                {"rid": recommendation_id, "cid": candidate_id},
            )
            await db.commit()
        return
    preference_eligible, preference_reasons = _preference_eligibility(
        signals,
        job_metadata,
        reconciliation["reconciled_experience_years"],
    )
    if not preference_eligible:
        async with SessionLocal() as db:
            await db.execute(
                text("""
                UPDATE candidate_job_recommendations
                SET hidden_at = COALESCE(hidden_at, NOW()), generated_at = NOW()
                WHERE id = :rid AND candidate_id = :cid
            """),
                {"rid": recommendation_id, "cid": candidate_id},
            )
            await db.commit()
        logger.info(
            "[matching] Stored recommendation hidden after preference revalidation "
            "candidate=%s recommendation=%s reasons=%s",
            candidate_id,
            recommendation_id,
            preference_reasons,
        )
        return
    raw_required_skills = selected["skills_required"]
    required_skills = _normalize_structured_required_skills(raw_required_skills)
    if raw_required_skills is None or raw_required_skills == []:
        required_skills = _extract_job_required_skills(
            _job_text(
                selected["title"] or "",
                selected["description"] or "",
                job_requirements,
                job_skills,
            )
        )
    responsibility_relevance = _responsibility_score(
        signals.get("responsibilities") or [],
        _job_responsibilities(job_metadata),
    )
    if not _job_passes_skills_or_role(
        signals,
        selected["title"] or "",
        _job_text(
            selected["title"] or "",
            selected["description"] or "",
            job_requirements,
            job_skills,
        ),
        required_skills,
        semantic_role_score,
        responsibility_relevance,
    ):
        async with SessionLocal() as db:
            await db.execute(
                text("""
                UPDATE candidate_job_recommendations
                SET hidden_at = COALESCE(hidden_at, NOW()), generated_at = NOW()
                WHERE id = :rid AND candidate_id = :cid
            """),
                {"rid": recommendation_id, "cid": candidate_id},
            )
            await db.commit()
        logger.info(
            "[matching] Stored recommendation hidden after skills/role revalidation "
            "candidate=%s recommendation=%s",
            candidate_id,
            recommendation_id,
        )
        return
    score, components = _hybrid_score(
        signals,
        selected["title"] or "",
        selected["description"] or "",
        job_requirements,
        job_skills,
        semantic_score,
        intelligence=_get_candidate_intelligence(candidate),
        job_metadata=job_metadata,
        candidate_years=reconciliation["reconciled_experience_years"],
        semantic_role_score=semantic_role_score,
    )
    async with SessionLocal() as db:
        await db.execute(
            text("""
            UPDATE candidate_job_recommendations
            SET match_score = :score,
                match_reason = CAST(:reason AS jsonb),
                generated_at = now()
            WHERE id = :rid AND candidate_id = :cid
        """),
            {
                "score": score,
                "reason": _json.dumps({"type": "hybrid_match", **components}),
                "rid": recommendation_id,
                "cid": candidate_id,
            },
        )
        await db.commit()


async def refresh_stale_candidate_job_scores(
    candidate_id: str,
    candidate: dict[str, Any],
    SessionLocal: async_sessionmaker,
    recommendation_id: str | None = None,
    opportunity_type: str | None = None,
) -> None:
    """Re-score stored recommendations whose authoritative job row changed.

    Job ingestion/update paths maintain ``job_descriptions.updated_at``. A
    stored score is current only when it was generated after that timestamp.
    Reuse the single-recommendation refresh path so current eligibility gates
    and the hybrid scorer remain the sole source of truth.
    """
    predicates = [
        "cjr.candidate_id = :cid",
        "cjr.hidden_at IS NULL",
        candidate_visible_where("jd"),
        "(cjr.generated_at IS NULL OR (jd.updated_at IS NOT NULL AND "
        "jd.updated_at > cjr.generated_at))",
    ]
    params: dict[str, Any] = {"cid": candidate_id}
    if recommendation_id is not None:
        predicates.append("cjr.id = :rid")
        params["rid"] = recommendation_id
    if opportunity_type is not None:
        predicates.append("COALESCE(jd.opportunity_type, 'jobs') = :opportunity_type")
        params["opportunity_type"] = opportunity_type
    async with SessionLocal() as db:
        result = await db.execute(
            text(f"""
                SELECT cjr.id, cjr.generated_at, jd.updated_at
                FROM candidate_job_recommendations cjr
                JOIN job_descriptions jd ON jd.id = cjr.job_id
                WHERE {" AND ".join(predicates)}
            """),
            params,
        )
        recommendation_ids = []
        for row in result.fetchall():
            mapping = getattr(row, "_mapping", row)
            try:
                if isinstance(mapping, dict):
                    # Test/session adapters and legacy callers may not expose
                    # the freshness columns; do not rescore without evidence
                    # that the row is stale.
                    if "generated_at" not in mapping or "updated_at" not in mapping:
                        continue
                    value = mapping.get("id")
                else:
                    value = row[0]
            except (IndexError, KeyError, TypeError):
                value = None
            if value is not None:
                recommendation_ids.append(str(value))
    for current_recommendation_id in recommendation_ids:
        await refresh_candidate_job_match(
            candidate_id,
            current_recommendation_id,
            candidate,
            SessionLocal,
        )
