"""Conservative, observability-only interpretation of role titles."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RoleAnalysis:
    normalized_title: str
    role_family: str | None
    role_families: tuple[str, ...]
    technology: str | None
    languages: tuple[str, ...]
    runtimes: tuple[str, ...]
    frameworks: tuple[str, ...]
    seniority: str | None
    seniority_indicators: tuple[str, ...]
    professional_track: str
    recognized_aliases: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "role_families",
            "languages",
            "runtimes",
            "frameworks",
            "seniority_indicators",
            "recognized_aliases",
        ):
            result[key] = list(getattr(self, key))
        return result


@dataclass(frozen=True)
class RoleRelationship:
    relationship: str
    candidate_role: RoleAnalysis
    job_role: RoleAnalysis
    role_family_match: str
    technology_compatibility: str
    seniority_compatibility: str
    professional_track_compatibility: str
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "relationship": self.relationship,
            "candidate_role": self.candidate_role.as_dict(),
            "job_role": self.job_role.as_dict(),
            "role_family_match": self.role_family_match,
            "technology_compatibility": self.technology_compatibility,
            "seniority_compatibility": self.seniority_compatibility,
            "professional_track_compatibility": self.professional_track_compatibility,
            "reason_codes": list(self.reason_codes),
        }


_FAMILY_PATTERNS = (
    ("fullstack", ("full stack", "fullstack")),
    ("backend", ("backend", "back end")),
    ("frontend", ("frontend", "front end")),
    ("mobile", ("mobile", "ios", "android")),
    ("devops", ("devops", "dev ops")),
    ("cloud", ("cloud",)),
    ("data", ("data scientist", "data engineer", "data analyst", "data")),
    ("qa", ("quality assurance", "qa", "test engineer", "tester")),
    ("security", ("cybersecurity", "cyber security", "security")),
    ("product", ("product manager", "product owner", "product")),
    ("design", ("designer", "design")),
    ("marketing", ("marketing",)),
    ("sales", ("sales",)),
)

_LANGUAGES = (
    ("Python", ("python",)),
    ("JavaScript", ("javascript", "java script", "js")),
    ("TypeScript", ("typescript", "type script", "ts")),
    ("C++", ("c++",)),
    ("C#", ("c#", "c sharp")),
    ("Java", ("java",)),
    ("Ruby", ("ruby",)),
    ("PHP", ("php",)),
    ("Go", ("golang", "go")),
)

_RUNTIMES = (("Node.js", ("node.js", "nodejs", "node js", "node")),)

_FRAMEWORKS = (
    ("React", ("react", "react.js", "reactjs")),
    ("Angular", ("angular",)),
    ("Vue", ("vue",)),
    ("Django", ("django",)),
    ("Flask", ("flask",)),
    ("FastAPI", ("fastapi", "fast api")),
    ("Spring Boot", ("spring boot", "springboot")),
    ("Spring", ("spring",)),
)

_SENIORITY = (
    ("intern", ("intern", "internship")),
    ("entry", ("entry level",)),
    ("junior", ("junior", "jr")),
    ("mid", ("mid level",)),
    ("senior", ("senior", "sr")),
    ("lead", ("lead",)),
    ("staff", ("staff",)),
    ("principal", ("principal",)),
)


def _empty_result() -> RoleAnalysis:
    return RoleAnalysis("", None, (), None, (), (), (), None, (), "unknown", ())


def _normalize_title(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().casefold().replace("–", "-").replace("—", "-")
    text = re.sub(r"[|/,;:_()\[\]]", " ", text)
    text = re.sub(r"\s*-\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term.casefold()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def _matches(text: str, patterns) -> tuple[str, ...]:
    matches = tuple(
        label
        for label, aliases in patterns
        if any(_contains_term(text, alias) for alias in aliases)
    )
    # Prefer the specific framework name over its shorter umbrella name.
    if "Spring Boot" in matches:
        matches = tuple(label for label in matches if label != "Spring")
    return matches


def _professional_track(title: str) -> str:
    architecture = any(
        _contains_term(title, term) for term in ("architect", "architecture")
    )
    management = any(
        _contains_term(title, term)
        for term in ("manager", "management", "director", "head of", "vp")
    )
    contributor = any(
        _contains_term(title, term)
        for term in ("developer", "engineer", "scientist", "analyst", "designer")
    )
    if architecture and management:
        return "ambiguous"
    if architecture:
        return "architecture"
    if management:
        return "management"
    if contributor:
        return "individual_contributor"
    return "unknown"


def _recognized_aliases(
    title: str, family: str | None, technology: str | None
) -> tuple[str, ...]:
    if family != "backend" or technology:
        return ()
    if title == "backend engineer":
        return ("backend developer",)
    if title == "backend developer":
        return ("backend engineer",)
    if title == "software engineer backend":
        return ("backend engineer", "backend developer")
    return ()


def analyze_role_title(title: Any) -> RoleAnalysis:
    """Analyze a string role title without making a matching decision."""
    normalized = _normalize_title(title)
    if not normalized:
        return _empty_result()
    families = _matches(normalized, _FAMILY_PATTERNS)
    languages = _matches(normalized, _LANGUAGES)
    runtimes = _matches(normalized, _RUNTIMES)
    frameworks = _matches(normalized, _FRAMEWORKS)
    seniority_indicators = _matches(normalized, _SENIORITY)
    seniority = seniority_indicators[0] if len(seniority_indicators) == 1 else None
    role_family = families[0] if len(families) == 1 else None
    all_technologies = languages + runtimes + frameworks
    technology = (
        all_technologies[0]
        if len(all_technologies) == 1
        else (languages[0] if languages else None)
    )
    return RoleAnalysis(
        normalized,
        role_family,
        families,
        technology,
        languages,
        runtimes,
        frameworks,
        seniority,
        seniority_indicators,
        _professional_track(normalized),
        _recognized_aliases(normalized, role_family, technology),
    )


def _technology_compatibility(candidate: RoleAnalysis, job: RoleAnalysis) -> str:
    candidate_values = set(
        candidate.languages + candidate.runtimes + candidate.frameworks
    )
    job_values = set(job.languages + job.runtimes + job.frameworks)
    if not candidate_values or not job_values:
        return "unknown"
    if candidate_values & job_values:
        return "compatible"
    language_overlap = set(candidate.languages) & set(job.languages)
    framework_overlap = set(candidate.frameworks) & set(job.frameworks)
    if language_overlap or framework_overlap:
        return "compatible"
    compatible_pairs = {
        frozenset(("Python", "FastAPI")),
        frozenset(("Python", "Django")),
        frozenset(("Python", "Flask")),
        frozenset(("Java", "Spring")),
        frozenset(("Java", "Spring Boot")),
        frozenset(("JavaScript", "TypeScript")),
        frozenset(("JavaScript", "React")),
        frozenset(("JavaScript", "Angular")),
        frozenset(("JavaScript", "Vue")),
        frozenset(("TypeScript", "React")),
        frozenset(("TypeScript", "Angular")),
        frozenset(("TypeScript", "Vue")),
    }
    if any(
        frozenset((left, right)) in compatible_pairs
        for left in candidate_values
        for right in job_values
    ):
        return "potentially_compatible"
    return "conflicting"


def _family_match(candidate: RoleAnalysis, job: RoleAnalysis) -> str:
    if not candidate.role_families or not job.role_families:
        return "unknown"
    if set(candidate.role_families) & set(job.role_families):
        return "same"
    return "different"


def _seniority_compatibility(candidate: RoleAnalysis, job: RoleAnalysis) -> str:
    if not candidate.seniority or not job.seniority:
        return "unknown"
    if candidate.seniority == job.seniority:
        return "compatible"
    return "conflicting"


def _track_compatibility(candidate: RoleAnalysis, job: RoleAnalysis) -> str:
    if candidate.professional_track == "unknown" or job.professional_track == "unknown":
        return "unknown"
    if candidate.professional_track == job.professional_track:
        return "compatible"
    return "conflicting"


def _is_api_title(role: RoleAnalysis) -> bool:
    return _contains_term(role.normalized_title, "api") and any(
        _contains_term(role.normalized_title, term)
        for term in ("developer", "engineer")
    )


def classify_role_relationship(
    candidate_title: Any, job_title: Any
) -> RoleRelationship:
    """Classify two titles conservatively without affecting production matching."""
    candidate = analyze_role_title(candidate_title)
    job = analyze_role_title(job_title)
    ambiguity_codes = (
        ("ambiguous_role_family",)
        if (len(candidate.role_families) > 1 or len(job.role_families) > 1)
        else ()
    )
    if not candidate.normalized_title or not job.normalized_title:
        return RoleRelationship(
            "unknown",
            candidate,
            job,
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            ("insufficient_role_evidence",),
        )
    if not candidate.role_families or not job.role_families:
        if _is_api_title(candidate) or _is_api_title(job):
            return RoleRelationship(
                "adjacent",
                candidate,
                job,
                "unknown",
                _technology_compatibility(candidate, job),
                _seniority_compatibility(candidate, job),
                _track_compatibility(candidate, job),
                ambiguity_codes + ("adjacent_role_title",),
            )
        technology = _technology_compatibility(candidate, job)
        if technology in {"compatible", "potentially_compatible"}:
            return RoleRelationship(
                "related",
                candidate,
                job,
                "unknown",
                technology,
                _seniority_compatibility(candidate, job),
                _track_compatibility(candidate, job),
                ambiguity_codes + ("related_role_title",),
            )
        return RoleRelationship(
            "unknown",
            candidate,
            job,
            "unknown",
            _technology_compatibility(candidate, job),
            _seniority_compatibility(candidate, job),
            _track_compatibility(candidate, job),
            ambiguity_codes + ("insufficient_role_evidence",),
        )
    family_match = _family_match(candidate, job)
    technology = _technology_compatibility(candidate, job)
    seniority = _seniority_compatibility(candidate, job)
    track = _track_compatibility(candidate, job)
    reasons: list[str] = []
    if len(candidate.role_families) > 1 or len(job.role_families) > 1:
        reasons.append("ambiguous_role_family")
    if candidate.seniority_indicators and not candidate.seniority:
        reasons.append("ambiguous_seniority")
    if job.seniority_indicators and not job.seniority:
        reasons.append("ambiguous_seniority")
    if track == "conflicting":
        reasons.append("professional_track_conflict")
    if seniority == "conflicting":
        reasons.append("seniority_conflict")
    if technology == "conflicting":
        reasons.append("conflicting_primary_technology")
    if "ambiguous_role_family" in reasons:
        return RoleRelationship(
            "unknown",
            candidate,
            job,
            family_match,
            technology,
            seniority,
            track,
            tuple(reasons + ["insufficient_role_evidence"]),
        )
    if family_match == "different":
        if set(candidate.role_families + job.role_families) == {"backend", "devops"}:
            return RoleRelationship(
                "transition",
                candidate,
                job,
                family_match,
                technology,
                seniority,
                track,
                tuple(reasons + ["related_role_title"]),
            )
        return RoleRelationship(
            "unrelated",
            candidate,
            job,
            family_match,
            technology,
            seniority,
            track,
            tuple(reasons or ["insufficient_role_evidence"]),
        )
    if reasons and (
        "professional_track_conflict" in reasons or "seniority_conflict" in reasons
    ):
        return RoleRelationship(
            "seniority_mismatch",
            candidate,
            job,
            family_match,
            technology,
            seniority,
            track,
            tuple(reasons),
        )
    if (
        candidate.normalized_title == job.normalized_title
        and technology != "conflicting"
    ):
        return RoleRelationship(
            "exact",
            candidate,
            job,
            family_match,
            technology,
            seniority,
            track,
            ("same_normalized_title", "same_role_family"),
        )
    if (
        job.normalized_title in candidate.recognized_aliases
        or candidate.normalized_title in job.recognized_aliases
    ):
        return RoleRelationship(
            "equivalent",
            candidate,
            job,
            family_match,
            technology,
            seniority,
            track,
            ("high_confidence_alias",),
        )
    if technology == "conflicting":
        return RoleRelationship(
            "different_technology",
            candidate,
            job,
            family_match,
            technology,
            seniority,
            track,
            ("same_role_family", "conflicting_primary_technology"),
        )
    if technology in {"compatible", "potentially_compatible"}:
        return RoleRelationship(
            "related",
            candidate,
            job,
            family_match,
            technology,
            seniority,
            track,
            ("same_role_family", "related_role_title"),
        )
    return RoleRelationship(
        "unknown",
        candidate,
        job,
        family_match,
        technology,
        seniority,
        track,
        tuple(reasons + ["insufficient_role_evidence"]),
    )


__all__ = [
    "RoleAnalysis",
    "RoleRelationship",
    "analyze_role_title",
    "classify_role_relationship",
]
