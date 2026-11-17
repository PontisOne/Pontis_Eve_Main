"""Schemas for standalone, field-specific candidate profile extraction."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TargetField = Literal["certifications", "skills", "education"]
ExtractionStatus = Literal["ok", "empty", "ambiguous", "invalid"]


class CertificationValue(BaseModel):
    """A candidate-named credential, course, or certification."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    issuer: str | None = None
    credential_type: Literal[
        "course", "formal_certification", "credential", "license", "unknown"
    ] = "unknown"


class EducationValue(BaseModel):
    """A degree or other education record explicitly stated by the candidate."""

    model_config = ConfigDict(extra="forbid")

    degree: str = Field(min_length=1)
    institution: str = Field(min_length=1)
    specialization: str | None = None
    graduation_year: int | None = Field(default=None, ge=1900, le=2200)
    status: Literal["completed", "pursuing", "discontinued", "unknown"] = "unknown"
    start_date: str = ""
    end_date: str = ""


class ExtractionResult(BaseModel):
    """Stable result contract for future chat-pipeline integration.

    ``confidence`` is informational metadata. Acceptance is determined by the
    extraction status and the field-specific schema, not by an arbitrary
    confidence threshold. It remains bounded to protect downstream consumers.
    """

    model_config = ConfigDict(extra="forbid")

    status: ExtractionStatus
    target_field: TargetField
    value: Any = None
    normalized: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_reason: str | None = None


class CertificationExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "empty", "ambiguous"]
    certification: CertificationValue | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_reason: str | None = None


class SkillsExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "empty", "ambiguous"]
    skills: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_reason: str | None = None


class EducationExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "empty", "ambiguous"]
    education: list[EducationValue] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_reason: str | None = None
