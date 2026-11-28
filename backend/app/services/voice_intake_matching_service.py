from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class VoiceIntakeMatchingDependencies:
    candidate_ready_for_matching: Callable[[dict], tuple[bool, str]]
    trigger_matching: Callable[[str], Any]
    schedule_task: Callable[[Any], Any]
    logger: logging.Logger


def voice_intake_completed_for_matching(
    candidate: dict,
    dependencies: VoiceIntakeMatchingDependencies,
) -> bool:
    """Return whether persisted profile data is usable for matching."""
    ready, reason = dependencies.candidate_ready_for_matching(candidate)
    dependencies.logger.info("[matching] API readiness=%s reason=%s", ready, reason)
    return ready


def schedule_voice_intake_matching(
    candidate_id: str,
    status: str,
    dependencies: VoiceIntakeMatchingDependencies,
) -> None:
    """Schedule matching only after voice intake reaches completed status."""
    if status == "completed":
        dependencies.schedule_task(dependencies.trigger_matching(candidate_id))
