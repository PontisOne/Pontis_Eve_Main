from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class MatchingTriggerDependencies:
    """Runtime dependencies for refreshing one candidate's job matches."""

    get_candidate: Callable[[str], Awaitable[dict]]
    load_preferences: Callable[[str], Awaitable[dict | None]]
    refresh_candidate_job_matches: Callable[[str, dict, Any], Awaitable[Any]]
    session_local: Callable[..., Any]
    logger: logging.Logger


async def trigger_candidate_matching(
    candidate_id: str,
    dependencies: MatchingTriggerDependencies,
) -> None:
    """Refresh semantic job matches while preserving the legacy failure policy."""
    try:
        candidate = await dependencies.get_candidate(candidate_id)
        candidate["_prefs_row"] = await dependencies.load_preferences(candidate_id)
        await dependencies.refresh_candidate_job_matches(
            candidate_id,
            candidate,
            dependencies.session_local,
        )
    except Exception as exc:
        dependencies.logger.warning(
            "[matching] Failed for candidate %s: %s", candidate_id, exc
        )
