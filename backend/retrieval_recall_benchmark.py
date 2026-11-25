"""Deterministic, offline Recall@K benchmark for retrieval diagnostics.

This module is deliberately independent from production matching and does not
call Qdrant, PostgreSQL, or any external service.
"""

from collections.abc import Sequence
from dataclasses import dataclass

RECALL_KS = (50, 100, 150, 250, 500)


@dataclass(frozen=True)
class RetrievalChannelDiagnostics:
    raw_hits: int
    valid_job_ids: int
    invalid_job_ids: int
    unique_job_ids: int


@dataclass(frozen=True)
class MergedRetrievalDiagnostics:
    broad: RetrievalChannelDiagnostics
    target_role: RetrievalChannelDiagnostics
    cross_channel_duplicate_job_ids: int
    final_merged_unique_job_ids: int


def _reject_production_results(value: object) -> None:
    if hasattr(value, "diagnostics") and hasattr(value, "metadata_by_job_id"):
        raise TypeError(
            "benchmark inputs must be raw ranked Qdrant point IDs, not "
            "deduplicated SearchResults"
        )


def _channel(
    values: Sequence[str | None], k: int
) -> tuple[list[str], RetrievalChannelDiagnostics]:
    window = list(values[:k])
    valid = [value for value in window if isinstance(value, str) and value.strip()]
    unique = list(dict.fromkeys(valid))
    return unique, RetrievalChannelDiagnostics(
        raw_hits=len(window),
        valid_job_ids=len(valid),
        invalid_job_ids=len(window) - len(valid),
        unique_job_ids=len(unique),
    )


def merge_ranked_retrieval_channels(
    broad_ranked_point_ids: Sequence[str | None],
    target_role_ranked_point_ids: Sequence[str | None],
    *,
    k: int,
) -> tuple[list[str], MergedRetrievalDiagnostics]:
    """Apply each channel's raw top-K boundary, then merge unique job IDs."""
    _reject_production_results(broad_ranked_point_ids)
    _reject_production_results(target_role_ranked_point_ids)
    broad, broad_diagnostics = _channel(broad_ranked_point_ids, k)
    target, target_diagnostics = _channel(target_role_ranked_point_ids, k)
    merged = list(dict.fromkeys([*broad, *target]))
    diagnostics = MergedRetrievalDiagnostics(
        broad=broad_diagnostics,
        target_role=target_diagnostics,
        cross_channel_duplicate_job_ids=len(set(broad) & set(target)),
        final_merged_unique_job_ids=len(merged),
    )
    return merged, diagnostics


def merge_scored_retrieval_channels(
    broad_ranked_points: Sequence[tuple[str | None, float]],
    target_role_ranked_points: Sequence[tuple[str | None, float]],
    *,
    k: int,
) -> tuple[list[tuple[str, float]], MergedRetrievalDiagnostics]:
    """Benchmark the production merge using raw scored points per channel."""
    _reject_production_results(broad_ranked_points)
    _reject_production_results(target_role_ranked_points)

    def channel(points):
        window = list(points[:k])
        valid = [
            point for point in window
            if isinstance(point, tuple)
            and len(point) == 2
            and isinstance(point[0], str)
            and point[0].strip()
        ]
        best: dict[str, float] = {}
        for job_id, score in valid:
            best[job_id] = max(best.get(job_id, float("-inf")), float(score))
        return best, RetrievalChannelDiagnostics(
            len(window), len(valid), len(window) - len(valid), len(best)
        )

    broad, broad_diagnostics = channel(broad_ranked_points)
    target, target_diagnostics = channel(target_role_ranked_points)
    merged = dict(broad)
    for job_id, score in target.items():
        merged[job_id] = max(merged.get(job_id, float("-inf")), score)
    ranked = sorted(merged.items(), key=lambda item: item[1], reverse=True)
    return ranked, MergedRetrievalDiagnostics(
        broad_diagnostics,
        target_diagnostics,
        len(set(broad) & set(target)),
        len(ranked),
    )


@dataclass(frozen=True)
class BenchmarkJob:
    job_id: str
    relevant: bool | None = None
    active: bool = True
    visible: bool = True
    opportunity_type: str = "jobs"
    passes_eligibility: bool = True
    country_code: str | None = None


@dataclass(frozen=True)
class RecallResult:
    k: int
    labeled_relevant: int
    retrieved_relevant_before_filtering: int | None
    retrieved_relevant_after_filtering: int | None

    @property
    def recall_before_filtering(self) -> float | None:
        if self.retrieved_relevant_before_filtering is None:
            return None
        return (
            self.retrieved_relevant_before_filtering / self.labeled_relevant
            if self.labeled_relevant
            else 0.0
        )

    @property
    def recall_after_filtering(self) -> float | None:
        if self.retrieved_relevant_after_filtering is None:
            return None
        return (
            self.retrieved_relevant_after_filtering / self.labeled_relevant
            if self.labeled_relevant
            else 0.0
        )


def benchmark_recall(
    ranked_point_ids: Sequence[str | None],
    jobs: dict[str, BenchmarkJob],
    relevant_job_ids: set[str] | None,
    *,
    opportunity_type: str = "jobs",
    candidate_country_code: str | None = None,
    ks: tuple[int, ...] = RECALL_KS,
) -> list[RecallResult]:
    """Calculate recall from raw ranked Qdrant point IDs.

    Each K is applied to raw ranked points before deduplication, matching
    production. Pass the raw point stream, not production ``SearchResults``
    (which are already deduplicated). Invalid IDs are ignored when
    deduplicating. Country matching is exact (no normalization), matching the
    production payload filter; an
    absent candidate country disables that filter. ``None`` labels remain
    unknown.
    """
    _reject_production_results(ranked_point_ids)
    if relevant_job_ids is None:
        return [RecallResult(k, 0, None, None) for k in ks]
    labeled = set(relevant_job_ids)
    def eligible(job_id: str) -> bool:
        job = jobs.get(job_id)
        return bool(
            job
            and job.active
            and job.visible
            and job.opportunity_type == opportunity_type
            and job.passes_eligibility
            and (
                candidate_country_code is None
                or job.country_code == candidate_country_code
            )
        )

    results = []
    for k in ks:
        unique_ids, _ = _channel(ranked_point_ids, k)
        before = {job_id for job_id in unique_ids if job_id in labeled}
        after = {job_id for job_id in before if eligible(job_id)}
        results.append(RecallResult(k, len(labeled), len(before), len(after)))
    return results
