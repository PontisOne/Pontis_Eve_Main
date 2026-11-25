import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
JOB_COLLECTION = "job_chunks"

_client = None


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Non-sensitive measurements for one Qdrant query."""

    raw_hits: int
    valid_job_ids: int
    invalid_job_ids: int
    unique_job_ids: int
    duplicate_points_removed: int


@dataclass(frozen=True)
class MergedRetrievalDiagnostics:
    """Diagnostics for merging the independently deduplicated channels."""

    broad: RetrievalDiagnostics
    target_role: RetrievalDiagnostics
    cross_channel_duplicate_job_ids: int
    final_merged_unique_job_ids: int


class SearchResults(list[tuple[str, float]]):
    """Backward-compatible result list carrying query diagnostics."""

    def __init__(
        self, values, diagnostics: RetrievalDiagnostics, metadata_by_job_id=None
    ):
        super().__init__(values)
        self.diagnostics = diagnostics
        self.metadata_by_job_id = metadata_by_job_id or {}


def merge_search_results(
    broad_results, target_role_results
) -> tuple[list[tuple[str, float]], MergedRetrievalDiagnostics]:
    """Merge broad and target-role results, retaining the strongest score."""
    broad_ids = {job_id for job_id, _ in broad_results}
    target_ids = {job_id for job_id, _ in target_role_results}
    combined: dict[str, float] = {}
    for job_id, score in [*broad_results, *target_role_results]:
        combined[job_id] = max(combined.get(job_id, float("-inf")), score)
    merged = sorted(combined.items(), key=lambda item: item[1], reverse=True)

    def diagnostics(results) -> RetrievalDiagnostics:
        value = getattr(results, "diagnostics", None)
        if isinstance(value, RetrievalDiagnostics):
            return value
        ids = [job_id for job_id, _ in results]
        return RetrievalDiagnostics(len(ids), len(ids), 0, len(set(ids)), 0)

    return merged, MergedRetrievalDiagnostics(
        broad=diagnostics(broad_results),
        target_role=diagnostics(target_role_results),
        cross_channel_duplicate_job_ids=len(broad_ids & target_ids),
        final_merged_unique_job_ids=len(merged),
    )


def _get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        if not QDRANT_URL:
            raise RuntimeError("QDRANT_URL environment variable is not set")
        kwargs: dict[str, Any] = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        logger.info("Connecting to Qdrant at %s", QDRANT_URL)
        _client = QdrantClient(**kwargs)
    return _client


def search_job_chunks(
    query_vector: list[float], limit: int = 50, country_code: str | None = None
) -> list[tuple[str, float]]:
    """
    Search the job_chunks Qdrant collection using COSINE similarity.
    Returns a list of (jobId, score) tuples, deduplicated by jobId (best score kept).
    """
    client = _get_client()
    # qdrant-client 1.17 removed ``search`` in favour of ``query_points``.
    # Retain the older call as a fallback for deployments using the pinned 1.9 client.
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=JOB_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="country_code", match=MatchValue(value=country_code)
                    )
                ]
            )
            if country_code
            else None,
        )
        results = response.points
    else:
        results = client.search(
            collection_name=JOB_COLLECTION,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="country_code", match=MatchValue(value=country_code)
                    )
                ]
            )
            if country_code
            else None,
        )
    logger.info("[matching] Qdrant raw hits=%d requested_limit=%d", len(results), limit)

    seen: dict[str, float] = {}
    metadata_by_job_id: dict[str, dict] = {}
    valid_job_ids = 0
    for hit in results:
        payload = hit.payload or {}
        if not isinstance(payload, Mapping):
            continue
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            continue
        valid_job_ids += 1
        score = float(hit.score)
        if job_id not in seen or score > seen[job_id]:
            seen[job_id] = score
            metadata_by_job_id[job_id] = {
                "embedding_version": payload.get("embedding_version"),
                "embedding_model": payload.get("embedding_model"),
            }

    deduplicated = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    diagnostics = RetrievalDiagnostics(
        raw_hits=len(results),
        valid_job_ids=valid_job_ids,
        invalid_job_ids=len(results) - valid_job_ids,
        unique_job_ids=len(deduplicated),
        duplicate_points_removed=max(0, valid_job_ids - len(deduplicated)),
    )
    logger.info(
        "[matching] Qdrant unique job IDs after deduplication=%d", len(deduplicated)
    )
    return SearchResults(deduplicated, diagnostics, metadata_by_job_id)
