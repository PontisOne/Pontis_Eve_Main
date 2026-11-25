import os
import sys
from types import SimpleNamespace
from typing import cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import qdrant_service


class _QueryClient:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.hits)


def test_query_points_deduplicates_by_stable_job_id(monkeypatch):
    client = _QueryClient(
        [
            SimpleNamespace(payload={"job_id": "job-1"}, score=0.4),
            SimpleNamespace(payload={"job_id": "job-2"}, score=0.7),
            SimpleNamespace(payload={"job_id": "job-1"}, score=0.9),
        ]
    )
    monkeypatch.setattr(qdrant_service, "_get_client", lambda: client)

    results = cast(qdrant_service.SearchResults, qdrant_service.search_job_chunks([0.1], limit=150))
    assert results == [
        ("job-1", 0.9),
        ("job-2", 0.7),
    ]
    assert results.diagnostics.raw_hits == 3
    assert results.diagnostics.unique_job_ids == 2
    assert results.diagnostics.duplicate_points_removed == 1
    assert client.calls[0]["limit"] == 150
    assert "score_threshold" not in client.calls[0]


def test_query_points_keeps_all_distinct_available_jobs(monkeypatch):
    client = _QueryClient(
        [
            SimpleNamespace(payload={"job_id": f"job-{index}"}, score=1 - index / 100)
            for index in range(5)
        ]
    )
    monkeypatch.setattr(qdrant_service, "_get_client", lambda: client)

    assert len(qdrant_service.search_job_chunks([0.1], limit=150)) == 5


def test_invalid_ids_are_not_counted_as_duplicates(monkeypatch):
    client = _QueryClient([
        SimpleNamespace(payload={}, score=0.9),
        SimpleNamespace(payload={"job_id": None}, score=0.8),
        SimpleNamespace(payload={"job_id": ""}, score=0.7),
        SimpleNamespace(payload={"job_id": "job-1"}, score=0.6),
        SimpleNamespace(payload={"job_id": "job-1"}, score=0.5),
    ])
    monkeypatch.setattr(qdrant_service, "_get_client", lambda: client)
    result = cast(qdrant_service.SearchResults, qdrant_service.search_job_chunks([0.1], limit=150))
    assert result == [("job-1", 0.6)]
    assert result.diagnostics.raw_hits == 5
    assert result.diagnostics.valid_job_ids == 2
    assert result.diagnostics.invalid_job_ids == 3
    assert result.diagnostics.duplicate_points_removed == 1
    assert result.diagnostics.unique_job_ids == 1


def test_non_mapping_payloads_are_invalid_not_duplicates(monkeypatch):
    client = _QueryClient([
        SimpleNamespace(payload={"job_id": "job-1"}, score=0.6),
        SimpleNamespace(payload={}, score=0.5),
        SimpleNamespace(payload=None, score=0.4),
        SimpleNamespace(payload=["job-1"], score=0.3),
        SimpleNamespace(payload="job-1", score=0.2),
        SimpleNamespace(payload=42, score=0.1),
        SimpleNamespace(payload={"job_id": "job-1"}, score=0.9),
    ])
    monkeypatch.setattr(qdrant_service, "_get_client", lambda: client)

    result = cast(qdrant_service.SearchResults, qdrant_service.search_job_chunks([0.1], limit=150))

    assert result == [("job-1", 0.9)]
    assert result.diagnostics.raw_hits == 7
    assert result.diagnostics.valid_job_ids == 2
    assert result.diagnostics.invalid_job_ids == 5
    assert result.diagnostics.duplicate_points_removed == 1
    assert result.diagnostics.unique_job_ids == 1


def test_empty_results_have_zero_diagnostics(monkeypatch):
    client = _QueryClient([])
    monkeypatch.setattr(qdrant_service, "_get_client", lambda: client)
    result = cast(qdrant_service.SearchResults, qdrant_service.search_job_chunks([0.1], limit=150))
    assert result == []
    assert result.diagnostics == qdrant_service.RetrievalDiagnostics(0, 0, 0, 0, 0)


def test_merge_channels_keeps_highest_score_and_separates_cross_channel_duplicates():
    broad = qdrant_service.SearchResults(
        [("shared", 0.6), ("broad-only", 0.5)],
        qdrant_service.RetrievalDiagnostics(4, 3, 1, 2, 1),
    )
    target = qdrant_service.SearchResults(
        [("shared", 0.9), ("target-only", 0.7)],
        qdrant_service.RetrievalDiagnostics(3, 2, 1, 2, 0),
    )

    merged, diagnostics = qdrant_service.merge_search_results(broad, target)

    assert merged == [("shared", 0.9), ("target-only", 0.7), ("broad-only", 0.5)]
    assert diagnostics.broad.raw_hits == 4
    assert diagnostics.target_role.raw_hits == 3
    assert diagnostics.cross_channel_duplicate_job_ids == 1
    assert diagnostics.final_merged_unique_job_ids == 3
