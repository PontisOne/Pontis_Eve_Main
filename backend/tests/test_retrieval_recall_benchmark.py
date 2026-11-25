import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from retrieval_recall_benchmark import (
    BenchmarkJob,
    benchmark_recall,
    merge_ranked_retrieval_channels,
    merge_scored_retrieval_channels,
)


def test_recall_before_and_after_filtering_and_duplicate_collapse():
    jobs = {
        "exact": BenchmarkJob("exact"),
        "inactive": BenchmarkJob("inactive", active=False),
        "wrong-type": BenchmarkJob("wrong-type", opportunity_type="internships"),
        "skills-fail": BenchmarkJob("skills-fail", passes_eligibility=False),
    }
    result = benchmark_recall(
        ["exact", "exact", "inactive", "wrong-type", "skills-fail"],
        jobs,
        {"exact", "inactive", "wrong-type", "skills-fail", "missing"},
        ks=(50,),
    )[0]
    assert result.retrieved_relevant_before_filtering == 4
    assert result.retrieved_relevant_after_filtering == 1
    assert result.recall_before_filtering == 0.8
    assert result.recall_after_filtering == 0.2


def test_missing_labels_are_explicitly_unknown():
    result = benchmark_recall(["job-1"], {}, None, ks=(50,))[0]
    assert result.labeled_relevant == 0
    assert result.recall_before_filtering is None
    assert result.recall_after_filtering is None


def test_recall_k_limits_and_empty_retrieval():
    results = benchmark_recall(
        [], {"relevant": BenchmarkJob("relevant")}, {"relevant"}, ks=(50, 100, 150, 250, 500)
    )
    assert len(results) == 5
    assert all(result.recall_before_filtering == 0.0 for result in results)
    assert all(result.recall_after_filtering == 0.0 for result in results)


def test_duplicates_consume_raw_top_k_boundary():
    jobs = {"relevant": BenchmarkJob("relevant")}
    results = benchmark_recall(["duplicate"] * 50 + ["relevant"], jobs, {"relevant"}, ks=(50, 100))
    assert results[0].retrieved_relevant_before_filtering == 0
    assert results[1].retrieved_relevant_before_filtering == 1


def test_invalid_ids_and_country_filter_use_exact_metadata():
    jobs = {
        "in": BenchmarkJob("in", country_code="IN"),
        "us": BenchmarkJob("us", country_code="US"),
        "missing": BenchmarkJob("missing"),
        "formatted": BenchmarkJob("formatted", country_code="in"),
    }
    results = benchmark_recall(
        [None, "", "in", "us", "missing", "formatted"], jobs,
        {"in", "us", "missing", "formatted"}, candidate_country_code="IN", ks=(50,)
    )
    assert results[0].retrieved_relevant_before_filtering == 4
    assert results[0].retrieved_relevant_after_filtering == 1
    no_country = benchmark_recall(["in", "missing"], jobs, {"in", "missing"}, ks=(50,))[0]
    assert no_country.retrieved_relevant_after_filtering == 2


def test_invalid_and_duplicate_points_consume_raw_boundary():
    jobs = {"relevant": BenchmarkJob("relevant")}
    points = [None, "duplicate", "duplicate", ""] + ["filler"] * 46 + ["relevant"]
    results = benchmark_recall(points, jobs, {"relevant"}, ks=(50, 51))
    assert results[0].retrieved_relevant_before_filtering == 0
    assert results[1].retrieved_relevant_before_filtering == 1


def test_benchmark_rejects_deduplicated_production_results():
    class ProductionResults(list):
        diagnostics = object()
        metadata_by_job_id = {}

    with pytest.raises(TypeError, match="raw ranked Qdrant point IDs"):
        benchmark_recall(ProductionResults(["job-1"]), {}, {"job-1"}, ks=(1,))


def test_two_raw_channels_report_cross_channel_duplicates():
    merged, diagnostics = merge_ranked_retrieval_channels(
        ["shared", None, "broad-only"],
        ["target-only", "shared", ""],
        k=3,
    )
    assert merged == ["shared", "broad-only", "target-only"]
    assert diagnostics.broad.raw_hits == 3
    assert diagnostics.broad.invalid_job_ids == 1
    assert diagnostics.target_role.invalid_job_ids == 1
    assert diagnostics.cross_channel_duplicate_job_ids == 1
    assert diagnostics.final_merged_unique_job_ids == 3


def test_scored_channel_merge_retains_highest_duplicate_score_and_ranks():
    merged, diagnostics = merge_scored_retrieval_channels(
        [("shared", 0.4), ("broad-only", 0.8)],
        [("target-only", 0.7), ("shared", 0.9)],
        k=2,
    )
    assert merged == [("shared", 0.9), ("broad-only", 0.8), ("target-only", 0.7)]
    assert diagnostics.cross_channel_duplicate_job_ids == 1
    assert diagnostics.final_merged_unique_job_ids == 3
