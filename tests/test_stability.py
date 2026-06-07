"""neighbor_stability tests."""

from __future__ import annotations

try:  # ensure src/ is importable under the stdlib runner (no-op once installed)
    from . import _path  # noqa: F401
except ImportError:
    import _path  # noqa: F401

from embspec import StabilityReport, neighbor_stability


def test_identical_results_give_perfect_overlap() -> None:
    old = {f"q{i}": [f"d{j}" for j in range(10)] for i in range(20)}
    new = {f"q{i}": [f"d{j}" for j in range(10)] for i in range(20)}
    report = neighbor_stability(old, new, k=10)
    assert report.n_probes == 20
    assert report.mean_overlap_at_k == 1.0
    assert report.mean_jaccard_at_k == 1.0
    assert report.regression_count == 0
    assert report.is_safe_to_deploy()


def test_disjoint_results_give_zero_overlap() -> None:
    old = {f"q{i}": [f"d{j}" for j in range(5)] for i in range(10)}
    new = {f"q{i}": [f"e{j}" for j in range(5)] for i in range(10)}
    report = neighbor_stability(old, new, k=5)
    assert report.mean_overlap_at_k == 0.0
    assert report.mean_jaccard_at_k == 0.0
    assert report.regression_count == 10
    assert not report.is_safe_to_deploy()


def test_partial_overlap_metrics() -> None:
    old = {"q1": ["a", "b", "c", "d", "e"]}
    new = {"q1": ["a", "b", "x", "y", "z"]}
    report = neighbor_stability(old, new, k=5)
    assert report.mean_overlap_at_k == 0.4  # 2 of 5
    assert report.mean_jaccard_at_k == 2 / 8  # |intersect=2| / |union=8|


def test_truncates_to_k_when_results_longer() -> None:
    old = {"q1": ["a", "b", "c", "d", "e", "f", "g"]}
    new = {"q1": ["a", "b", "z", "y", "x", "w", "v"]}
    report = neighbor_stability(old, new, k=2)
    assert report.mean_overlap_at_k == 1.0  # only top-2 considered, both match


def test_only_intersecting_probe_ids_counted() -> None:
    old = {"q1": ["a"], "q2": ["b"]}
    new = {"q2": ["b"], "q3": ["c"]}
    report = neighbor_stability(old, new, k=1)
    assert report.n_probes == 1
    assert report.mean_overlap_at_k == 1.0


def test_empty_inputs_return_zero_report() -> None:
    report = neighbor_stability({}, {}, k=10)
    assert isinstance(report, StabilityReport)
    assert report.n_probes == 0
    assert not report.is_safe_to_deploy()


def test_regression_threshold_marks_low_overlap_probes() -> None:
    old = {
        "good_q": ["a", "b", "c", "d"],
        "bad_q": ["a", "b", "c", "d"],
    }
    new = {
        "good_q": ["a", "b", "c", "d"],
        "bad_q": ["x", "y", "z", "w"],
    }
    report = neighbor_stability(old, new, k=4, regression_threshold=0.5)
    assert "bad_q" in report.regression_probe_ids
    assert "good_q" not in report.regression_probe_ids


def test_is_safe_to_deploy_with_custom_thresholds() -> None:
    report = StabilityReport(
        n_probes=100,
        k=10,
        mean_overlap_at_k=0.90,
        mean_jaccard_at_k=0.85,
        regression_probe_ids=tuple(f"p{i}" for i in range(3)),
    )
    assert report.is_safe_to_deploy(min_mean_overlap=0.85, max_regression_fraction=0.05)
    assert not report.is_safe_to_deploy(min_mean_overlap=0.95, max_regression_fraction=0.05)
    assert not report.is_safe_to_deploy(min_mean_overlap=0.85, max_regression_fraction=0.02)
