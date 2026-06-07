"""Stdlib ``unittest`` coverage for ``neighbor_stability``.

Mirrors ``test_stability.py`` using ``unittest.TestCase`` so it runs under
``python3 -m unittest discover -s tests``. ``neighbor_stability`` is pure and
numpy-free, so this exercises the real implementation directly.
"""

from __future__ import annotations

import unittest

try:  # ensure src/ is importable under the stdlib runner (no-op once installed)
    from . import _path  # noqa: F401
except ImportError:
    import _path  # noqa: F401

from embspec import StabilityReport, neighbor_stability


class NeighborStabilityTests(unittest.TestCase):
    def test_identical_results_give_perfect_overlap(self) -> None:
        old = {f"q{i}": [f"d{j}" for j in range(10)] for i in range(20)}
        new = {f"q{i}": [f"d{j}" for j in range(10)] for i in range(20)}
        report = neighbor_stability(old, new, k=10)
        self.assertEqual(report.n_probes, 20)
        self.assertEqual(report.mean_overlap_at_k, 1.0)
        self.assertEqual(report.mean_jaccard_at_k, 1.0)
        self.assertEqual(report.regression_count, 0)
        self.assertTrue(report.is_safe_to_deploy())

    def test_disjoint_results_give_zero_overlap(self) -> None:
        old = {f"q{i}": [f"d{j}" for j in range(5)] for i in range(10)}
        new = {f"q{i}": [f"e{j}" for j in range(5)] for i in range(10)}
        report = neighbor_stability(old, new, k=5)
        self.assertEqual(report.mean_overlap_at_k, 0.0)
        self.assertEqual(report.mean_jaccard_at_k, 0.0)
        self.assertEqual(report.regression_count, 10)
        self.assertFalse(report.is_safe_to_deploy())

    def test_partial_overlap_metrics(self) -> None:
        old = {"q1": ["a", "b", "c", "d", "e"]}
        new = {"q1": ["a", "b", "x", "y", "z"]}
        report = neighbor_stability(old, new, k=5)
        self.assertAlmostEqual(report.mean_overlap_at_k, 0.4)
        self.assertAlmostEqual(report.mean_jaccard_at_k, 2 / 8)

    def test_truncates_to_k_when_results_longer(self) -> None:
        old = {"q1": ["a", "b", "c", "d", "e", "f", "g"]}
        new = {"q1": ["a", "b", "z", "y", "x", "w", "v"]}
        report = neighbor_stability(old, new, k=2)
        self.assertEqual(report.mean_overlap_at_k, 1.0)

    def test_only_intersecting_probe_ids_counted(self) -> None:
        old = {"q1": ["a"], "q2": ["b"]}
        new = {"q2": ["b"], "q3": ["c"]}
        report = neighbor_stability(old, new, k=1)
        self.assertEqual(report.n_probes, 1)
        self.assertEqual(report.mean_overlap_at_k, 1.0)

    def test_empty_inputs_return_zero_report(self) -> None:
        report = neighbor_stability({}, {}, k=10)
        self.assertIsInstance(report, StabilityReport)
        self.assertEqual(report.n_probes, 0)
        self.assertFalse(report.is_safe_to_deploy())

    def test_regression_threshold_marks_low_overlap_probes(self) -> None:
        old = {"good_q": ["a", "b", "c", "d"], "bad_q": ["a", "b", "c", "d"]}
        new = {"good_q": ["a", "b", "c", "d"], "bad_q": ["x", "y", "z", "w"]}
        report = neighbor_stability(old, new, k=4, regression_threshold=0.5)
        self.assertIn("bad_q", report.regression_probe_ids)
        self.assertNotIn("good_q", report.regression_probe_ids)

    def test_is_safe_to_deploy_with_custom_thresholds(self) -> None:
        report = StabilityReport(
            n_probes=100,
            k=10,
            mean_overlap_at_k=0.90,
            mean_jaccard_at_k=0.85,
            regression_probe_ids=tuple(f"p{i}" for i in range(3)),
        )
        self.assertTrue(
            report.is_safe_to_deploy(
                min_mean_overlap=0.85, max_regression_fraction=0.05
            )
        )
        self.assertFalse(
            report.is_safe_to_deploy(
                min_mean_overlap=0.95, max_regression_fraction=0.05
            )
        )
        self.assertFalse(
            report.is_safe_to_deploy(
                min_mean_overlap=0.85, max_regression_fraction=0.02
            )
        )


if __name__ == "__main__":
    unittest.main()
