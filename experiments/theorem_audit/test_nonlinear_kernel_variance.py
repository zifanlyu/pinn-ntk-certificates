"""Numerical-contract tests for :mod:`nonlinear_kernel_variance`."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .nonlinear_kernel_variance import (
    SCHEMA_VERSION,
    load_config,
    measure,
    row_quantities,
)
from .run import AuditError

BASE_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "output_dir": "experiments/theorem_audit/results/unit_test",
    "gamma": 1.0,
    "collocation_points": [0.3, -0.7],
    "widths": [64, 256, 1024],
    "seeds": list(range(120)),
    "polynomial_degree": 2,
    "hidden_parameter_low": -1.0,
    "hidden_parameter_high": 1.0,
    "uncentered_output_weight_mean": 1.0,
    "shifted_bias_low": 0.0,
    "shifted_bias_high": 2.0,
}


def _write(config: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(config, handle)
    handle.close()
    return Path(handle.name)


class ConfigValidation(unittest.TestCase):
    def test_rejects_wrong_schema_version(self) -> None:
        path = _write({**BASE_CONFIG, "schema_version": 999})
        with self.assertRaises(AuditError):
            load_config(path)

    def test_rejects_missing_field(self) -> None:
        broken = {key: value for key, value in BASE_CONFIG.items() if key != "gamma"}
        with self.assertRaises(AuditError):
            load_config(_write(broken))

    def test_rejects_wrong_number_of_collocation_points(self) -> None:
        path = _write({**BASE_CONFIG, "collocation_points": [0.3]})
        with self.assertRaises(AuditError):
            load_config(path)

    def test_rejects_degree_below_two(self) -> None:
        path = _write({**BASE_CONFIG, "polynomial_degree": 1})
        with self.assertRaises(AuditError):
            load_config(path)

    def test_accepts_the_tracked_configs(self) -> None:
        root = Path(__file__).resolve().parent / "configs"
        for name in ("nonlinear_kernel_variance.json", "nonlinear_kernel_variance_smoke.json"):
            config = load_config(root / name)
            self.assertEqual(config["schema_version"], SCHEMA_VERSION)


class GradientContract(unittest.TestCase):
    """The analytic gradients must match finite differences."""

    def test_gradients_match_central_differences(self) -> None:
        rng = np.random.default_rng(0)
        width = 8
        w = rng.uniform(-1.0, 1.0, size=width)
        b = rng.uniform(-1.0, 1.0, size=width)
        a = rng.uniform(-1.0, 1.0, size=width)
        _, _, g1, _ = row_quantities(w, b, a, BASE_CONFIG)

        gamma = BASE_CONFIG["gamma"]
        x1 = BASE_CONFIG["collocation_points"][0]
        step = 1e-6
        for index in range(width):
            shifted_up = w.copy()
            shifted_down = w.copy()
            shifted_up[index] += step
            shifted_down[index] -= step
            up = a[index] * np.tanh(shifted_up[index] * x1 + gamma * b[index])
            down = a[index] * np.tanh(shifted_down[index] * x1 + gamma * b[index])
            self.assertAlmostEqual((up - down) / (2 * step), g1[0][index], places=6)


class Predictions(unittest.TestCase):
    """The three predictions of ``sec:nonlinear_and_depth``."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = measure(BASE_CONFIG)["per_width"]

    def test_linear_row_variance_decays_like_one_over_width(self) -> None:
        scaled = [row["width_times_variance_linear_row"] for row in self.rows]
        self.assertLess(max(scaled) / min(scaled), 4.0)

    def test_product_row_variance_does_not_decay(self) -> None:
        first = self.rows[0]["variance_product_row"]
        last = self.rows[-1]["variance_product_row"]
        widths = self.rows[-1]["width"] / self.rows[0]["width"]
        # A concentrating quantity would fall by the full width ratio.
        self.assertGreater(last, first / (0.1 * widths))

    def test_product_variance_dominates_linear_variance_at_the_largest_width(self) -> None:
        last = self.rows[-1]
        self.assertGreater(last["variance_product_row"], 100.0 * last["variance_linear_row"])

    def test_degree_matched_block_decays_like_one_over_width(self) -> None:
        scaled = [row["width_times_degree_matched_block"] for row in self.rows]
        self.assertLess(max(scaled) / min(scaled), 4.0)

    def test_shifted_bias_prediction_grows_like_width_to_the_one_over_two_p(self) -> None:
        ratios = [row["shifted_prediction_over_width_power"] for row in self.rows]
        self.assertLess(max(ratios) / min(ratios), 1.3)

    def test_symmetric_bias_prediction_decays_instead(self) -> None:
        """``rem:uncentered_reachability``: an odd activation with a symmetric
        hidden law has zero feature mean, so uncentering ``a`` is not enough."""
        first = self.rows[0]["mean_uncentered_symmetric_prediction"]
        last = self.rows[-1]["mean_uncentered_symmetric_prediction"]
        self.assertLess(last, first)


if __name__ == "__main__":
    unittest.main()
