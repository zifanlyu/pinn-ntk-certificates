"""Numerical-contract tests for :mod:`per_iterate_monotonicity`."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .per_iterate_monotonicity import (
    SCHEMA_VERSION,
    load_config,
    rows_and_gradients,
    run_cell,
)
from .run import AuditError

BASE_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "output_dir": "experiments/theorem_audit/results/unit_test",
    "gamma": 1.0,
    "collocation_point": 0.35,
    "widths": [64],
    "epsilons": [0.1],
    "seeds": [0, 1],
    "steps": 200,
    "learning_rate_factor": 0.2,
    "target": [0.6, -0.6],
    "hidden_parameter_low": -1.0,
    "hidden_parameter_high": 1.0,
    "precision_floor_relative": 1e-26,
    "increase_tolerance_relative": 1e-12,
}


def _write(config: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(config, handle)
    handle.close()
    return Path(handle.name)


class ConfigValidation(unittest.TestCase):
    def test_rejects_wrong_schema_version(self) -> None:
        with self.assertRaises(AuditError):
            load_config(_write({**BASE_CONFIG, "schema_version": 999}))

    def test_rejects_missing_field(self) -> None:
        broken = {key: value for key, value in BASE_CONFIG.items() if key != "steps"}
        with self.assertRaises(AuditError):
            load_config(_write(broken))

    def test_rejects_wrong_target_length(self) -> None:
        with self.assertRaises(AuditError):
            load_config(_write({**BASE_CONFIG, "target": [0.6]}))

    def test_rejects_nonpositive_steps(self) -> None:
        with self.assertRaises(AuditError):
            load_config(_write({**BASE_CONFIG, "steps": 0}))

    def test_accepts_the_tracked_configs(self) -> None:
        root = Path(__file__).resolve().parent / "configs"
        for name in ("per_iterate_monotonicity.json", "per_iterate_monotonicity_smoke.json"):
            self.assertEqual(load_config(root / name)["schema_version"], SCHEMA_VERSION)


class GradientContract(unittest.TestCase):
    def test_mixed_row_gradient_matches_central_differences(self) -> None:
        rng = np.random.default_rng(1)
        width = 8
        mixing = 0.1
        w = rng.uniform(-1.0, 1.0, size=width)
        b = rng.uniform(-1.0, 1.0, size=width)
        a = rng.uniform(-1.0, 1.0, size=width)
        _, _, grad_mixed = rows_and_gradients(w, b, a, mixing, width, BASE_CONFIG)

        step = 1e-6
        for index in range(width):
            up = w.copy()
            down = w.copy()
            up[index] += step
            down[index] -= step
            pred_up, _, _ = rows_and_gradients(up, b, a, mixing, width, BASE_CONFIG)
            pred_down, _, _ = rows_and_gradients(down, b, a, mixing, width, BASE_CONFIG)
            numerical = (pred_up[1] - pred_down[1]) / (2 * step) * np.sqrt(width)
            self.assertAlmostEqual(numerical, grad_mixed[0][index], places=5)

    def test_mixing_of_zero_makes_both_rows_the_value_row(self) -> None:
        rng = np.random.default_rng(2)
        width = 16
        w = rng.uniform(-1.0, 1.0, size=width)
        b = rng.uniform(-1.0, 1.0, size=width)
        a = rng.uniform(-1.0, 1.0, size=width)
        prediction, _, _ = rows_and_gradients(w, b, a, 0.0, width, BASE_CONFIG)
        self.assertAlmostEqual(prediction[0], prediction[1], places=12)


class MonotonicityContract(unittest.TestCase):
    def test_no_violations_on_a_well_conditioned_cell(self) -> None:
        cell = run_cell(64, 0.1, BASE_CONFIG)
        self.assertEqual(cell["violations"], 0)
        self.assertEqual(cell["scored_iterates"], cell["possible_iterates"])

    def test_worst_ratio_matches_the_frozen_kernel_prediction(self) -> None:
        cell = run_cell(64, 0.1, BASE_CONFIG)
        self.assertIsNotNone(cell["worst_ratio_first_seed"])
        self.assertAlmostEqual(
            cell["worst_ratio_first_seed"], cell["predicted_ratio_first_seed"], places=3
        )

    def test_scored_iterates_never_exceed_possible(self) -> None:
        cell = run_cell(64, 1.0, BASE_CONFIG)
        self.assertLessEqual(cell["scored_iterates"], cell["possible_iterates"])

    def test_precision_floor_truncates_and_suppresses_the_ratio(self) -> None:
        """A floor just below the initial loss must truncate immediately and
        withhold the worst-ratio figure rather than report an artifact."""
        cell = run_cell(64, 0.1, {**BASE_CONFIG, "precision_floor_relative": 0.5})
        self.assertTrue(cell["reached_precision_floor"])
        self.assertIsNone(cell["worst_ratio_first_seed"])
        self.assertLess(cell["scored_iterates"], cell["possible_iterates"])


if __name__ == "__main__":
    unittest.main()
