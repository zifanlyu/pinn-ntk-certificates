"""Regression tests for the three-row joint-nullspace controls."""

import json
from pathlib import Path
import unittest

import torch

from .constraints import functional_rank_certificate
from .engine import evaluate_finite, make_initialization_bank
from .features import compile_constraints
from .joint_nullspace_controls import (
    CASES,
    NULL_VECTOR,
    gd_pair_record,
    make_problem,
)


class JointNullspaceControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_coefficient_design_has_the_predicted_exact_rank(self):
        for case in CASES:
            for epsilon, expected in ((0.0, 2), (1.0e-3, 3)):
                problem, metadata = make_problem(case, epsilon)
                certificate = functional_rank_certificate(
                    problem, require_measure_atoms=case == "nonlocal_boundary"
                )
                self.assertEqual(metadata["expected_structural_rank"], expected)
                self.assertEqual(certificate["rank"], expected)

    def test_exact_relation_annihilates_predictions_and_empirical_ntk(self):
        for case in CASES:
            problem, _ = make_problem(case, 0.0)
            state = make_initialization_bank(
                24, problem.input_dim, problem.output_dim, 17
            ).state
            evaluation = evaluate_finite(state, compile_constraints(problem))
            self.assertLess(abs(float(NULL_VECTOR @ evaluation.prediction)), 2.0e-15)
            self.assertLess(
                float(torch.linalg.vector_norm(evaluation.kernel @ NULL_VECTOR)),
                2.0e-14 * max(1.0, float(torch.linalg.matrix_norm(evaluation.kernel))),
            )

    def test_incompatible_target_has_exact_floor_and_same_gd_path(self):
        case = "weak_boundary"
        problem, _ = make_problem(case, 0.0)
        state = make_initialization_bank(
            32, problem.input_dim, problem.output_dim, 5
        ).state
        record = gd_pair_record(
            case,
            32,
            5,
            state,
            steps=25,
            learning_rate_factor=0.2,
            rank_tolerance=1.0e-12,
            incompatibility_amplitude=0.25,
        )
        self.assertLess(record["paired_initial_loss_offset_error"], 2.0e-15)
        self.assertLess(record["paired_final_loss_offset_error"], 2.0e-14)
        self.assertLess(record["paired_final_parameter_l2_difference"], 2.0e-14)
        self.assertAlmostEqual(
            record["measured_final_null_projection_loss"], 0.25**2, places=13
        )
        self.assertEqual([row["step"] for row in record["paired_history"]], [0, 25])
        self.assertLess(
            max(row["paired_loss_offset_error"] for row in record["paired_history"]),
            2.0e-14,
        )
        self.assertTrue(
            all(
                row["incompatible_nullspace_loss"] == 0.25**2
                and row["compatible_nullspace_loss"] == 0.0
                for row in record["paired_history"]
            )
        )

    def test_tracked_aggregate_has_complete_configured_grid(self):
        path = Path(__file__).with_name("data") / "joint_nullspace_controls.json"
        if not path.exists():
            self.skipTest("tracked aggregate has not yet been generated")
        data = json.loads(path.read_text(encoding="utf-8"))
        sizes = data["sample_sizes"]
        expected_kernels = (
            sizes["cases"]
            * sizes["epsilon_values"]
            * sizes["widths"]
            * sizes["seeds"]
        )
        self.assertEqual(sizes["kernel_initializations"], expected_kernels)
        self.assertEqual(len(data["rank_records"]), expected_kernels)
        self.assertEqual(len(data["gd_pair_records"]), sizes["paired_gd_settings"])
        self.assertEqual(sizes["gd_trajectories"], 2 * sizes["paired_gd_settings"])
        expected_steps = data["configuration"]["checkpoint_steps"]
        self.assertEqual(expected_steps[0], 0)
        self.assertEqual(expected_steps[-1], data["configuration"]["steps"])
        for record in data["gd_pair_records"]:
            self.assertEqual(
                [row["step"] for row in record["paired_history"]], expected_steps
            )


if __name__ == "__main__":
    unittest.main()
