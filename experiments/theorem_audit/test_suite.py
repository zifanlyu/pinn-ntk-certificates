"""Regression tests for the theorem-audit numerical implementation.

Run from the repository root with

    python -m unittest experiments.theorem_audit.test_suite

The most important checks use PyTorch autograd as an implementation-independent
oracle for the hand-derived feature/Jacobian and gradient-descent code.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from .constraints import local_operator_matrix, positivity_certificate, validate_problem
from .aggregate import _group_width_metric, _reference_row, _run_row
from .engine import (
    NetworkState,
    estimate_infinite_ntk,
    evaluate_finite,
    full_loss_gd_step,
    make_initialization_bank,
    train_full_batch_gd,
)
from .features import DTYPE, compile_constraints, tanh_derivative
from .problems import get_problem, list_problems, validate_registry
from .run import _classify_reference


def _autograd_constraint_predictions(problem: object, packed: torch.Tensor) -> torch.Tensor:
    """Evaluate normalized constraints without using the analytic feature code."""

    width = packed.shape[0]
    input_dim = int(problem.input_dim)
    w = packed[:, :input_dim]
    b = packed[:, input_dim]
    a = packed[:, input_dim + 1 :]
    group_sizes = Counter(constraint.group for constraint in problem.constraints)
    values: list[torch.Tensor] = []

    for constraint in problem.constraints:
        total = torch.zeros((), dtype=DTYPE, device=packed.device)
        for term in constraint.terms:
            x = torch.tensor(
                term.point, dtype=DTYPE, device=packed.device, requires_grad=True
            )
            hidden = torch.tanh(w @ x + b)
            derivative = torch.dot(a[:, term.output], hidden) / math.sqrt(width)
            for coordinate, order in enumerate(term.alpha):
                for _ in range(order):
                    derivative = torch.autograd.grad(
                        derivative,
                        x,
                        create_graph=True,
                        retain_graph=True,
                    )[0][coordinate]
            total = total + float(term.coefficient) * derivative
        values.append(total / math.sqrt(group_sizes[constraint.group]))
    return torch.stack(values)


class ProblemRegistryTests(unittest.TestCase):
    def test_local_nullspace_uses_normalized_prediction_rows(self) -> None:
        base = get_problem("poisson_1d")
        first = replace(base.constraints[0], name="row_a", group="group_a")
        duplicate = replace(first, name="row_b", group="group_b")
        other = replace(base.constraints[1], group="group_a")
        problem = replace(base, constraints=(first, duplicate, other))
        matrix, _, _ = local_operator_matrix(problem, first.physical_point)
        # Group A has two rows, B only one. The raw-row vector (1,-1)
        # therefore does not annihilate normalized predictions.
        null_vector = torch.tensor([math.sqrt(2.0), -1.0], dtype=DTYPE)
        torch.testing.assert_close(null_vector @ matrix, torch.zeros(1, dtype=DTYPE))
        state = make_initialization_bank(5, 1, 1, seed=934).state
        evaluation = evaluate_finite(state, compile_constraints(problem))
        torch.testing.assert_close(
            null_vector @ evaluation.prediction[:2], torch.zeros((), dtype=DTYPE)
        )
        self.assertEqual(
            positivity_certificate(problem)["locations"][0]["smallest_singular_value"],
            0.0,
        )

    def test_all_eleven_problems_validate(self) -> None:
        problems = validate_registry()
        self.assertEqual(len(problems), 11)
        self.assertEqual(tuple(problem.name for problem in problems), list_problems())
        self.assertEqual(len({problem.name for problem in problems}), 11)
        self.assertTrue(
            all(problem.positivity_regime in {"automatic", "numerical"} for problem in problems)
        )
        for problem in problems:
            validate_problem(problem)
            compiled = compile_constraints(problem)
            self.assertEqual(compiled.count, len(problem.constraints))
            self.assertTrue(torch.isfinite(compiled.targets).all())

    def test_every_automatic_positivity_certificate_passes(self) -> None:
        for problem in validate_registry():
            if problem.positivity_regime != "automatic":
                continue
            certificate = positivity_certificate(problem)
            self.assertTrue(certificate["passed"], problem.name)
            self.assertIn(
                certificate["certificate_type"],
                {
                    "local_operator_coefficient_rank",
                    "finite_jet_functional_coefficient_rank",
                    "signed_measure_coefficient_rank",
                },
            )
            if "locations" in certificate:
                self.assertGreater(len(certificate["locations"]), 0)
                for location in certificate["locations"]:
                    self.assertTrue(location["full_row_rank"], problem.name)
                    self.assertEqual(location["rank"], location["num_rows"])
            else:
                self.assertTrue(certificate["full_row_rank"], problem.name)
                self.assertEqual(certificate["rank"], certificate["num_rows"])


class AnalyticFeatureTests(unittest.TestCase):
    def test_jacobian_drift_matches_svd_and_bounds_kernel_gap(self) -> None:
        compiled = compile_constraints(get_problem("poisson_1d"))
        state = make_initialization_bank(13, 1, 1, seed=319).state
        result = train_full_batch_gd(
            state, compiled, steps=3, learning_rate=1e-3, checkpoint_every=3
        )
        initial = evaluate_finite(state, compiled)
        final = evaluate_finite(result.final_state, compiled)
        difference = (
            (final.per_neuron_gradient - initial.per_neuron_gradient)
            .permute(1, 0, 2).reshape(compiled.count, -1) / math.sqrt(state.width)
        )
        # An explicit rectangular SVD is an independent oracle for the small
        # Gram-eigenvalue calculation used by the production engine.
        expected_drift = float(torch.linalg.matrix_norm(difference, ord=2))
        initial_gap = float(torch.linalg.eigvalsh(initial.kernel)[0])
        self.assertGreater(initial_gap, 0.0)
        record = result.history[-1]
        self.assertAlmostEqual(record.jacobian_drift_operator, expected_drift, places=13)
        self.assertAlmostEqual(
            record.jacobian_drift_over_initial_sqrt_gap,
            expected_drift / math.sqrt(initial_gap), places=12,
        )
        expected_lower = max(math.sqrt(initial_gap) - expected_drift, 0.0) ** 2
        self.assertGreater(expected_lower, 0.0)  # Exercise a non-vacuous certificate.
        self.assertAlmostEqual(record.jacobian_perturbation_gap_lower_bound, expected_lower, places=13)
        self.assertGreaterEqual(record.lambda_min + 1e-13, expected_lower)
        self.assertAlmostEqual(
            record.kernel_drift_over_initial_gap,
            float(torch.linalg.matrix_norm(final.kernel - initial.kernel, ord=2)) / initial_gap,
            places=10,
        )
        self.assertEqual(result.history[0].jacobian_drift_operator, 0.0)
        self.assertEqual(result.history[0].kernel_drift_over_initial_gap, 0.0)
        self.assertIn("jacobian_perturbation_gap_lower_bound", record.as_dict())

    def test_zero_initial_gap_has_no_gap_normalized_diagnostics(self) -> None:
        problem = get_problem("poisson_1d")
        # An exactly zero functional creates a zero initial Gram eigenvalue;
        # this measurement test does not assert that the PDE is well posed.
        row = replace(problem.constraints[0], terms=(replace(problem.constraints[0].terms[0], coefficient=0.0),))
        compiled = compile_constraints(replace(problem, constraints=(row,)))
        state = make_initialization_bank(3, 1, 1, seed=17).state
        record = train_full_batch_gd(state, compiled, steps=0, learning_rate=0.01).history[0]
        self.assertEqual(record.lambda_min, 0.0)
        self.assertIsNone(record.jacobian_drift_over_initial_sqrt_gap)
        self.assertIsNone(record.jacobian_perturbation_gap_lower_bound)
        self.assertIsNone(record.kernel_drift_over_initial_gap)

    def test_checkpoint_spacing_does_not_change_gd_trajectory(self) -> None:
        compiled = compile_constraints(get_problem("poisson_1d"))
        state = make_initialization_bank(7, 1, 1, seed=288).state
        dense = train_full_batch_gd(
            state, compiled, steps=4, learning_rate=0.03, checkpoint_every=1
        )
        sparse = train_full_batch_gd(
            state, compiled, steps=4, learning_rate=0.03, checkpoint_every=3
        )
        torch.testing.assert_close(
            dense.final_state.packed_per_neuron(),
            sparse.final_state.packed_per_neuron(),
            rtol=0, atol=0,
        )
        self.assertEqual([record.step for record in sparse.history], [0, 3, 4])
        self.assertEqual(dense.history[-1].loss, sparse.history[-1].loss)

    def test_nonuniform_checkpoint_schedule(self) -> None:
        compiled = compile_constraints(get_problem("poisson_1d"))
        state = make_initialization_bank(8, 1, 1, seed=19).state
        result = train_full_batch_gd(
            state,
            compiled,
            steps=7,
            learning_rate=0.001,
            checkpoint_steps=(0, 1, 3, 6),
        )
        self.assertEqual([record.step for record in result.history], [0, 1, 3, 6, 7])

    def test_tanh_derivatives_match_recursive_autograd(self) -> None:
        max_operator_order = max(
            sum(term.alpha)
            for problem in validate_registry()
            for constraint in problem.constraints
            for term in constraint.terms
        )
        # The hidden-parameter formulas require one derivative beyond the PDE.
        maximum_order = max_operator_order + 1
        base = torch.tensor(
            [-1.7, -0.8, -0.17, 0.0, 0.31, 1.1, 1.9],
            dtype=DTYPE,
            requires_grad=True,
        )
        derivative = torch.tanh(base)
        for order in range(maximum_order + 1):
            actual = tanh_derivative(base.detach(), order)
            torch.testing.assert_close(actual, derivative.detach(), rtol=2e-12, atol=2e-12)
            if order < maximum_order:
                derivative = torch.autograd.grad(
                    derivative.sum(), base, create_graph=True, retain_graph=True
                )[0]

    def test_vector_constraint_jacobian_and_ntk_match_autograd(self) -> None:
        # Stokes simultaneously tests vector outputs, shared physical points,
        # mixed derivatives, and multiple normalized loss groups.
        problem = get_problem("stokes_2d")
        compiled = compile_constraints(problem)
        state = make_initialization_bank(
            2, problem.input_dim, problem.output_dim, seed=913
        ).state
        evaluation = evaluate_finite(state, compiled)

        packed = state.packed_per_neuron().detach().clone().requires_grad_(True)
        prediction = _autograd_constraint_predictions(problem, packed)
        jacobian = torch.autograd.functional.jacobian(
            lambda parameters: _autograd_constraint_predictions(problem, parameters),
            packed,
            vectorize=False,
        )
        jacobian_matrix = jacobian.reshape(compiled.count, -1)
        analytic_jacobian = (
            evaluation.per_neuron_gradient.permute(1, 0, 2).reshape(compiled.count, -1)
            / math.sqrt(state.width)
        )

        torch.testing.assert_close(evaluation.prediction, prediction, rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(
            analytic_jacobian, jacobian_matrix, rtol=3e-10, atol=3e-11
        )
        torch.testing.assert_close(
            evaluation.kernel,
            jacobian_matrix @ jacobian_matrix.T,
            rtol=3e-10,
            atol=3e-11,
        )

    def test_full_loss_gd_step_matches_autograd(self) -> None:
        problem = get_problem("stokes_2d")
        compiled = compile_constraints(problem)
        state = make_initialization_bank(
            2, problem.input_dim, problem.output_dim, seed=191
        ).state
        learning_rate = 7.0e-4

        packed = state.packed_per_neuron().detach().clone().requires_grad_(True)
        prediction = _autograd_constraint_predictions(problem, packed)
        loss = torch.sum((prediction - compiled.targets) ** 2)
        gradient = torch.autograd.grad(loss, packed)[0]
        expected = packed.detach() - learning_rate * gradient.detach()

        updated = full_loss_gd_step(state, compiled, learning_rate)
        torch.testing.assert_close(
            updated.packed_per_neuron(), expected, rtol=3e-10, atol=3e-11
        )

    def test_weak_nonlocal_and_fractional_features_match_autograd(self) -> None:
        # These exercise aggregation across many spatial points inside one
        # functional, including value, first-derivative, and second-derivative
        # atoms.  They are distinct from the co-located Stokes test above.
        for problem_name in (
            "weak_poisson_1d",
            "nonlocal_diffusion_1d",
            "integro_diff_1d",
            "caputo_diffusion_1d",
        ):
            with self.subTest(problem=problem_name):
                problem = get_problem(problem_name)
                compiled = compile_constraints(problem)
                state = make_initialization_bank(
                    2, problem.input_dim, problem.output_dim, seed=417
                ).state
                evaluation = evaluate_finite(state, compiled)
                packed = (
                    state.packed_per_neuron().detach().clone().requires_grad_(True)
                )
                prediction = _autograd_constraint_predictions(problem, packed)
                jacobian = torch.autograd.functional.jacobian(
                    lambda parameters: _autograd_constraint_predictions(
                        problem, parameters
                    ),
                    packed,
                    vectorize=False,
                ).reshape(compiled.count, -1)
                analytic_jacobian = (
                    evaluation.per_neuron_gradient.permute(1, 0, 2).reshape(
                        compiled.count, -1
                    )
                    / math.sqrt(state.width)
                )
                torch.testing.assert_close(
                    evaluation.prediction, prediction, rtol=3e-10, atol=3e-11
                )
                torch.testing.assert_close(
                    analytic_jacobian, jacobian, rtol=5e-10, atol=5e-11
                )


class InitializationAndReferenceTests(unittest.TestCase):
    def test_nested_initialization_prefixes_are_deterministic_and_bounded(self) -> None:
        bank = make_initialization_bank(19, 2, 3, seed=2027)
        repeated = make_initialization_bank(19, 2, 3, seed=2027)
        torch.testing.assert_close(
            bank.state.packed_per_neuron(), repeated.state.packed_per_neuron(), rtol=0, atol=0
        )
        self.assertTrue((bank.state.packed_per_neuron() >= -1.0).all())
        self.assertTrue((bank.state.packed_per_neuron() <= 1.0).all())
        for width in (1, 4, 11, 19):
            prefix = bank.prefix(width)
            self.assertIsInstance(prefix, NetworkState)
            torch.testing.assert_close(
                prefix.packed_per_neuron(),
                bank.state.packed_per_neuron()[:width],
                rtol=0,
                atol=0,
            )
            self.assertFalse(prefix.w.requires_grad)
            self.assertFalse(prefix.b.requires_grad)
            self.assertFalse(prefix.a.requires_grad)

    def test_small_scrambled_sobol_reference_is_symmetric_psd(self) -> None:
        compiled = compile_constraints(get_problem("poisson_1d"))
        reference = estimate_infinite_ntk(
            compiled,
            coarse_samples=32,
            fine_samples=64,
            replicates=2,
            seed=73,
            chunk_size=16,
            positivity_relative_tolerance=1e-10,
        )
        for matrix in (
            reference.K_full,
            reference.K_out,
            reference.K_full_coarse,
            reference.K_out_coarse,
            reference.K_hidden,
        ):
            torch.testing.assert_close(matrix, matrix.T, rtol=0, atol=2e-14)
            self.assertTrue(torch.isfinite(matrix).all())
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(matrix)[0]), -2e-12)
        self.assertEqual(reference.K_full.dtype, DTYPE)
        self.assertIn(reference.full_positivity.status, {"positive", "inconclusive"})
        self.assertTrue(math.isfinite(reference.full_positivity.lower_bound))
        self.assertEqual(reference.full_replicate_lambda_min.shape, (2,))


class PositivityGateTests(unittest.TestCase):
    def test_automatic_gate_uses_only_the_single_analytic_certificate(self) -> None:
        problem = get_problem("poisson_1d")

        # Deliberately omit output/hidden diagnostics: an automatic case must
        # need only its one output-feature rank certificate.
        inconclusive_reference = SimpleNamespace(
            full_positivity=SimpleNamespace(status="inconclusive")
        )
        accepted = _classify_reference(
            problem, inconclusive_reference, {"passed": True}
        )
        self.assertTrue(accepted["allows_training"])
        self.assertFalse(accepted["qmc_is_training_gate"])

        positive_reference = SimpleNamespace(
            full_positivity=SimpleNamespace(status="positive")
        )
        rejected = _classify_reference(problem, positive_reference, {"passed": False})
        self.assertFalse(rejected["allows_training"])


class AggregationRegressionTests(unittest.TestCase):
    def _run_artifact(self) -> dict[str, object]:
        return {
            "status": "complete", "config_hash": "test", "problem": "poisson_1d",
            "width": 8, "seed": 0, "optimization": {"steps": 4},
            "history": [
                {"step": 0, "loss": 1.0, "lambda_min": 0.9},
                {"step": 4, "loss": 0.5, "lambda_min": 0.9},
            ],
        }

    def _aggregate_artifact(self, artifact: dict[str, object]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            return _run_row(
                "poisson_1d", 8, 0, path, "test", None,
                {"lambda_min": 1.0}, expected_steps=4,
            )

    def test_completed_histories_must_reach_requested_step(self) -> None:
        artifact = self._run_artifact()
        artifact["history"][-1]["step"] = 3
        row, history, issue = self._aggregate_artifact(artifact)
        self.assertEqual(row["status"], "invalid_history")
        self.assertEqual(history, [])
        self.assertIsNotNone(issue)

    def test_completed_histories_require_finite_losses_and_integer_steps(self) -> None:
        for field, value in (("loss", None), ("loss", -0.1), ("step", 4.2)):
            with self.subTest(field=field, value=value):
                artifact = self._run_artifact()
                artifact["history"][-1][field] = value
                row, history, issue = self._aggregate_artifact(artifact)
                self.assertEqual(row["status"], "invalid_history")
                self.assertEqual(history, [])
                self.assertIsNotNone(issue)

    def test_scalar_gap_ratio_is_not_a_loewner_event(self) -> None:
        row, _, issue = self._aggregate_artifact(self._run_artifact())
        self.assertIsNone(issue)
        self.assertEqual(row["initial_minimum_eigenvalue_ratio"], 0.9)
        self.assertIsNone(row["initial_generalized_gap_ratio"])
        self.assertIsNone(row["initial_gap_criterion_passed"])

    def test_failed_reference_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps({
                "status": "failed", "config_hash": "test", "problem": "poisson_1d",
            }), encoding="utf-8")
            row, data, issue = _reference_row("poisson_1d", path, "test", None)
            self.assertEqual(row["artifact_status"], "failed")
            self.assertIsNone(data)
            self.assertIsNotNone(issue)

    def test_stale_source_and_mixed_reference_are_rejected(self) -> None:
        artifact = self._run_artifact()
        artifact["source"] = {"sha256": "old"}
        artifact["reference"] = {"npz_sha256": "different"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            row, _, issue = _run_row(
                "poisson_1d", 8, 0, path, "test", None, {"lambda_min": 1.0},
                expected_source_hash="new",
            )
            self.assertEqual(row["status"], "rejected_source_hash")
            self.assertIsNotNone(issue)
            row, _, issue = _run_row(
                "poisson_1d", 8, 0, path, "test", None,
                {"lambda_min": 1.0, "npz_sha256": "expected"},
                expected_source_hash="old",
            )
            self.assertEqual(row["status"], "rejected_reference_hash")
            self.assertIsNotNone(issue)

    def test_wilson_interval_contains_observed_proportion(self) -> None:
        # Protect the all-fail/all-pass small-seed cases used in smoke plots;
        # roundoff must not create negative Matplotlib error-bar lengths.
        for passed in (False, True):
            rows = [
                {
                    "problem": "poisson_1d",
                    "status": "complete",
                    "width": 16,
                    "probability": float(passed),
                }
                for _ in range(3)
            ]
            points = _group_width_metric(
                rows, "poisson_1d", "probability", probability=True
            )
            self.assertEqual(len(points), 1)
            _, center, lower, upper, count = points[0]
            self.assertEqual(count, 3)
            self.assertLessEqual(lower, center)
            self.assertLessEqual(center, upper)


if __name__ == "__main__":
    unittest.main()
