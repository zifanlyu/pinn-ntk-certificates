"""Independent mathematical checks for deliberate rank-deficient controls."""

import math
import unittest

import torch

from .constraints import functional_rank_certificate
from .engine import evaluate_finite, make_initialization_bank, train_full_batch_gd
from .features import compile_constraints
from .rank_controls import FAMILIES, make_control, structural_null_projector


class RankControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_structural_rank_and_normalized_floor_all_families(self):
        for family in FAMILIES:
            for epsilon in (None, 0.1, 0.0):
                for mode in ("manufactured", "contrast"):
                    problem, metadata = make_control(family, epsilon, mode)
                    compiled = compile_constraints(problem)
                    certificate = functional_rank_certificate(problem)
                    self.assertEqual(certificate["rank"], 1 if epsilon == 0 else 2)
                    state = make_initialization_bank(32, problem.input_dim, problem.output_dim, 19).state
                    evaluation = evaluate_finite(state, compiled)
                    projector = structural_null_projector(epsilon)
                    torch.testing.assert_close(projector @ evaluation.prediction, torch.zeros(2, dtype=torch.float64), atol=3e-15, rtol=0)
                    self.assertAlmostEqual(float((projector @ evaluation.error).square().sum()), metadata["exact_structural_loss_floor"], places=13)
                    if epsilon == 0:
                        torch.testing.assert_close(evaluation.kernel @ projector, torch.zeros((2, 2), dtype=torch.float64), atol=3e-15, rtol=0)

    def test_co_located_independent_rows_are_positive_definite(self):
        for family in ("pointwise_poisson", "vector_stokes"):
            problem, metadata = make_control(family, None)
            self.assertTrue(metadata["same_physical_location"])
            state = make_initialization_bank(128, problem.input_dim, problem.output_dim, 41).state
            evaluation = evaluate_finite(state, compile_constraints(problem))
            self.assertGreater(float(torch.linalg.eigvalsh(evaluation.kernel)[0]), 1e-4)
            output_jacobian = evaluation.per_neuron_gradient[:, :, -problem.output_dim:]
            output_kernel = torch.einsum("miq,mjq->ij", output_jacobian, output_jacobian) / state.width
            self.assertGreater(float(torch.linalg.eigvalsh(output_kernel)[0]), 1e-5)
            self.assertGreater(float(torch.linalg.eigvalsh(evaluation.kernel - output_kernel)[0]), -1e-14)

    def test_singular_contrast_has_same_gd_trajectory_and_correct_floor(self):
        good, _ = make_control("pointwise_poisson", 0.0, "manufactured")
        bad, _ = make_control("pointwise_poisson", 0.0, "contrast", contrast_amplitude=0.5)
        state = make_initialization_bank(128, 1, 1, 3).state
        initial = evaluate_finite(state, compile_constraints(good))
        eta = 0.1 / float(torch.linalg.eigvalsh(initial.kernel)[-1])
        kwargs = dict(steps=100, learning_rate=eta, checkpoint_every=10)
        good_result = train_full_batch_gd(state, compile_constraints(good), **kwargs)
        bad_result = train_full_batch_gd(state, compile_constraints(bad), **kwargs)
        torch.testing.assert_close(good_result.final_state.packed_per_neuron(), bad_result.final_state.packed_per_neuron(), atol=2e-14, rtol=2e-14)
        for good_record, bad_record in zip(good_result.history, bad_result.history, strict=True):
            self.assertAlmostEqual(bad_record.loss - good_record.loss, 0.25, places=13)
            self.assertAlmostEqual(bad_record.frozen_kernel_loss - good_record.frozen_kernel_loss, 0.25, places=13)
        self.assertLess(good_result.history[-1].loss, 1e-10)
        self.assertAlmostEqual(bad_result.history[-1].loss, 0.25, places=9)

    def test_near_dependent_gap_scales_quadratically(self):
        gaps = []
        state = make_initialization_bank(256, 1, 1, 22).state
        for epsilon in (0.1, 0.01, 0.001):
            problem, _ = make_control("pointwise_poisson", epsilon)
            kernel = evaluate_finite(state, compile_constraints(problem)).kernel
            gaps.append(float(torch.linalg.eigvalsh(kernel)[0]))
        for first, second in zip(gaps, gaps[1:]):
            self.assertLess(abs(math.log10(first / second) - 2.0), 0.08)

    def test_frozen_dynamics_equal_eigenmode_solution(self):
        problem, _ = make_control("vector_stokes", 0.1, "contrast")
        state = make_initialization_bank(64, 2, 3, 17).state
        compiled = compile_constraints(problem)
        initial = evaluate_finite(state, compiled)
        values, vectors = torch.linalg.eigh(initial.kernel)
        eta = 0.2 / float(values[-1])
        result = train_full_batch_gd(state, compiled, steps=15, learning_rate=eta, checkpoint_every=3)
        coefficients = vectors.T @ initial.error
        for row in result.history:
            predicted_loss = float(((1 - 2 * eta * values) ** (2 * row.step) * coefficients.square()).sum())
            self.assertAlmostEqual(row.frozen_kernel_loss, predicted_loss, places=13)


if __name__ == "__main__":
    unittest.main()
