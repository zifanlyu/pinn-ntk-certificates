"""Exact finite-width and limiting-NTK engine for the theorem audit.

The implemented network is exactly the paper's conventional biased shallow
architecture (``gamma = 1``), with no global output bias,

    u_q(x) = m^{-1/2} sum_j a[j, q] tanh(w[j]^T x + b[j]).

All coordinates of ``(w, b, a)`` are trainable and initialized independently
from ``Uniform[-1, 1]``.  Finite-width gradients and full-loss gradient-descent
steps are evaluated analytically, without autograd.  Every public numerical
path requires FP64.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

from .features import CompiledConstraints, DTYPE, FeatureBatch, evaluate_features


def _symmetric(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.T)


def _relative_frobenius(matrix: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.matrix_norm(reference, ord="fro")
    tiny = torch.finfo(DTYPE).tiny
    return float(
        (torch.linalg.matrix_norm(matrix - reference, ord="fro") / denominator.clamp_min(tiny))
        .detach()
        .cpu()
    )


def _relative_operator_norm(matrix: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = _operator_norm_symmetric(reference)
    tiny = torch.finfo(DTYPE).tiny
    return _operator_norm_symmetric(matrix - reference) / max(denominator, tiny)


def _operator_norm_symmetric(matrix: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(_symmetric(matrix))
    return float(eigenvalues.abs().max().detach().cpu())


@dataclass(frozen=True)
class NetworkState:
    """Trainable tensors of the theorem architecture.

    ``w``, ``b`` and ``a`` have shapes ``[m,d]``, ``[m]`` and ``[m,q]``.
    Tensors never require gradients because this engine uses exact manual
    derivatives.
    """

    w: torch.Tensor
    b: torch.Tensor
    a: torch.Tensor

    def __post_init__(self) -> None:
        if self.w.dtype != DTYPE or self.b.dtype != DTYPE or self.a.dtype != DTYPE:
            raise TypeError("NetworkState tensors must all use torch.float64")
        if self.w.ndim != 2 or self.a.ndim != 2 or self.b.ndim != 1:
            raise ValueError("expected w[m,d], b[m], and a[m,q]")
        if self.w.shape[0] == 0 or self.w.shape[1] == 0 or self.a.shape[1] == 0:
            raise ValueError("width, input dimension, and output dimension must be positive")
        if self.b.shape[0] != self.w.shape[0] or self.a.shape[0] != self.w.shape[0]:
            raise ValueError("w, b, and a must have the same width")
        if not (self.w.device == self.b.device == self.a.device):
            raise ValueError("w, b, and a must be on the same device")
        if self.w.requires_grad or self.b.requires_grad or self.a.requires_grad:
            raise ValueError("NetworkState tensors must not require autograd")

    @property
    def width(self) -> int:
        return int(self.w.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.w.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.a.shape[1])

    @property
    def device(self) -> torch.device:
        return self.w.device

    def packed_per_neuron(self) -> torch.Tensor:
        """Return parameters ordered as ``(w_1,...,w_d,b,a_1,...,a_q)``."""

        return torch.cat((self.w, self.b[:, None], self.a), dim=1)

    def clone(self) -> "NetworkState":
        return NetworkState(
            self.w.detach().clone(), self.b.detach().clone(), self.a.detach().clone()
        )

    def tensor_dict(self) -> dict[str, torch.Tensor]:
        """Return detached tensor copies suitable for NPZ/checkpoint serialization."""

        return {
            "w": self.w.detach().clone(),
            "b": self.b.detach().clone(),
            "a": self.a.detach().clone(),
        }

    def prefix(self, width: int) -> "NetworkState":
        """Return an independent clone of the first ``width`` neurons."""

        if not 1 <= width <= self.width:
            raise ValueError(f"width must lie in [1, {self.width}], got {width}")
        return NetworkState(
            self.w[:width].detach().clone(),
            self.b[:width].detach().clone(),
            self.a[:width].detach().clone(),
        )

    def to(self, device: torch.device | str) -> "NetworkState":
        return NetworkState(
            self.w.detach().to(device=device, dtype=DTYPE),
            self.b.detach().to(device=device, dtype=DTYPE),
            self.a.detach().to(device=device, dtype=DTYPE),
        )


@dataclass(frozen=True)
class InitializationBank:
    """A maximum-width IID draw from which every tested width takes a prefix."""

    state: NetworkState
    seed: int

    def prefix(self, width: int) -> NetworkState:
        return self.state.prefix(width)


def make_initialization_bank(
    max_width: int,
    input_dim: int,
    output_dim: int,
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> InitializationBank:
    """Draw one nested-prefix ``Uniform[-1,1]`` initialization bank.

    A single ``[max_width, d+1+q]`` CPU draw makes the result independent of
    how widths are later requested and reproducible across compute devices.
    """

    if max_width <= 0 or input_dim <= 0 or output_dim <= 0:
        raise ValueError("max_width, input_dim, and output_dim must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    combined = 2.0 * torch.rand(
        (max_width, input_dim + 1 + output_dim),
        generator=generator,
        dtype=DTYPE,
        device="cpu",
    ) - 1.0
    combined = combined.to(device=device, dtype=DTYPE)
    state = NetworkState(
        w=combined[:, :input_dim].clone(),
        b=combined[:, input_dim].clone(),
        a=combined[:, input_dim + 1 :].clone(),
    )
    return InitializationBank(state=state, seed=int(seed))


def _compiled_on(state: NetworkState, compiled: CompiledConstraints) -> CompiledConstraints:
    if state.input_dim != compiled.input_dim or state.output_dim != compiled.output_dim:
        raise ValueError(
            "state dimensions do not match compiled constraints: "
            f"state ({state.input_dim}, {state.output_dim}) versus "
            f"constraints ({compiled.input_dim}, {compiled.output_dim})"
        )
    return compiled if compiled.device == state.device else compiled.to(state.device)


@torch.no_grad()
def network_output(state: NetworkState, points: torch.Tensor) -> torch.Tensor:
    """Evaluate ``u`` at arbitrary points, returning an ``[n_points,q]`` tensor."""

    points = points.to(device=state.device, dtype=DTYPE)
    if points.ndim != 2 or points.shape[1] != state.input_dim:
        raise ValueError(
            f"points must have shape [n, {state.input_dim}], got {tuple(points.shape)}"
        )
    hidden = torch.tanh(points @ state.w.T + state.b[None, :])
    return (hidden @ state.a) / math.sqrt(state.width)


@dataclass(frozen=True)
class FiniteEvaluation:
    """Prediction, exact Jacobian building blocks, and finite empirical NTK."""

    prediction: torch.Tensor
    error: torch.Tensor
    per_neuron_gradient: torch.Tensor
    kernel: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        return torch.dot(self.error, self.error)


@torch.no_grad()
def prediction_and_per_neuron_gradient(
    state: NetworkState,
    compiled: CompiledConstraints,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized predictions and ``G_j = grad_(w_j,b_j,a_j) h_j``.

    The full prediction Jacobian is ``J_j = G_j / sqrt(m)``.  The last axis of
    ``G`` follows ``NetworkState.packed_per_neuron``.
    """

    compiled = _compiled_on(state, compiled)
    features = evaluate_features(compiled, state.w, state.b)
    neuron_predictions = torch.einsum("miq,mq->mi", features.output_basis, state.a)
    prediction = neuron_predictions.sum(dim=0) / math.sqrt(state.width)
    hidden_gradient = torch.einsum("miqh,mq->mih", features.hidden_basis, state.a)
    per_neuron_gradient = torch.cat((hidden_gradient, features.output_basis), dim=-1)
    return prediction, per_neuron_gradient


@torch.no_grad()
def evaluate_finite(
    state: NetworkState,
    compiled: CompiledConstraints,
) -> FiniteEvaluation:
    """Evaluate prediction, exact full-loss error, per-neuron gradient and NTK."""

    compiled = _compiled_on(state, compiled)
    prediction, per_neuron_gradient = prediction_and_per_neuron_gradient(state, compiled)
    error = prediction - compiled.targets
    kernel = torch.einsum(
        "mip,mjp->ij", per_neuron_gradient, per_neuron_gradient
    ) / state.width
    return FiniteEvaluation(
        prediction=prediction,
        error=error,
        per_neuron_gradient=per_neuron_gradient,
        kernel=_symmetric(kernel),
    )


@torch.no_grad()
def finite_ntk(state: NetworkState, compiled: CompiledConstraints) -> torch.Tensor:
    """Return the exact empirical NTK ``J J^T``."""

    return evaluate_finite(state, compiled).kernel


def _analytic_a_kernel_sums(features: FeatureBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unnormalized ``(K_full, K_out)`` sums over sampled ``(w,b)``.

    Hidden-gradient products integrate the IID output weights analytically via
    ``E[a_q a_r] = delta_qr / 3`` for ``a_q ~ Uniform[-1,1]``.
    """

    output_sum = torch.einsum(
        "miq,mjq->ij", features.output_basis, features.output_basis
    )
    hidden_sum = torch.einsum(
        "miqh,mjqh->ij", features.hidden_basis, features.hidden_basis
    ) / 3.0
    return _symmetric(output_sum + hidden_sum), _symmetric(output_sum)


@torch.no_grad()
def analytic_a_kernel(
    w: torch.Tensor,
    b: torch.Tensor,
    compiled: CompiledConstraints,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average over sampled ``(w,b)`` and integrate output weights analytically.

    Returns ``(K_full, K_out)``.  ``K_out`` is the output-weight-gradient
    contribution; ``K_full - K_out`` is the hidden-parameter contribution.
    """

    if w.shape[0] == 0:
        raise ValueError("at least one (w,b) sample is required")
    features = evaluate_features(compiled, w, b)
    full_sum, output_sum = _analytic_a_kernel_sums(features)
    return full_sum / w.shape[0], output_sum / w.shape[0]


@dataclass(frozen=True)
class SpectrumSummary:
    eigenvalues: torch.Tensor
    lambda_min: float
    lambda_max: float
    condition_number: float
    numerical_rank: int
    rank_tolerance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "eigenvalues": self.eigenvalues.detach().cpu().tolist(),
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "condition_number": (
                self.condition_number if math.isfinite(self.condition_number) else None
            ),
            "numerical_rank": self.numerical_rank,
            "rank_tolerance": self.rank_tolerance,
        }


@torch.no_grad()
def summarize_spectrum(
    matrix: torch.Tensor,
    *,
    relative_tolerance: float = 1.0e-10,
) -> SpectrumSummary:
    """Return the complete symmetric spectrum and transparent rank metrics."""

    if matrix.dtype != DTYPE or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square torch.float64 tensor")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    eigenvalues = torch.linalg.eigvalsh(_symmetric(matrix))
    lambda_min = float(eigenvalues[0].detach().cpu())
    lambda_max = float(eigenvalues[-1].detach().cpu())
    scale = max(abs(lambda_max), torch.finfo(DTYPE).tiny)
    rank_tolerance = float(relative_tolerance * scale)
    numerical_rank = int((eigenvalues > rank_tolerance).sum().detach().cpu())
    condition_number = (
        lambda_max / lambda_min if lambda_min > rank_tolerance else math.inf
    )
    return SpectrumSummary(
        eigenvalues=eigenvalues.detach().clone(),
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        condition_number=float(condition_number),
        numerical_rank=numerical_rank,
        rank_tolerance=rank_tolerance,
    )


@dataclass(frozen=True)
class PositivityDiagnostics:
    """Conservative two-level randomized-QMC positivity calculation."""

    status: str
    lambda_min: float
    lambda_max: float
    replicate_lambda_min_standard_error: float
    level_difference_operator_norm: float
    confidence_multiplier: float
    lower_bound: float
    relative_lower_bound: float
    relative_tolerance: float
    positivity_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "replicate_lambda_min_standard_error": (
                self.replicate_lambda_min_standard_error
            ),
            "level_difference_operator_norm": self.level_difference_operator_norm,
            "confidence_multiplier": self.confidence_multiplier,
            "lower_bound": self.lower_bound,
            "relative_lower_bound": self.relative_lower_bound,
            "relative_tolerance": self.relative_tolerance,
            "positivity_threshold": self.positivity_threshold,
        }


@dataclass(frozen=True)
class InfiniteNTKReference:
    """Replicated, scrambled-Sobol two-level estimate of the limiting NTK.

    Raw matrices/eigenvalues are tensors suitable for direct NPZ export.  Call
    ``summary_dict`` for a JSON-serializable record of all scalar diagnostics
    and complete mean spectra.
    """

    K_full: torch.Tensor
    K_out: torch.Tensor
    K_full_coarse: torch.Tensor
    K_out_coarse: torch.Tensor
    K_full_standard_error: torch.Tensor
    K_out_standard_error: torch.Tensor
    full_spectrum: SpectrumSummary
    out_spectrum: SpectrumSummary
    full_positivity: PositivityDiagnostics
    out_positivity: PositivityDiagnostics
    full_replicate_lambda_min: torch.Tensor
    out_replicate_lambda_min: torch.Tensor
    coarse_samples: int
    fine_samples: int
    replicates: int
    seed: int

    @property
    def K_hidden(self) -> torch.Tensor:
        return self.K_full - self.K_out

    def summary_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata; raw matrices remain available as attributes."""

        return {
            "method": "replicated_scrambled_sobol_two_level_analytic_a",
            "coarse_samples": self.coarse_samples,
            "fine_samples": self.fine_samples,
            "replicates": self.replicates,
            "seed": self.seed,
            "full_spectrum": self.full_spectrum.as_dict(),
            "out_spectrum": self.out_spectrum.as_dict(),
            "full_positivity": self.full_positivity.as_dict(),
            "out_positivity": self.out_positivity.as_dict(),
            "full_replicate_lambda_min": (
                self.full_replicate_lambda_min.detach().cpu().tolist()
            ),
            "out_replicate_lambda_min": (
                self.out_replicate_lambda_min.detach().cpu().tolist()
            ),
            "full_elementwise_se_frobenius_norm": float(
                torch.linalg.matrix_norm(self.K_full_standard_error, ord="fro")
                .detach()
                .cpu()
            ),
            "out_elementwise_se_frobenius_norm": float(
                torch.linalg.matrix_norm(self.K_out_standard_error, ord="fro")
                .detach()
                .cpu()
            ),
        }


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _positivity_diagnostics(
    fine_mean: torch.Tensor,
    coarse_mean: torch.Tensor,
    replicate_lambda_min: torch.Tensor,
    spectrum: SpectrumSummary,
    relative_tolerance: float,
    confidence_multiplier: float,
) -> PositivityDiagnostics:
    standard_error = float(
        (replicate_lambda_min.std(unbiased=True) / math.sqrt(replicate_lambda_min.numel()))
        .detach()
        .cpu()
    )
    level_difference = _operator_norm_symmetric(fine_mean - coarse_mean)
    lower_bound = (
        spectrum.lambda_min
        - confidence_multiplier * standard_error
        - level_difference
    )
    positivity_threshold = relative_tolerance * max(spectrum.lambda_max, 0.0)
    scale = max(abs(spectrum.lambda_max), torch.finfo(DTYPE).tiny)
    return PositivityDiagnostics(
        status="positive" if lower_bound > positivity_threshold else "inconclusive",
        lambda_min=spectrum.lambda_min,
        lambda_max=spectrum.lambda_max,
        replicate_lambda_min_standard_error=standard_error,
        level_difference_operator_norm=level_difference,
        confidence_multiplier=confidence_multiplier,
        lower_bound=lower_bound,
        relative_lower_bound=lower_bound / scale,
        relative_tolerance=relative_tolerance,
        positivity_threshold=positivity_threshold,
    )


@torch.no_grad()
def estimate_infinite_ntk(
    compiled: CompiledConstraints,
    *,
    fine_samples: int = 65_536,
    coarse_samples: int | None = None,
    replicates: int = 8,
    seed: int = 20_270_001,
    chunk_size: int = 8_192,
    positivity_relative_tolerance: float = 1.0e-10,
    confidence_multiplier: float = 2.365,
) -> InfiniteNTKReference:
    """Estimate ``K^infinity`` by replicated scrambled Sobol integration.

    Each scramble draws one nested fine sequence; its first half is the coarse
    level.  The output weights are not sampled: their contribution to the
    hidden-parameter Gram matrix is integrated exactly using
    ``E[a_q a_r] = delta_qr/3``.  The numerical positivity gate is

    ``lambda_min(K_fine) - c*SE_rep(lambda_min)``
    ``- ||K_fine-K_coarse||_op > tau_rel*lambda_max(K_fine)``.

    Failure of this conservative inequality is reported as ``inconclusive``,
    never as proof of singularity.
    """

    if coarse_samples is None:
        coarse_samples = fine_samples // 2
    if not _is_power_of_two(fine_samples) or not _is_power_of_two(coarse_samples):
        raise ValueError("fine_samples and coarse_samples must be powers of two")
    if fine_samples != 2 * coarse_samples:
        raise ValueError("the nested two-level estimate requires fine_samples=2*coarse_samples")
    if replicates < 2:
        raise ValueError("at least two independent scrambles are required for uncertainty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if positivity_relative_tolerance < 0 or confidence_multiplier < 0:
        raise ValueError("positivity tolerances and multipliers must be non-negative")

    n_constraints = compiled.count
    device = compiled.device
    fine_full_replicates: list[torch.Tensor] = []
    fine_out_replicates: list[torch.Tensor] = []
    coarse_full_replicates: list[torch.Tensor] = []
    coarse_out_replicates: list[torch.Tensor] = []

    for replicate in range(replicates):
        sobol_seed = (int(seed) + replicate) % (2**31 - 1)
        sobol = torch.quasirandom.SobolEngine(
            dimension=compiled.input_dim + 1, scramble=True, seed=sobol_seed
        )
        fine_full_sum = torch.zeros(
            (n_constraints, n_constraints), dtype=DTYPE, device=device
        )
        fine_out_sum = torch.zeros_like(fine_full_sum)
        coarse_full_sum = torch.zeros_like(fine_full_sum)
        coarse_out_sum = torch.zeros_like(fine_full_sum)
        offset = 0
        while offset < fine_samples:
            draw_count = min(chunk_size, fine_samples - offset)
            samples = 2.0 * sobol.draw(draw_count, dtype=DTYPE) - 1.0
            samples = samples.to(device=device, dtype=DTYPE)
            w = samples[:, : compiled.input_dim]
            b = samples[:, compiled.input_dim]
            features = evaluate_features(compiled, w, b)
            chunk_full, chunk_out = _analytic_a_kernel_sums(features)
            fine_full_sum += chunk_full
            fine_out_sum += chunk_out

            coarse_count = max(0, min(draw_count, coarse_samples - offset))
            if coarse_count == draw_count:
                coarse_full_sum += chunk_full
                coarse_out_sum += chunk_out
            elif coarse_count > 0:
                prefix = FeatureBatch(
                    output_basis=features.output_basis[:coarse_count],
                    hidden_basis=features.hidden_basis[:coarse_count],
                )
                prefix_full, prefix_out = _analytic_a_kernel_sums(prefix)
                coarse_full_sum += prefix_full
                coarse_out_sum += prefix_out
            offset += draw_count

        fine_full_replicates.append(_symmetric(fine_full_sum / fine_samples))
        fine_out_replicates.append(_symmetric(fine_out_sum / fine_samples))
        coarse_full_replicates.append(_symmetric(coarse_full_sum / coarse_samples))
        coarse_out_replicates.append(_symmetric(coarse_out_sum / coarse_samples))

    fine_full_stack = torch.stack(fine_full_replicates)
    fine_out_stack = torch.stack(fine_out_replicates)
    coarse_full_stack = torch.stack(coarse_full_replicates)
    coarse_out_stack = torch.stack(coarse_out_replicates)
    K_full = _symmetric(fine_full_stack.mean(dim=0))
    K_out = _symmetric(fine_out_stack.mean(dim=0))
    K_full_coarse = _symmetric(coarse_full_stack.mean(dim=0))
    K_out_coarse = _symmetric(coarse_out_stack.mean(dim=0))
    K_full_standard_error = fine_full_stack.std(dim=0, unbiased=True) / math.sqrt(replicates)
    K_out_standard_error = fine_out_stack.std(dim=0, unbiased=True) / math.sqrt(replicates)

    full_spectrum = summarize_spectrum(
        K_full, relative_tolerance=positivity_relative_tolerance
    )
    out_spectrum = summarize_spectrum(
        K_out, relative_tolerance=positivity_relative_tolerance
    )
    full_replicate_lambda_min = torch.stack(
        [torch.linalg.eigvalsh(matrix)[0] for matrix in fine_full_stack]
    )
    out_replicate_lambda_min = torch.stack(
        [torch.linalg.eigvalsh(matrix)[0] for matrix in fine_out_stack]
    )
    full_positivity = _positivity_diagnostics(
        K_full,
        K_full_coarse,
        full_replicate_lambda_min,
        full_spectrum,
        positivity_relative_tolerance,
        confidence_multiplier,
    )
    out_positivity = _positivity_diagnostics(
        K_out,
        K_out_coarse,
        out_replicate_lambda_min,
        out_spectrum,
        positivity_relative_tolerance,
        confidence_multiplier,
    )
    return InfiniteNTKReference(
        K_full=K_full,
        K_out=K_out,
        K_full_coarse=K_full_coarse,
        K_out_coarse=K_out_coarse,
        K_full_standard_error=K_full_standard_error,
        K_out_standard_error=K_out_standard_error,
        full_spectrum=full_spectrum,
        out_spectrum=out_spectrum,
        full_positivity=full_positivity,
        out_positivity=out_positivity,
        full_replicate_lambda_min=full_replicate_lambda_min,
        out_replicate_lambda_min=out_replicate_lambda_min,
        coarse_samples=coarse_samples,
        fine_samples=fine_samples,
        replicates=replicates,
        seed=int(seed),
    )


@dataclass(frozen=True)
class TrainingRecord:
    """A JSON-serializable (via ``as_dict``) theorem diagnostic checkpoint."""

    step: int
    loss: float
    error_norm: float
    lambda_min: float
    lambda_max: float
    condition_number: float
    numerical_rank: int
    eigenvalues: torch.Tensor
    relative_kernel_drift: float
    relative_kernel_drift_operator: float
    relative_reference_kernel_error: float | None
    relative_reference_kernel_error_operator: float | None
    max_per_neuron_drift: float
    sqrt_width_max_per_neuron_drift: float
    total_parameter_drift: float
    frozen_kernel_loss: float
    reference_kernel_loss: float | None
    relative_error_to_frozen_dynamics: float
    relative_update_taylor_defect: float | None
    jacobian_drift_operator: float = 0.0
    jacobian_drift_over_initial_sqrt_gap: float | None = None
    jacobian_perturbation_gap_lower_bound: float | None = None
    kernel_drift_over_initial_gap: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "error_norm": self.error_norm,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "condition_number": (
                self.condition_number if math.isfinite(self.condition_number) else None
            ),
            "numerical_rank": self.numerical_rank,
            "eigenvalues": self.eigenvalues.detach().cpu().tolist(),
            # The unqualified historical key is the relative Frobenius norm.
            "relative_kernel_drift": self.relative_kernel_drift,
            "relative_kernel_drift_frobenius": self.relative_kernel_drift,
            "relative_kernel_drift_operator": self.relative_kernel_drift_operator,
            "relative_reference_kernel_error": self.relative_reference_kernel_error,
            "relative_reference_kernel_error_frobenius": (
                self.relative_reference_kernel_error
            ),
            "relative_reference_kernel_error_operator": (
                self.relative_reference_kernel_error_operator
            ),
            "max_per_neuron_drift": self.max_per_neuron_drift,
            "sqrt_width_max_per_neuron_drift": (
                self.sqrt_width_max_per_neuron_drift
            ),
            "total_parameter_drift": self.total_parameter_drift,
            "frozen_kernel_loss": self.frozen_kernel_loss,
            "reference_kernel_loss": self.reference_kernel_loss,
            "relative_error_to_frozen_dynamics": self.relative_error_to_frozen_dynamics,
            "relative_update_taylor_defect": self.relative_update_taylor_defect,
            "jacobian_drift_operator": self.jacobian_drift_operator,
            "jacobian_drift_over_initial_sqrt_gap": self.jacobian_drift_over_initial_sqrt_gap,
            "jacobian_perturbation_gap_lower_bound": self.jacobian_perturbation_gap_lower_bound,
            "kernel_drift_over_initial_gap": self.kernel_drift_over_initial_gap,
        }


@dataclass(frozen=True)
class TrainingResult:
    initial_state: NetworkState
    final_state: NetworkState
    initial_kernel: torch.Tensor
    initial_error: torch.Tensor
    history: tuple[TrainingRecord, ...]
    learning_rate: float
    steps: int

    def history_dicts(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.history]

    def summary_dict(self) -> dict[str, Any]:
        """Return JSON-safe run metadata and checkpoint histories."""

        return {
            "architecture": "u_q(x)=sum_j a[j,q]*tanh(w[j]^T*x+b[j])/sqrt(m)",
            "initialization": "iid_uniform[-1,1]_nested_prefix",
            "dtype": "float64",
            "width": self.initial_state.width,
            "input_dim": self.initial_state.input_dim,
            "output_dim": self.initial_state.output_dim,
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "history": self.history_dicts(),
        }


@torch.no_grad()
def full_loss_gd_step(
    state: NetworkState,
    compiled: CompiledConstraints,
    learning_rate: float,
) -> NetworkState:
    """Apply exactly ``theta <- theta - 2 eta J^T(P(theta)-target)``."""

    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    compiled = _compiled_on(state, compiled)
    prediction, per_neuron_gradient = prediction_and_per_neuron_gradient(state, compiled)
    return _apply_full_loss_gradient(
        state, prediction - compiled.targets, per_neuron_gradient, learning_rate
    )


def _apply_full_loss_gradient(
    state: NetworkState,
    error: torch.Tensor,
    per_neuron_gradient: torch.Tensor,
    learning_rate: float,
) -> NetworkState:
    """Update from existing features without forming an unused ``J J^T``."""

    gradient = (2.0 / math.sqrt(state.width)) * torch.einsum(
        "i,mip->mp", error, per_neuron_gradient
    )
    updated = state.packed_per_neuron() - learning_rate * gradient
    d = state.input_dim
    return NetworkState(
        w=updated[:, :d].detach().clone(),
        b=updated[:, d].detach().clone(),
        a=updated[:, d + 1 :].detach().clone(),
    )


def _training_record(
    step: int,
    state: NetworkState,
    initial_state: NetworkState,
    evaluation: FiniteEvaluation,
    initial_kernel: torch.Tensor,
    initial_per_neuron_gradient: torch.Tensor,
    frozen_error: torch.Tensor,
    reference_error: torch.Tensor | None,
    reference_kernel: torch.Tensor | None,
    relative_rank_tolerance: float,
    relative_update_taylor_defect: float | None,
) -> TrainingRecord:
    spectrum = summarize_spectrum(
        evaluation.kernel, relative_tolerance=relative_rank_tolerance
    )
    displacement = state.packed_per_neuron() - initial_state.packed_per_neuron()
    per_neuron = torch.linalg.vector_norm(displacement, dim=1)
    frozen_norm = torch.linalg.vector_norm(frozen_error)
    actual_to_frozen = torch.linalg.vector_norm(evaluation.error - frozen_error)
    frozen_denominator = frozen_norm.clamp_min(torch.finfo(DTYPE).tiny)
    # J has one normalized feature block G_j / sqrt(m) per neuron. Form
    # (J_k-J_0)(J_k-J_0)^T in constraint space instead of a large matrix SVD.
    gradient_difference = evaluation.per_neuron_gradient - initial_per_neuron_gradient
    difference_gram = torch.einsum(
        "mip,mjp->ij", gradient_difference, gradient_difference
    ) / state.width
    jacobian_drift = math.sqrt(max(
        float(torch.linalg.eigvalsh(_symmetric(difference_gram))[-1].detach().cpu()), 0.0
    ))
    initial_gap = float(torch.linalg.eigvalsh(initial_kernel)[0].detach().cpu())
    initial_sqrt_gap = math.sqrt(initial_gap) if initial_gap > 0.0 else None
    kernel_drift = _operator_norm_symmetric(evaluation.kernel - initial_kernel)
    return TrainingRecord(
        step=step,
        loss=float(evaluation.loss.detach().cpu()),
        error_norm=float(torch.linalg.vector_norm(evaluation.error).detach().cpu()),
        lambda_min=spectrum.lambda_min,
        lambda_max=spectrum.lambda_max,
        condition_number=spectrum.condition_number,
        numerical_rank=spectrum.numerical_rank,
        eigenvalues=spectrum.eigenvalues,
        relative_kernel_drift=_relative_frobenius(evaluation.kernel, initial_kernel),
        relative_kernel_drift_operator=_relative_operator_norm(
            evaluation.kernel, initial_kernel
        ),
        relative_reference_kernel_error=(
            None
            if reference_kernel is None
            else _relative_frobenius(evaluation.kernel, reference_kernel)
        ),
        relative_reference_kernel_error_operator=(
            None
            if reference_kernel is None
            else _relative_operator_norm(evaluation.kernel, reference_kernel)
        ),
        max_per_neuron_drift=float(per_neuron.max().detach().cpu()),
        sqrt_width_max_per_neuron_drift=float(
            (math.sqrt(state.width) * per_neuron.max()).detach().cpu()
        ),
        total_parameter_drift=float(torch.linalg.vector_norm(displacement).detach().cpu()),
        frozen_kernel_loss=float(torch.dot(frozen_error, frozen_error).detach().cpu()),
        reference_kernel_loss=(
            None
            if reference_error is None
            else float(torch.dot(reference_error, reference_error).detach().cpu())
        ),
        relative_error_to_frozen_dynamics=float(
            (actual_to_frozen / frozen_denominator).detach().cpu()
        ),
        relative_update_taylor_defect=relative_update_taylor_defect,
        jacobian_drift_operator=jacobian_drift,
        jacobian_drift_over_initial_sqrt_gap=(
            jacobian_drift / initial_sqrt_gap if initial_sqrt_gap is not None else None
        ),
        jacobian_perturbation_gap_lower_bound=(
            max(initial_sqrt_gap - jacobian_drift, 0.0) ** 2
            if initial_sqrt_gap is not None else None
        ),
        kernel_drift_over_initial_gap=(
            kernel_drift / initial_gap if initial_gap > 0.0 else None
        ),
    )


@torch.no_grad()
def train_full_batch_gd(
    initial_state: NetworkState,
    compiled: CompiledConstraints,
    *,
    steps: int,
    learning_rate: float,
    checkpoint_every: int = 1,
    checkpoint_steps: Sequence[int] | None = None,
    reference_kernel: torch.Tensor | None = None,
    relative_rank_tolerance: float = 1.0e-10,
) -> TrainingResult:
    """Run exact full-batch GD and record theorem-facing diagnostics.

    The loss is ``||P(theta)-target||_2**2``.  Histories always contain steps
    zero and ``steps``; each checkpoint contains the full empirical spectrum,
    kernel drift, frozen-kernel comparison, and maximum per-neuron movement.
    ``reference_kernel`` can be the independent Sobol ``K^infinity`` estimate.
    Pass ``checkpoint_steps`` for a fixed, possibly nonuniform schedule; step zero
    and the final step are always included.
    """

    if steps < 0 or checkpoint_every <= 0:
        raise ValueError("steps must be non-negative and checkpoint_every positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    checkpoint_set: set[int] | None = None
    if checkpoint_steps is not None:
        if any(
            not isinstance(checkpoint, int) or checkpoint < 0 or checkpoint > steps
            for checkpoint in checkpoint_steps
        ):
            raise ValueError("checkpoint_steps must contain integers in [0, steps]")
        checkpoint_set = set(checkpoint_steps)
        checkpoint_set.update((0, steps))
    compiled = _compiled_on(initial_state, compiled)
    if reference_kernel is not None:
        reference_kernel = reference_kernel.to(device=initial_state.device, dtype=DTYPE)
        if reference_kernel.shape != (compiled.count, compiled.count):
            raise ValueError("reference_kernel has the wrong shape")
        reference_kernel = _symmetric(reference_kernel)

    origin = initial_state.clone()
    state = initial_state.clone()
    initial_evaluation = evaluate_finite(state, compiled)
    initial_kernel = initial_evaluation.kernel.detach().clone()
    initial_error = initial_evaluation.error.detach().clone()
    frozen_error = initial_error.clone()
    reference_error = initial_error.clone() if reference_kernel is not None else None
    records: list[TrainingRecord] = [
        _training_record(
            0,
            state,
            origin,
            initial_evaluation,
            initial_kernel,
            initial_evaluation.per_neuron_gradient,
            frozen_error,
            reference_error,
            reference_kernel,
            relative_rank_tolerance,
            None,
        )
    ]

    for step in range(1, steps + 1):
        is_checkpoint = (
            step in checkpoint_set
            if checkpoint_set is not None
            else step % checkpoint_every == 0 or step == steps
        )
        before = evaluate_finite(state, compiled) if is_checkpoint else None
        state = (
            full_loss_gd_step(state, compiled, learning_rate)
            if before is None
            else _apply_full_loss_gradient(
                state, before.error, before.per_neuron_gradient, learning_rate
            )
        )
        frozen_error = frozen_error - 2.0 * learning_rate * (initial_kernel @ frozen_error)
        if reference_error is not None and reference_kernel is not None:
            reference_error = reference_error - 2.0 * learning_rate * (
                reference_kernel @ reference_error
            )
        if is_checkpoint:
            evaluation = evaluate_finite(state, compiled)
            assert before is not None
            linear_update = -2.0 * learning_rate * (before.kernel @ before.error)
            update_defect = evaluation.prediction - before.prediction - linear_update
            denominator = torch.linalg.vector_norm(linear_update).clamp_min(
                torch.finfo(DTYPE).tiny
            )
            relative_update_taylor_defect = float(
                (torch.linalg.vector_norm(update_defect) / denominator).detach().cpu()
            )
            records.append(
                _training_record(
                    step,
                    state,
                    origin,
                    evaluation,
                    initial_kernel,
                    initial_evaluation.per_neuron_gradient,
                    frozen_error,
                    reference_error,
                    reference_kernel,
                    relative_rank_tolerance,
                    relative_update_taylor_defect,
                )
            )

    return TrainingResult(
        initial_state=origin,
        final_state=state,
        initial_kernel=initial_kernel,
        initial_error=initial_error,
        history=tuple(records),
        learning_rate=float(learning_rate),
        steps=int(steps),
    )


__all__ = [
    "FiniteEvaluation",
    "InfiniteNTKReference",
    "InitializationBank",
    "NetworkState",
    "PositivityDiagnostics",
    "SpectrumSummary",
    "TrainingRecord",
    "TrainingResult",
    "analytic_a_kernel",
    "estimate_infinite_ntk",
    "evaluate_finite",
    "finite_ntk",
    "full_loss_gd_step",
    "make_initialization_bank",
    "network_output",
    "prediction_and_per_neuron_gradient",
    "summarize_spectrum",
    "train_full_batch_gd",
]
