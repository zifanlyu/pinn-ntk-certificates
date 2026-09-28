"""Differential features for the finite-width PINN theorem audit.

The module compiles a collection of scalar linear differential constraints into
dense tensors and evaluates their exact one-neuron features.  It deliberately
does not use autograd: all input and parameter derivatives follow from

    d^alpha_x tanh(w^T x + b)
      = w^alpha tanh^{(|alpha|)}(w^T x + b).

Every constraint is divided by the square root of the number of constraints in
its loss group, matching the normalization in the paper.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Sequence

import torch


DTYPE = torch.float64


@lru_cache(maxsize=None)
def tanh_derivative_polynomial(order: int) -> tuple[int, ...]:
    """Return coefficients of ``d**order tanh(z) / dz**order`` in ``tanh(z)``.

    Coefficients are in ascending order.  If ``P_n(t)`` represents the
    ``n``-th derivative, the exact integer recurrence is

    ``P_{n+1}(t) = (1 - t**2) P_n'(t)``, with ``P_0(t) = t``.

    The recurrence supports any non-negative order needed by a registered PDE
    without hard-coding derivative formulas.
    """

    if not isinstance(order, int) or order < 0:
        raise ValueError(f"order must be a non-negative integer, got {order!r}")
    coefficients: tuple[int, ...] = (0, 1)
    for _ in range(order):
        derivative = [power * value for power, value in enumerate(coefficients)][1:]
        next_coefficients = [0] * (len(derivative) + 2)
        for power, value in enumerate(derivative):
            next_coefficients[power] += value
            next_coefficients[power + 2] -= value
        while len(next_coefficients) > 1 and next_coefficients[-1] == 0:
            next_coefficients.pop()
        coefficients = tuple(next_coefficients)
    return coefficients


def tanh_derivative(z: torch.Tensor, order: int) -> torch.Tensor:
    """Evaluate an arbitrary derivative of tanh in FP64 using exact polynomials."""

    if z.dtype != DTYPE:
        raise TypeError(f"tanh derivatives require torch.float64, got {z.dtype}")
    values = torch.tanh(z)
    coefficients = tanh_derivative_polynomial(order)
    result = torch.zeros_like(values)
    for coefficient in reversed(coefficients):
        result = result * values + coefficient
    return result


@dataclass(frozen=True)
class CompiledConstraints:
    """Tensor representation of group-normalized linear constraints.

    A term index ``t`` belongs to constraint ``constraint_index[t]`` and has
    the form

    ``coefficient[t] * partial**alpha[t] u_output[t](points[t])``.

    ``coefficient`` already includes the loss-group scale.  ``targets`` are
    scaled by the same factor.
    """

    names: tuple[str, ...]
    groups: tuple[str, ...]
    group_scales: torch.Tensor
    targets: torch.Tensor
    points: torch.Tensor
    alphas: torch.Tensor
    orders: torch.Tensor
    outputs: torch.Tensor
    coefficients: torch.Tensor
    constraint_index: torch.Tensor
    input_dim: int
    output_dim: int

    @property
    def count(self) -> int:
        return len(self.names)

    @property
    def term_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def device(self) -> torch.device:
        return self.points.device

    def to(self, device: torch.device | str) -> "CompiledConstraints":
        """Return a copy whose tensors live on ``device`` (always in FP64)."""

        return CompiledConstraints(
            names=self.names,
            groups=self.groups,
            group_scales=self.group_scales.to(device=device, dtype=DTYPE),
            targets=self.targets.to(device=device, dtype=DTYPE),
            points=self.points.to(device=device, dtype=DTYPE),
            alphas=self.alphas.to(device=device, dtype=torch.long),
            orders=self.orders.to(device=device, dtype=torch.long),
            outputs=self.outputs.to(device=device, dtype=torch.long),
            coefficients=self.coefficients.to(device=device, dtype=DTYPE),
            constraint_index=self.constraint_index.to(device=device, dtype=torch.long),
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )


@dataclass(frozen=True)
class TermFeatures:
    """Hidden-unit values and exact parameter derivatives for every term."""

    phi: torch.Tensor
    dphi_dw: torch.Tensor
    dphi_db: torch.Tensor


@dataclass(frozen=True)
class FeatureBatch:
    """Per-neuron bases for output-weight and hidden-parameter derivatives.

    Shapes for ``m`` neurons, ``n`` constraints, ``q`` outputs and input
    dimension ``d`` are

    - ``output_basis``: ``[m, n, q]`` (derivative with respect to ``a``),
    - ``hidden_basis``: ``[m, n, q, d + 1]`` (coefficient multiplying each
      output weight in the derivatives with respect to ``(w, b)``).
    """

    output_basis: torch.Tensor
    hidden_basis: torch.Tensor


def _constraints_and_dimensions(
    source: Any,
    input_dim: int | None,
    output_dim: int | None,
) -> tuple[Sequence[Any], int, int]:
    if hasattr(source, "constraints"):
        constraints = tuple(source.constraints)
        input_dim = int(source.input_dim if input_dim is None else input_dim)
        output_dim = int(source.output_dim if output_dim is None else output_dim)
    else:
        constraints = tuple(source)
        if input_dim is None or output_dim is None:
            raise ValueError("input_dim and output_dim are required for a constraint sequence")
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")
    if not constraints:
        raise ValueError("at least one constraint is required")
    return constraints, int(input_dim), int(output_dim)


def compile_constraints(
    source: Any,
    input_dim: int | None = None,
    output_dim: int | None = None,
    *,
    device: torch.device | str = "cpu",
) -> CompiledConstraints:
    """Compile a ``Problem`` or sequence of ``Constraint`` objects.

    The function is intentionally structural: the registry's dataclasses need
    only expose the fields documented in ``constraints.py``.  This keeps the
    feature engine independent of a particular problem catalogue.
    """

    constraints, input_dim, output_dim = _constraints_and_dimensions(
        source, input_dim, output_dim
    )
    names = tuple(str(constraint.name) for constraint in constraints)
    groups = tuple(str(constraint.group) for constraint in constraints)
    group_sizes = Counter(groups)
    scales = [1.0 / math.sqrt(group_sizes[group]) for group in groups]

    term_points: list[tuple[float, ...]] = []
    term_alphas: list[tuple[int, ...]] = []
    term_outputs: list[int] = []
    term_coefficients: list[float] = []
    term_constraints: list[int] = []
    targets: list[float] = []

    for constraint_id, (constraint, scale) in enumerate(zip(constraints, scales)):
        target = float(constraint.target)
        if not math.isfinite(target):
            raise ValueError(f"constraint {constraint.name!r} has a non-finite target")
        targets.append(scale * target)
        terms = tuple(constraint.terms)
        if not terms:
            raise ValueError(f"constraint {constraint.name!r} has no terms")
        for term in terms:
            point = tuple(float(value) for value in term.point)
            alpha = tuple(int(value) for value in term.alpha)
            output = int(term.output)
            coefficient = float(term.coefficient)
            if len(point) != input_dim:
                raise ValueError(
                    f"term point in {constraint.name!r} has dimension {len(point)}, "
                    f"expected {input_dim}"
                )
            if len(alpha) != input_dim or any(value < 0 for value in alpha):
                raise ValueError(
                    f"term alpha in {constraint.name!r} must contain {input_dim} "
                    "non-negative integers"
                )
            if not 0 <= output < output_dim:
                raise ValueError(
                    f"term output {output} in {constraint.name!r} is outside "
                    f"[0, {output_dim})"
                )
            if not all(math.isfinite(value) for value in point):
                raise ValueError(f"constraint {constraint.name!r} has a non-finite point")
            if not math.isfinite(coefficient):
                raise ValueError(f"constraint {constraint.name!r} has a non-finite coefficient")
            term_points.append(point)
            term_alphas.append(alpha)
            term_outputs.append(output)
            term_coefficients.append(scale * coefficient)
            term_constraints.append(constraint_id)

    points = torch.tensor(term_points, dtype=DTYPE, device=device)
    alphas = torch.tensor(term_alphas, dtype=torch.long, device=device)
    orders = alphas.sum(dim=1)
    return CompiledConstraints(
        names=names,
        groups=groups,
        group_scales=torch.tensor(scales, dtype=DTYPE, device=device),
        targets=torch.tensor(targets, dtype=DTYPE, device=device),
        points=points,
        alphas=alphas,
        orders=orders,
        outputs=torch.tensor(term_outputs, dtype=torch.long, device=device),
        coefficients=torch.tensor(term_coefficients, dtype=DTYPE, device=device),
        constraint_index=torch.tensor(term_constraints, dtype=torch.long, device=device),
        input_dim=input_dim,
        output_dim=output_dim,
    )


def _validate_hidden_state(w: torch.Tensor, b: torch.Tensor, input_dim: int) -> None:
    if w.dtype != DTYPE or b.dtype != DTYPE:
        raise TypeError("the theorem audit uses torch.float64 throughout")
    if w.ndim != 2 or w.shape[1] != input_dim:
        raise ValueError(f"w must have shape [width, {input_dim}], got {tuple(w.shape)}")
    if b.shape != (w.shape[0],):
        raise ValueError(f"b must have shape [{w.shape[0]}], got {tuple(b.shape)}")
    if w.device != b.device:
        raise ValueError("w and b must be on the same device")


def evaluate_term_features(
    w: torch.Tensor,
    b: torch.Tensor,
    compiled: CompiledConstraints,
) -> TermFeatures:
    """Evaluate all hidden differential terms and their exact ``(w,b)`` gradients."""

    _validate_hidden_state(w, b, compiled.input_dim)
    if compiled.device != w.device:
        compiled = compiled.to(w.device)

    z = w @ compiled.points.T + b[:, None]
    powers = w[:, None, :].pow(compiled.alphas[None, :, :])
    monomial = powers.prod(dim=-1)

    sigma_order = torch.empty_like(z)
    sigma_next = torch.empty_like(z)
    for order in torch.unique(compiled.orders).tolist():
        mask = compiled.orders == order
        sigma_order[:, mask] = tanh_derivative(z[:, mask], int(order))
        sigma_next[:, mask] = tanh_derivative(z[:, mask], int(order) + 1)

    phi = monomial * sigma_order
    dphi_db = monomial * sigma_next
    dphi_dw = torch.empty(
        (w.shape[0], compiled.term_count, compiled.input_dim),
        dtype=DTYPE,
        device=w.device,
    )
    for coordinate in range(compiled.input_dim):
        reduced_alphas = compiled.alphas.clone()
        reduced_alphas[:, coordinate] = torch.clamp(
            reduced_alphas[:, coordinate] - 1, min=0
        )
        reduced_monomial = w[:, None, :].pow(reduced_alphas[None, :, :]).prod(dim=-1)
        polynomial_derivative = (
            compiled.alphas[:, coordinate].to(DTYPE)[None, :]
            * reduced_monomial
            * sigma_order
        )
        affine_derivative = (
            compiled.points[:, coordinate][None, :] * monomial * sigma_next
        )
        dphi_dw[:, :, coordinate] = polynomial_derivative + affine_derivative

    return TermFeatures(phi=phi, dphi_dw=dphi_dw, dphi_db=dphi_db)


def evaluate_features(
    compiled: CompiledConstraints,
    w: torch.Tensor,
    b: torch.Tensor,
) -> FeatureBatch:
    """Aggregate exact term features into normalized constraint features."""

    if compiled.device != w.device:
        compiled = compiled.to(w.device)
    term_features = evaluate_term_features(w, b, compiled)
    width = w.shape[0]
    constraint_output_index = (
        compiled.constraint_index * compiled.output_dim + compiled.outputs
    )
    scatter_index = constraint_output_index[None, :].expand(width, -1)

    weighted_phi = term_features.phi * compiled.coefficients[None, :]
    output_flat = torch.zeros(
        (width, compiled.count * compiled.output_dim), dtype=DTYPE, device=w.device
    )
    output_flat.scatter_add_(1, scatter_index, weighted_phi)
    output_basis = output_flat.reshape(width, compiled.count, compiled.output_dim)

    term_hidden = torch.cat(
        (term_features.dphi_dw, term_features.dphi_db[:, :, None]), dim=-1
    )
    term_hidden = term_hidden * compiled.coefficients[None, :, None]
    hidden_flat = torch.zeros(
        (width, compiled.count * compiled.output_dim, compiled.input_dim + 1),
        dtype=DTYPE,
        device=w.device,
    )
    hidden_flat.scatter_add_(
        1,
        scatter_index[:, :, None].expand(-1, -1, compiled.input_dim + 1),
        term_hidden,
    )
    hidden_basis = hidden_flat.reshape(
        width, compiled.count, compiled.output_dim, compiled.input_dim + 1
    )
    return FeatureBatch(output_basis=output_basis, hidden_basis=hidden_basis)


__all__ = [
    "CompiledConstraints",
    "DTYPE",
    "FeatureBatch",
    "TermFeatures",
    "compile_constraints",
    "evaluate_features",
    "evaluate_term_features",
    "tanh_derivative",
    "tanh_derivative_polynomial",
]
