"""Constraint data structures and exact-target utilities for theorem audits.

Every scalar constraint is represented as a finite linear combination of
pointwise derivatives.  This representation covers both the pointwise
differential operators in the paper and quadrature discretizations of weak or
nonlocal operators.  It also gives the experiment engine one common route for
forming prediction vectors and empirical NTKs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable

import torch


ExactSolution = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Term:
    """One coefficient times one output derivative at one input point."""

    point: tuple[float, ...]
    output: int
    alpha: tuple[int, ...]
    coefficient: float


@dataclass(frozen=True)
class Constraint:
    """A scalar linear functional, its target, and its normalization group."""

    name: str
    group: str
    terms: tuple[Term, ...]
    target: float
    physical_point: tuple[float, ...] | None


@dataclass(frozen=True)
class Problem:
    """A deterministic manufactured PDE problem used by the audit suite."""

    name: str
    family: str
    positivity_regime: str
    input_dim: int
    output_dim: int
    constraints: tuple[Constraint, ...]
    exact_solution: ExactSolution
    evaluation_points: torch.Tensor
    description: str
    positivity_route: str = "pointwise_dntk"
    trial_space: str | None = None
    independence_witness: str | None = None
    positivity_scope: str = "exact_discrete_prediction_map"


@dataclass(frozen=True)
class LocalRankCertificate:
    """Finite-dimensional certificate used by the automatic positivity result."""

    physical_point: tuple[float, ...]
    constraint_names: tuple[str, ...]
    columns: tuple[tuple[int, tuple[int, ...]], ...]
    matrix: torch.Tensor
    rank: int
    smallest_singular_value: float
    full_row_rank: bool


def _solution_batch(
    exact_solution: ExactSolution,
    point: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor([point], dtype=torch.float64, requires_grad=True)
    values = exact_solution(inputs)
    if not isinstance(values, torch.Tensor) or values.ndim != 2 or values.shape[0] != 1:
        raise ValueError("exact_solution must return a torch tensor of shape [n, q]")
    if values.dtype != torch.float64:
        raise ValueError("exact_solution must preserve torch.float64 inputs")
    return inputs, values


def differentiate_exact(
    exact_solution: ExactSolution,
    point: tuple[float, ...],
    output: int,
    alpha: tuple[int, ...],
) -> torch.Tensor:
    """Evaluate ``D**alpha exact_solution[..., output]`` using autograd.

    The returned value is a detached scalar tensor in FP64.  The manufactured
    solutions in :mod:`problems` are deliberately smooth and non-polynomial,
    so all requested higher derivatives retain an autograd graph.
    """

    if len(point) != len(alpha):
        raise ValueError("point and alpha must have the same dimension")
    if output < 0 or any(order < 0 for order in alpha):
        raise ValueError("output and derivative orders must be non-negative")

    inputs, values = _solution_batch(exact_solution, point)
    if output >= values.shape[1]:
        raise ValueError(f"output index {output} is outside q={values.shape[1]}")
    derivative = values[0, output]
    for dimension, order in enumerate(alpha):
        for _ in range(order):
            if not derivative.requires_grad:
                # A genuinely zero derivative may lose its graph (for example,
                # differentiating a component independent of one coordinate).
                return torch.zeros((), dtype=torch.float64)
            gradient = torch.autograd.grad(
                derivative,
                inputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if gradient is None:
                return torch.zeros((), dtype=torch.float64)
            derivative = gradient[0, dimension]
    return derivative.detach().to(dtype=torch.float64)


def evaluate_constraint(
    exact_solution: ExactSolution,
    constraint: Constraint,
) -> float:
    """Evaluate a constraint on the exact solution using its stored terms."""

    value = torch.zeros((), dtype=torch.float64)
    for term in constraint.terms:
        value = value + float(term.coefficient) * differentiate_exact(
            exact_solution,
            term.point,
            term.output,
            term.alpha,
        )
    return float(value)


def compute_exact_targets(
    exact_solution: ExactSolution,
    constraints: Iterable[Constraint],
) -> torch.Tensor:
    """Return exact (or exactly discretized) targets in constraint order."""

    return torch.tensor(
        [evaluate_constraint(exact_solution, constraint) for constraint in constraints],
        dtype=torch.float64,
    )


def constraint_group_sizes(problem: Problem) -> dict[str, int]:
    """Count scalar rows in each loss-normalization group."""

    sizes: dict[str, int] = {}
    for constraint in problem.constraints:
        sizes[constraint.group] = sizes.get(constraint.group, 0) + 1
    return sizes


def with_exact_targets(
    exact_solution: ExactSolution,
    constraints: Iterable[Constraint],
) -> tuple[Constraint, ...]:
    """Copy constraints and populate targets from the same stored functional."""

    return tuple(
        replace(constraint, target=evaluate_constraint(exact_solution, constraint))
        for constraint in constraints
    )


def local_operator_matrix(
    problem: Problem,
    physical_point: tuple[float, ...],
) -> tuple[torch.Tensor, tuple[str, ...], tuple[tuple[int, tuple[int, ...]], ...]]:
    """Build the loss-scaled local coefficient matrix at one location.

    Positive row scales preserve rank but change left null vectors. Including
    them is essential when comparing coefficient and empirical-NTK nullspaces.
    """

    local = tuple(
        constraint
        for constraint in problem.constraints
        if constraint.physical_point == physical_point
    )
    if not local:
        raise KeyError(f"no constraints at physical point {physical_point}")

    for constraint in local:
        if any(term.point != physical_point for term in constraint.terms):
            raise ValueError(
                f"constraint {constraint.name!r} is marked pointwise but contains "
                "a term at a different point"
            )

    columns = tuple(
        sorted(
            {(term.output, term.alpha) for constraint in local for term in constraint.terms},
            key=lambda item: (item[0], item[1]),
        )
    )
    column_index = {column: index for index, column in enumerate(columns)}
    group_sizes = constraint_group_sizes(problem)
    matrix = torch.zeros((len(local), len(columns)), dtype=torch.float64)
    for row, constraint in enumerate(local):
        scale = 1.0 / math.sqrt(group_sizes[constraint.group])
        for term in constraint.terms:
            matrix[row, column_index[(term.output, term.alpha)]] += scale * term.coefficient
    return matrix, tuple(constraint.name for constraint in local), columns


def _local_operator_rank_certificate_objects(
    problem: Problem,
) -> tuple[LocalRankCertificate, ...]:
    """Build typed rank certificates before converting them for persistence."""

    points = sorted(
        {
            constraint.physical_point
            for constraint in problem.constraints
            if constraint.physical_point is not None
        }
    )
    certificates: list[LocalRankCertificate] = []
    for point in points:
        assert point is not None
        matrix, names, columns = local_operator_matrix(problem, point)
        singular_values = torch.linalg.svdvals(matrix)
        rank = int(torch.linalg.matrix_rank(matrix).item())
        sigma_min = (
            float(singular_values[-1])
            if singular_values.numel() and matrix.shape[0] <= matrix.shape[1]
            else 0.0
        )
        certificates.append(
            LocalRankCertificate(
                physical_point=point,
                constraint_names=names,
                columns=columns,
                matrix=matrix,
                rank=rank,
                smallest_singular_value=sigma_min,
                full_row_rank=rank == matrix.shape[0],
            )
        )
    return tuple(certificates)


def local_operator_rank_certificates(problem: Problem) -> dict[str, object]:
    """Return a JSON-serializable local-rank certificate.

    ``passed`` is the exact design-side condition used by the paper's automatic
    positivity theorem: every local coefficient family is linearly independent
    (equivalently, each matrix has full row rank).  The matrices themselves are
    included as plain nested lists to make the audit artifact independently
    inspectable.
    """

    objects = _local_operator_rank_certificate_objects(problem)
    locations = [
        {
            "physical_point": list(certificate.physical_point),
            "constraint_names": list(certificate.constraint_names),
            "columns": [
                {"output": output, "alpha": list(alpha)}
                for output, alpha in certificate.columns
            ],
            "matrix": certificate.matrix.tolist(),
            "num_rows": int(certificate.matrix.shape[0]),
            "num_columns": int(certificate.matrix.shape[1]),
            "rank": certificate.rank,
            "smallest_singular_value": certificate.smallest_singular_value,
            "full_row_rank": certificate.full_row_rank,
        }
        for certificate in objects
    ]
    return {
        "problem": problem.name,
        "positivity_regime": problem.positivity_regime,
        "positivity_route": problem.positivity_route,
        "certificate_type": "local_operator_coefficient_rank",
        "trial_space": problem.trial_space,
        "independence_witness": problem.independence_witness,
        "positivity_scope": problem.positivity_scope,
        "passed": bool(locations) and all(
            bool(location["full_row_rank"]) for location in locations
        ),
        "locations": locations,
    }


def functional_rank_certificate(
    problem: Problem,
    *,
    require_measure_atoms: bool = False,
) -> dict[str, object]:
    """Certify independence of the exact finite functionals in the loss.

    Each distinct ``(point, output, derivative multiindex)`` is an evaluation
    atom.  Such finite jet evaluations are linearly independent on ``C^r``;
    for value-only rows they are Dirac measures on ``C(D)``.  Therefore full
    row rank of this coefficient matrix is precisely the design-side
    independence condition in the paper's weak-functional theorem, and in the
    value-only case in its signed-measure nonlocal theorem.  Positive
    loss-group scales are included so the stored matrix represents the actual
    prediction map.
    """

    group_sizes = constraint_group_sizes(problem)
    atoms = tuple(
        sorted(
            {
                (term.point, term.output, term.alpha)
                for constraint in problem.constraints
                for term in constraint.terms
            },
            key=lambda item: (item[0], item[1], item[2]),
        )
    )
    if require_measure_atoms and any(any(alpha) for _, _, alpha in atoms):
        return {
            "problem": problem.name,
            "positivity_regime": problem.positivity_regime,
            "positivity_route": problem.positivity_route,
            "certificate_type": "signed_measure_coefficient_rank",
            "trial_space": problem.trial_space,
            "independence_witness": problem.independence_witness,
            "positivity_scope": problem.positivity_scope,
            "passed": False,
            "reason": "derivative atoms are distributions, not finite signed measures",
            "constraint_names": [constraint.name for constraint in problem.constraints],
            "atoms": [],
            "matrix": [],
            "num_rows": len(problem.constraints),
            "num_columns": 0,
            "rank": 0,
            "smallest_singular_value": 0.0,
            "full_row_rank": False,
        }

    atom_index = {atom: column for column, atom in enumerate(atoms)}
    matrix = torch.zeros(
        (len(problem.constraints), len(atoms)), dtype=torch.float64
    )
    for row, constraint in enumerate(problem.constraints):
        scale = 1.0 / math.sqrt(group_sizes[constraint.group])
        for term in constraint.terms:
            atom = (term.point, term.output, term.alpha)
            matrix[row, atom_index[atom]] += scale * term.coefficient

    singular_values = torch.linalg.svdvals(matrix)
    rank = int(torch.linalg.matrix_rank(matrix).item())
    sigma_min = (
        float(singular_values[-1])
        if singular_values.numel() and matrix.shape[0] <= matrix.shape[1]
        else 0.0
    )
    full_row_rank = rank == matrix.shape[0]
    return {
        "problem": problem.name,
        "positivity_regime": problem.positivity_regime,
        "positivity_route": problem.positivity_route,
        "certificate_type": (
            "signed_measure_coefficient_rank"
            if require_measure_atoms
            else "finite_jet_functional_coefficient_rank"
        ),
        "trial_space": problem.trial_space,
        "independence_witness": problem.independence_witness,
        "positivity_scope": problem.positivity_scope,
        "passed": full_row_rank,
        "constraint_names": [constraint.name for constraint in problem.constraints],
        "atoms": [
            {
                "point": list(point),
                "output": output,
                "alpha": list(alpha),
            }
            for point, output, alpha in atoms
        ],
        "matrix": matrix.tolist(),
        "num_rows": int(matrix.shape[0]),
        "num_columns": int(matrix.shape[1]),
        "rank": rank,
        "smallest_singular_value": sigma_min,
        "full_row_rank": full_row_rank,
    }


def positivity_certificate(problem: Problem) -> dict[str, object]:
    """Return the theorem-specific design certificate for one problem."""

    if problem.positivity_route == "pointwise_dntk":
        return local_operator_rank_certificates(problem)
    if problem.positivity_route == "weak_functional":
        return functional_rank_certificate(problem)
    if problem.positivity_route == "nonlocal_measure":
        return functional_rank_certificate(problem, require_measure_atoms=True)
    if problem.positivity_route == "numerical":
        return {
            "problem": problem.name,
            "positivity_regime": problem.positivity_regime,
            "positivity_route": problem.positivity_route,
            "certificate_type": "none",
            "trial_space": problem.trial_space,
            "independence_witness": problem.independence_witness,
            "positivity_scope": problem.positivity_scope,
            "passed": False,
            "reason": "no analytic positivity theorem has been assigned",
        }
    raise ValueError(f"unknown positivity route {problem.positivity_route!r}")


def _canonical_terms(constraint: Constraint) -> tuple[tuple[object, ...], ...]:
    """Combine equal terms into a stable signature for duplicate-row checks."""

    combined: dict[tuple[tuple[float, ...], int, tuple[int, ...]], float] = {}
    for term in constraint.terms:
        key = (term.point, term.output, term.alpha)
        combined[key] = combined.get(key, 0.0) + term.coefficient
    return tuple(
        sorted(
            (
                point,
                output,
                alpha,
                round(coefficient, 15),
            )
            for (point, output, alpha), coefficient in combined.items()
            if coefficient != 0.0
        )
    )


def validate_problem(problem: Problem, target_atol: float = 2e-11) -> None:
    """Raise ``ValueError`` when registry invariants are violated."""

    if not problem.name or not problem.family:
        raise ValueError("problem name and family must be non-empty")
    if problem.positivity_regime not in {"automatic", "numerical"}:
        raise ValueError("positivity_regime must be 'automatic' or 'numerical'")
    valid_routes = {
        "pointwise_dntk",
        "weak_functional",
        "nonlocal_measure",
        "numerical",
    }
    if problem.positivity_route not in valid_routes:
        raise ValueError(
            f"positivity_route must be one of {sorted(valid_routes)}, "
            f"got {problem.positivity_route!r}"
        )
    if (problem.positivity_regime == "automatic") != (
        problem.positivity_route != "numerical"
    ):
        raise ValueError("automatic regime and theorem-backed positivity route disagree")
    if problem.input_dim < 1 or problem.output_dim < 1:
        raise ValueError("input_dim and output_dim must be positive")
    if not problem.constraints:
        raise ValueError("a problem must contain at least one constraint")
    points = problem.evaluation_points
    if points.dtype != torch.float64 or points.ndim != 2 or points.shape[1] != problem.input_dim:
        raise ValueError("evaluation_points must be an FP64 tensor of shape [n, input_dim]")
    if not torch.isfinite(points).all():
        raise ValueError("evaluation_points contain a non-finite value")
    values = problem.exact_solution(points.clone().requires_grad_(True))
    if (
        not isinstance(values, torch.Tensor)
        or values.dtype != torch.float64
        or values.shape != (points.shape[0], problem.output_dim)
        or not torch.isfinite(values).all()
    ):
        raise ValueError("exact_solution must return finite FP64 values of shape [n, output_dim]")

    names: set[str] = set()
    signatures: dict[tuple[tuple[object, ...], ...], str] = {}
    for constraint in problem.constraints:
        if not constraint.name or constraint.name in names:
            raise ValueError(f"constraint names must be unique; got {constraint.name!r}")
        names.add(constraint.name)
        if (
            not constraint.group
            or constraint.group != constraint.group.strip()
            or not constraint.terms
            or not math.isfinite(constraint.target)
        ):
            raise ValueError(f"invalid group, terms, or target in {constraint.name!r}")
        for term in constraint.terms:
            if len(term.point) != problem.input_dim or len(term.alpha) != problem.input_dim:
                raise ValueError(f"dimension mismatch in constraint {constraint.name!r}")
            if term.output < 0 or term.output >= problem.output_dim:
                raise ValueError(f"invalid output in constraint {constraint.name!r}")
            if any(order < 0 for order in term.alpha):
                raise ValueError(f"negative derivative order in {constraint.name!r}")
            if not math.isfinite(term.coefficient) or term.coefficient == 0.0:
                raise ValueError(f"zero/non-finite coefficient in {constraint.name!r}")
            if not all(math.isfinite(coordinate) for coordinate in term.point):
                raise ValueError(f"non-finite point in {constraint.name!r}")
        if constraint.physical_point is not None:
            if len(constraint.physical_point) != problem.input_dim:
                raise ValueError(f"invalid physical_point in {constraint.name!r}")
            if any(term.point != constraint.physical_point for term in constraint.terms):
                raise ValueError(f"nonlocal terms in pointwise constraint {constraint.name!r}")
        if (
            problem.positivity_route == "pointwise_dntk"
            and constraint.physical_point is None
        ):
            raise ValueError("pointwise-DNTK problems may contain only pointwise constraints")

        signature = _canonical_terms(constraint)
        if signature in signatures:
            raise ValueError(
                f"duplicate operator rows {signatures[signature]!r} and {constraint.name!r}"
            )
        signatures[signature] = constraint.name
        expected = evaluate_constraint(problem.exact_solution, constraint)
        if not math.isclose(expected, constraint.target, rel_tol=2e-11, abs_tol=target_atol):
            raise ValueError(
                f"target mismatch in {constraint.name!r}: stored={constraint.target}, exact={expected}"
            )

    group_sizes = constraint_group_sizes(problem)
    if (
        not group_sizes
        or any(size <= 0 for size in group_sizes.values())
        or sum(group_sizes.values()) != len(problem.constraints)
    ):
        raise ValueError("invalid constraint-group sizes")

    if problem.positivity_regime == "automatic":
        certificate = positivity_certificate(problem)
        if not certificate["passed"]:
            raise ValueError(
                f"automatic positivity certificate failed for {problem.name}: "
                f"{certificate.get('reason', certificate.get('locations', 'rank deficient'))}"
            )
