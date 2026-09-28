"""Deterministic manufactured PDE registry for the finite-width theorem audit.

The pointwise cases satisfy the paper's biased-network DNTK criterion by
construction.  Weak and quadrature-discretized nonlocal cases use the paper's
new exact weak-functional nullspace theorem, with a global jet-atom rank
certificate.  The registry records the applicable theorem route explicitly.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

try:
    from .constraints import Constraint, Problem, Term, validate_problem, with_exact_targets
except ImportError:  # Permit direct execution/import from this directory.
    from constraints import Constraint, Problem, Term, validate_problem, with_exact_targets


DTYPE = torch.float64


def _term(
    point: tuple[float, ...],
    output: int,
    alpha: tuple[int, ...],
    coefficient: float = 1.0,
) -> Term:
    return Term(point, output, alpha, float(coefficient))


def _constraint(
    name: str,
    group: str,
    terms: tuple[Term, ...],
    physical_point: tuple[float, ...] | None,
) -> Constraint:
    return Constraint(name, group, terms, 0.0, physical_point)


def _finish(
    *,
    name: str,
    family: str,
    positivity_regime: str,
    input_dim: int,
    output_dim: int,
    constraints: list[Constraint],
    exact_solution: Callable[[torch.Tensor], torch.Tensor],
    evaluation_points: torch.Tensor,
    description: str,
    positivity_route: str | None = None,
    trial_space: str | None = None,
    independence_witness: str | None = None,
    positivity_scope: str = "exact_discrete_prediction_map",
) -> Problem:
    problem = Problem(
        name=name,
        family=family,
        positivity_regime=positivity_regime,
        input_dim=input_dim,
        output_dim=output_dim,
        constraints=with_exact_targets(exact_solution, constraints),
        exact_solution=exact_solution,
        evaluation_points=evaluation_points.to(dtype=DTYPE),
        description=description,
        positivity_route=(
            positivity_route
            if positivity_route is not None
            else ("pointwise_dntk" if positivity_regime == "automatic" else "numerical")
        ),
        trial_space=trial_space,
        independence_witness=independence_witness,
        positivity_scope=positivity_scope,
    )
    validate_problem(problem)
    return problem


def _grid_1d(left: float = -1.0, right: float = 1.0, count: int = 161) -> torch.Tensor:
    return torch.linspace(left, right, count, dtype=DTYPE).reshape(-1, 1)


def _grid_2d(
    first_bounds: tuple[float, float],
    second_bounds: tuple[float, float],
    first_count: int = 31,
    second_count: int = 33,
) -> torch.Tensor:
    first = torch.linspace(*first_bounds, first_count, dtype=DTYPE)
    second = torch.linspace(*second_bounds, second_count, dtype=DTYPE)
    mesh = torch.meshgrid(first, second, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, 2)


def _gauss_legendre(left: float, right: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    scale = 0.5 * (right - left)
    return scale * nodes + 0.5 * (right + left), scale * weights


def _poisson_1d() -> Problem:
    def exact(x: torch.Tensor) -> torch.Tensor:
        value = torch.sin(1.3 * x[:, 0] + 0.2) + 0.17 * torch.cos(2.1 * x[:, 0] - 0.1)
        return value[:, None]

    constraints: list[Constraint] = []
    for index, coordinate in enumerate((-0.79, -0.34, 0.07, 0.43, 0.82)):
        point = (coordinate,)
        constraints.append(
            _constraint(
                f"poisson_{index}", "interior", (_term(point, 0, (2,), -1.0),), point
            )
        )
    for side, coordinate in (("left", -1.0), ("right", 1.0)):
        point = (coordinate,)
        constraints.append(
            _constraint(f"dirichlet_{side}", "boundary", (_term(point, 0, (0,)),), point)
        )
    return _finish(
        name="poisson_1d",
        family="elliptic",
        positivity_regime="automatic",
        input_dim=1,
        output_dim=1,
        constraints=constraints,
        exact_solution=exact,
        evaluation_points=_grid_1d(),
        description="1D Poisson equation -u_xx=f with two Dirichlet constraints.",
    )


def _variable_elliptic_2d() -> Problem:
    def exact(xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, 0], xy[:, 1]
        value = torch.sin(0.8 * x + 0.3) * torch.cos(1.1 * y - 0.2) + 0.13 * torch.sin(x * y + 0.4)
        return value[:, None]

    constraints: list[Constraint] = []
    interior = ((-0.71, -0.48), (-0.38, 0.26), (0.06, -0.17), (0.41, 0.63), (0.77, -0.31))
    for index, point in enumerate(interior):
        x, y = point
        diffusion = 2.0 + 0.3 * x + 0.2 * y
        terms = (
            _term(point, 0, (2, 0), -diffusion),
            _term(point, 0, (0, 2), -diffusion),
            _term(point, 0, (1, 0), -0.3),
            _term(point, 0, (0, 1), -0.2),
            _term(point, 0, (0, 0), 0.35),
        )
        constraints.append(_constraint(f"elliptic_{index}", "interior", terms, point))

    # n . grad(u) + beta(x,y) u = g on an asymmetric boundary sample.
    boundary = (
        ((-1.0, -0.64), (-1.0, 0.0)),
        ((-1.0, 0.23), (-1.0, 0.0)),
        ((1.0, -0.19), (1.0, 0.0)),
        ((1.0, 0.72), (1.0, 0.0)),
        ((-0.57, -1.0), (0.0, -1.0)),
        ((0.18, -1.0), (0.0, -1.0)),
        ((-0.24, 1.0), (0.0, 1.0)),
        ((0.69, 1.0), (0.0, 1.0)),
    )
    for index, (point, normal) in enumerate(boundary):
        beta = 0.75 + 0.08 * point[0] - 0.05 * point[1]
        terms = [
            _term(point, 0, (0, 0), beta),
        ]
        if normal[0] != 0.0:
            terms.append(_term(point, 0, (1, 0), normal[0]))
        if normal[1] != 0.0:
            terms.append(_term(point, 0, (0, 1), normal[1]))
        constraints.append(_constraint(f"robin_{index}", "robin_boundary", tuple(terms), point))
    return _finish(
        name="variable_elliptic_2d",
        family="elliptic",
        positivity_regime="automatic",
        input_dim=2,
        output_dim=1,
        constraints=constraints,
        exact_solution=exact,
        evaluation_points=_grid_2d((-1.0, 1.0), (-1.0, 1.0)),
        description="Variable-coefficient 2D elliptic equation -div(a grad u)+0.35u=f with Robin data.",
    )


def _heat_1d() -> Problem:
    def exact(tx: torch.Tensor) -> torch.Tensor:
        t, x = tx[:, 0], tx[:, 1]
        value = torch.exp(-0.7 * t) * (torch.sin(1.2 * x + 0.25) + 0.2 * torch.cos(0.7 * x - 0.1))
        return value[:, None]

    constraints: list[Constraint] = []
    interior = ((0.13, -0.72), (0.24, 0.18), (0.39, -0.27), (0.56, 0.69), (0.73, -0.51), (0.91, 0.34))
    for index, point in enumerate(interior):
        constraints.append(
            _constraint(
                f"heat_{index}",
                "interior",
                (_term(point, 0, (1, 0)), _term(point, 0, (0, 2), -0.35)),
                point,
            )
        )
    for index, x in enumerate((-0.83, -0.29, 0.14, 0.58, 0.91)):
        point = (0.0, x)
        constraints.append(_constraint(f"initial_{index}", "initial", (_term(point, 0, (0, 0)),), point))
    for index, (t, x) in enumerate(((0.17, -1.0), (0.48, -1.0), (0.86, -1.0), (0.29, 1.0), (0.67, 1.0), (0.94, 1.0))):
        point = (t, x)
        constraints.append(_constraint(f"boundary_{index}", "boundary", (_term(point, 0, (0, 0)),), point))
    return _finish(
        name="heat_1d",
        family="parabolic",
        positivity_regime="automatic",
        input_dim=2,
        output_dim=1,
        constraints=constraints,
        exact_solution=exact,
        evaluation_points=_grid_2d((0.0, 1.0), (-1.0, 1.0)),
        description="Manufactured 1D heat equation u_t-0.35u_xx=f with initial and Dirichlet data.",
    )


def _transport_1d() -> Problem:
    def exact(tx: torch.Tensor) -> torch.Tensor:
        t, x = tx[:, 0], tx[:, 1]
        value = torch.sin(0.9 * x - 0.8 * t + 0.35) + 0.16 * torch.cos(1.3 * x + 0.4 * t)
        return value[:, None]

    constraints: list[Constraint] = []
    interior = ((0.11, -0.66), (0.22, 0.09), (0.37, -0.24), (0.53, 0.73), (0.76, -0.39), (0.92, 0.41))
    for index, point in enumerate(interior):
        t, x = point
        speed = 0.65 + 0.12 * x + 0.08 * t
        constraints.append(
            _constraint(
                f"transport_{index}",
                "interior",
                (
                    _term(point, 0, (1, 0)),
                    _term(point, 0, (0, 1), speed),
                    _term(point, 0, (0, 0), 0.25),
                ),
                point,
            )
        )
    for index, x in enumerate((-0.78, -0.31, 0.12, 0.49, 0.87)):
        point = (0.0, x)
        constraints.append(_constraint(f"initial_{index}", "initial", (_term(point, 0, (0, 0)),), point))
    for index, t in enumerate((0.16, 0.43, 0.71, 0.93)):
        point = (t, -1.0)
        constraints.append(_constraint(f"inflow_{index}", "inflow", (_term(point, 0, (0, 0)),), point))
    return _finish(
        name="transport_1d",
        family="hyperbolic",
        positivity_regime="automatic",
        input_dim=2,
        output_dim=1,
        constraints=constraints,
        exact_solution=exact,
        evaluation_points=_grid_2d((0.0, 1.0), (-1.0, 1.0)),
        description="Variable-speed 1D transport-reaction equation with initial and inflow data.",
    )


def _wave_1d() -> Problem:
    def exact(tx: torch.Tensor) -> torch.Tensor:
        t, x = tx[:, 0], tx[:, 1]
        value = torch.cos(1.1 * t + 0.1) * torch.sin(0.9 * x + 0.4) + 0.15 * torch.sin(0.7 * t + 1.3 * x)
        return value[:, None]

    constraints: list[Constraint] = []
    interior = ((0.12, -0.58), (0.28, 0.21), (0.44, -0.33), (0.61, 0.74), (0.79, -0.47), (0.93, 0.38))
    for index, point in enumerate(interior):
        constraints.append(
            _constraint(
                f"wave_{index}",
                "interior",
                (_term(point, 0, (2, 0)), _term(point, 0, (0, 2), -(0.8**2))),
                point,
            )
        )
    for index, x in enumerate((-0.82, -0.36, 0.09, 0.53, 0.88)):
        point = (0.0, x)
        constraints.append(_constraint(f"displacement_{index}", "initial_displacement", (_term(point, 0, (0, 0)),), point))
        constraints.append(_constraint(f"velocity_{index}", "initial_velocity", (_term(point, 0, (1, 0)),), point))
    for index, (t, x) in enumerate(((0.18, -1.0), (0.52, -1.0), (0.87, -1.0), (0.31, 1.0), (0.68, 1.0), (0.95, 1.0))):
        point = (t, x)
        constraints.append(_constraint(f"boundary_{index}", "boundary", (_term(point, 0, (0, 0)),), point))
    return _finish(
        name="wave_1d",
        family="hyperbolic",
        positivity_regime="automatic",
        input_dim=2,
        output_dim=1,
        constraints=constraints,
        exact_solution=exact,
        evaluation_points=_grid_2d((0.0, 1.0), (-1.0, 1.0)),
        description="1D wave equation u_tt-0.8^2u_xx=f with co-located displacement/velocity data.",
    )


def _biharmonic_2d() -> Problem:
    def exact(xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, 0], xy[:, 1]
        value = torch.sin(0.8 * x + 0.2) * torch.cos(1.05 * y - 0.3) + 0.11 * torch.cos(0.6 * x * y + 0.2)
        return value[:, None]

    constraints: list[Constraint] = []
    interior = ((-0.68, -0.43), (-0.29, 0.24), (0.08, -0.16), (0.39, 0.67), (0.74, -0.28))
    for index, point in enumerate(interior):
        constraints.append(
            _constraint(
                f"biharmonic_{index}",
                "interior",
                (
                    _term(point, 0, (4, 0)),
                    _term(point, 0, (2, 2), 2.0),
                    _term(point, 0, (0, 4)),
                ),
                point,
            )
        )
    boundary = (
        ((-1.0, -0.61), (-1.0, 0.0)),
        ((-1.0, 0.32), (-1.0, 0.0)),
        ((1.0, -0.22), (1.0, 0.0)),
        ((1.0, 0.73), (1.0, 0.0)),
        ((-0.52, -1.0), (0.0, -1.0)),
        ((0.19, -1.0), (0.0, -1.0)),
        ((-0.27, 1.0), (0.0, 1.0)),
        ((0.66, 1.0), (0.0, 1.0)),
    )
    for index, (point, normal) in enumerate(boundary):
        constraints.append(_constraint(f"clamped_value_{index}", "boundary_value", (_term(point, 0, (0, 0)),), point))
        alpha = (1, 0) if normal[0] else (0, 1)
        coefficient = normal[0] if normal[0] else normal[1]
        constraints.append(_constraint(f"clamped_normal_{index}", "boundary_normal", (_term(point, 0, alpha, coefficient),), point))
    return _finish(
        name="biharmonic_2d",
        family="fourth_order_elliptic",
        positivity_regime="automatic",
        input_dim=2,
        output_dim=1,
        constraints=constraints,
        exact_solution=exact,
        evaluation_points=_grid_2d((-1.0, 1.0), (-1.0, 1.0)),
        description="Clamped 2D biharmonic equation Delta^2u=f with co-located value/normal data.",
    )


def _stokes_2d() -> Problem:
    def exact(xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, 0], xy[:, 1]
        angle_x = 0.8 * x + 0.2
        angle_y = 1.1 * y - 0.3
        velocity_x = 1.1 * torch.sin(angle_x) * torch.cos(angle_y) + 0.1 * x
        velocity_y = -0.8 * torch.cos(angle_x) * torch.sin(angle_y) - 0.1 * y
        pressure = 0.3 * torch.cos(0.7 * x - 0.4 * y + 0.1) + 0.08 * x - 0.05 * y
        return torch.stack((velocity_x, velocity_y, pressure), dim=1)

    viscosity = 0.2
    constraints: list[Constraint] = []
    interior = ((-0.66, -0.41), (-0.31, 0.28), (0.07, -0.14), (0.42, 0.64), (0.76, -0.27))
    for index, point in enumerate(interior):
        constraints.extend(
            (
                _constraint(
                    f"momentum_x_{index}",
                    "momentum",
                    (
                        _term(point, 0, (2, 0), -viscosity),
                        _term(point, 0, (0, 2), -viscosity),
                        _term(point, 2, (1, 0)),
                    ),
                    point,
                ),
                _constraint(
                    f"momentum_y_{index}",
                    "momentum",
                    (
                        _term(point, 1, (2, 0), -viscosity),
                        _term(point, 1, (0, 2), -viscosity),
                        _term(point, 2, (0, 1)),
                    ),
                    point,
                ),
                _constraint(
                    f"continuity_{index}",
                    "continuity",
                    (_term(point, 0, (1, 0)), _term(point, 1, (0, 1))),
                    point,
                ),
            )
        )
    # Anchor pressure at a PDE location to exercise four independent rows at x.
    anchor = interior[2]
    constraints.append(_constraint("pressure_anchor", "pressure_anchor", (_term(anchor, 2, (0, 0)),), anchor))
    boundary = ((-1.0, -0.56), (-1.0, 0.37), (1.0, -0.18), (1.0, 0.69), (-0.47, -1.0), (0.21, -1.0), (-0.25, 1.0), (0.63, 1.0))
    for index, point in enumerate(boundary):
        constraints.append(_constraint(f"velocity_x_boundary_{index}", "velocity_boundary", (_term(point, 0, (0, 0)),), point))
        constraints.append(_constraint(f"velocity_y_boundary_{index}", "velocity_boundary", (_term(point, 1, (0, 0)),), point))
    return _finish(
        name="stokes_2d",
        family="vector_elliptic_system",
        positivity_regime="automatic",
        input_dim=2,
        output_dim=3,
        constraints=constraints,
        exact_solution=exact,
        evaluation_points=_grid_2d((-1.0, 1.0), (-1.0, 1.0)),
        description="Steady 2D Stokes system with co-located momentum/continuity rows and a pressure anchor.",
    )


def _weak_poisson_1d() -> Problem:
    def exact(x: torch.Tensor) -> torch.Tensor:
        value = torch.sin(1.15 * x[:, 0] + 0.3) + 0.19 * torch.cos(0.65 * x[:, 0] - 0.2)
        return value[:, None]

    nodes, weights = _gauss_legendre(-1.0, 1.0, 16)
    constraints: list[Constraint] = []
    # Energy-orthonormal Galerkin tests: v_n' is the L2-normalized Legendre
    # polynomial P_n.  For n >= 1, v_n(x)=int_{-1}^x v_n'(s) ds vanishes at
    # both endpoints because P_n is orthogonal to constants.
    for degree in range(1, 5):
        legendre_coefficients = np.zeros(degree + 1, dtype=np.float64)
        legendre_coefficients[degree] = 1.0
        normalization = math.sqrt((2.0 * degree + 1.0) / 2.0)
        terms: list[Term] = []
        for node, weight in zip(nodes, weights, strict=True):
            test_derivative = normalization * np.polynomial.legendre.legval(
                node, legendre_coefficients
            )
            terms.append(_term((float(node),), 0, (1,), float(weight * test_derivative)))
        constraints.append(
            _constraint(
                f"weak_legendre_test_{degree}",
                "weak_residual",
                tuple(terms),
                None,
            )
        )
    for side, coordinate in (("left", -1.0), ("right", 1.0)):
        point = (coordinate,)
        constraints.append(_constraint(f"dirichlet_{side}", "boundary", (_term(point, 0, (0,)),), point))
    return _finish(
        name="weak_poisson_1d",
        family="weak_elliptic",
        positivity_regime="automatic",
        input_dim=1,
        output_dim=1,
        constraints=constraints,
        exact_solution=exact,
        evaluation_points=_grid_1d(),
        description=(
            "Weak 1D Poisson constraints int u'v_n' with four "
            "energy-orthonormal Legendre tests and 16-point Gauss-Legendre "
            "quadrature."
        ),
        positivity_route="weak_functional",
        trial_space="C^1([-1,1])",
        independence_witness=(
            "The four normalized Legendre derivatives P_1 through P_4 are "
            "linearly independent and no nonzero combination of degree at "
            "most four vanishes at all 16 Gauss-Legendre nodes; the two "
            "endpoint value atoms are distinct."
        ),
    )


def _nonlocal_diffusion_1d() -> Problem:
    """Value-only nonlocal diffusion covered by the signed-measure theorem."""

    def exact(x: torch.Tensor) -> torch.Tensor:
        coordinate = x[:, 0]
        value = (
            torch.sin(0.95 * coordinate + 0.23)
            + 0.21 * torch.cos(1.45 * coordinate - 0.17)
        )
        return value[:, None]

    nodes, weights = _gauss_legendre(-1.0, 1.0, 18)
    length_scale = 0.5
    anchors = (-0.5, 0.0, 0.5)
    constraints: list[Constraint] = []
    for index, anchor in enumerate(anchors):
        kernel_weights = [
            float(weight)
            * math.exp(-0.5 * ((anchor - float(node)) / length_scale) ** 2)
            for node, weight in zip(nodes, weights, strict=True)
        ]
        # Quadrature form of int k(x,y)[u(x)-u(y)]dy.  Every residual
        # measure has a nonzero atom at its own anchor, which is distinct from
        # all quadrature nodes and boundary atoms; these pivots certify joint
        # independence of residual and Dirichlet measures.
        terms = [_term((anchor,), 0, (0,), sum(kernel_weights))]
        terms.extend(
            _term((float(node),), 0, (0,), -kernel_weight)
            for node, kernel_weight in zip(nodes, kernel_weights, strict=True)
        )
        constraints.append(
            _constraint(
                f"nonlocal_diffusion_{index}",
                "nonlocal_residual",
                tuple(terms),
                None,
            )
        )
    for side, coordinate in (("left", -1.0), ("right", 1.0)):
        point = (coordinate,)
        constraints.append(
            _constraint(
                f"dirichlet_{side}", "boundary", (_term(point, 0, (0,)),), point
            )
        )
    return _finish(
        name="nonlocal_diffusion_1d",
        family="nonlocal_integral",
        positivity_regime="automatic",
        input_dim=1,
        output_dim=1,
        constraints=constraints,
        exact_solution=exact,
        evaluation_points=_grid_1d(),
        description=(
            "Value-only nonlocal diffusion with an 18-node Gaussian-kernel "
            "quadrature and two Dirichlet constraints."
        ),
        positivity_route="nonlocal_measure",
        trial_space="C([-1,1])",
        independence_witness=(
            "Each residual measure has a nonzero Dirac atom at its own "
            "interior anchor, distinct from all quadrature and endpoint "
            "atoms; the two boundary Dirac measures are also distinct."
        ),
    )


def _integro_diff_1d() -> Problem:
    def exact(x: torch.Tensor) -> torch.Tensor:
        coordinate = x[:, 0]
        value = torch.exp(0.2 * coordinate) * torch.sin(1.1 * coordinate + 0.4) + 0.3 * torch.cos(0.6 * coordinate - 0.1)
        return value[:, None]

    nodes, weights = _gauss_legendre(-1.0, 1.0, 18)
    length_scale = 0.42
    constraints: list[Constraint] = []
    anchors = (-0.76, -0.33, 0.08, 0.47, 0.81)
    for index, anchor in enumerate(anchors):
        point = (anchor,)
        terms = [
            _term(point, 0, (1,)),
            _term(point, 0, (0,), 0.4 + 0.1 * anchor),
        ]
        for node, weight in zip(nodes, weights, strict=True):
            kernel = math.exp(-0.5 * ((anchor - node) / length_scale) ** 2)
            terms.append(_term((float(node),), 0, (0,), -0.55 * float(weight) * kernel))
        constraints.append(_constraint(f"nonlocal_{index}", "nonlocal_residual", tuple(terms), None))
    boundary = (-1.0,)
    constraints.append(_constraint("left_boundary", "boundary", (_term(boundary, 0, (0,)),), boundary))
    return _finish(
        name="integro_diff_1d",
        family="nonlocal_integro_differential",
        positivity_regime="automatic",
        input_dim=1,
        output_dim=1,
        constraints=constraints,
        exact_solution=exact,
        evaluation_points=_grid_1d(),
        description="First-order integro-differential equation with an 18-node Gaussian-kernel quadrature.",
        positivity_route="weak_functional",
        trial_space="C^1([-1,1])",
        independence_witness=(
            "Each residual has a unique first-derivative evaluation at its "
            "anchor; after those coefficients vanish, the boundary value "
            "atom is unique."
        ),
    )


def _caputo_diffusion_1d() -> Problem:
    def exact(tx: torch.Tensor) -> torch.Tensor:
        t, x = tx[:, 0], tx[:, 1]
        value = (1.0 + t**2) * torch.sin(0.9 * x + 0.2) + 0.1 * torch.exp(-0.3 * t) * torch.cos(0.7 * x - 0.1)
        return value[:, None]

    alpha = 0.6
    diffusivity = 0.3
    unit_nodes, unit_weights = _gauss_legendre(0.0, 1.0, 16)
    constraints: list[Constraint] = []
    anchors = ((0.16, -0.68), (0.29, 0.17), (0.43, -0.26), (0.61, 0.72), (0.78, -0.44), (0.93, 0.36))
    for index, (time, space) in enumerate(anchors):
        factor = time ** (1.0 - alpha) / ((1.0 - alpha) * math.gamma(1.0 - alpha))
        terms: list[Term] = []
        for node, weight in zip(unit_nodes, unit_weights, strict=True):
            # r=z^(1/(1-alpha)) cancels the endpoint singularity exactly.
            transformed_time = time * (1.0 - node ** (1.0 / (1.0 - alpha)))
            terms.append(
                _term(
                    (float(transformed_time), space),
                    0,
                    (1, 0),
                    factor * float(weight),
                )
            )
        terms.append(_term((time, space), 0, (0, 2), -diffusivity))
        constraints.append(_constraint(f"caputo_{index}", "fractional_residual", tuple(terms), None))
    for index, space in enumerate((-0.81, -0.32, 0.11, 0.55, 0.89)):
        point = (0.0, space)
        constraints.append(_constraint(f"initial_{index}", "initial", (_term(point, 0, (0, 0)),), point))
    for index, (time, space) in enumerate(((0.19, -1.0), (0.51, -1.0), (0.84, -1.0), (0.34, 1.0), (0.69, 1.0), (0.96, 1.0))):
        point = (time, space)
        constraints.append(_constraint(f"boundary_{index}", "boundary", (_term(point, 0, (0, 0)),), point))
    return _finish(
        name="caputo_diffusion_1d",
        family="fractional_parabolic",
        positivity_regime="automatic",
        input_dim=2,
        output_dim=1,
        constraints=constraints,
        exact_solution=exact,
        evaluation_points=_grid_2d((0.0, 1.0), (-1.0, 1.0)),
        description="Time-fractional diffusion with a singularity-cancelling transformed Caputo quadrature.",
        positivity_route="weak_functional",
        trial_space="C^2([0,1] x [-1,1])",
        independence_witness=(
            "Each fractional residual has a unique second-space-derivative "
            "atom at its anchor; all remaining initial and boundary value "
            "atoms are distinct."
        ),
        positivity_scope=(
            "exact_quadrature_prediction_map; continuum Caputo positivity "
            "requires separate Banach-space boundedness and ridge-density checks"
        ),
    )


_BUILDERS: dict[str, Callable[[], Problem]] = {
    "poisson_1d": _poisson_1d,
    "variable_elliptic_2d": _variable_elliptic_2d,
    "heat_1d": _heat_1d,
    "transport_1d": _transport_1d,
    "wave_1d": _wave_1d,
    "biharmonic_2d": _biharmonic_2d,
    "stokes_2d": _stokes_2d,
    "weak_poisson_1d": _weak_poisson_1d,
    "nonlocal_diffusion_1d": _nonlocal_diffusion_1d,
    "integro_diff_1d": _integro_diff_1d,
    "caputo_diffusion_1d": _caputo_diffusion_1d,
}


def list_problems() -> tuple[str, ...]:
    """Return stable registry names in the intended reporting order."""

    return tuple(_BUILDERS)


def get_problem(name: str) -> Problem:
    """Construct and validate one fresh problem instance."""

    try:
        builder = _BUILDERS[name]
    except KeyError as error:
        choices = ", ".join(list_problems())
        raise KeyError(f"unknown problem {name!r}; choose one of: {choices}") from error
    return builder()


def problem_registry() -> dict[str, Problem]:
    """Construct the complete validated registry."""

    return {name: get_problem(name) for name in list_problems()}


def validate_registry() -> tuple[Problem, ...]:
    """Build and validate every registered problem, returning them in order."""

    problems = tuple(get_problem(name) for name in list_problems())
    if len({problem.name for problem in problems}) != len(problems):
        raise ValueError("registry contains duplicate problem names")
    return problems


if __name__ == "__main__":
    for registered_problem in validate_registry():
        print(
            f"{registered_problem.name:24s} "
            f"{registered_problem.positivity_regime:9s} "
            f"N={len(registered_problem.constraints):2d} "
            f"d={registered_problem.input_dim} q={registered_problem.output_dim}"
        )
