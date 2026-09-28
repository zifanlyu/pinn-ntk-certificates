"""Three-row controls for joint NTK rank, nullspaces, and singular GD.

The experiment isolates three genuinely different functional designs:

* co-located rows of a two-output network,
* a quadrature-discretized weak residual jointly with a boundary row, and
* a finite-signed-measure nonlocal residual jointly with a boundary row.

For base functionals ``A``, ``B``, and an independent perturbation ``C``, the
three rows are ``A``, ``B``, and ``A+B+epsilon*C``.  Thus epsilon zero has the
known left-null vector ``z=(1,1,-1)`` and exact structural rank two, whereas
every configured positive epsilon has structural rank three.  Numerical rank
is deliberately reported at several relative tolerances rather than treated as
an exact algebraic statement.

Run from the repository root with

``python -m experiments.theorem_audit.joint_nullspace_controls --config experiments/theorem_audit/configs/joint_nullspace_controls.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Sequence

import numpy as np
import torch

from .constraints import Constraint, Problem, Term, functional_rank_certificate
from .engine import evaluate_finite, make_initialization_bank, train_full_batch_gd
from .features import compile_constraints
from .problems import get_problem


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
CASES = ("colocated_vector", "weak_boundary", "nonlocal_boundary")
NULL_VECTOR = torch.tensor((1.0, 1.0, -1.0), dtype=torch.float64) / math.sqrt(3.0)


def _combine(
    rows: Sequence[Constraint],
    weights: Sequence[float],
    name: str,
    *,
    group: str = "joint_control",
) -> Constraint:
    """Return the exact finite linear combination of stored functionals."""

    if len(rows) != len(weights):
        raise ValueError("rows and weights must have the same length")
    combined: dict[tuple[tuple[float, ...], int, tuple[int, ...]], float] = {}
    for row, weight in zip(rows, weights, strict=True):
        for term in row.terms:
            key = (term.point, term.output, term.alpha)
            combined[key] = combined.get(key, 0.0) + float(weight) * term.coefficient
    terms = tuple(
        Term(point, output, alpha, coefficient)
        for (point, output, alpha), coefficient in sorted(combined.items())
        if coefficient != 0.0
    )
    if not terms:
        raise ValueError("a control row cannot be identically zero")
    active_locations = {
        row.physical_point
        for row, weight in zip(rows, weights, strict=True)
        if weight != 0.0
    }
    physical_point = next(iter(active_locations)) if len(active_locations) == 1 else None
    return Constraint(
        name=name,
        group=group,
        terms=terms,
        target=sum(
            float(weight) * row.target
            for row, weight in zip(rows, weights, strict=True)
        ),
        physical_point=physical_point,
    )


def _normalized(row: Constraint, name: str) -> Constraint:
    """Normalize the coefficient vector over distinct stored jet atoms."""

    norm = math.sqrt(sum(term.coefficient**2 for term in row.terms))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"row {row.name!r} has invalid coefficient norm")
    return _combine((row,), (1.0 / norm,), name)


def _vector_exact(x: torch.Tensor) -> torch.Tensor:
    coordinate = x[:, 0]
    first = torch.sin(1.1 * coordinate + 0.2) + 0.13 * coordinate
    second = torch.cos(0.8 * coordinate - 0.3) - 0.17 * coordinate
    return torch.stack((first, second), dim=1)


def _base_functionals(case: str) -> tuple[Problem, tuple[Constraint, Constraint, Constraint], dict[str, str]]:
    """Return independent, coefficient-normalized base rows ``(A,B,C)``."""

    if case == "colocated_vector":
        point = (0.17,)
        base = Problem(
            name="colocated_two_output_source",
            family="vector_pointwise_control",
            positivity_regime="diagnostic",
            input_dim=1,
            output_dim=2,
            constraints=(),
            exact_solution=_vector_exact,
            evaluation_points=torch.linspace(-1.0, 1.0, 33, dtype=torch.float64)[:, None],
            description="Source problem for a co-located two-output rank control.",
            positivity_route="pointwise_dntk",
            trial_space="C^1([-1,1];R^2)",
            independence_witness="At one site, u_1, u_2, and partial_x u_1 are distinct jet coordinates.",
        )
        raw = (
            Constraint("u1_at_x0", "joint_control", (Term(point, 0, (0,), 1.0),), 0.0, point),
            Constraint("u2_at_x0", "joint_control", (Term(point, 1, (0,), 1.0),), 0.0, point),
            Constraint("dx_u1_at_x0", "joint_control", (Term(point, 0, (1,), 1.0),), 0.0, point),
        )
        # Populate exact targets without relying on private problem constructors.
        from .constraints import with_exact_targets

        rows = with_exact_targets(base.exact_solution, raw)
        labels = {
            "A": r"$u_1(x_0)$",
            "B": r"$u_2(x_0)$",
            "C": r"$\partial_xu_1(x_0)$",
        }
    elif case == "weak_boundary":
        base = get_problem("weak_poisson_1d")
        rows = (base.constraints[0], base.constraints[-2], base.constraints[1])
        labels = {
            "A": "first 16-node weak-residual quadrature row",
            "B": "left Dirichlet boundary row",
            "C": "second 16-node weak-residual quadrature row",
        }
    elif case == "nonlocal_boundary":
        base = get_problem("nonlocal_diffusion_1d")
        rows = (base.constraints[0], base.constraints[-2], base.constraints[1])
        labels = {
            "A": "first 18-node finite-measure nonlocal row",
            "B": "left Dirichlet boundary row",
            "C": "second 18-node finite-measure nonlocal row",
        }
    else:
        raise ValueError(f"unknown case {case!r}")
    normalized = tuple(
        _normalized(row, f"base_{label}_{row.name}")
        for label, row in zip(("A", "B", "C"), rows, strict=True)
    )
    return base, normalized, labels


def make_problem(
    case: str,
    epsilon: float,
    *,
    target_mode: str = "compatible",
    incompatibility_amplitude: float = 0.25,
) -> tuple[Problem, dict[str, Any]]:
    """Construct one exact/near-dependent three-row prediction problem.

    For epsilon zero, incompatible targets add ``amplitude*(1,1,-1)`` to the
    unnormalized target rows.  All three rows share the scale ``1/sqrt(3)``;
    hence this is an amplitude-sized displacement along the unit null vector
    and gives the exact least-squares loss floor ``amplitude**2``.
    """

    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative")
    if target_mode not in {"compatible", "incompatible"}:
        raise ValueError("target_mode must be compatible or incompatible")
    if target_mode == "incompatible" and epsilon != 0.0:
        raise ValueError("incompatible targets are defined only for the exact relation")
    if not math.isfinite(incompatibility_amplitude) or incompatibility_amplitude < 0.0:
        raise ValueError("incompatibility_amplitude must be finite and nonnegative")

    base, (row_a, row_b, row_c), labels = _base_functionals(case)
    rows = (
        _combine((row_a,), (1.0,), "row_A"),
        _combine((row_b,), (1.0,), "row_B"),
        _combine((row_a, row_b, row_c), (1.0, 1.0, epsilon), "row_A_plus_B_plus_epsilon_C"),
    )
    if target_mode == "incompatible":
        signs = (1.0, 1.0, -1.0)
        rows = tuple(
            replace(row, target=row.target + incompatibility_amplitude * sign)
            for row, sign in zip(rows, signs, strict=True)
        )

    route = {
        "colocated_vector": "pointwise_dntk",
        "weak_boundary": "weak_functional",
        "nonlocal_boundary": "nonlocal_measure",
    }[case]
    problem = replace(
        base,
        name=f"joint_nullspace__{case}__epsilon_{epsilon:g}__{target_mode}",
        family=f"joint_nullspace_{case}",
        positivity_regime="diagnostic",
        constraints=rows,
        description=(
            f"Three-row joint-nullspace control: A, B, A+B+epsilon*C; "
            f"case={case}, epsilon={epsilon:g}, targets={target_mode}."
        ),
        positivity_route=route,
        positivity_scope="exact stored finite prediction map",
    )
    expected_rank = 2 if epsilon == 0.0 else 3
    metadata = {
        "case": case,
        "epsilon": epsilon,
        "target_mode": target_mode,
        "row_formula": "(A, B, A+B+epsilon*C)",
        "base_functionals": labels,
        "coefficient_normalization": "Each of A, B, C has unit Euclidean coefficient norm before mixing.",
        "loss_group_scale": 1.0 / math.sqrt(3.0),
        "candidate_left_null_vector": NULL_VECTOR.tolist(),
        "expected_structural_rank": expected_rank,
        "target_compatible_with_exact_relation": target_mode == "compatible",
        "predicted_irreducible_loss_floor": (
            incompatibility_amplitude**2 if target_mode == "incompatible" else 0.0
        ),
    }
    return problem, metadata


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version {SCHEMA_VERSION}")
    for key in ("cases", "epsilons", "widths", "seeds", "relative_rank_tolerances"):
        if not isinstance(config.get(key), list) or not config[key]:
            raise ValueError(f"{key} must be a nonempty list")
        if len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must not contain duplicates")
    if tuple(config["cases"]) != CASES:
        raise ValueError(f"cases must be exactly {list(CASES)} in this order")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in config["epsilons"]):
        raise ValueError("epsilons must be finite nonnegative numbers")
    if 0.0 not in config["epsilons"] or not any(value > 0.0 for value in config["epsilons"]):
        raise ValueError("epsilons must contain zero and at least one positive value")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in config["widths"]):
        raise ValueError("widths must be positive integers")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in config["seeds"]):
        raise ValueError("seeds must be integers")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 < value < 1.0 for value in config["relative_rank_tolerances"]):
        raise ValueError("relative_rank_tolerances must lie in (0,1)")
    if int(config.get("steps", -1)) <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 < float(config.get("learning_rate_factor", 0.0)) < 0.5:
        raise ValueError("learning_rate_factor must lie in (0,0.5)")
    if float(config.get("incompatibility_amplitude", -1.0)) <= 0.0:
        raise ValueError("incompatibility_amplitude must be positive")
    if int(config.get("threads", 0)) <= 0:
        raise ValueError("threads must be positive")
    return config


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    return _validate_config(json.loads(resolved.read_text(encoding="utf-8"))), resolved


def _resolve_output(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _operator_norm(kernel: torch.Tensor) -> float:
    return float(torch.linalg.eigvalsh(0.5 * (kernel + kernel.T)).abs().max())


def kernel_record(
    case: str,
    epsilon: float,
    width: int,
    seed: int,
    state: Any,
    tolerances: Sequence[float],
    incompatibility_amplitude: float,
) -> dict[str, Any]:
    problem, metadata = make_problem(
        case, epsilon, incompatibility_amplitude=incompatibility_amplitude
    )
    certificate = functional_rank_certificate(
        problem, require_measure_atoms=case == "nonlocal_boundary"
    )
    if certificate["rank"] != metadata["expected_structural_rank"]:
        raise AssertionError(
            f"coefficient certificate disagrees for {case}, epsilon={epsilon}"
        )
    evaluation = evaluate_finite(state, compile_constraints(problem))
    kernel = evaluation.kernel
    eigenvalues = torch.linalg.eigvalsh(kernel)
    scale = max(float(eigenvalues.abs().max()), torch.finfo(torch.float64).tiny)
    candidate = NULL_VECTOR.to(kernel.device)
    candidate_residual = float(torch.linalg.vector_norm(kernel @ candidate))
    kernel_norm = _operator_norm(kernel)
    normalized_residual = candidate_residual / (
        max(kernel_norm, torch.finfo(torch.float64).tiny)
        * float(torch.linalg.vector_norm(candidate))
    )
    rayleigh = float(candidate @ kernel @ candidate) / max(
        kernel_norm * float(candidate @ candidate), torch.finfo(torch.float64).tiny
    )
    ranks = {
        f"{tolerance:.0e}": int((eigenvalues > float(tolerance) * scale).sum())
        for tolerance in tolerances
    }
    return {
        "case": case,
        "epsilon": epsilon,
        "width": width,
        "seed": seed,
        "predicted_structural_rank": metadata["expected_structural_rank"],
        "coefficient_matrix_rank": certificate["rank"],
        "coefficient_matrix_smallest_singular_value": certificate["smallest_singular_value"],
        "kernel_eigenvalues": eigenvalues.tolist(),
        "kernel_operator_norm": kernel_norm,
        "tolerance_classified_rank": ranks,
        "candidate_relation_Kz_normalized_residual": normalized_residual,
        "candidate_relation_normalized_rayleigh_quotient": rayleigh,
    }


def gd_pair_record(
    case: str,
    width: int,
    seed: int,
    state: Any,
    *,
    steps: int,
    learning_rate_factor: float,
    rank_tolerance: float,
    incompatibility_amplitude: float,
    checkpoint_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    compatible, _ = make_problem(
        case,
        0.0,
        target_mode="compatible",
        incompatibility_amplitude=incompatibility_amplitude,
    )
    incompatible, incompatible_metadata = make_problem(
        case,
        0.0,
        target_mode="incompatible",
        incompatibility_amplitude=incompatibility_amplitude,
    )
    compatible_compiled = compile_constraints(compatible)
    incompatible_compiled = compile_constraints(incompatible)
    initial = evaluate_finite(state, compatible_compiled)
    lambda_max = float(torch.linalg.eigvalsh(initial.kernel)[-1])
    if not math.isfinite(lambda_max) or lambda_max <= 0.0:
        raise AssertionError("initial kernel has invalid largest eigenvalue")
    learning_rate = learning_rate_factor / lambda_max
    common = {
        "steps": steps,
        "learning_rate": learning_rate,
        "checkpoint_every": steps,
        "checkpoint_steps": checkpoint_steps,
        "relative_rank_tolerance": rank_tolerance,
    }
    compatible_result = train_full_batch_gd(state, compatible_compiled, **common)
    incompatible_result = train_full_batch_gd(state, incompatible_compiled, **common)
    compatible_final = evaluate_finite(
        compatible_result.final_state, compatible_compiled
    )
    incompatible_final = evaluate_finite(
        incompatible_result.final_state, incompatible_compiled
    )
    initial_incompatible = evaluate_finite(state, incompatible_compiled)
    floor = float(incompatible_metadata["predicted_irreducible_loss_floor"])
    parameter_difference = torch.linalg.vector_norm(
        compatible_result.final_state.packed_per_neuron()
        - incompatible_result.final_state.packed_per_neuron()
    )
    null_projection = float(
        torch.dot(NULL_VECTOR, incompatible_final.error) ** 2
    )
    paired_history: list[dict[str, float | int]] = []
    for compatible_checkpoint, incompatible_checkpoint in zip(
        compatible_result.history,
        incompatible_result.history,
        strict=True,
    ):
        if compatible_checkpoint.step != incompatible_checkpoint.step:
            raise AssertionError("paired GD histories have different checkpoint grids")
        offset_error = abs(
            incompatible_checkpoint.loss - compatible_checkpoint.loss - floor
        )
        paired_history.append(
            {
                "step": compatible_checkpoint.step,
                # The exact row relation makes this the common active-subspace
                # loss for both targets; the two GD parameter paths coincide.
                "shared_active_subspace_loss": compatible_checkpoint.loss,
                "compatible_nullspace_loss": 0.0,
                "compatible_total_loss": compatible_checkpoint.loss,
                "incompatible_nullspace_loss": floor,
                "incompatible_total_loss": incompatible_checkpoint.loss,
                "paired_loss_offset_error": offset_error,
            }
        )
    return {
        "case": case,
        "width": width,
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "learning_rate_rule": "eta=learning_rate_factor/lambda_max(K0)",
        "learning_rate_factor": learning_rate_factor,
        "theorem_B_star_step_condition_verified": False,
        "compatible_initial_loss": float(initial.loss),
        "compatible_final_loss": float(compatible_final.loss),
        "incompatible_initial_loss": float(initial_incompatible.loss),
        "incompatible_final_loss": float(incompatible_final.loss),
        "predicted_incompatible_loss_floor": floor,
        "measured_final_null_projection_loss": null_projection,
        "incompatible_final_loss_above_floor": float(incompatible_final.loss) - floor,
        "paired_initial_loss_offset_error": abs(
            float(initial_incompatible.loss - initial.loss) - floor
        ),
        "paired_final_loss_offset_error": abs(
            float(incompatible_final.loss - compatible_final.loss) - floor
        ),
        "paired_final_parameter_l2_difference": float(parameter_difference),
        "checkpoint_steps": [row["step"] for row in paired_history],
        "paired_history": paired_history,
    }


def _aggregate(
    config: dict[str, Any],
    config_path: Path,
    kernel_records: list[dict[str, Any]],
    gd_records: list[dict[str, Any]],
) -> dict[str, Any]:
    tolerance_keys = [f"{value:.0e}" for value in config["relative_rank_tolerances"]]
    rank_summary: list[dict[str, Any]] = []
    for case in config["cases"]:
        for epsilon in config["epsilons"]:
            selected = [
                record
                for record in kernel_records
                if record["case"] == case and record["epsilon"] == epsilon
            ]
            predicted = 2 if epsilon == 0.0 else 3
            rank_summary.append(
                {
                    "case": case,
                    "epsilon": epsilon,
                    "predicted_structural_rank": predicted,
                    "initializations": len(selected),
                    "classified_ranks_by_tolerance": {
                        key: sorted(
                            {
                                record["tolerance_classified_rank"][key]
                                for record in selected
                            }
                        )
                        for key in tolerance_keys
                    },
                    "prediction_match_count_by_tolerance": {
                        key: sum(
                            record["tolerance_classified_rank"][key] == predicted
                            for record in selected
                        )
                        for key in tolerance_keys
                    },
                    "max_candidate_relation_Kz_normalized_residual": max(
                        record["candidate_relation_Kz_normalized_residual"]
                        for record in selected
                    ),
                }
            )

    exact_records = [record for record in kernel_records if record["epsilon"] == 0.0]
    positive_records = [record for record in kernel_records if record["epsilon"] > 0.0]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "three_row_joint_nullspace_control_aggregate",
        "status": "complete",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "dtype": "float64",
            "deterministic_algorithms": True,
            "threads": torch.get_num_threads(),
        },
        "design": {
            "rows": "(A, B, A+B+epsilon*C)",
            "candidate_left_null_vector_at_epsilon_zero": NULL_VECTOR.tolist(),
            "expected_rank_at_epsilon_zero": 2,
            "expected_rank_at_positive_epsilon": 3,
            "cases": {
                case: make_problem(case, 0.0)[1]["base_functionals"]
                for case in config["cases"]
            },
            "network": "u_q(x)=sum_j a[j,q] tanh(w[j]^T x+b[j])/sqrt(width); all parameters trained; no output bias",
            "initialization": "iid Uniform[-1,1], nested width prefixes within each case and seed",
            "loss": "sum of squares of three equally group-normalized rows",
        },
        "configuration": config,
        "sample_sizes": {
            "cases": len(config["cases"]),
            "epsilon_values": len(config["epsilons"]),
            "widths": len(config["widths"]),
            "seeds": len(config["seeds"]),
            "kernel_initializations": len(kernel_records),
            "exact_relation_kernel_initializations": len(exact_records),
            "positive_epsilon_kernel_initializations": len(positive_records),
            "paired_gd_settings": len(gd_records),
            "gd_trajectories": 2 * len(gd_records),
        },
        "summary": {
            "rank_by_case_and_epsilon": rank_summary,
            "exact_relation_max_normalized_Kz_residual": max(
                record["candidate_relation_Kz_normalized_residual"]
                for record in exact_records
            ),
            "compatible_gd_max_final_loss": max(
                record["compatible_final_loss"] for record in gd_records
            ),
            "incompatible_gd_predicted_floor": float(
                config["incompatibility_amplitude"] ** 2
            ),
            "incompatible_gd_max_final_loss_above_floor": max(
                record["incompatible_final_loss_above_floor"] for record in gd_records
            ),
            "paired_gd_max_final_loss_offset_error": max(
                record["paired_final_loss_offset_error"] for record in gd_records
            ),
            "paired_gd_max_final_parameter_l2_difference": max(
                record["paired_final_parameter_l2_difference"] for record in gd_records
            ),
            "paired_gd_max_checkpoint_loss_offset_error": max(
                checkpoint["paired_loss_offset_error"]
                for record in gd_records
                for checkpoint in record["paired_history"]
            ),
        },
        "rank_records": kernel_records,
        "gd_pair_records": gd_records,
        "limitations": [
            "This is a finite-width mechanistic control, not a PDE solution-error or continuum-convergence experiment.",
            "The weak and nonlocal functionals are the exact stored 16- and 18-node quadrature rules; no claim is made here for the corresponding continuum operators.",
            "Tolerance-classified numerical rank is diagnostic and can label a small positive eigenvalue as zero; structural rank comes from the coefficient functional design.",
            "The spectral learning-rate rule verifies frozen-kernel stability only; the theorem's deterministic B_* step-size and width conditions are not numerically certified.",
        ],
    }


def _rank_set_text(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _power_of_ten_tex(value: float) -> str:
    """Format zero or an exact configured power of ten in math mode."""

    if value == 0.0:
        return "$0$"
    exponent = round(math.log10(value))
    if not math.isclose(value, 10.0**exponent, rel_tol=2.0e-15, abs_tol=0.0):
        raise ValueError(f"expected an exact power of ten, got {value}")
    return rf"$10^{{{exponent}}}$"


def _scientific_tex(value: float, digits: int = 2) -> str:
    """Format a measured scalar as publication-safe scientific notation."""

    if value == 0.0:
        return "$0$"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10.0**exponent)
    return rf"${mantissa:.{digits}f}\times 10^{{{exponent}}}$"


def render_tex(data: dict[str, Any]) -> str:
    """Render a compact, data-derived LaTeX snippet."""

    config = data["configuration"]
    tolerances = [f"{value:.0e}" for value in config["relative_rank_tolerances"]]
    tolerance_headers = [
        rf"$r_{{10^{{{round(math.log10(value))}}}}}$"
        for value in config["relative_rank_tolerances"]
    ]
    case_labels = {
        "colocated_vector": "Co-located",
        "weak_boundary": r"Weak+$\partial$",
        "nonlocal_boundary": r"Nonlocal+$\partial$",
    }
    header = " & ".join(
        ["Design", r"$\epsilon$", r"$r_\star$"]
        + tolerance_headers
        + [r"max. $\rho_z$"]
    )
    rows: list[str] = []
    for record in data["summary"]["rank_by_case_and_epsilon"]:
        n = record["initializations"]
        observed = []
        for key in tolerances:
            ranks = _rank_set_text(record["classified_ranks_by_tolerance"][key])
            matches = record["prediction_match_count_by_tolerance"][key]
            observed.append(rf"${ranks}\;({matches}/{n})$")
        fields = [
            case_labels[record["case"]],
            _power_of_ten_tex(record["epsilon"]),
            f"${record['predicted_structural_rank']}$",
            *observed,
            _scientific_tex(
                record["max_candidate_relation_Kz_normalized_residual"]
            ),
        ]
        rows.append(" & ".join(fields) + r" \\")

    gd_rows: list[str] = []
    for case in config["cases"]:
        selected = [row for row in data["gd_pair_records"] if row["case"] == case]
        gd_rows.append(
            " & ".join(
                (
                    case_labels[case],
                    str(len(selected)),
                    _scientific_tex(
                        max(row["compatible_final_loss"] for row in selected)
                    ),
                    _scientific_tex(
                        max(row["incompatible_final_loss"] for row in selected)
                    ),
                    _scientific_tex(
                        max(row["paired_final_loss_offset_error"] for row in selected)
                    ),
                )
            )
            + r" \\"
        )

    sample = data["sample_sizes"]
    floor = data["summary"]["incompatible_gd_predicted_floor"]
    checkpoint_count = len(data["gd_pair_records"][0]["paired_history"])
    maximum_checkpoint_offset_error = data["summary"][
        "paired_gd_max_checkpoint_loss_offset_error"
    ]
    maximum_final_parameter_difference = data["summary"][
        "paired_gd_max_final_parameter_l2_difference"
    ]
    return "\n".join(
        (
            "% Generated by experiments/theorem_audit/joint_nullspace_controls.py; do not edit numerical values manually.",
            r"\paragraph{Three-row joint-nullspace controls}",
            (
                "We tested rows $A$, $B$, and $A+B+\\epsilon C$ for a co-located "
                "two-output pointwise design, a weak-quadrature residual jointly with a "
                "boundary value, and a finite-measure nonlocal residual jointly with a "
                f"boundary value.  The audit contains {sample['kernel_initializations']} "
                f"initial kernels ({sample['cases']} cases, {sample['epsilon_values']} "
                f"values of $\\epsilon$, {sample['widths']} widths, and {sample['seeds']} "
                "seeds) in float64.  The parenthesized counts below are agreements with "
                "the exact coefficient-design rank $r_\\star$; $r_\\tau$ is numerical "
                "rank at relative tolerance $\\tau$, and "
                "$\\rho_z=\\|Kz\\|/(\\|K\\|\\|z\\|)$.  The displayed rank set records "
                "all values observed over the configured widths and seeds."
            ),
            r"\begin{center}\scriptsize",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\setlength{\tabcolsep}{2.8pt}",
            r"\begin{tabular}{@{}lcc" + "c" * len(tolerances) + "c@{}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            (
                f"For the exact relation ($\\epsilon=0$), we additionally ran "
                f"{sample['paired_gd_settings']} paired settings, hence "
                f"{sample['gd_trajectories']} trajectories: manufactured compatible "
                "targets and targets displaced only along the known left null direction. "
                f"The predicted incompatible loss floor was {_scientific_tex(floor)}. "
                f"Across all {checkpoint_count} saved steps, the largest error in the "
                "predicted paired loss offset was "
                f"{_scientific_tex(maximum_checkpoint_offset_error)}; the largest final "
                "paired parameter difference was "
                f"{_scientific_tex(maximum_final_parameter_difference)}."
            ),
            r"\begin{center}\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            r"Design & pairs & max. $\mathcal L_T^{\mathrm{comp}}$ & max. $\mathcal L_T^{\mathrm{inc}}$ & max. offset error \\",
            r"\midrule",
            *gd_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            (
                r"These controls diagnose the exact stored finite prediction maps.  "
                r"The weak and nonlocal rows use fixed 16- and 18-node quadrature, "
                r"respectively; they do not establish continuum discretization error.  "
                r"Numerical rank depends on the displayed tolerance, and the GD runs do "
                r"not certify the theorem's deterministic width or $B_*$ step-size bounds."
            ),
            "",
        )
    )


def run(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(config["threads"]))
    torch.use_deterministic_algorithms(True)
    maximum_width = max(config["widths"])
    amplitude = float(config["incompatibility_amplitude"])
    kernel_records: list[dict[str, Any]] = []
    gd_records: list[dict[str, Any]] = []

    for case in config["cases"]:
        source, _, _ = _base_functionals(case)
        for seed in config["seeds"]:
            bank = make_initialization_bank(
                maximum_width,
                source.input_dim,
                source.output_dim,
                seed,
            )
            for width in config["widths"]:
                state = bank.prefix(width)
                for epsilon in config["epsilons"]:
                    kernel_records.append(
                        kernel_record(
                            case,
                            float(epsilon),
                            width,
                            seed,
                            state,
                            config["relative_rank_tolerances"],
                            amplitude,
                        )
                    )
                gd_records.append(
                    gd_pair_record(
                        case,
                        width,
                        seed,
                        state,
                        steps=int(config["steps"]),
                        learning_rate_factor=float(config["learning_rate_factor"]),
                        rank_tolerance=min(config["relative_rank_tolerances"]),
                        incompatibility_amplitude=amplitude,
                        checkpoint_steps=tuple(int(step) for step in config["checkpoint_steps"]),
                    )
                )
    return _aggregate(config, config_path, kernel_records, gd_records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/theorem_audit/configs/joint_nullspace_controls.json",
    )
    parser.add_argument(
        "--output",
        default="experiments/theorem_audit/data/joint_nullspace_controls.json",
    )
    parser.add_argument(
        "--tex-output",
        default="experiments/theorem_audit/joint_nullspace_controls.tex",
    )
    args = parser.parse_args(argv)
    config, config_path = load_config(args.config)
    data = run(config, config_path)
    output = _resolve_output(args.output)
    tex_output = _resolve_output(args.tex_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tex_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    tex_output.write_text(render_tex(data), encoding="utf-8")
    print(
        f"wrote {output.relative_to(ROOT)} and {tex_output.relative_to(ROOT)}; "
        f"{data['sample_sizes']['kernel_initializations']} kernels, "
        f"{data['sample_sizes']['gd_trajectories']} GD trajectories"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
