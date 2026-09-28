"""Paired rank and conditioning controls for the exact theorem architecture.

Run ``python -m experiments.theorem_audit.rank_controls --help`` from the repo.
These two-row diagnostic problems intentionally include dependent constraints;
they are separate from the PDE registry, whose validator rejects duplicates.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Sequence

import numpy as np
import torch

from .constraints import Constraint, Problem, Term, functional_rank_certificate, positivity_certificate, with_exact_targets
from .engine import evaluate_finite, make_initialization_bank, summarize_spectrum, train_full_batch_gd
from .features import compile_constraints
from .problems import get_problem


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("pointwise_poisson", "vector_stokes", "weak_poisson", "nonlocal_diffusion", "weak_boundary", "nonlocal_boundary")
SCHEMA = 1


def _combine(rows: Sequence[Constraint], weights: Sequence[float], name: str) -> Constraint:
    combined: dict[tuple, float] = {}
    for row, weight in zip(rows, weights, strict=True):
        for term in row.terms:
            atom = (term.point, term.output, term.alpha)
            combined[atom] = combined.get(atom, 0.0) + weight * term.coefficient
    terms = tuple(Term(point, output, alpha, value) for (point, output, alpha), value in sorted(combined.items()) if value != 0.0)
    locations = {row.physical_point for row, weight in zip(rows, weights, strict=True) if weight != 0.0}
    physical_point = next(iter(locations)) if len(locations) == 1 else None
    return Constraint(name, "control", terms, sum(weight * row.target for row, weight in zip(rows, weights, strict=True)), physical_point)


def base_pair(family: str) -> tuple[Problem, tuple[Constraint, Constraint]]:
    """Choose independent PDE functionals, preserving exact quadrature atoms."""
    if family == "pointwise_poisson":
        base = get_problem("poisson_1d")
        point = (0.07,)
        rows = with_exact_targets(base.exact_solution, (
            Constraint("poisson_at_x", "control", (Term(point, 0, (2,), -1.0),), 0.0, point),
            Constraint("value_at_same_x", "control", (Term(point, 0, (0,), 1.0),), 0.0, point),
        ))
    elif family == "vector_stokes":
        base = get_problem("stokes_2d")
        rows = base.constraints[:2]
    elif family in {"weak_poisson", "weak_boundary"}:
        base = get_problem("weak_poisson_1d")
        rows = base.constraints[:2] if family == "weak_poisson" else (base.constraints[-2], base.constraints[0])
    elif family in {"nonlocal_diffusion", "nonlocal_boundary"}:
        base = get_problem("nonlocal_diffusion_1d")
        rows = (base.constraints[0], base.constraints[2]) if family == "nonlocal_diffusion" else (base.constraints[-2], base.constraints[0])
    else:
        raise ValueError(f"unknown control family {family!r}")
    # Normalize coefficients before mixing; apply the SAME scaling to targets.
    normalized = tuple(_combine((row,), (1.0 / math.sqrt(sum(t.coefficient ** 2 for t in row.terms)),), f"base_{index}_{row.name}") for index, row in enumerate(rows))
    return base, normalized


def make_control(family: str, epsilon: float | None, target_mode: str = "manufactured", contrast_amplitude: float = 0.5) -> tuple[Problem, dict[str, Any]]:
    """None selects the independent pair; zero duplicates the first row.

    The loss is the mean squared error of two scalar rows. A contrast target
    shift ``amplitude*(-1,+1)`` in physical coordinates has normalized norm
    ``amplitude``. At epsilon=0 its irreducible squared loss is amplitude^2.
    """
    if epsilon is not None and (not math.isfinite(epsilon) or epsilon < 0):
        raise ValueError("epsilon must be nonnegative and finite")
    if target_mode not in {"manufactured", "contrast"}:
        raise ValueError("target_mode must be manufactured or contrast")
    base, pair = base_pair(family)
    mixing = torch.eye(2, dtype=torch.float64)
    if epsilon is not None:
        mixing[1] = torch.tensor([1.0, epsilon], dtype=torch.float64) / math.sqrt(1.0 + epsilon ** 2)
    rows = tuple(_combine(pair, weights.tolist(), f"control_row_{i}") for i, weights in enumerate(mixing))
    if target_mode == "contrast":
        rows = tuple(replace(row, target=row.target + contrast_amplitude * sign) for row, sign in zip(rows, (-1.0, 1.0), strict=True))
    label = "independent" if epsilon is None else f"epsilon_{epsilon:g}"
    problem = replace(base, name=f"{family}__{label}__{target_mode}", constraints=rows,
        description=f"Two-row rank control from {base.name}; {label}; {target_mode} targets. This is a functional diagnostic, not a full PDE solution benchmark.")
    singular = epsilon == 0.0
    metadata = {
        "family": family, "source_problem": base.name, "row_design": label, "epsilon": epsilon,
        "target_mode": target_mode, "contrast_amplitude": contrast_amplitude,
        "mixing_matrix": mixing.tolist(), "base_constraint_names": [row.name for row in pair],
        "base_functional_roles": ["Dirichlet boundary", "weak PDE residual" if family == "weak_boundary" else "nonlocal PDE residual"] if family in {"weak_boundary", "nonlocal_boundary"} else ["PDE functional 1", "PDE functional 2"],
        "joint_boundary_residual_control": family in {"weak_boundary", "nonlocal_boundary"},
        "loss_normalization_note": "Both diagnostic rows share the control group (scale 1/sqrt(2)); boundary/residual roles are preserved in metadata. Mixing applies to the exact stored finite functionals.",
        "exact_structural_rank": 1 if singular else 2,
        "target_functionally_compatible": not (singular and target_mode == "contrast" and contrast_amplitude != 0),
        "exact_structural_loss_floor": contrast_amplitude ** 2 if singular and target_mode == "contrast" else 0.0,
        "same_physical_location": rows[0].physical_point is not None and rows[0].physical_point == rows[1].physical_point,
        "interpretation": "Positive definiteness guarantees contraction for every residual; singularity need not prevent convergence for compatible targets. Small positive gaps can give slow convergence.",
    }
    return problem, metadata


def structural_null_projector(epsilon: float | None, device: torch.device | str = "cpu") -> torch.Tensor:
    return (torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float64, device=device)
            if epsilon == 0.0 else torch.zeros((2, 2), dtype=torch.float64, device=device))


def kernel_diagnostics(kernel: torch.Tensor, error: torch.Tensor, tolerance: float) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (kernel + kernel.T))
    spectrum = summarize_spectrum(kernel, relative_tolerance=tolerance).as_dict()
    numerical_null = eigenvectors[:, eigenvalues <= spectrum["rank_tolerance"]]
    projector = numerical_null @ numerical_null.T
    projection = projector @ error
    spectrum.update({
        "initial_error_eigenmode_coefficients": (eigenvectors.T @ error).cpu().tolist(),
        "numerical_nullspace_loss": float(torch.dot(projection, projection)),
        "numerical_nullspace_note": "Threshold diagnostic for frozen K0 only; tiny positive eigenvalues below tolerance are not exact null modes, and finite-width nullspaces can move under nonlinear GD.",
    })
    return spectrum, eigenvectors, projector


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError(f"expected schema_version {SCHEMA}")
    for key in ("families", "epsilons", "target_modes", "widths", "seeds"):
        if not isinstance(config.get(key), list) or not config[key]:
            raise ValueError(f"{key} must be a nonempty list")
        if len(set(config[key])) != len(config[key]):
            raise ValueError(f"duplicate values in {key}")
    if not set(config["families"]) <= set(FAMILIES):
        raise ValueError("unrecognized family")
    if not set(config["target_modes"]) <= {"manufactured", "contrast"}:
        raise ValueError("unrecognized target mode")
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in config["widths"]):
        raise ValueError("widths must be positive integers")
    if any(not isinstance(x, int) or isinstance(x, bool) for x in config["seeds"]):
        raise ValueError("seeds must be integers")
    if any(x is not None and (not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0) for x in config["epsilons"]):
        raise ValueError("epsilons must be nonnegative numbers or null (independent)")
    if not 0 < float(config["learning_rate_factor"]) < 1:
        raise ValueError("learning_rate_factor must lie in (0,1)")
    if int(config["steps"]) < 0 or int(config["checkpoint_every"]) <= 0:
        raise ValueError("steps must be nonnegative and checkpoint_every positive")
    if not math.isfinite(float(config.get("contrast_amplitude", 0.5))):
        raise ValueError("contrast_amplitude must be finite")
    if not isinstance(config.get("include_incompatible_targets", False), bool):
        raise ValueError("include_incompatible_targets must be boolean")
    if not math.isfinite(float(config.get("relative_rank_tolerance", 1e-12))) or float(config.get("relative_rank_tolerance", 1e-12)) < 0:
        raise ValueError("relative_rank_tolerance must be finite and nonnegative")
    config["_config_path"] = str(path)
    return config


def target_is_compatible(epsilon: float | None, mode: str, amplitude: float) -> bool:
    """Independent rows admit any target; identical rows require equal targets."""
    return not (epsilon == 0.0 and mode == "contrast" and amplitude != 0.0)


def task_grid(config: dict[str, Any], *, include_incompatible: bool | None = None) -> list[tuple[str, float | None, str, int, int]]:
    """Default to compatible targets; explicit override reconstructs archived grids.

    The override does not mutate the config or its digest. Old frozen campaigns
    had the full Cartesian grid, which the separate reporting overlay validates
    with ``include_incompatible=True`` before selecting compatible artifacts.
    """
    include = config.get("include_incompatible_targets", False) if include_incompatible is None else include_incompatible
    amplitude = float(config.get("contrast_amplitude", 0.5))
    return [(family, epsilon, mode, width, seed) for family in config["families"] for epsilon in config["epsilons"] for mode in config["target_modes"] for width in config["widths"] for seed in config["seeds"]
            if include or target_is_compatible(epsilon, mode, amplitude)]


def output_dir(config: dict[str, Any]) -> Path:
    path = Path(config["output_dir"])
    return path if path.is_absolute() else ROOT / path


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in config.items() if not k.startswith("_")}, sort_keys=True).encode()).hexdigest()


def source_digest() -> str:
    digest = hashlib.sha256()
    for name in ("rank_controls.py", "constraints.py", "problems.py", "features.py", "engine.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_task(config: dict[str, Any], index: int, *, overwrite: bool = False) -> Path:
    tasks = task_grid(config)
    if not 0 <= index < len(tasks):
        raise ValueError(f"task index {index} out of range [0,{len(tasks)-1}]")
    directory = output_dir(config) / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"task_{index:05d}.json"
    array_path = path.with_suffix(".npz")
    fingerprint = source_digest()
    if path.exists() and not overwrite:
        previous = json.loads(path.read_text())
        if previous.get("config_sha256") != config_digest(config) or previous.get("source_sha256") != fingerprint:
            raise ValueError(f"{path} exists with a different config/source; use a fresh output_dir or --overwrite")
        if array_path.exists() and hashlib.sha256(array_path.read_bytes()).hexdigest() == previous["arrays_sha256"]:
            print(f"[rank-control] task={index} already complete", flush=True)
            return path
        raise ValueError(f"{path} has missing/corrupt arrays; rerun with --overwrite")
    device = torch.device(config.get("device", "cpu"))
    torch.set_num_threads(int(config.get("threads", 1)))
    torch.use_deterministic_algorithms(True)
    family, epsilon, mode, width, seed = tasks[index]
    problem, metadata = make_control(family, epsilon, mode, float(config.get("contrast_amplitude", 0.5)))
    compiled = compile_constraints(problem, device=device)
    state = make_initialization_bank(max(config["widths"]), problem.input_dim, problem.output_dim, seed, device=device).prefix(width)
    start = time.monotonic()
    initial = evaluate_finite(state, compiled)
    tolerance = float(config.get("relative_rank_tolerance", 1e-12))
    spectrum, eigenvectors, numerical_null = kernel_diagnostics(initial.kernel, initial.error, tolerance)
    if spectrum["lambda_max"] <= 0 or not math.isfinite(spectrum["lambda_max"]):
        raise ValueError("invalid initial kernel")
    eta = float(config["learning_rate_factor"]) / spectrum["lambda_max"]
    output_gradient = initial.per_neuron_gradient[:, :, -problem.output_dim:]
    output_kernel = torch.einsum("miq,mjq->ij", output_gradient, output_gradient) / width
    null_projector = structural_null_projector(epsilon, device)
    structural_projection = null_projector @ initial.error
    numerical_floor = spectrum["numerical_nullspace_loss"]
    print(f"[rank-control] task={index} {problem.name} width={width} seed={seed} eta={eta:.3g}", flush=True)
    result = train_full_batch_gd(state, compiled, steps=int(config["steps"]), learning_rate=eta,
        checkpoint_every=int(config["checkpoint_every"]), relative_rank_tolerance=tolerance)
    final = evaluate_finite(result.final_state, compiled)
    history = result.history_dicts()
    floor = metadata["exact_structural_loss_floor"]
    for row in history:
        row["loss_above_structural_floor"] = row["loss"] - floor
        row["frozen_loss_above_numerical_nullspace_floor"] = row["frozen_kernel_loss"] - numerical_floor
    scaled_gap = 2 * eta * max(0.0, spectrum["lambda_min"])
    mode_factors = 1.0 - 2.0 * eta * torch.linalg.eigvalsh(initial.kernel)
    positive_modes = torch.linalg.eigvalsh(initial.kernel) > spectrum["rank_tolerance"]
    active_factor = float(mode_factors[positive_modes].abs().max()) if positive_modes.any() else 1.0
    arrays = {
        "K_initial": initial.kernel.cpu().numpy(), "K_final": final.kernel.cpu().numpy(),
        "K_output_initial": output_kernel.cpu().numpy(),
        "K_hidden_initial": (initial.kernel - output_kernel).cpu().numpy(),
        "eigenvectors_initial": eigenvectors.cpu().numpy(),
        "normalized_targets": compiled.targets.cpu().numpy(), "initial_error": initial.error.cpu().numpy(),
        "final_error": final.error.cpu().numpy(), "structural_null_projector": null_projector.cpu().numpy(),
        "frozen_numerical_null_projector": numerical_null.cpu().numpy(),
        "initial_parameters": state.packed_per_neuron().cpu().numpy(),
        "final_parameters": result.final_state.packed_per_neuron().cpu().numpy(),
        "history_step": np.asarray([h["step"] for h in history], dtype=np.int64),
        "history_loss": np.asarray([h["loss"] for h in history]),
        "history_frozen_kernel_loss": np.asarray([h["frozen_kernel_loss"] for h in history]),
    }
    temporary = array_path.with_name(array_path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(array_path)
    manifest = {
        "schema_version": SCHEMA, "artifact_type": "rank_conditioning_control", "status": "complete",
        "task_index": index, "config_sha256": config_digest(config), "source_sha256": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - start,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "device": str(device), "dtype": "float64", "threads": torch.get_num_threads(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "deterministic": True},
        "task": {**metadata, "width": width, "seed": seed, "maximum_width_initialization_bank": max(config["widths"]),
            "architecture": "u_q(x)=sum_j a[j,q]*tanh(w[j]^T*x+b[j])/sqrt(m); all w,b,a trained; no output bias",
            "initialization": "iid_uniform[-1,1]; nested width prefixes; paired across row/target controls",
            "input_dim": problem.input_dim, "output_dim": problem.output_dim, "constraint_count": 2},
        "constraints": [{"name": row.name, "group": row.group, "target": row.target,
            "physical_point": row.physical_point, "terms": [{"point": term.point, "output": term.output,
            "alpha": term.alpha, "coefficient": term.coefficient} for term in row.terms]} for row in problem.constraints],
        "functional_certificate": functional_rank_certificate(problem),
        "theorem_route_certificate": positivity_certificate(problem),
        "initial_spectrum": spectrum, "output_kernel_spectrum": summarize_spectrum(output_kernel, relative_tolerance=tolerance).as_dict(),
        "hidden_kernel_min_eigenvalue": float(torch.linalg.eigvalsh(initial.kernel - output_kernel)[0]),
        "optimization": {"loss": "sum_i (ell_i[u]-y_i)^2 = ||P-y_normalized||^2", "optimizer": "full_batch_GD_all_parameters",
            "learning_rate": eta, "learning_rate_factor": config["learning_rate_factor"], "steps": config["steps"],
            "update": "theta_next=theta-2*eta*J(theta)^T*error", "frozen_error_update": "e_next=(I-2*eta*K0)e",
            "paper_B_star_step_condition_verified": False,
            "step_note": "eta normalizes initial lambda_max; the frozen spectral stability condition is verified, the theorem's existential B_* condition is not numerically verified.",
            "scaled_initial_gap_2_eta_lambda_min": scaled_gap,
            "frozen_slowest_numerically_active_loss_factor": active_factor ** 2},
        "summary": {"initial_loss": float(initial.loss), "final_loss": float(final.loss),
            "final_to_initial_loss_ratio": float(final.loss / initial.loss) if initial.loss > 0 else None,
            "exact_structural_loss_floor": floor,
            "measured_initial_structural_projection_loss": float(torch.dot(structural_projection, structural_projection)),
            "structural_projection_invariance_error": float(torch.linalg.vector_norm(null_projector @ (final.error - initial.error))),
            "final_loss_above_structural_floor": float(final.loss) - floor,
            "final_frozen_kernel_loss": history[-1]["frozen_kernel_loss"],
            "final_relative_kernel_drift_operator": history[-1]["relative_kernel_drift_operator"],
            "checkpoint_loss_monotone": all(b["loss"] <= a["loss"] + 1e-12 * max(1, a["loss"]) for a, b in zip(history, history[1:]))},
        "history": history, "arrays_file": array_path.name, "arrays_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
    }
    atomic_json(path, manifest)
    print(f"[rank-control] task={index} loss {float(initial.loss):.4g} -> {float(final.loss):.4g}; floor={floor:.4g}; {manifest['elapsed_seconds']:.1f}s", flush=True)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "count", "aggregate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "count":
        print(len(task_grid(config)))
    elif args.command == "aggregate":
        from .rank_controls_aggregate import aggregate
        aggregate(config)
    elif args.task_index is not None:
        run_task(config, args.task_index, overwrite=args.overwrite)
    else:
        for index in range(len(task_grid(config))):
            run_task(config, index, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
