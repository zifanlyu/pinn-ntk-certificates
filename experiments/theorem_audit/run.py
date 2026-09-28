#!/usr/bin/env python3
"""Run limiting-kernel references and finite-width theorem audits.

This file is deliberately the only orchestration entry point used by the
local and Slurm instructions.  Numerical work is delegated to ``engine.py``;
this module validates configurations, selects independent array tasks, gates
the non-automatic cases, and writes self-describing artifacts atomically.

Examples (from the repository root)::

    python experiments/theorem_audit/run.py reference \
        --config experiments/theorem_audit/configs/smoke.json
    python experiments/theorem_audit/run.py train \
        --config experiments/theorem_audit/configs/smoke.json
    python experiments/theorem_audit/run.py all \
        --config experiments/theorem_audit/configs/smoke.json

The implemented network is exactly

    u_q(x) = m^{-1/2} sum_j a[j,q] tanh(w[j]^T x + b[j]),

with no global output bias.  All coordinates of ``(w,b,a)`` are trainable and
are initialized independently from ``Uniform[-1,1]``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch


# ``python experiments/theorem_audit/run.py`` does not set a package context.
# Insert only the known repository root so the documented invocation and
# ``python -m experiments.theorem_audit.run`` share the same imports.
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.theorem_audit.constraints import (  # noqa: E402
    Problem,
    constraint_group_sizes,
    positivity_certificate,
)
from experiments.theorem_audit.engine import (  # noqa: E402
    InfiniteNTKReference,
    NetworkState,
    estimate_infinite_ntk,
    evaluate_finite,
    make_initialization_bank,
    network_output,
    summarize_spectrum,
    train_full_batch_gd,
)
from experiments.theorem_audit.features import DTYPE, compile_constraints  # noqa: E402
from experiments.theorem_audit.problems import get_problem, list_problems  # noqa: E402


SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
ARCHITECTURE = "u_q(x)=m^(-1/2)*sum_j a[j,q]*tanh(w[j]^T*x+b[j])"
INITIALIZATION = "all coordinates of (w,b,a) iid Uniform[-1,1]"
LOSS = "||group_normalized_prediction-group_normalized_target||_2^2"
OPTIMIZER = "deterministic full-batch plain gradient descent"
SOURCE_FILES = (
    "experiments/theorem_audit/constraints.py",
    "experiments/theorem_audit/problems.py",
    "experiments/theorem_audit/features.py",
    "experiments/theorem_audit/engine.py",
    "experiments/theorem_audit/run.py",
)


class AuditError(RuntimeError):
    """A configuration, artifact, or task-selection error with a concise message."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def _strict_json_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise AuditError(f"could not read strict JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"expected a JSON object in {path}")
    return value


def _json_safe(value: Any) -> Any:
    """Convert common numerical objects and replace non-finite floats by null."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"cannot serialize {type(value).__name__} to strict JSON")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(payload)
    encoded = (
        json.dumps(safe, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Hash the semantic JSON configuration (ignoring runner-private keys)."""

    public = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(public, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(payload.encode("utf-8"))


def _source_fingerprint() -> dict[str, Any]:
    files: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise AuditError(f"required source file is missing: {relative}")
        data = path.read_bytes()
        digest = _sha256_bytes(data)
        files[relative] = digest
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(data)
        combined.update(b"\0")
    return {"sha256": combined.hexdigest(), "files": files}


def _git_metadata() -> dict[str, Any]:
    def run_git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "status_entry_count": None if status is None else len(status.splitlines()),
    }


def _portable_path(value: str) -> str:
    """Return ``value`` repository-relative when it lies inside the repository.

    Absolute paths outside the repository are reduced to their final component
    so that recorded provenance never contains user directories or machine
    names; relative arguments are returned unchanged.
    """
    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return candidate.name


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    lock = PROJECT_ROOT / "uv.lock"
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_CPUS_PER_TASK",
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.executable).name, *(_portable_path(arg) for arg in sys.argv)],
        "working_directory": _portable_path(str(Path.cwd())),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_count": os.cpu_count(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "uv_lock_sha256": _sha256_file(lock) if lock.is_file() else None,
        "git": _git_metadata(),
        "slurm": {key: os.environ[key] for key in slurm_keys if key in os.environ},
    }


def _require_mapping(container: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise AuditError(f"configuration field {key!r} must be an object")
    return value


def _require_int(container: Mapping[str, Any], key: str, minimum: int = 0) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AuditError(f"configuration field {key!r} must be an integer >= {minimum}")
    return value


def _require_float(
    container: Mapping[str, Any], key: str, *, positive: bool = False
) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"configuration field {key!r} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise AuditError(f"configuration field {key!r} must be {qualifier}")
    return converted


def _validate_config(config: dict[str, Any], path: Path) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise AuditError(
            f"{path}: schema_version must be {SCHEMA_VERSION}, got "
            f"{config.get('schema_version')!r}"
        )
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise AuditError("configuration name must be a non-empty string")
    output_value = config.get("output_dir")
    if not isinstance(output_value, str) or not output_value.strip():
        raise AuditError("configuration output_dir must be a non-empty string")
    output = Path(output_value)
    output = output.resolve() if output.is_absolute() else (PROJECT_ROOT / output).resolve()
    if not _inside_project(output):
        raise AuditError(f"output_dir must remain inside {PROJECT_ROOT}; got {output}")

    runtime = _require_mapping(config, "runtime")
    if runtime.get("dtype") != "float64":
        raise AuditError("the theorem audit requires runtime.dtype='float64'")
    if not isinstance(runtime.get("device"), str) or not runtime["device"]:
        raise AuditError("runtime.device must be a non-empty torch device string")
    if not isinstance(runtime.get("deterministic"), bool):
        raise AuditError("runtime.deterministic must be boolean")

    problem_names = config.get("problems")
    known = set(list_problems())
    if (
        not isinstance(problem_names, list)
        or not problem_names
        or any(not isinstance(name, str) for name in problem_names)
    ):
        raise AuditError("configuration problems must be a non-empty string list")
    duplicates = sorted({name for name in problem_names if problem_names.count(name) > 1})
    unknown = sorted(set(problem_names) - known)
    if duplicates:
        raise AuditError(f"configuration contains duplicate problems: {duplicates}")
    if unknown:
        raise AuditError(f"configuration contains unknown problems: {unknown}")

    reference = _require_mapping(config, "reference")
    coarse = _require_int(reference, "coarse_samples", 1)
    fine = _require_int(reference, "fine_samples", 1)
    replicates = _require_int(reference, "replicates", 2)
    _require_int(reference, "chunk_size", 1)
    _require_int(reference, "seed", 0)
    tolerance = _require_float(reference, "relative_tolerance")
    if tolerance < 0.0:
        raise AuditError("reference.relative_tolerance must be non-negative")
    confidence = float(reference.get("confidence_multiplier", 2.365))
    if not math.isfinite(confidence) or confidence < 0.0:
        raise AuditError("reference.confidence_multiplier must be finite and non-negative")
    if fine != 2 * coarse or coarse & (coarse - 1) or fine & (fine - 1):
        raise AuditError(
            "reference sample counts must be powers of two with fine_samples=2*coarse_samples"
        )

    training = _require_mapping(config, "training")
    widths = training.get("widths")
    seeds = training.get("seeds")
    if (
        not isinstance(widths, list)
        or not widths
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in widths)
        or len(widths) != len(set(widths))
    ):
        raise AuditError("training.widths must contain unique positive integers")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise AuditError("training.seeds must contain unique non-negative integers")
    _require_int(training, "steps", 0)
    _require_int(training, "checkpoint_every", 1)
    factor = _require_float(training, "learning_rate_factor", positive=True)
    if factor >= 1.0:
        raise AuditError(
            "training.learning_rate_factor must be <1 for stability of the frozen full-loss update"
        )
    if training.get("nested_prefix") is not True:
        raise AuditError("training.nested_prefix must be true for paired width comparisons")

    # Store normalized paths/hashes under reserved keys in the in-memory copy.
    config = dict(config)
    config["_resolved_output_dir"] = str(output)
    config["_config_path"] = str(path.resolve())
    config["_config_hash"] = canonical_config_hash(config)
    return config


def load_config(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    path = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    if not path.is_file():
        raise AuditError(f"configuration file does not exist: {path}")
    return _validate_config(_strict_json_load(path), path)


def _configure_runtime(config: Mapping[str, Any]) -> torch.device:
    runtime = config["runtime"]
    assert isinstance(runtime, dict)
    try:
        device = torch.device(runtime["device"])
    except (RuntimeError, TypeError) as error:
        raise AuditError(f"invalid torch device {runtime['device']!r}: {error}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise AuditError("a CUDA device was requested but torch.cuda.is_available() is false")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise AuditError("an MPS device was requested but it is unavailable")
    torch.set_default_dtype(DTYPE)
    torch.use_deterministic_algorithms(bool(runtime["deterministic"]))
    return device


def _config_for_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _problem_manifest(problem: Problem) -> dict[str, Any]:
    return {
        "name": problem.name,
        "family": problem.family,
        "description": problem.description,
        "positivity_regime": problem.positivity_regime,
        "positivity_route": problem.positivity_route,
        "trial_space": problem.trial_space,
        "independence_witness": problem.independence_witness,
        "positivity_scope": problem.positivity_scope,
        "input_dim": problem.input_dim,
        "output_dim": problem.output_dim,
        "constraint_count": len(problem.constraints),
        "constraint_group_sizes": constraint_group_sizes(problem),
        "evaluation_point_count": int(problem.evaluation_points.shape[0]),
        "constraints": [
            {
                "name": constraint.name,
                "group": constraint.group,
                "target": constraint.target,
                "physical_point": constraint.physical_point,
                "terms": [
                    {
                        "point": term.point,
                        "output": term.output,
                        "alpha": term.alpha,
                        "coefficient": term.coefficient,
                    }
                    for term in constraint.terms
                ],
            }
            for constraint in problem.constraints
        ],
    }


def _experiment_specification() -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "activation": "tanh",
        "normalization_gamma": 1,
        "global_output_bias": False,
        "trainable_parameters": ["w", "b", "a"],
        "initialization": INITIALIZATION,
        "initialization_support": [-1.0, 1.0],
        "initialization_centered_output_weights": True,
        "dtype": "float64",
        "constraint_group_scaling": "each scalar row divided by sqrt(number of rows in its group)",
        "loss": LOSS,
        "optimizer": OPTIMIZER,
    }


def _common_manifest(
    config: Mapping[str, Any], device: torch.device, source: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "strict_json_nonfinite_policy": "non-finite floating diagnostics are serialized as null",
        "configuration": _config_for_manifest(config),
        "configuration_path": _relative_path(Path(str(config["_config_path"]))),
        "config_hash": config["_config_hash"],
        "configuration_sha256": config["_config_hash"],
        "source": source,
        "runtime": _runtime_metadata(device),
        "experiment": _experiment_specification(),
    }


def _analytic_certificate(problem: Problem) -> dict[str, Any]:
    """Return the registry-selected pointwise, functional, or measure certificate."""

    return dict(positivity_certificate(problem))


def _classify_reference(
    problem: Problem,
    reference: InfiniteNTKReference,
    analytic_certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Central policy for training eligibility; update here if theory expands.

    Automatic cases use their registry-selected exact design certificate from
    the paper: pointwise DNTK rank, weak-functional independence, or signed-
    measure independence.  Only cases explicitly registered as numerical use
    the conservative replicated-QMC gate.  A positive numerical gate is
    evidence, not a mathematical proof.
    """

    qmc_status = reference.full_positivity.status
    if problem.positivity_regime == "automatic":
        passed = analytic_certificate.get("passed") is True
        route_basis = {
            "pointwise_dntk": "paper pointwise DNTK positivity theorem plus local operator-rank certificate",
            "weak_functional": "paper weak-functional positivity theorem plus finite-jet functional-independence certificate",
            "nonlocal_measure": "paper signed-measure nonlocal positivity theorem plus measure-independence certificate",
        }.get(problem.positivity_route, "paper analytic positivity theorem plus stored design certificate")
        return {
            "policy_version": "registry_regime_v1",
            "decision": "theory_guaranteed" if passed else "invalid_automatic_certificate",
            "allows_training": passed,
            "basis": route_basis,
            "positivity_route": problem.positivity_route,
            "positivity_scope": problem.positivity_scope,
            "qmc_status": qmc_status,
            "qmc_is_training_gate": False,
            "interpretation": "QMC estimates the gap size but does not determine positivity for this case",
        }
    supported = qmc_status == "positive"
    return {
        "policy_version": "registry_regime_v1",
        "decision": "numerically_supported_positive" if supported else "inconclusive",
        "allows_training": supported,
        "basis": "conservative replicated scrambled-Sobol two-level positivity gate",
        "positivity_route": problem.positivity_route,
        "positivity_scope": problem.positivity_scope,
        "qmc_status": qmc_status,
        "qmc_is_training_gate": True,
        "interpretation": (
            "numerical evidence of positivity, not a proof"
            if supported
            else "the calculation did not certify positivity; it does not prove singularity"
        ),
    }


def _reference_paths(config: Mapping[str, Any], problem_name: str) -> tuple[Path, Path]:
    directory = Path(str(config["_resolved_output_dir"])) / "references"
    return directory / f"{problem_name}.json", directory / f"{problem_name}.npz"


def _run_paths(
    config: Mapping[str, Any], problem_name: str, width: int, seed: int
) -> tuple[Path, Path]:
    directory = Path(str(config["_resolved_output_dir"])) / "runs" / problem_name
    stem = f"w{width}_seed{seed}"
    return directory / f"{stem}.json", directory / f"{stem}.npz"


def _tensor_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def run_reference_task(
    config: Mapping[str, Any], problem_name: str, device: torch.device
) -> Path:
    started = time.perf_counter()
    problem = get_problem(problem_name)
    compiled = compile_constraints(problem, device=device)
    reference_config = config["reference"]
    assert isinstance(reference_config, dict)
    source = _source_fingerprint()
    print(
        f"[reference] {problem_name}: N={compiled.count}, "
        f"fine={reference_config['fine_samples']}, replicates={reference_config['replicates']}",
        flush=True,
    )
    reference = estimate_infinite_ntk(
        compiled,
        fine_samples=int(reference_config["fine_samples"]),
        coarse_samples=int(reference_config["coarse_samples"]),
        replicates=int(reference_config["replicates"]),
        seed=int(reference_config["seed"]),
        chunk_size=int(reference_config["chunk_size"]),
        positivity_relative_tolerance=float(reference_config["relative_tolerance"]),
        confidence_multiplier=float(reference_config.get("confidence_multiplier", 2.365)),
    )
    certificate = _analytic_certificate(problem)
    classification = _classify_reference(problem, reference, certificate)
    json_path, npz_path = _reference_paths(config, problem_name)
    arrays = {
        "K_full": _tensor_numpy(reference.K_full),
        "K_out": _tensor_numpy(reference.K_out),
        "K_hidden": _tensor_numpy(reference.K_hidden),
        "K_full_coarse": _tensor_numpy(reference.K_full_coarse),
        "K_out_coarse": _tensor_numpy(reference.K_out_coarse),
        "K_full_standard_error": _tensor_numpy(reference.K_full_standard_error),
        "K_out_standard_error": _tensor_numpy(reference.K_out_standard_error),
        "full_eigenvalues": _tensor_numpy(reference.full_spectrum.eigenvalues),
        "out_eigenvalues": _tensor_numpy(reference.out_spectrum.eigenvalues),
        "full_replicate_lambda_min": _tensor_numpy(reference.full_replicate_lambda_min),
        "out_replicate_lambda_min": _tensor_numpy(reference.out_replicate_lambda_min),
        "normalized_targets": _tensor_numpy(compiled.targets),
        "group_scales": _tensor_numpy(compiled.group_scales),
    }
    _atomic_npz(npz_path, arrays)
    manifest = {
        **_common_manifest(config, device, source),
        "artifact_type": "limiting_ntk_reference",
        "elapsed_seconds": time.perf_counter() - started,
        "status": "complete",
        "problem": _problem_manifest(problem),
        "positivity_certificate": certificate,
        "reference": reference.summary_dict(),
        "positivity_classification": classification,
        "artifacts": {
            "npz": {
                "path": _relative_path(npz_path),
                "sha256": _sha256_file(npz_path),
                "arrays": sorted(arrays),
            }
        },
    }
    _atomic_json(json_path, manifest)
    print(
        f"[reference] {problem_name}: {classification['decision']}; "
        f"lambda_min={reference.full_spectrum.lambda_min:.6e} -> {_relative_path(json_path)}",
        flush=True,
    )
    return json_path


def _load_reference(
    config: Mapping[str, Any], problem: Problem, device: torch.device
) -> tuple[dict[str, Any], torch.Tensor, Path]:
    json_path, npz_path = _reference_paths(config, problem.name)
    if not json_path.is_file() or not npz_path.is_file():
        raise AuditError(
            f"missing reference for {problem.name}; run the reference command first "
            f"(expected {json_path} and {npz_path})"
        )
    manifest = _strict_json_load(json_path)
    if manifest.get("artifact_type") != "limiting_ntk_reference" or manifest.get("status") != "complete":
        raise AuditError(f"invalid or incomplete reference manifest: {json_path}")
    if manifest.get("configuration_sha256") != config["_config_hash"]:
        raise AuditError(
            f"reference {json_path} was made with another configuration; rerun reference"
        )
    expected_source = _source_fingerprint()["sha256"]
    actual_source = manifest.get("source", {}).get("sha256")
    if actual_source != expected_source:
        raise AuditError(
            f"reference {json_path} was made with different audit source code; rerun reference"
        )
    expected_npz_hash = manifest.get("artifacts", {}).get("npz", {}).get("sha256")
    actual_npz_hash = _sha256_file(npz_path)
    if expected_npz_hash != actual_npz_hash:
        raise AuditError(f"reference array hash mismatch: {npz_path}")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if "K_full" not in archive:
                raise AuditError(f"reference archive lacks K_full: {npz_path}")
            matrix = np.asarray(archive["K_full"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise AuditError(f"could not load reference array {npz_path}: {error}") from error
    count = len(problem.constraints)
    if matrix.shape != (count, count) or not np.isfinite(matrix).all():
        raise AuditError(
            f"reference K_full has shape/data error: expected {(count, count)}, got {matrix.shape}"
        )
    kernel = torch.as_tensor(matrix, dtype=DTYPE, device=device)
    kernel = 0.5 * (kernel + kernel.T)
    return manifest, kernel, npz_path


def _operator_norm_symmetric(matrix: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    return float(eigenvalues.abs().max().detach().cpu())


def _kernel_concentration(
    initial_kernel: torch.Tensor,
    reference_kernel: torch.Tensor,
    relative_tolerance: float,
) -> dict[str, Any]:
    difference = 0.5 * (
        initial_kernel - reference_kernel + (initial_kernel - reference_kernel).T
    )
    tiny = torch.finfo(DTYPE).tiny
    ref_fro = torch.linalg.matrix_norm(reference_kernel, ord="fro")
    diff_fro = torch.linalg.matrix_norm(difference, ord="fro")
    ref_op = _operator_norm_symmetric(reference_kernel)
    diff_op = _operator_norm_symmetric(difference)
    ref_spectrum = summarize_spectrum(
        reference_kernel, relative_tolerance=relative_tolerance
    )
    initial_spectrum = summarize_spectrum(
        initial_kernel, relative_tolerance=relative_tolerance
    )

    generalized_min: float | None = None
    generalized_eigenvalues: list[float] | None = None
    loewner_three_quarters: bool | None = None
    threshold = relative_tolerance * max(ref_spectrum.lambda_max, 0.0)
    if ref_spectrum.lambda_min > threshold:
        eigenvalues, eigenvectors = torch.linalg.eigh(
            0.5 * (reference_kernel + reference_kernel.T)
        )
        inverse_root = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T
        whitened = inverse_root @ initial_kernel @ inverse_root
        whitened = 0.5 * (whitened + whitened.T)
        generalized = torch.linalg.eigvalsh(whitened)
        generalized_min = float(generalized[0].detach().cpu())
        generalized_eigenvalues = generalized.detach().cpu().tolist()
        loewner_three_quarters = generalized_min >= 0.75

    return {
        "initial_spectrum": initial_spectrum.as_dict(),
        "reference_spectrum_recomputed": ref_spectrum.as_dict(),
        "absolute_frobenius_error": float(diff_fro.detach().cpu()),
        "relative_frobenius_error": float((diff_fro / ref_fro.clamp_min(tiny)).detach().cpu()),
        "absolute_operator_error": diff_op,
        "relative_operator_error": diff_op / max(ref_op, float(tiny)),
        "lambda_min_ratio_to_reference": (
            initial_spectrum.lambda_min / ref_spectrum.lambda_min
            if ref_spectrum.lambda_min > 0.0
            else None
        ),
        "generalized_eigenvalues_of_Kinf_inverse_half_K0": generalized_eigenvalues,
        "generalized_lambda_min": generalized_min,
        "loewner_K0_ge_three_quarters_Kinf": loewner_three_quarters,
        "loewner_test_applicable": generalized_min is not None,
    }


@torch.no_grad()
def _solution_metrics(state: NetworkState, problem: Problem) -> tuple[dict[str, Any], np.ndarray]:
    points = problem.evaluation_points.to(device=state.device, dtype=DTYPE)
    prediction = network_output(state, points)
    exact = problem.exact_solution(points).to(device=state.device, dtype=DTYPE)
    error = prediction - exact
    absolute_l2 = torch.linalg.vector_norm(error)
    exact_l2 = torch.linalg.vector_norm(exact)
    relative_l2 = absolute_l2 / exact_l2.clamp_min(torch.finfo(DTYPE).tiny)
    rmse = torch.sqrt(torch.mean(error.square()))
    return (
        {
            "absolute_discrete_l2": float(absolute_l2.detach().cpu()),
            "relative_discrete_l2": float(relative_l2.detach().cpu()),
            "rmse": float(rmse.detach().cpu()),
            "point_count": int(points.shape[0]),
            "note": "secondary manufactured-solution metric; the theorem concerns the finite constraint loss",
        },
        _tensor_numpy(prediction),
    )


def _augmented_history(
    raw_history: Sequence[Mapping[str, Any]],
    *,
    width: int,
    learning_rate: float,
    reference_lambda_min_estimate: float,
    reference_lambda_lower_bound: float | None,
) -> list[dict[str, Any]]:
    initial_loss = float(raw_history[0]["loss"])
    contraction_base = (
        None
        if reference_lambda_lower_bound is None
        else 1.0 - learning_rate * reference_lambda_lower_bound
    )
    history: list[dict[str, Any]] = []
    for raw in raw_history:
        record = dict(raw)
        step = int(record["step"])
        record["loss_relative_to_initial"] = (
            float(record["loss"]) / initial_loss if initial_loss > 0.0 else None
        )
        # The engine already records sqrt(m)*max_j ||theta_j-theta_j(0)||.
        if "sqrt_width_max_per_neuron_drift" not in record:
            record["sqrt_width_max_per_neuron_drift"] = math.sqrt(width) * float(
                record["max_per_neuron_drift"]
            )
        record["lambda_min_ratio_to_reference_estimate"] = (
            float(record["lambda_min"]) / reference_lambda_min_estimate
            if reference_lambda_min_estimate > 0.0
            else None
        )
        record["half_estimated_reference_gap_preserved"] = (
            float(record["lambda_min"]) >= 0.5 * reference_lambda_min_estimate
            if reference_lambda_min_estimate > 0.0
            else None
        )
        record["theorem_conservative_loss_envelope"] = (
            initial_loss * contraction_base**step
            if contraction_base is not None and 0.0 <= contraction_base <= 1.0
            else None
        )
        history.append(record)
    return history


def _skip_training_manifest(
    config: Mapping[str, Any],
    device: torch.device,
    source: Mapping[str, Any],
    problem: Problem,
    width: int,
    seed: int,
    reason_code: str,
    reason: str,
    reference_manifest: Mapping[str, Any],
) -> Path:
    json_path, _ = _run_paths(config, problem.name, width, seed)
    manifest = {
        **_common_manifest(config, device, source),
        "artifact_type": "finite_width_training",
        "status": "skipped",
        "skip_reason_code": reason_code,
        "skip_reason": reason,
        "problem": _problem_manifest(problem),
        "task": {"width": width, "seed": seed},
        "reference": {
            "positivity_classification": reference_manifest.get("positivity_classification"),
            "manifest_path": _relative_path(_reference_paths(config, problem.name)[0]),
        },
        "artifacts": {"npz": None},
    }
    _atomic_json(json_path, manifest)
    print(f"[train] {problem.name} width={width} seed={seed}: skipped ({reason})", flush=True)
    return json_path


def run_training_task(
    config: Mapping[str, Any],
    problem_name: str,
    width: int,
    seed: int,
    device: torch.device,
) -> Path:
    started = time.perf_counter()
    problem = get_problem(problem_name)
    reference_manifest, reference_kernel, reference_npz_path = _load_reference(
        config, problem, device
    )
    source = _source_fingerprint()
    classification = reference_manifest.get("positivity_classification")
    if not isinstance(classification, dict):
        raise AuditError(f"reference classification is absent for {problem.name}")
    if classification.get("allows_training") is not True:
        if problem.positivity_regime == "numerical":
            reason_code = "positivity_gate_inconclusive"
            reason = "the registered numerical case did not pass the reference positivity gate"
        else:
            reason_code = "analytic_positivity_certificate_failed"
            reason = "the registry-selected analytic positivity certificate did not pass"
        return _skip_training_manifest(
            config,
            device,
            source,
            problem,
            width,
            seed,
            reason_code,
            reason,
            reference_manifest,
        )

    parameter_count = width * (problem.input_dim + 1 + problem.output_dim)
    constraint_count = len(problem.constraints)
    if parameter_count < constraint_count and not config["training"].get("allow_rank_infeasible", False):
        return _skip_training_manifest(
            config,
            device,
            source,
            problem,
            width,
            seed,
            "rank_infeasible",
            f"necessary Jacobian-rank condition fails: {parameter_count} parameters < {constraint_count} rows",
            reference_manifest,
        )

    training_config = config["training"]
    assert isinstance(training_config, dict)
    widths = [int(value) for value in training_config["widths"]]
    maximum_width = max(widths)
    if width not in widths:
        raise AuditError(f"width {width} is not listed in the configuration")
    if seed not in training_config["seeds"]:
        raise AuditError(f"seed {seed} is not listed in the configuration")
    bank = make_initialization_bank(
        maximum_width,
        problem.input_dim,
        problem.output_dim,
        seed,
        device=device,
    )
    initial_state = bank.prefix(width)
    compiled = compile_constraints(problem, device=device)
    initial_evaluation = evaluate_finite(initial_state, compiled)
    relative_tolerance = float(config["reference"]["relative_tolerance"])
    initial_spectrum = summarize_spectrum(
        initial_evaluation.kernel, relative_tolerance=relative_tolerance
    )
    if not math.isfinite(initial_spectrum.lambda_max) or initial_spectrum.lambda_max <= 0.0:
        raise AuditError(
            f"initial kernel for {problem.name}, width={width}, seed={seed} has invalid lambda_max"
        )
    factor = float(training_config["learning_rate_factor"])
    learning_rate = factor / initial_spectrum.lambda_max
    reference_spectrum = summarize_spectrum(
        reference_kernel, relative_tolerance=relative_tolerance
    )
    lower_bound_value = (
        reference_manifest.get("reference", {})
        .get("full_positivity", {})
        .get("lower_bound")
    )
    reference_lambda_lower_bound = (
        float(lower_bound_value)
        if isinstance(lower_bound_value, (int, float))
        and not isinstance(lower_bound_value, bool)
        and math.isfinite(float(lower_bound_value))
        and float(lower_bound_value) > 0.0
        else None
    )
    concentration = _kernel_concentration(
        initial_evaluation.kernel, reference_kernel, relative_tolerance
    )
    initial_solution, initial_prediction = _solution_metrics(initial_state, problem)

    print(
        f"[train] {problem.name}: width={width}, seed={seed}, "
        f"steps={training_config['steps']}, eta={learning_rate:.6e}",
        flush=True,
    )
    result = train_full_batch_gd(
        initial_state,
        compiled,
        steps=int(training_config["steps"]),
        learning_rate=learning_rate,
        checkpoint_every=int(training_config["checkpoint_every"]),
        reference_kernel=reference_kernel,
        relative_rank_tolerance=relative_tolerance,
    )
    final_evaluation = evaluate_finite(result.final_state, compiled)
    final_solution, final_prediction = _solution_metrics(result.final_state, problem)
    raw_history = result.history_dicts()
    history = _augmented_history(
        raw_history,
        width=width,
        learning_rate=learning_rate,
        reference_lambda_min_estimate=reference_spectrum.lambda_min,
        reference_lambda_lower_bound=reference_lambda_lower_bound,
    )
    initial_loss = float(history[0]["loss"])
    final_loss = float(history[-1]["loss"])
    losses = [float(record["loss"]) for record in history]
    half_gap_values = [
        record["half_estimated_reference_gap_preserved"] for record in history
    ]

    json_path, npz_path = _run_paths(config, problem.name, width, seed)
    arrays = {
        "initial_w": _tensor_numpy(result.initial_state.w),
        "initial_b": _tensor_numpy(result.initial_state.b),
        "initial_a": _tensor_numpy(result.initial_state.a),
        "final_w": _tensor_numpy(result.final_state.w),
        "final_b": _tensor_numpy(result.final_state.b),
        "final_a": _tensor_numpy(result.final_state.a),
        "K_initial": _tensor_numpy(result.initial_kernel),
        "K_final": _tensor_numpy(final_evaluation.kernel),
        "K_infinity_reference": _tensor_numpy(reference_kernel),
        "initial_constraint_error": _tensor_numpy(result.initial_error),
        "final_constraint_error": _tensor_numpy(final_evaluation.error),
        "normalized_targets": _tensor_numpy(compiled.targets),
        "evaluation_points": _tensor_numpy(problem.evaluation_points),
        "exact_solution_on_evaluation_points": _tensor_numpy(
            problem.exact_solution(problem.evaluation_points).detach()
        ),
        "initial_solution_prediction": initial_prediction,
        "final_solution_prediction": final_prediction,
        "history_step": np.asarray([record["step"] for record in history], dtype=np.int64),
        "history_loss": np.asarray([record["loss"] for record in history], dtype=np.float64),
        "history_lambda_min": np.asarray(
            [record["lambda_min"] for record in history], dtype=np.float64
        ),
        "history_lambda_max": np.asarray(
            [record["lambda_max"] for record in history], dtype=np.float64
        ),
        "history_relative_kernel_drift_frobenius": np.asarray(
            [record["relative_kernel_drift"] for record in history], dtype=np.float64
        ),
        "history_relative_kernel_drift_operator": np.asarray(
            [record["relative_kernel_drift_operator"] for record in history],
            dtype=np.float64,
        ),
        "history_relative_reference_kernel_error_frobenius": np.asarray(
            [record["relative_reference_kernel_error"] for record in history],
            dtype=np.float64,
        ),
        "history_relative_reference_kernel_error_operator": np.asarray(
            [record["relative_reference_kernel_error_operator"] for record in history],
            dtype=np.float64,
        ),
        "history_sqrt_width_max_per_neuron_drift": np.asarray(
            [record["sqrt_width_max_per_neuron_drift"] for record in history],
            dtype=np.float64,
        ),
        "history_relative_update_taylor_defect": np.asarray(
            [
                np.nan
                if record["relative_update_taylor_defect"] is None
                else record["relative_update_taylor_defect"]
                for record in history
            ],
            dtype=np.float64,
        ),
        "history_frozen_kernel_loss": np.asarray(
            [record["frozen_kernel_loss"] for record in history], dtype=np.float64
        ),
        "history_reference_kernel_loss": np.asarray(
            [record["reference_kernel_loss"] for record in history], dtype=np.float64
        ),
        "history_theorem_conservative_loss_envelope": np.asarray(
            [
                np.nan
                if record["theorem_conservative_loss_envelope"] is None
                else record["theorem_conservative_loss_envelope"]
                for record in history
            ],
            dtype=np.float64,
        ),
    }
    _atomic_npz(npz_path, arrays)
    manifest = {
        **_common_manifest(config, device, source),
        "artifact_type": "finite_width_training",
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "problem": _problem_manifest(problem),
        "task": {
            "width": width,
            "seed": seed,
            "maximum_width_initialization_bank": maximum_width,
            "uses_nested_prefix": True,
            "parameter_count": parameter_count,
            "constraint_count": constraint_count,
            "necessary_rank_feasible": parameter_count >= constraint_count,
        },
        "reference": {
            "manifest_path": _relative_path(_reference_paths(config, problem.name)[0]),
            "npz_path": _relative_path(reference_npz_path),
            "npz_sha256": _sha256_file(reference_npz_path),
            "positivity_classification": classification,
            "full_spectrum": reference_spectrum.as_dict(),
        },
        "optimization": {
            "loss": LOSS,
            "optimizer": OPTIMIZER,
            "gradient_update": "theta_(k+1)=theta_k-2*eta*J(theta_k)^T*(P(theta_k)-y)",
            "steps": int(training_config["steps"]),
            "checkpoint_every": int(training_config["checkpoint_every"]),
            "learning_rate_factor": factor,
            "learning_rate": learning_rate,
            "eta_times_lambda_max_K0": learning_rate * initial_spectrum.lambda_max,
            "frozen_quadratic_spectral_stability_checked": factor < 1.0,
            "paper_B_star_step_condition_verified": False,
            "paper_B_star_step_condition_note": "B_* is an existential proof constant and is not numerically available; the spectral factor is reported without claiming it verifies eta*B_*^2<=1/2",
            "theorem_envelope_gap_lower_bound": reference_lambda_lower_bound,
            "theorem_envelope_gap_source": "positive conservative QMC lower_bound; null means no quantitative envelope was certified",
        },
        "initial_kernel_concentration": concentration,
        "solution_error": {"initial": initial_solution, "final": final_solution},
        "training_summary": {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "final_to_initial_loss_ratio": final_loss / initial_loss if initial_loss > 0.0 else None,
            "checkpoint_loss_monotone_nonincreasing": all(
                following <= previous * (1.0 + 1.0e-12)
                for previous, following in zip(losses, losses[1:])
            ),
            "half_estimated_reference_gap_preserved_at_all_checkpoints": (
                all(bool(value) for value in half_gap_values)
                if all(value is not None for value in half_gap_values)
                else None
            ),
            "history": history,
        },
        "artifacts": {
            "npz": {
                "path": _relative_path(npz_path),
                "sha256": _sha256_file(npz_path),
                "arrays": sorted(arrays),
            }
        },
    }
    _atomic_json(json_path, manifest)
    print(
        f"[train] {problem.name} width={width} seed={seed}: "
        f"loss {initial_loss:.6e} -> {final_loss:.6e} -> {_relative_path(json_path)}",
        flush=True,
    )
    return json_path


def _select_reference_tasks(
    config: Mapping[str, Any], requested: Sequence[str] | None, task_index: int | None
) -> list[str]:
    configured = list(config["problems"])
    if task_index is not None:
        if requested:
            raise AuditError("--task-index cannot be combined with --problem")
        if not 0 <= task_index < len(configured):
            raise AuditError(
                f"reference task index {task_index} is outside [0, {len(configured) - 1}]"
            )
        return [configured[task_index]]
    if not requested:
        return configured
    unknown = sorted(set(requested) - set(configured))
    if unknown:
        raise AuditError(f"requested problems are not in the configuration: {unknown}")
    requested_set = set(requested)
    return [name for name in configured if name in requested_set]


def _all_training_tasks(config: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    training = config["training"]
    return [
        (problem, int(width), int(seed))
        for problem in config["problems"]
        for width in training["widths"]
        for seed in training["seeds"]
    ]


def _select_training_tasks(
    config: Mapping[str, Any],
    requested_problems: Sequence[str] | None,
    requested_widths: Sequence[int] | None,
    requested_seeds: Sequence[int] | None,
    task_index: int | None,
) -> list[tuple[str, int, int]]:
    tasks = _all_training_tasks(config)
    if task_index is not None:
        if requested_problems or requested_widths or requested_seeds:
            raise AuditError("--task-index cannot be combined with --problem, --width, or --seed")
        if not 0 <= task_index < len(tasks):
            raise AuditError(
                f"training task index {task_index} is outside [0, {len(tasks) - 1}]"
            )
        return [tasks[task_index]]

    configured_problems = set(config["problems"])
    configured_widths = set(config["training"]["widths"])
    configured_seeds = set(config["training"]["seeds"])
    if requested_problems and not set(requested_problems) <= configured_problems:
        raise AuditError("one or more requested problems are not in the configuration")
    if requested_widths and not set(requested_widths) <= configured_widths:
        raise AuditError("one or more requested widths are not in the configuration")
    if requested_seeds and not set(requested_seeds) <= configured_seeds:
        raise AuditError("one or more requested seeds are not in the configuration")
    problem_filter = set(requested_problems) if requested_problems else configured_problems
    width_filter = set(requested_widths) if requested_widths else configured_widths
    seed_filter = set(requested_seeds) if requested_seeds else configured_seeds
    return [
        task
        for task in tasks
        if task[0] in problem_filter and task[1] in width_filter and task[2] in seed_filter
    ]


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="JSON experiment configuration")
    parser.add_argument(
        "--problem",
        action="append",
        dest="problems",
        help="restrict to one configured problem; repeat to select several",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact-architecture finite-width PINN NTK theorem audit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser("reference", help="estimate limiting NTK references")
    _add_common_arguments(reference)
    reference.add_argument(
        "--task-index", type=int, help="zero-based configured-problem index for a job array"
    )

    train = subparsers.add_parser("train", help="run finite-width full-batch GD tasks")
    _add_common_arguments(train)
    train.add_argument("--width", action="append", type=int, dest="widths")
    train.add_argument("--seed", action="append", type=int, dest="seeds")
    train.add_argument(
        "--task-index", type=int, help="zero-based problem-major (problem,width,seed) array index"
    )

    all_parser = subparsers.add_parser(
        "all", help="run selected references, then their selected training tasks"
    )
    _add_common_arguments(all_parser)
    all_parser.add_argument("--width", action="append", type=int, dest="widths")
    all_parser.add_argument("--seed", action="append", type=int, dest="seeds")

    aggregate = subparsers.add_parser(
        "aggregate", help="validate completed artifacts and create tables/figures"
    )
    aggregate.add_argument("--config", required=True, help="JSON experiment configuration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        device = _configure_runtime(config)
        if arguments.command == "reference":
            tasks = _select_reference_tasks(config, arguments.problems, arguments.task_index)
            for problem_name in tasks:
                run_reference_task(config, problem_name, device)
        elif arguments.command == "train":
            tasks = _select_training_tasks(
                config,
                arguments.problems,
                arguments.widths,
                arguments.seeds,
                arguments.task_index,
            )
            for problem_name, width, seed in tasks:
                run_training_task(config, problem_name, width, seed, device)
        elif arguments.command == "all":
            reference_tasks = _select_reference_tasks(config, arguments.problems, None)
            for problem_name in reference_tasks:
                run_reference_task(config, problem_name, device)
            training_tasks = _select_training_tasks(
                config, arguments.problems, arguments.widths, arguments.seeds, None
            )
            for problem_name, width, seed in training_tasks:
                run_training_task(config, problem_name, width, seed, device)
        elif arguments.command == "aggregate":
            # Lazy import keeps array workers independent of the plotting stack.
            from experiments.theorem_audit.aggregate import (  # noqa: PLC0415
                AggregateError,
                aggregate_results,
            )

            try:
                aggregate = aggregate_results(arguments.config)
            except AggregateError as error:
                raise AuditError(str(error)) from error
            coverage = aggregate.get("coverage", {})
            print(
                "[aggregate] wrote tables/figures; "
                f"accepted references={coverage.get('accepted_reference_artifacts')}/"
                f"{coverage.get('expected_references')}, "
                f"training statuses={coverage.get('training_status_counts')}",
                flush=True,
            )
        else:  # pragma: no cover - argparse enforces the choices.
            raise AssertionError(arguments.command)
    except AuditError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
