"""Strict aggregation and publication figures for the theorem audit.

The array jobs deliberately write disjoint JSON artifacts.  This module is the
only place that combines them.  Every accepted artifact must carry the active
configuration hash, so stale runs from a previous sweep cannot silently enter
tables or figures.  Missing, skipped, and malformed jobs are retained as audit
rows rather than being dropped.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .problems import get_problem


SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
COMPLETE_STATUSES = {"complete", "completed", "ok", "success"}
SKIPPED_STATUSES = {"skip", "skipped", "gated", "positivity_gated"}

DISPLAY_NAMES = {
    "poisson_1d": "Poisson (1D)",
    "variable_elliptic_2d": "Variable elliptic (2D)",
    "heat_1d": "Heat (1+1D)",
    "transport_1d": "Transport (1+1D)",
    "wave_1d": "Wave (1+1D)",
    "biharmonic_2d": "Biharmonic (2D)",
    "stokes_2d": "Stokes (2D, vector)",
    "weak_poisson_1d": "Weak Poisson (1D)",
    "nonlocal_diffusion_1d": "Nonlocal diffusion (1D)",
    "integro_diff_1d": "Integro-differential (1D)",
    "caputo_diffusion_1d": "Caputo diffusion (1+1D)",
}


class AggregateError(RuntimeError):
    """Raised when manifest identity is ambiguous or inconsistent."""


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 identity used by jobs and aggregation."""

    public = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(
        public, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _json_safe(value),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(path)


def _nested(source: Any, dotted_path: str, default: Any = None) -> Any:
    value = source
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _first(source: Mapping[str, Any], paths: Sequence[str], default: Any = None) -> Any:
    for path in paths:
        value = _nested(source, path, default=None)
        if value is not None:
            return value
    return default


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
        if float(value) != result:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return result


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "pass", "passed", "positive"}:
            return True
        if lowered in {"false", "no", "fail", "failed", "inconclusive"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return None


def _status(value: Any, *, default: str = "unknown") -> str:
    if value is None:
        return default
    return str(value).strip().lower()


def _artifact_hash(data: Mapping[str, Any]) -> str | None:
    value = _first(
        data,
        (
            "config_hash",
            "configuration_sha256",
            "manifest.config_hash",
            "provenance.config_hash",
            "metadata.config_hash",
        ),
    )
    return None if value is None else str(value)


def _artifact_problem_name(data: Mapping[str, Any], fallback: str) -> str:
    value = _first(data, ("problem_name", "problem.name", "problem"), fallback)
    return str(value) if not isinstance(value, Mapping) else fallback


def _locate_unique(candidates: Iterable[Path]) -> tuple[Path | None, str | None]:
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        return None, None
    if len(existing) > 1:
        return None, "multiple candidate artifacts: " + ", ".join(map(str, existing))
    return existing[0], None


def _reference_candidates(output_dir: Path, problem: str) -> tuple[Path, ...]:
    base = output_dir / "references"
    return (
        base / f"{problem}.json",
        base / problem / "reference.json",
        base / problem / "result.json",
    )


def _run_candidates(output_dir: Path, problem: str, width: int, seed: int) -> tuple[Path, ...]:
    base = output_dir / "runs" / problem
    return (
        base / f"w{width}_seed{seed}.json",
        base / f"width{width}_seed{seed}.json",
        base / f"width_{width}_seed_{seed}.json",
        base / f"w{width}" / f"seed{seed}.json",
        base / f"width_{width}" / f"seed_{seed}.json",
    )


def _problem_metadata(problem_name: str) -> dict[str, Any]:
    problem = get_problem(problem_name)
    return {
        "family": problem.family,
        "positivity_regime": problem.positivity_regime,
        "positivity_route": problem.positivity_route,
        "positivity_scope": problem.positivity_scope,
        "trial_space": problem.trial_space,
        "independence_witness": problem.independence_witness,
        "constraint_count": len(problem.constraints),
        "input_dim": problem.input_dim,
        "output_dim": problem.output_dim,
    }


def _reference_row(
    problem_name: str,
    path: Path | None,
    expected_hash: str,
    locate_error: str | None,
    *,
    expected_source_hash: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    metadata = _problem_metadata(problem_name)
    row: dict[str, Any] = {
        "problem": problem_name,
        "display_name": DISPLAY_NAMES.get(problem_name, problem_name),
        **metadata,
        "artifact_status": "missing",
        "artifact_path": None,
        "source_hash": None,
        "npz_sha256": None,
        "verdict": None,
        "allows_training": None,
        "certificate_type": None,
        "certificate_passed": None,
        "qmc_status": None,
        "lambda_min": None,
        "lambda_max": None,
        "relative_gap": None,
        "conservative_lower_bound": None,
        "relative_lower_bound": None,
        "lambda_min_standard_error": None,
        "level_difference_operator_norm": None,
    }
    if locate_error is not None:
        row["artifact_status"] = "ambiguous"
        return row, None, locate_error
    if path is None:
        return row, None, f"missing reference for {problem_name}"

    row["artifact_path"] = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
    try:
        data = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        row["artifact_status"] = "invalid_json"
        return row, None, f"{path}: {error}"

    actual_hash = _artifact_hash(data)
    if actual_hash != expected_hash:
        row["artifact_status"] = "rejected_config_hash"
        return row, None, (
            f"{path}: config_hash={actual_hash!r}, expected {expected_hash!r}"
        )
    row["source_hash"] = _nested(data, "source.sha256")
    row["npz_sha256"] = _nested(data, "artifacts.npz.sha256")
    if expected_source_hash is not None and row["source_hash"] != expected_source_hash:
        row["artifact_status"] = "rejected_source_hash"
        return row, None, f"{path}: reference source fingerprint differs from active code"
    recorded_problem = _artifact_problem_name(data, problem_name)
    if recorded_problem != problem_name:
        row["artifact_status"] = "rejected_problem_mismatch"
        return row, None, f"{path}: records problem {recorded_problem!r}"

    artifact_status = _status(_first(data, ("status", "reference_status")))
    row["artifact_status"] = artifact_status
    if artifact_status not in COMPLETE_STATUSES:
        return row, None, f"{path}: incomplete reference status {artifact_status!r}"
    row["verdict"] = str(
        _first(
            data,
            (
                "positivity_classification.decision",
                "verdict.status",
                "verdict",
                "positivity_verdict",
                "gate_verdict",
            ),
            "unknown",
        )
    )
    row["qmc_status"] = str(
        _first(
            data,
            (
                "reference.full_positivity.status",
                "estimate.full_positivity.status",
                "full_positivity.status",
                "positivity_classification.qmc_status",
            ),
            "unknown",
        )
    )
    row["allows_training"] = _as_bool(
        _nested(data, "positivity_classification.allows_training")
    )
    row["certificate_type"] = _nested(data, "positivity_certificate.certificate_type")
    row["certificate_passed"] = _as_bool(
        _nested(data, "positivity_certificate.passed")
    )
    spectrum_prefixes = (
        "reference.full_spectrum",
        "estimate.full_spectrum",
        "full_spectrum",
    )
    positivity_prefixes = (
        "reference.full_positivity",
        "estimate.full_positivity",
        "full_positivity",
    )

    def from_prefixes(prefixes: Sequence[str], key: str) -> Any:
        return _first(data, tuple(f"{prefix}.{key}" for prefix in prefixes))

    row["lambda_min"] = _as_float(from_prefixes(spectrum_prefixes, "lambda_min"))
    row["lambda_max"] = _as_float(from_prefixes(spectrum_prefixes, "lambda_max"))
    if (
        row["lambda_min"] is None
        or row["lambda_max"] is None
        or row["lambda_max"] < row["lambda_min"]
    ):
        row["artifact_status"] = "invalid_spectrum"
        return row, None, f"{path}: completed reference lacks a finite ordered spectrum"
    row["conservative_lower_bound"] = _as_float(
        from_prefixes(positivity_prefixes, "lower_bound")
    )
    row["relative_lower_bound"] = _as_float(
        from_prefixes(positivity_prefixes, "relative_lower_bound")
    )
    row["lambda_min_standard_error"] = _as_float(
        from_prefixes(positivity_prefixes, "replicate_lambda_min_standard_error")
    )
    row["level_difference_operator_norm"] = _as_float(
        from_prefixes(positivity_prefixes, "level_difference_operator_norm")
    )
    if row["lambda_min"] is not None and row["lambda_max"] not in {None, 0.0}:
        row["relative_gap"] = row["lambda_min"] / row["lambda_max"]
    if row["relative_lower_bound"] is None and (
        row["conservative_lower_bound"] is not None
        and row["lambda_max"] not in {None, 0.0}
    ):
        row["relative_lower_bound"] = (
            row["conservative_lower_bound"] / row["lambda_max"]
        )
    return row, data, None


def _extract_history(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = _first(
        data,
        ("training_summary.history", "training.history", "result.history", "history"),
        [],
    )
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        return []
    return sorted(value, key=lambda item: _as_int(item.get("step")) or 0)


def _history_float(record: Mapping[str, Any], names: Sequence[str]) -> float | None:
    return _as_float(_first(record, names))


def _normalize_history(
    problem: str,
    width: int,
    seed: int,
    data: Mapping[str, Any],
    reference_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = _extract_history(data)
    if not source:
        return []
    initial_loss = _history_float(source[0], ("loss", "training_loss"))
    learning_rate = _as_float(
        _first(
            data,
            (
                "optimization.learning_rate",
                "training.learning_rate",
                "result.learning_rate",
                "learning_rate",
            ),
        )
    )
    lambda_infinity = _as_float(reference_row.get("lambda_min"))
    theorem_gap_lower_bound = _as_float(
        _nested(data, "optimization.theorem_envelope_gap_lower_bound")
    )
    normalized: list[dict[str, Any]] = []
    for record in source:
        step = _as_int(record.get("step"))
        loss = _history_float(record, ("loss", "training_loss"))
        frozen_loss = _history_float(
            record, ("frozen_kernel_loss", "frozen_k0_loss", "linearized_loss")
        )
        reference_loss = _history_float(
            record, ("reference_kernel_loss", "infinite_kernel_loss", "kinf_loss")
        )
        theorem_loss = _history_float(
            record,
            (
                "theorem_envelope_loss",
                "theorem_loss_upper_bound",
                "theorem_envelope",
                "theorem_conservative_loss_envelope",
            ),
        )
        if (
            theorem_loss is None
            and step is not None
            and initial_loss is not None
            and learning_rate is not None
            and theorem_gap_lower_bound is not None
            and theorem_gap_lower_bound > 0
        ):
            contraction = 1.0 - learning_rate * theorem_gap_lower_bound
            if 0.0 <= contraction <= 1.0:
                theorem_loss = initial_loss * contraction**step

        explicit_operator_drift = _history_float(
            record,
            (
                "relative_kernel_drift_operator",
                "relative_kernel_operator_drift",
                "kernel_drift_operator_relative",
            ),
        )
        generic_drift = _history_float(record, ("relative_kernel_drift",))
        explicit_frobenius_drift = _history_float(
            record,
            (
                "relative_kernel_drift_frobenius",
                "relative_kernel_frobenius_drift",
            ),
        )
        if explicit_frobenius_drift is None:
            # The engine's historical unqualified quantity is Frobenius based.
            explicit_frobenius_drift = generic_drift

        movement = _history_float(
            record,
            (
                "sqrt_width_max_per_neuron_drift",
                "sqrt_width_times_max_per_neuron_drift",
                "scaled_max_per_neuron_drift",
            ),
        )
        raw_movement = _history_float(record, ("max_per_neuron_drift",))
        if movement is None and raw_movement is not None:
            movement = math.sqrt(width) * raw_movement

        lambda_min = _history_float(record, ("lambda_min", "kernel_lambda_min"))
        lambda_ratio = _history_float(
            record,
            (
                "lambda_min_over_reference",
                "lambda_min_ratio_to_reference",
                "lambda_min_ratio_to_reference_estimate",
            ),
        )
        if lambda_ratio is None and lambda_min is not None and lambda_infinity not in {None, 0.0}:
            lambda_ratio = lambda_min / lambda_infinity
        gap_preserved = _as_bool(
            _first(
                record,
                (
                    "gap_preserved",
                    "half_gap_preserved",
                    "reference_half_gap_preserved",
                    "half_estimated_reference_gap_preserved",
                ),
            )
        )
        if gap_preserved is None and lambda_ratio is not None:
            gap_preserved = lambda_ratio >= 0.5

        def relative(value: float | None) -> float | None:
            if value is None or initial_loss in {None, 0.0}:
                return None
            return value / initial_loss

        normalized.append(
            {
                "problem": problem,
                "width": width,
                "seed": seed,
                "step": step,
                "loss": loss,
                "normalized_loss": relative(loss),
                "frozen_kernel_loss": frozen_loss,
                "normalized_frozen_kernel_loss": relative(frozen_loss),
                "reference_kernel_loss": reference_loss,
                "normalized_reference_kernel_loss": relative(reference_loss),
                "theorem_envelope_loss": theorem_loss,
                "normalized_theorem_envelope": relative(theorem_loss),
                "lambda_min": lambda_min,
                "lambda_max": _history_float(record, ("lambda_max", "kernel_lambda_max")),
                "lambda_min_ratio_to_reference": lambda_ratio,
                "gap_preserved": gap_preserved,
                "relative_kernel_drift_operator": explicit_operator_drift,
                "relative_kernel_drift_frobenius": explicit_frobenius_drift,
                "relative_reference_kernel_error": _history_float(
                    record,
                    (
                        "relative_reference_kernel_error",
                        "relative_kernel_reference_error",
                    ),
                ),
                "relative_reference_kernel_error_frobenius": _history_float(
                    record,
                    (
                        "relative_reference_kernel_error_frobenius",
                        "relative_reference_kernel_error",
                    ),
                ),
                "relative_reference_kernel_error_operator": _history_float(
                    record, ("relative_reference_kernel_error_operator",)
                ),
                "sqrt_width_max_per_neuron_drift": movement,
                "max_per_neuron_drift": raw_movement,
                "total_parameter_drift": _history_float(
                    record, ("total_parameter_drift",)
                ),
                "relative_error_to_frozen_dynamics": _history_float(
                    record, ("relative_error_to_frozen_dynamics",)
                ),
                "relative_update_taylor_defect": _history_float(
                    record, ("relative_update_taylor_defect",)
                ),
                "jacobian_drift_operator": _history_float(record, ("jacobian_drift_operator",)),
                "jacobian_drift_over_initial_sqrt_gap": _history_float(
                    record, ("jacobian_drift_over_initial_sqrt_gap",)
                ),
                "jacobian_perturbation_gap_lower_bound": _history_float(
                    record, ("jacobian_perturbation_gap_lower_bound",)
                ),
                "kernel_drift_over_initial_gap": _history_float(record, ("kernel_drift_over_initial_gap",)),
            }
        )
    return normalized


def _run_row(
    problem: str,
    width: int,
    seed: int,
    path: Path | None,
    expected_hash: str,
    locate_error: str | None,
    reference_row: Mapping[str, Any],
    *,
    expected_steps: int | None = None,
    expected_source_hash: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    row: dict[str, Any] = {
        "problem": problem,
        "display_name": DISPLAY_NAMES.get(problem, problem),
        "width": width,
        "seed": seed,
        "status": "missing",
        "skip_reason": None,
        "skip_reason_code": None,
        "artifact_path": None,
        "source_hash": None,
        "learning_rate": None,
        "checkpoint_count": 0,
        "initial_loss": None,
        "final_loss": None,
        "final_over_initial_loss": None,
        "initial_lambda_min": None,
        "final_lambda_min": None,
        "initial_reference_error_operator": None,
        "initial_reference_error_frobenius": None,
        "initial_generalized_gap_ratio": None,
        "initial_minimum_eigenvalue_ratio": None,
        "initial_gap_criterion_passed": None,
        "gap_criterion_source": None,
        "max_kernel_drift_operator": None,
        "max_kernel_drift_frobenius": None,
        "max_sqrt_width_per_neuron_drift": None,
        "minimum_gap_ratio_during_training": None,
        "gap_preserved_through_training": None,
        "final_solution_relative_l2": None,
        "max_relative_update_taylor_defect": None,
        "final_relative_error_to_frozen_dynamics": None,
        "max_jacobian_drift_operator": None,
        "max_jacobian_drift_over_initial_sqrt_gap": None,
        "min_jacobian_perturbation_gap_lower_bound": None,
        "max_kernel_drift_over_initial_gap": None,
    }
    if locate_error is not None:
        row["status"] = "ambiguous"
        return row, [], locate_error
    if path is None:
        return row, [], f"missing run for {problem}, width={width}, seed={seed}"
    row["artifact_path"] = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
    try:
        data = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        row["status"] = "invalid_json"
        return row, [], f"{path}: {error}"
    actual_hash = _artifact_hash(data)
    if actual_hash != expected_hash:
        row["status"] = "rejected_config_hash"
        return row, [], f"{path}: config_hash={actual_hash!r}, expected {expected_hash!r}"
    row["source_hash"] = _nested(data, "source.sha256")
    if expected_source_hash is not None and row["source_hash"] != expected_source_hash:
        row["status"] = "rejected_source_hash"
        return row, [], f"{path}: training source fingerprint differs from active code"
    if _artifact_problem_name(data, problem) != problem:
        row["status"] = "rejected_problem_mismatch"
        return row, [], f"{path}: problem identity mismatch"
    recorded_width = _as_int(
        _first(data, ("task.width", "width", "training.width", "result.width"))
    )
    recorded_seed = _as_int(_first(data, ("task.seed", "seed", "training_seed")))
    if recorded_width != width:
        row["status"] = "rejected_width_mismatch"
        return row, [], f"{path}: records width={recorded_width}, expected {width}"
    if recorded_seed != seed:
        row["status"] = "rejected_seed_mismatch"
        return row, [], f"{path}: records seed={recorded_seed}, expected {seed}"

    status = _status(_first(data, ("status", "training_status")))
    row["status"] = status
    row["skip_reason"] = _first(data, ("skip_reason", "reason", "message"))
    row["skip_reason_code"] = _first(data, ("skip_reason_code", "reason_code"))
    if status in SKIPPED_STATUSES:
        return row, [], None
    if status not in COMPLETE_STATUSES:
        return row, [], f"{path}: incomplete status {status!r}"
    if reference_row.get("npz_sha256") is not None and (
        _nested(data, "reference.npz_sha256") != reference_row["npz_sha256"]
    ):
        row["status"] = "rejected_reference_hash"
        return row, [], f"{path}: training used a different limiting-NTK reference artifact"
    if (
        "artifact_status" in reference_row
        and reference_row["artifact_status"] not in COMPLETE_STATUSES
    ):
        row["status"] = "invalid_reference"
        return row, [], f"{path}: associated reference was not accepted"

    history = _normalize_history(problem, width, seed, data, reference_row)
    if not history or history[0]["step"] != 0:
        row["status"] = "invalid_history"
        return row, [], f"{path}: completed run has no step-zero history"
    steps = [item["step"] for item in history]
    if any(step is None for step in steps) or len(set(steps)) != len(steps):
        row["status"] = "invalid_history"
        return row, [], f"{path}: checkpoint steps are missing or duplicated"
    recorded_steps = _as_int(
        _first(data, ("optimization.steps", "training.steps", "result.steps", "steps"))
    )
    required_steps = expected_steps if expected_steps is not None else recorded_steps
    if (
        any(step < 0 for step in steps)
        or (required_steps is not None and steps[-1] != required_steps)
        or (
            expected_steps is not None
            and recorded_steps is not None
            and recorded_steps != expected_steps
        )
    ):
        row["status"] = "invalid_history"
        return row, [], f"{path}: completed history does not reach requested step {required_steps}"
    if any(item["loss"] is None or item["loss"] < 0 for item in history):
        row["status"] = "invalid_history"
        return row, [], f"{path}: completed history contains missing, non-finite, or negative loss"

    initial, final = history[0], history[-1]
    row["checkpoint_count"] = len(history)
    row["learning_rate"] = _as_float(
        _first(
            data,
            (
                "optimization.learning_rate",
                "training.learning_rate",
                "result.learning_rate",
                "learning_rate",
            ),
        )
    )
    row["initial_loss"] = initial["loss"]
    row["final_loss"] = final["loss"]
    if initial["loss"] not in {None, 0.0} and final["loss"] is not None:
        row["final_over_initial_loss"] = final["loss"] / initial["loss"]
    row["initial_lambda_min"] = initial["lambda_min"]
    row["final_lambda_min"] = final["lambda_min"]

    initial_metrics = _first(
        data,
        ("initial_kernel_concentration", "initial_metrics", "diagnostics.initial"),
        {},
    )
    if not isinstance(initial_metrics, Mapping):
        initial_metrics = {}
    row["initial_reference_error_operator"] = _as_float(
        _first(
            initial_metrics,
            (
                "relative_reference_kernel_error_operator",
                "relative_kernel_error_operator",
                "relative_operator_error",
            ),
        )
    )
    row["initial_reference_error_frobenius"] = _as_float(
        _first(
            initial_metrics,
            (
                "relative_reference_kernel_error_frobenius",
                "relative_kernel_error_frobenius",
                "relative_frobenius_error",
            ),
        )
    )
    if row["initial_reference_error_frobenius"] is None:
        row["initial_reference_error_frobenius"] = initial[
            "relative_reference_kernel_error"
        ]

    gap_ratio = _as_float(
        _first(
            initial_metrics,
            (
                "generalized_minimum_eigenvalue",
                "generalized_min_eigenvalue",
                "generalized_lambda_min",
                "whitened_minimum_eigenvalue",
                "loewner_min_ratio",
                "initial_generalized_gap_ratio",
            ),
        )
    )
    gap_pass = _as_bool(
        _first(
            initial_metrics,
            (
                "three_quarter_loewner_bound_passed",
                "initial_gap_criterion_passed",
                "loewner_gap_passed",
                "loewner_K0_ge_three_quarters_Kinf",
            ),
        )
    )
    if gap_ratio is not None:
        row["gap_criterion_source"] = "generalized_loewner_ratio"
        if gap_pass is None:
            gap_pass = gap_ratio >= 0.75
    reference_lambda = _as_float(reference_row.get("lambda_min"))
    if initial["lambda_min"] is not None and reference_lambda is not None and reference_lambda > 0:
        row["initial_minimum_eigenvalue_ratio"] = initial["lambda_min"] / reference_lambda
    # A scalar minimum-eigenvalue ratio does not establish a Loewner order.
    # Keep it as a distinct descriptive measurement, never a fallback event.
    row["initial_generalized_gap_ratio"] = gap_ratio
    row["initial_gap_criterion_passed"] = gap_pass

    operator_drifts = [
        item["relative_kernel_drift_operator"]
        for item in history
        if item["relative_kernel_drift_operator"] is not None
    ]
    frobenius_drifts = [
        item["relative_kernel_drift_frobenius"]
        for item in history
        if item["relative_kernel_drift_frobenius"] is not None
    ]
    movements = [
        item["sqrt_width_max_per_neuron_drift"]
        for item in history
        if item["sqrt_width_max_per_neuron_drift"] is not None
    ]
    gap_ratios = [
        item["lambda_min_ratio_to_reference"]
        for item in history
        if item["lambda_min_ratio_to_reference"] is not None
    ]
    row["max_kernel_drift_operator"] = max(operator_drifts, default=None)
    row["max_kernel_drift_frobenius"] = max(frobenius_drifts, default=None)
    row["max_sqrt_width_per_neuron_drift"] = max(movements, default=None)
    taylor_defects = [
        item["relative_update_taylor_defect"]
        for item in history
        if item["relative_update_taylor_defect"] is not None
    ]
    row["max_relative_update_taylor_defect"] = max(taylor_defects, default=None)
    for field in ("jacobian_drift_operator", "jacobian_drift_over_initial_sqrt_gap", "kernel_drift_over_initial_gap"):
        row[f"max_{field}"] = max(
            (item[field] for item in history if item[field] is not None), default=None,
        )
    row["min_jacobian_perturbation_gap_lower_bound"] = min(
        (item["jacobian_perturbation_gap_lower_bound"] for item in history
         if item["jacobian_perturbation_gap_lower_bound"] is not None), default=None,
    )
    row["final_relative_error_to_frozen_dynamics"] = final[
        "relative_error_to_frozen_dynamics"
    ]
    row["minimum_gap_ratio_during_training"] = min(gap_ratios, default=None)
    if gap_ratios:
        row["gap_preserved_through_training"] = min(gap_ratios) >= 0.5
    row["final_solution_relative_l2"] = _as_float(
        _first(
            data,
            (
                "solution_metrics.final_relative_l2",
                "final_metrics.solution_relative_l2",
                "solution_error.final.relative_discrete_l2",
                "final_solution_relative_l2",
                "relative_l2_error",
            ),
        )
    )
    return row, history, None


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], leading: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = set().union(*(row.keys() for row in rows)) if rows else set(leading)
    fieldnames = list(dict.fromkeys([*leading, *sorted(keys - set(leading))]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    temporary.replace(path)


def _plot_setup() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "savefig.dpi": 300,
        }
    )
    return plt


def _save_figure(figure: Any, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, bbox_inches="tight", dpi=300, metadata={"Software": "theorem_audit"})
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "theorem_audit", "Title": stem.name},
    )
    return [str(png), str(pdf)]


ROUTE_COLORS = {
    "pointwise_dntk": "#20639B",
    "weak_functional": "#2A9D55",
    "nonlocal_measure": "#7B2CBF",
    "numerical": "#ED8B2D",
}


def _route_color(route: str) -> str:
    return ROUTE_COLORS.get(route, "#666666")


def _route_linestyle(route: str) -> str:
    return {
        "pointwise_dntk": "-",
        "weak_functional": "--",
        "nonlocal_measure": ":",
        "numerical": "-.",
    }.get(route, "-")


def _plot_positivity(
    output_dir: Path,
    problem_order: Sequence[str],
    reference_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    plt = _plot_setup()
    from matplotlib.lines import Line2D

    rows_by_problem = {str(row["problem"]): row for row in reference_rows}
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), sharey=True)
    y = np.arange(len(problem_order))
    for index, problem in enumerate(problem_order):
        row = rows_by_problem[problem]
        color = _route_color(str(row["positivity_route"]))
        marker = "o" if row.get("qmc_status") == "positive" else "X"
        gap = _as_float(row.get("relative_gap"))
        lower = _as_float(row.get("relative_lower_bound"))
        if gap is not None and gap > 0:
            axes[0].scatter(gap, index, color=color, marker=marker, s=46, zorder=3)
        else:
            axes[0].scatter(1e-16, index, facecolors="none", edgecolors=color, marker="o", s=46)
        if lower is not None:
            axes[1].scatter(lower, index, color=color, marker=marker, s=46, zorder=3)
            if gap is not None:
                axes[1].plot([lower, gap], [index, index], color=color, alpha=0.45, linewidth=1.2)

    axes[0].set_xscale("log")
    axes[0].set_xlabel(
        r"estimated relative gap $\widehat{\lambda}_{\min}/\widehat{\lambda}_{\max}$"
    )
    axes[0].set_title("(a) Limiting-NTK eigengap estimate")
    axes[0].set_yticks(y, [DISPLAY_NAMES.get(name, name) for name in problem_order])
    axes[0].invert_yaxis()
    finite_bounds = [
        abs(float(row["relative_lower_bound"]))
        for row in reference_rows
        if _as_float(row.get("relative_lower_bound")) not in {None, 0.0}
    ]
    # Size the linear neighborhood from the displayed range rather than the
    # smallest bound.  Otherwise one unusually precise near-zero estimate
    # creates a dense stack of unreadable symlog tick labels around zero.
    linear_threshold = max(max(finite_bounds, default=1e-6) * 1e-2, 1e-8)
    axes[1].set_xscale("symlog", linthresh=linear_threshold)
    if finite_bounds:
        outer_exponent = math.floor(math.log10(max(finite_bounds)))
        inner_exponent = math.ceil(math.log10(linear_threshold))
        displayed_exponents = list(
            range(outer_exponent, inner_exponent - 1, -1)
        )
        positive_ticks = [10.0**exponent for exponent in displayed_exponents]
        axes[1].set_xticks(
            sorted([-tick for tick in positive_ticks] + [0.0] + positive_ticks)
        )
    axes[1].axvline(0.0, color="0.25", linewidth=1.0)
    axes[1].set_xlabel(
        r"conservative relative lower bound $\lambda_{\mathrm{LB}}/\widehat{\lambda}_{\max}$"
    )
    axes[1].set_title("(b) QMC uncertainty-aware positivity gate")
    route_labels = {
        "pointwise_dntk": "pointwise DNTK theorem",
        "weak_functional": "weak-functional theorem",
        "nonlocal_measure": "signed-measure theorem",
        "numerical": "numerical precheck route",
    }
    present_routes = list(
        dict.fromkeys(str(row["positivity_route"]) for row in reference_rows)
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=_route_color(route),
            marker="o",
            linestyle="none",
            label=route_labels.get(route, route),
        )
        for route in present_routes
    ] + [
        Line2D([0], [0], color="0.25", marker="o", linestyle="none", label="QMC gate passed"),
        Line2D([0], [0], color="0.25", marker="X", linestyle="none", label="QMC gate inconclusive"),
    ]
    axes[1].legend(handles=handles, loc="best", frameon=False)
    figure.suptitle(
        "Positivity audit of the limiting neural tangent kernel",
        fontsize=12,
        y=1.01,
    )
    figure.subplots_adjust(wspace=0.15)
    paths = _save_figure(figure, output_dir / "figures" / "positivity_overview")
    plt.close(figure)
    return paths


def _curve_summary(
    rows: Sequence[Mapping[str, Any]], field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    values_by_step: dict[int, list[float]] = {}
    for row in rows:
        step = _as_int(row.get("step"))
        value = _as_float(row.get(field))
        if step is not None and value is not None and value >= 0:
            values_by_step.setdefault(step, []).append(value)
    if not values_by_step:
        return None
    steps = np.asarray(sorted(values_by_step), dtype=float)
    groups = [np.asarray(values_by_step[int(step)], dtype=float) for step in steps]
    median = np.asarray([np.median(group) for group in groups])
    lower = np.asarray([np.quantile(group, 0.25) for group in groups])
    upper = np.asarray([np.quantile(group, 0.75) for group in groups])
    return steps, median, lower, upper


def _plot_training(
    output_dir: Path,
    problem_order: Sequence[str],
    run_rows: Sequence[Mapping[str, Any]],
    histories: Sequence[Mapping[str, Any]],
) -> list[str]:
    plt = _plot_setup()
    column_count = 4 if len(problem_order) > 10 else 5
    row_count = int(math.ceil(len(problem_order) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.0 * column_count, 3.25 * row_count),
        sharex=False,
        sharey=True,
        squeeze=False,
    )
    curve_styles = (
        ("normalized_loss", "nonlinear GD", "#151515", "-"),
        ("normalized_frozen_kernel_loss", r"frozen $K_0$", "#377EB8", "--"),
        ("normalized_reference_kernel_loss", r"limiting $K^\infty$", "#2A9D55", "-."),
        ("normalized_theorem_envelope", "theorem envelope", "#D1495B", ":"),
    )
    legend_by_label: dict[str, Any] = {}
    for panel, problem in enumerate(problem_order):
        axis = axes.flat[panel]
        completed = [
            row
            for row in run_rows
            if row["problem"] == problem and row["status"] in COMPLETE_STATUSES
        ]
        if not completed:
            axis.text(0.5, 0.5, "No completed run", ha="center", va="center", transform=axis.transAxes, color="0.4")
            axis.set_title(DISPLAY_NAMES.get(problem, problem))
            continue
        width = max(int(row["width"]) for row in completed)
        selected = [
            row
            for row in histories
            if row["problem"] == problem and int(row["width"]) == width
        ]
        seeds = {int(row["seed"]) for row in selected}
        for field, label, color, linestyle in curve_styles:
            summary = _curve_summary(selected, field)
            if summary is None:
                continue
            steps, median, lower, upper = summary
            floor = 1e-16
            (handle,) = axis.plot(
                steps,
                np.clip(median, floor, None),
                color=color,
                linestyle=linestyle,
                label=label,
            )
            if field == "normalized_loss" and len(seeds) > 1:
                axis.fill_between(
                    steps,
                    np.clip(lower, floor, None),
                    np.clip(upper, floor, None),
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
            legend_by_label.setdefault(label, handle)
        axis.set_yscale("log")
        axis.set_title(f"{DISPLAY_NAMES.get(problem, problem)}\n$m={width}$, {len(seeds)} seed(s)")
        axis.set_xlabel("GD step")
        if panel % column_count == 0:
            axis.set_ylabel(r"normalized constraint loss $\mathcal{L}_k/\mathcal{L}_0$")
        axis.grid(True, which="both", alpha=0.22)
    for panel in range(len(problem_order), row_count * column_count):
        axes.flat[panel].set_visible(False)
    if legend_by_label:
        figure.legend(
            list(legend_by_label.values()),
            list(legend_by_label),
            loc="lower center",
            ncol=len(legend_by_label),
            frameon=False,
            bbox_to_anchor=(0.5, -0.015),
        )
    figure.suptitle(
        "Full-batch GD versus frozen finite-width and limiting-kernel dynamics",
        fontsize=12,
        y=1.01,
    )
    figure.subplots_adjust(hspace=0.38, wspace=0.23, bottom=0.12)
    paths = _save_figure(figure, output_dir / "figures" / "training_dynamics")
    plt.close(figure)
    return paths


def _group_width_metric(
    run_rows: Sequence[Mapping[str, Any]],
    problem: str,
    field: str,
    *,
    probability: bool = False,
) -> list[tuple[int, float, float, float, int]]:
    groups: dict[int, list[float]] = {}
    for row in run_rows:
        if row["problem"] != problem or row["status"] not in COMPLETE_STATUSES:
            continue
        value = _as_float(row.get(field))
        if value is not None:
            groups.setdefault(int(row["width"]), []).append(value)
    result = []
    for width, values in sorted(groups.items()):
        array = np.asarray(values, dtype=float)
        if probability:
            center = float(np.mean(array))
            # Wilson score interval is informative even for all-pass/all-fail
            # outcomes and behaves better than a normal interval for few seeds.
            z = 1.959963984540054
            count = len(values)
            denominator = 1.0 + z**2 / count
            wilson_center = (center + z**2 / (2.0 * count)) / denominator
            half_width = (
                z
                * math.sqrt(center * (1.0 - center) / count + z**2 / (4.0 * count**2))
                / denominator
            )
            lower_value = max(0.0, wilson_center - half_width)
            upper_value = min(1.0, wilson_center + half_width)
            # Roundoff can put the nominal Wilson endpoint a few ulps on the
            # wrong side of the observed proportion (notably for zero
            # successes), which Matplotlib correctly rejects as a negative
            # error-bar length.
            lower_value = min(lower_value, center)
            upper_value = max(upper_value, center)
        else:
            center = float(np.median(array))
            lower_value = float(np.quantile(array, 0.25))
            upper_value = float(np.quantile(array, 0.75))
        result.append(
            (
                width,
                center,
                lower_value,
                upper_value,
                len(values),
            )
        )
    return result


def _plot_width_metric(
    axis: Any,
    run_rows: Sequence[Mapping[str, Any]],
    problem_order: Sequence[str],
    field: str,
    colors: Mapping[str, Any],
    *,
    logarithmic_y: bool,
    probability: bool = False,
) -> None:
    for problem in problem_order:
        points = _group_width_metric(
            run_rows, problem, field, probability=probability
        )
        if not points:
            continue
        x = np.asarray([point[0] for point in points], dtype=float)
        median = np.asarray([point[1] for point in points], dtype=float)
        lower = np.asarray([point[2] for point in points], dtype=float)
        upper = np.asarray([point[3] for point in points], dtype=float)
        if logarithmic_y:
            floor = 1e-16
            median = np.clip(median, floor, None)
            lower = np.clip(lower, floor, None)
            upper = np.clip(upper, floor, None)
        route = _problem_metadata(problem)["positivity_route"]
        axis.errorbar(
            x,
            median,
            yerr=np.vstack(
                (
                    np.maximum(median - lower, 0.0),
                    np.maximum(upper - median, 0.0),
                )
            ),
            color=colors[problem],
            marker="o",
            markersize=4,
            capsize=2,
            linestyle=_route_linestyle(str(route)),
            label=DISPLAY_NAMES.get(problem, problem),
        )
    axis.set_xscale("log", base=2)
    if logarithmic_y:
        axis.set_yscale("log")
    axis.set_xlabel("width $m$")


def _plot_finite_width(
    output_dir: Path,
    problem_order: Sequence[str],
    run_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    plt = _plot_setup()
    usable = [row for row in run_rows if row["status"] in COMPLETE_STATUSES]
    concentration_field = (
        "initial_reference_error_operator"
        if any(row.get("initial_reference_error_operator") is not None for row in usable)
        else "initial_reference_error_frobenius"
    )
    drift_field = (
        "max_kernel_drift_operator"
        if any(row.get("max_kernel_drift_operator") is not None for row in usable)
        else "max_kernel_drift_frobenius"
    )
    cmap = plt.get_cmap("tab10")
    colors = {problem: cmap(index % 10) for index, problem in enumerate(problem_order)}
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))

    _plot_width_metric(
        axes[0, 0], run_rows, problem_order, concentration_field, colors, logarithmic_y=True
    )
    concentration_norm = "operator" if concentration_field.endswith("operator") else "Frobenius"
    axes[0, 0].set_ylabel(
        rf"relative $K_0$--$K^\infty$ error ({concentration_norm} norm)"
    )
    axes[0, 0].set_title("(a) Finite-width kernel concentration")

    # Boolean means are empirical probabilities.  Reuse the generic summary by
    # materializing 0/1 values under a private plotting field.
    probability_rows = [dict(row) for row in run_rows]
    for row in probability_rows:
        passed = _as_bool(row.get("initial_gap_criterion_passed"))
        row["_gap_probability"] = None if passed is None else float(passed)
    _plot_width_metric(
        axes[0, 1],
        probability_rows,
        problem_order,
        "_gap_probability",
        colors,
        logarithmic_y=False,
        probability=True,
    )
    axes[0, 1].axhline(1.0, color="0.3", linestyle=":", linewidth=1.0)
    axes[0, 1].set_ylim(-0.04, 1.04)
    axes[0, 1].set_ylabel("fraction of seeds passing initialization gap criterion")
    axes[0, 1].set_title(r"(b) Initialization event ($K_0\succeq0.75\widehat K^\infty$)")

    _plot_width_metric(
        axes[1, 0], run_rows, problem_order, drift_field, colors, logarithmic_y=True
    )
    drift_norm = "operator" if drift_field.endswith("operator") else "Frobenius"
    axes[1, 0].set_ylabel(rf"max training kernel drift ({drift_norm} norm, relative)")
    axes[1, 0].set_title("(c) Kernel stability along the GD trajectory")

    _plot_width_metric(
        axes[1, 1],
        run_rows,
        problem_order,
        "max_sqrt_width_per_neuron_drift",
        colors,
        logarithmic_y=True,
    )
    axes[1, 1].set_ylabel(r"$\sqrt{m}\,\max_{j,k}\|\theta_{j,k}-\theta_{j,0}\|_2$")
    axes[1, 1].set_title("(d) Theorem-scaled maximum neuron movement")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
    figure.suptitle(
        "Finite-width concentration, spectral-gap preservation, and lazy-training diagnostics",
        fontsize=12,
        y=1.01,
    )
    figure.subplots_adjust(hspace=0.34, wspace=0.27, bottom=0.17)
    paths = _save_figure(figure, output_dir / "figures" / "finite_width_stability")
    plt.close(figure)
    return paths, {
        "concentration_metric": concentration_field,
        "kernel_drift_metric": drift_field,
    }


def _write_captions(
    path: Path,
    run_rows: Sequence[Mapping[str, Any]],
    metric_choices: Mapping[str, str],
) -> None:
    completed = sum(row["status"] in COMPLETE_STATUSES for row in run_rows)
    skipped = sum(row["status"] in SKIPPED_STATUSES for row in run_rows)
    missing = sum(row["status"] == "missing" for row in run_rows)
    text = f"""# Theorem-audit figure captions

**Positivity overview.** Estimated limiting-NTK relative eigengap (left) and the
replicated, scrambled-Sobol lower bound after subtracting both the
between-scramble uncertainty term and the coarse/fine operator-norm difference
(right). Color distinguishes the pointwise DNTK, weak-functional,
signed-measure nonlocal, and any numerical-precheck routes. A circle means the
conservative QMC gate passed; an X means it was inconclusive. For analytically
covered rows QMC estimates the gap size but is not the logical positivity test.
An inconclusive numerical gate is not evidence of singularity, and a numerical
pass is evidence rather than a proof.

**Training dynamics.** Constraint loss for full-batch nonlinear gradient
descent under the paper's exact shallow biased architecture, normalized by its
initial value, at the largest completed width for each PDE. Curves show the
median over seeds; the translucent band is the nonlinear-GD interquartile
range. Dashed and dash-dotted curves propagate the same initial residual with
the frozen empirical kernel and independently estimated limiting kernel. The
dotted curve is the theorem's conservative geometric envelope
`L_0 (1 - eta lambda_LB)^k`, shown only when the uncertainty-adjusted QMC
lower quantity `lambda_LB` is positive. Values below `1e-16` are clipped only for
logarithmic display.

**Finite-width stability.** Medians with seedwise interquartile bars as width
increases. Panel (a) uses `{metric_choices['concentration_metric']}`; panel (b)
is the empirical fraction satisfying the recorded three-quarter initialization
gap event (Wilson 95% interval; seeds without a generalized Loewner diagnostic
are omitted from that panel, and scalar eigenvalue ratios are stored separately);
panel (c) uses
`{metric_choices['kernel_drift_metric']}` and takes the maximum over recorded GD
checkpoints; panel (d) shows the theorem-scaled maximum per-neuron displacement.
Line style distinguishes the pointwise, weak-functional, signed-measure, and
numerical-precheck routes.

Aggregation coverage: {completed} completed, {skipped} positivity-gated or
explicitly skipped, and {missing} missing expected training artifacts. See
`runs.csv` and `aggregate.json` for every expected job and exclusion reason.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _resolve_manifest(config: Mapping[str, Any], output_dir: Path) -> tuple[str, dict[str, Any] | None]:
    computed_hash = canonical_config_hash(config)
    candidates = (output_dir / "manifest.json", output_dir / "run_manifest.json")
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise AggregateError(f"multiple manifests found: {existing}")
    if not existing:
        return computed_hash, None
    manifest = _load_json(existing[0])
    manifest_hash = _artifact_hash(manifest)
    if manifest_hash is None:
        raise AggregateError(f"manifest {existing[0]} has no config_hash")
    snapshot = manifest.get("config")
    if snapshot is not None:
        if not isinstance(snapshot, Mapping) or canonical_config_hash(snapshot) != computed_hash:
            raise AggregateError("manifest configuration snapshot differs from the active config")
    elif manifest_hash != computed_hash:
        raise AggregateError(
            "manifest hash differs from the active config and has no verifiable config snapshot"
        )
    if manifest_hash != computed_hash:
        # Permit a runner-specific hash encoding only when the complete snapshot
        # above proves semantic identity; all artifacts must still match it.
        return manifest_hash, manifest
    return computed_hash, manifest


def aggregate_results(config_path: str | Path) -> dict[str, Any]:
    """Aggregate exactly one configured sweep and return the written summary."""

    from .run import _source_fingerprint

    expected_source_hash = str(_source_fingerprint()["sha256"])
    config_path = Path(config_path).resolve()
    config = _load_json(config_path)
    output_setting = config.get("output_dir")
    if not isinstance(output_setting, str) or not output_setting:
        raise AggregateError("configuration must define a non-empty output_dir")
    output_dir = Path(output_setting)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    problems = config.get("problems")
    training = config.get("training")
    if not isinstance(problems, list) or not all(isinstance(item, str) for item in problems):
        raise AggregateError("configuration problems must be a list of names")
    if not problems:
        raise AggregateError("configuration problems must be non-empty")
    if len(set(problems)) != len(problems):
        raise AggregateError("configuration contains duplicate problem names")
    if not isinstance(training, Mapping):
        raise AggregateError("configuration has no training object")
    widths = training.get("widths")
    seeds = training.get("seeds")
    if not isinstance(widths, list) or not widths or not all(type(value) is int and value > 0 for value in widths):
        raise AggregateError("training.widths must contain positive integers")
    if not isinstance(seeds, list) or not seeds or not all(type(value) is int for value in seeds):
        raise AggregateError("training.seeds must contain integers")
    if len(set(widths)) != len(widths) or len(set(seeds)) != len(seeds):
        raise AggregateError("training widths and seeds must be unique")
    expected_steps = training.get("steps")
    if type(expected_steps) is not int or expected_steps < 0:
        raise AggregateError("training.steps must be a non-negative integer")

    expected_hash, manifest = _resolve_manifest(config, output_dir)
    reference_rows: list[dict[str, Any]] = []
    reference_data: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    used_paths: set[Path] = set()
    for problem in problems:
        path, locate_error = _locate_unique(_reference_candidates(output_dir, problem))
        if path is not None:
            used_paths.add(path.resolve())
        row, data, issue = _reference_row(
            problem, path, expected_hash, locate_error,
            expected_source_hash=expected_source_hash,
        )
        reference_rows.append(row)
        if data is not None:
            reference_data[problem] = data
        if issue is not None:
            issues.append(issue)

    reference_by_problem = {str(row["problem"]): row for row in reference_rows}
    run_rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for problem in problems:
        for width in widths:
            for seed in seeds:
                path, locate_error = _locate_unique(
                    _run_candidates(output_dir, problem, width, seed)
                )
                if path is not None:
                    used_paths.add(path.resolve())
                row, run_history, issue = _run_row(
                    problem,
                    width,
                    seed,
                    path,
                    expected_hash,
                    locate_error,
                    reference_by_problem[problem],
                    expected_steps=expected_steps,
                    expected_source_hash=expected_source_hash,
                )
                run_rows.append(row)
                histories.extend(run_history)
                if issue is not None:
                    issues.append(issue)

    unexpected_paths: list[str] = []
    for subdirectory in (output_dir / "references", output_dir / "runs"):
        if subdirectory.is_dir():
            for candidate in subdirectory.rglob("*.json"):
                if candidate.resolve() not in used_paths:
                    unexpected_paths.append(
                        str(candidate.relative_to(REPO_ROOT))
                        if candidate.is_relative_to(REPO_ROOT)
                        else str(candidate)
                    )

    table_dir = output_dir / "tables"
    positivity_csv = table_dir / "positivity.csv"
    runs_csv = table_dir / "runs.csv"
    histories_csv = table_dir / "histories.csv"
    _write_csv(positivity_csv, reference_rows, ("problem", "artifact_status", "verdict"))
    _write_csv(runs_csv, run_rows, ("problem", "width", "seed", "status"))
    _write_csv(histories_csv, histories, ("problem", "width", "seed", "step"))

    figure_paths: dict[str, list[str]] = {}
    figure_paths["positivity"] = _plot_positivity(output_dir, problems, reference_rows)
    figure_paths["training"] = _plot_training(output_dir, problems, run_rows, histories)
    finite_paths, metric_choices = _plot_finite_width(output_dir, problems, run_rows)
    figure_paths["finite_width"] = finite_paths
    captions_path = output_dir / "figures" / "figure_captions.md"
    _write_captions(captions_path, run_rows, metric_choices)

    status_counts: dict[str, int] = {}
    for row in run_rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    aggregate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_name": config.get("name"),
        "config_path": str(config_path.relative_to(REPO_ROOT)) if config_path.is_relative_to(REPO_ROOT) else str(config_path),
        "config_hash": expected_hash,
        "source_hash": expected_source_hash,
        "aggregation_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numerical_project_root": str(REPO_ROOT.resolve()),
            "note": "Numerical artifacts retain their source hash; aggregation code may be a separately recorded rendering overlay.",
        },
        "manifest_present": manifest is not None,
        "coverage": {
            "expected_references": len(problems),
            "accepted_reference_artifacts": sum(
                row["artifact_status"] in COMPLETE_STATUSES for row in reference_rows
            ),
            "expected_training_runs": len(problems) * len(widths) * len(seeds),
            "training_status_counts": status_counts,
            "history_rows": len(histories),
        },
        "metric_choices": metric_choices,
        "references": reference_rows,
        "runs": run_rows,
        "issues": issues,
        "unexpected_json_artifacts_not_aggregated": sorted(unexpected_paths),
        "tables": {
            "positivity": str(positivity_csv),
            "runs": str(runs_csv),
            "histories": str(histories_csv),
        },
        "figures": figure_paths,
        "figure_captions": str(captions_path),
        "interpretation_guardrails": [
            "A numerical positivity pass is finite-precision evidence, not a proof.",
            "An inconclusive numerical gate is not evidence that the limiting NTK is singular.",
            "The configured spectral learning rate does not itself verify the theorem's unknown global smoothness constant.",
            "Solution error is secondary; the theorem controls the finite constraint loss.",
        ],
    }
    aggregate_path = output_dir / "aggregate.json"
    _write_json(aggregate_path, aggregate)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON sweep configuration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    aggregate = aggregate_results(args.config)
    coverage = aggregate["coverage"]
    print(
        "Aggregated "
        f"{coverage['accepted_reference_artifacts']}/{coverage['expected_references']} references and "
        f"{coverage['training_status_counts'].get('completed', 0) + coverage['training_status_counts'].get('complete', 0)}/{coverage['expected_training_runs']} completed training runs."
    )
    print(f"Output: {Path(args.config).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AggregateError", "aggregate_results", "canonical_config_hash"]
