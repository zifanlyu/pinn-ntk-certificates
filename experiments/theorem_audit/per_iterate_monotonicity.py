"""Per-iterate monotonicity audit for ``thm:linearPDEsconvergence``.

The main campaigns store the loss only at saved checkpoints, so they cannot
speak to the theorem's claim of a decrease at *every* iterate.  This script
records the loss at every step on a reduced two-row design and reports, per
cell, how many steps increased the loss.

Design.  Two-layer ``tanh`` network of ``def:ntk_parameterization``, all
parameters trained, per-neuron parameters i.i.d. uniform on a compact box.
The rows are the value row ``u(x_star)`` and the mixed row
``(u + eps u') / sqrt(1 + eps^2)``, so ``eps`` is the row-mixing parameter of
the control study.  Full-batch gradient descent on the unscaled loss of
``def:pinn_loss`` at ``eta = learning_rate_factor / lambda_max(K_0)`` against a
contrast target.

Scoring.  An iterate is scored while ``L_k > precision_floor_relative * L_0``.
Past that point the residual is a difference of two quantities agreeing to
machine precision, the per-step ratios quantize to small integers, and nothing
is being measured.

All settings come from the configuration file.  Run::

    uv run python experiments/theorem_audit/per_iterate_monotonicity.py \\
        --config experiments/theorem_audit/configs/per_iterate_monotonicity_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .run import AuditError, PROJECT_ROOT, canonical_config_hash

SCHEMA_VERSION = 1
DTYPE = np.float64


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise AuditError("configuration must be a JSON object")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise AuditError(f"unsupported schema_version: {config.get('schema_version')!r}")
    for key in (
        "output_dir",
        "gamma",
        "collocation_point",
        "widths",
        "epsilons",
        "seeds",
        "steps",
        "learning_rate_factor",
        "target",
        "hidden_parameter_low",
        "hidden_parameter_high",
        "precision_floor_relative",
        "increase_tolerance_relative",
    ):
        if key not in config:
            raise AuditError(f"configuration is missing required field {key!r}")
    if len(config["target"]) != 2:
        raise AuditError("target must contain exactly two entries")
    if int(config["steps"]) < 1:
        raise AuditError("steps must be positive")
    return config


def rows_and_gradients(w, b, a, mixing: float, width: int, config: Mapping[str, Any]):
    """Predictions and per-neuron gradients of the two rows."""
    gamma = DTYPE(config["gamma"])
    x = DTYPE(config["collocation_point"])
    scale = DTYPE(width) ** DTYPE(-0.5)
    mix = DTYPE(1.0) / np.sqrt(DTYPE(1.0) + DTYPE(mixing) * DTYPE(mixing))

    t = np.tanh(w * x + gamma * b)
    dt = DTYPE(1.0) - t * t
    ddt = DTYPE(-2.0) * t * dt

    prediction = np.array(
        [
            scale * float((a * t).sum()),
            scale * mix * float((a * (t + mixing * w * dt)).sum()),
        ],
        dtype=DTYPE,
    )
    grad_value = np.stack([a * dt * x, a * dt * gamma, t])
    grad_mixed = mix * np.stack(
        [
            a * (dt * x + mixing * (dt + w * ddt * x)),
            a * (dt * gamma + mixing * w * ddt * gamma),
            t + mixing * w * dt,
        ]
    )
    return prediction, grad_value, grad_mixed


def run_cell(width: int, mixing: float, config: Mapping[str, Any]) -> dict[str, Any]:
    steps = int(config["steps"])
    target = np.array(config["target"], dtype=DTYPE)
    floor_relative = DTYPE(config["precision_floor_relative"])
    tolerance = DTYPE(config["increase_tolerance_relative"])
    factor = DTYPE(config["learning_rate_factor"])
    low = DTYPE(config["hidden_parameter_low"])
    high = DTYPE(config["hidden_parameter_high"])
    scale = DTYPE(width) ** DTYPE(-0.5)

    violations = 0
    scored = 0
    worst_first_seed: float | None = None
    predicted_first_seed: float | None = None
    truncated = False

    for position, seed in enumerate(int(v) for v in config["seeds"]):
        rng = np.random.default_rng([seed, width, int(round(mixing * 1000))])
        w = rng.uniform(low, high, size=width).astype(DTYPE)
        b = rng.uniform(low, high, size=width).astype(DTYPE)
        a = rng.uniform(low, high, size=width).astype(DTYPE)

        prediction, grad_value, grad_mixed = rows_and_gradients(w, b, a, mixing, width, config)
        residual = target - prediction
        loss = float(residual @ residual)
        floor = float(floor_relative) * loss

        gram = (
            np.array(
                [
                    [float((grad_value * grad_value).sum()), float((grad_value * grad_mixed).sum())],
                    [float((grad_value * grad_mixed).sum()), float((grad_mixed * grad_mixed).sum())],
                ],
                dtype=DTYPE,
            )
            / width
        )
        eigenvalues = np.linalg.eigvalsh(gram)
        eta = float(factor) / float(eigenvalues[-1])
        predicted = (1.0 - 2.0 * eta * float(eigenvalues[0])) ** 2
        if position == 0:
            predicted_first_seed = predicted

        previous = loss
        cell_worst = 0.0
        for _ in range(steps):
            step = 2.0 * eta * float(scale)
            w = w + step * (residual[0] * grad_value[0] + residual[1] * grad_mixed[0])
            b = b + step * (residual[0] * grad_value[1] + residual[1] * grad_mixed[1])
            a = a + step * (residual[0] * grad_value[2] + residual[1] * grad_mixed[2])

            prediction, grad_value, grad_mixed = rows_and_gradients(
                w, b, a, mixing, width, config
            )
            residual = target - prediction
            loss = float(residual @ residual)

            if previous > floor:
                scored += 1
                if loss > previous * (1.0 + float(tolerance)):
                    violations += 1
                cell_worst = max(cell_worst, loss / previous)
            else:
                truncated = True
            previous = loss

        if position == 0 and cell_worst > 0.0:
            worst_first_seed = cell_worst

    if truncated:
        # The loss reached the precision floor inside the horizon, so the
        # worst-ratio figure would be dominated by cancellation artifacts.
        worst_first_seed = None
        predicted_first_seed = None

    return {
        "width": width,
        "epsilon": mixing,
        "violations": violations,
        "scored_iterates": scored,
        "possible_iterates": steps * len(config["seeds"]),
        "reached_precision_floor": truncated,
        "worst_ratio_first_seed": worst_first_seed,
        "predicted_ratio_first_seed": predicted_first_seed,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    threads = str(int(config.get("threads", 1)))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, threads)

    started = time.perf_counter()
    cells = [
        run_cell(int(width), float(mixing), config)
        for width in config["widths"]
        for mixing in config["epsilons"]
    ]
    elapsed = time.perf_counter() - started

    from .run import _git_metadata  # noqa: PLC0415 - provenance only

    import matplotlib
    import torch

    total_violations = sum(cell["violations"] for cell in cells)
    total_scored = sum(cell["scored_iterates"] for cell in cells)

    record = {
        "schema_version": SCHEMA_VERSION,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "config_path": str(arguments.config),
        "config_sha256": canonical_config_hash(config),
        "total_violations": total_violations,
        "total_scored_iterates": total_scored,
        "elapsed_seconds": elapsed,
        "environment": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
            "git": _git_metadata(),
        },
        "cells": cells,
    }

    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "per_iterate_monotonicity_summary.json"
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(f"{'d_1':>7} {'eps':>7} {'viol':>6} {'scored':>9} {'worst':>12} {'predicted':>12}")
    for cell in cells:
        worst = "---" if cell["worst_ratio_first_seed"] is None else f"{cell['worst_ratio_first_seed']:.6f}"
        predicted = (
            "---"
            if cell["predicted_ratio_first_seed"] is None
            else f"{cell['predicted_ratio_first_seed']:.6f}"
        )
        print(
            f"{cell['width']:>7} {cell['epsilon']:>7} {cell['violations']:>6} "
            f"{cell['scored_iterates']:>9} {worst:>12} {predicted:>12}"
        )
    print(f"\ntotal: {total_violations} violations in {total_scored} scored iterates")
    print(f"wrote {destination} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
