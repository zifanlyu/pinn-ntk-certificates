"""Numerical check of the nonlinear-row obstructions of ``sec:nonlinear_and_depth``.

Three predictions are measured on the two-layer ``tanh`` network of
``def:ntk_parameterization`` with scalar input and output and per-neuron
parameters i.i.d. uniform on a compact box, as ``assump:main`` requires.

1.  ``cor:nonlinear_diagonal_dichotomy``.  For a row linear in the network
    output the tangent Gram entry concentrates, so its variance over
    initializations decays as ``1 / d_1``.  For a product of two such rows the
    entry has a non-degenerate weak limit and its variance is ``Theta(1)``.

2.  ``cor:no_deterministic_polynomial_ntk``.  Under conditional output-weight
    centering, at the degree-matched exponent ``rho_p = (2p-1)/(2p)`` the
    degree-``(p, p)`` block vanishes at rate ``d_1^{-(p-1)}``.

3.  ``prop:uncentered_branch`` and ``rem:uncentered_reachability``.  Without
    centering the prediction grows as ``d_1^{1/(2p)}``, but only when the
    feature mean is itself nonzero.  With an odd activation and a hidden law
    symmetric about the origin, ``E[h] = E[a] E[f]`` vanishes through
    ``E[f] = 0`` however the output weight is distributed, and the prediction
    instead decays as ``d_1^{-1/(2p)}``.  Both branches are measured.

All settings come from the configuration file.  Run::

    uv run python experiments/theorem_audit/nonlinear_kernel_variance.py \\
        --config experiments/theorem_audit/configs/nonlinear_kernel_variance_smoke.json
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
        "collocation_points",
        "widths",
        "seeds",
        "polynomial_degree",
        "hidden_parameter_low",
        "hidden_parameter_high",
        "uncentered_output_weight_mean",
        "shifted_bias_low",
        "shifted_bias_high",
    ):
        if key not in config:
            raise AuditError(f"configuration is missing required field {key!r}")
    if len(config["collocation_points"]) != 2:
        raise AuditError("collocation_points must contain exactly two entries")
    if not config["widths"] or not config["seeds"]:
        raise AuditError("widths and seeds must both be non-empty")
    if int(config["polynomial_degree"]) < 2:
        raise AuditError("polynomial_degree must be at least two")
    return config


def draw(rng: np.random.Generator, config: Mapping[str, Any], width: int):
    """One initialization: hidden weights, biases, and centered output weights."""
    low = DTYPE(config["hidden_parameter_low"])
    high = DTYPE(config["hidden_parameter_high"])
    w = rng.uniform(low, high, size=width).astype(DTYPE)
    b = rng.uniform(low, high, size=width).astype(DTYPE)
    a = rng.uniform(low, high, size=width).astype(DTYPE)
    return w, b, a


def row_quantities(w, b, a, config: Mapping[str, Any]):
    """Per-neuron features and gradients for the two base rows."""
    gamma = DTYPE(config["gamma"])
    x1, x2 = (DTYPE(v) for v in config["collocation_points"])
    t1 = np.tanh(w * x1 + gamma * b)
    t2 = np.tanh(w * x2 + gamma * b)
    d1_ = DTYPE(1.0) - t1 * t1
    d2_ = DTYPE(1.0) - t2 * t2
    g1 = np.stack([a * d1_ * x1, a * d1_ * gamma, t1])
    g2 = np.stack([a * d2_ * x2, a * d2_ * gamma, t2])
    return t1, t2, g1, g2


def measure(config: Mapping[str, Any]) -> dict[str, Any]:
    widths = [int(v) for v in config["widths"]]
    seeds = [int(v) for v in config["seeds"]]
    degree = int(config["polynomial_degree"])
    rho = DTYPE(2 * degree - 1) / DTYPE(2 * degree)
    gamma = DTYPE(config["gamma"])
    x1 = DTYPE(config["collocation_points"][0])
    mean_shift = DTYPE(config["uncentered_output_weight_mean"])
    bias_low = DTYPE(config["shifted_bias_low"])
    bias_high = DTYPE(config["shifted_bias_high"])

    rows: list[dict[str, Any]] = []
    for width in widths:
        linear = np.empty(len(seeds), dtype=DTYPE)
        product = np.empty(len(seeds), dtype=DTYPE)
        block = np.empty(len(seeds), dtype=DTYPE)
        symmetric = np.empty(len(seeds), dtype=DTYPE)
        shifted = np.empty(len(seeds), dtype=DTYPE)

        scale_half = DTYPE(width) ** DTYPE(-0.5)
        scale_rho = DTYPE(width) ** (-rho)
        prefactor = DTYPE(width) ** (DTYPE(1.0) - DTYPE(2.0) * rho)

        for index, seed in enumerate(seeds):
            rng = np.random.default_rng([seed, width])
            w, b, a = draw(rng, config, width)
            t1, t2, g1, g2 = row_quantities(w, b, a, config)

            gram11 = float((g1 * g1).sum()) / width
            gram12 = float((g1 * g2).sum()) / width
            gram22 = float((g2 * g2).sum()) / width

            amp1_half = float((a * t1).sum()) * scale_half
            amp2_half = float((a * t2).sum()) * scale_half
            linear[index] = gram11
            product[index] = (
                amp2_half * amp2_half * gram11
                + 2.0 * amp1_half * amp2_half * gram12
                + amp1_half * amp1_half * gram22
            )

            amp1_rho = float((a * t1).sum()) * scale_rho
            amp2_rho = float((a * t2).sum()) * scale_rho
            block[index] = abs(
                prefactor
                * (
                    amp2_rho * amp2_rho * gram11
                    + 2.0 * amp1_rho * amp2_rho * gram12
                    + amp1_rho * amp1_rho * gram22
                )
            )

            # Uncentered output weight against a symmetric and a shifted hidden
            # bias law.  Only the shifted law gives a nonzero feature mean.
            rng_unc = np.random.default_rng([seed, width, 1])
            w_u = rng_unc.uniform(
                DTYPE(config["hidden_parameter_low"]),
                DTYPE(config["hidden_parameter_high"]),
                size=width,
            ).astype(DTYPE)
            b_sym = rng_unc.uniform(
                DTYPE(config["hidden_parameter_low"]),
                DTYPE(config["hidden_parameter_high"]),
                size=width,
            ).astype(DTYPE)
            b_shift = rng_unc.uniform(bias_low, bias_high, size=width).astype(DTYPE)
            a_unc = (
                rng_unc.uniform(
                    DTYPE(config["hidden_parameter_low"]),
                    DTYPE(config["hidden_parameter_high"]),
                    size=width,
                ).astype(DTYPE)
                + mean_shift
            )
            symmetric[index] = abs(
                float((a_unc * np.tanh(w_u * x1 + gamma * b_sym)).sum()) * scale_rho
            )
            shifted[index] = abs(
                float((a_unc * np.tanh(w_u * x1 + gamma * b_shift)).sum()) * scale_rho
            )

        growth = DTYPE(width) ** (DTYPE(1.0) / DTYPE(2 * degree))
        rows.append(
            {
                "width": width,
                "variance_linear_row": float(linear.var(ddof=1)),
                "width_times_variance_linear_row": float(width * linear.var(ddof=1)),
                "variance_product_row": float(product.var(ddof=1)),
                "mean_product_row": float(product.mean()),
                "mean_degree_matched_block": float(block.mean()),
                "width_times_degree_matched_block": float(width * block.mean()),
                "mean_uncentered_symmetric_prediction": float(symmetric.mean()),
                "mean_uncentered_shifted_prediction": float(shifted.mean()),
                "shifted_prediction_over_width_power": float(shifted.mean() / growth),
            }
        )
    return {"per_width": rows}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    threads = str(int(config.get("threads", 1)))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, threads)

    started = time.perf_counter()
    measurements = measure(config)
    elapsed = time.perf_counter() - started

    from .run import _git_metadata  # noqa: PLC0415 - provenance only

    import matplotlib
    import torch

    record = {
        "schema_version": SCHEMA_VERSION,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "config_path": str(arguments.config),
        "config_sha256": canonical_config_hash(config),
        "seed_count": len(config["seeds"]),
        "degree_matched_exponent": (2 * int(config["polynomial_degree"]) - 1)
        / (2 * int(config["polynomial_degree"])),
        "elapsed_seconds": elapsed,
        "environment": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
            "git": _git_metadata(),
        },
        **measurements,
    }

    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "nonlinear_kernel_variance_summary.json"
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=False)
        handle.write("\n")

    header = (
        f"{'d_1':>7} {'Var(K_lin)':>13} {'d1*Var(lin)':>13} {'Var(K_prod)':>13} "
        f"{'d1*|block|':>13} {'|u| shifted':>13} {'/d1^(1/2p)':>13}"
    )
    print(header)
    for row in measurements["per_width"]:
        print(
            f"{row['width']:>7} {row['variance_linear_row']:>13.6e} "
            f"{row['width_times_variance_linear_row']:>13.6e} "
            f"{row['variance_product_row']:>13.6e} "
            f"{row['width_times_degree_matched_block']:>13.6e} "
            f"{row['mean_uncentered_shifted_prediction']:>13.6e} "
            f"{row['shifted_prediction_over_width_power']:>13.6e}"
        )
    print(f"\nwrote {destination} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
