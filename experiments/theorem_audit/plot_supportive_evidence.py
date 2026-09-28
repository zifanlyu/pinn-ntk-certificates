"""Plot frozen aggregate evidence for finite-width GD and conditioning.

The default input contains the counts and five-seed medians displayed in the two
main-text panels.  Its source SHA256 is retained in the figure summary.  Raw
histories are never inferred from endpoints.

Run from any directory with the repository's locked Python environment::

    python experiments/theorem_audit/plot_supportive_evidence.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


EXPECTED_FAMILIES = (
    "pointwise_poisson",
    "vector_stokes",
    "weak_poisson",
    "nonlocal_diffusion",
    "weak_boundary",
    "nonlocal_boundary",
)
FAMILY_LABELS = {
    "pointwise_poisson": "Pointwise Poisson",
    "vector_stokes": "Vector Stokes",
    "weak_poisson": "Weak Poisson",
    "nonlocal_diffusion": "Nonlocal diffusion",
    "weak_boundary": "Weak + boundary",
    "nonlocal_boundary": "Nonlocal + boundary",
}
WIDTHS = (16, 256, 4096, 16384)
EPSILONS = (1.0, 0.1, 0.01)
EPSILON_MARKERS = {1.0: "o", 0.1: "s", 0.01: "^"}
FAMILY_COLORS = {
    "pointwise_poisson": "#0072B2",
    "vector_stokes": "#E69F00",
    "weak_poisson": "#009E73",
    "nonlocal_diffusion": "#CC79A7",
    "weak_boundary": "#56B4E9",
    "nonlocal_boundary": "#D55E00",
}
FAMILY_LINESTYLES = {
    "pointwise_poisson": "-",
    "vector_stokes": "--",
    "weak_poisson": "-.",
    "nonlocal_diffusion": ":",
    "weak_boundary": (0, (5, 1, 1, 1)),
    "nonlocal_boundary": (0, (3, 1, 1, 1, 1, 1)),
}
DISPLAY_FLOOR = 1.0e-16
TRAJECTORY_DISPLAY_FLOOR = 1.0e-30
LOSS_THRESHOLD = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_aggregate(path: Path) -> dict[str, Any]:
    """Load and strictly validate the tracked figure-level aggregate."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("unsupported frozen-aggregate schema")

    raw_source = record.get("raw_source", {})
    if raw_source.get("rows") != 1080:
        raise ValueError("the frozen aggregate must document 1080 source rows")
    source_hash = raw_source.get("sha256", "")
    if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
        raise ValueError("invalid raw-source SHA256 in frozen aggregate")

    panels = record.get("panels", {})
    if set(panels) != {
        "width_diagnostics",
        "compatible_singular",
        "conditioning_width_16384",
    }:
        raise ValueError("unexpected or missing figure panel in frozen aggregate")

    width_diagnostics = panels["width_diagnostics"]
    if set(width_diagnostics) != {"provenance", *(str(width) for width in WIDTHS)}:
        raise ValueError("unexpected finite-width diagnostic grid")
    provenance = width_diagnostics["provenance"]
    if not (
        provenance.get("pde_systems") == 11
        and provenance.get("runs_per_width") == 110
        and "completed_width_results" in provenance.get("description", "")
    ):
        raise ValueError("invalid finite-width diagnostic provenance")
    for width in WIDTHS:
        cell = width_diagnostics[str(width)]
        if set(cell) != {
            "median_initial_relative_kernel_error",
            "maximum_relative_kernel_drift",
            "final_frozen_trajectory_discrepancy",
            "gap_retained_at_saved_checkpoints",
            "gap_retained_denominator",
        }:
            raise ValueError(f"unexpected finite-width diagnostics at width {width}")
        for name in (
            "median_initial_relative_kernel_error",
            "maximum_relative_kernel_drift",
            "final_frozen_trajectory_discrepancy",
        ):
            value = float(cell[name])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid {name} at width {width}")
        retained = cell["gap_retained_at_saved_checkpoints"]
        denominator = cell["gap_retained_denominator"]
        if denominator != 110 or not isinstance(retained, int) or not 0 <= retained <= denominator:
            raise ValueError(f"invalid saved-checkpoint gap count at width {width}")

    singular = panels["compatible_singular"]
    if set(singular) != {str(width) for width in WIDTHS}:
        raise ValueError("unexpected compatible-singular width grid")
    for width in WIDTHS:
        cell = singular[str(width)]
        if cell.get("runs") != 30:
            raise ValueError(f"expected 30 compatible-singular runs at width {width}")
        for name in ("loss_reduced_at_least_100x", "final_loss_below_1e_minus_12"):
            count = cell.get(name)
            if not isinstance(count, int) or not 0 <= count <= 30:
                raise ValueError(f"invalid {name} count at width {width}")
        for name in ("median_final_loss", "maximum_final_loss"):
            value = float(cell.get(name, -1.0))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid {name} at width {width}")

    conditioning = panels["conditioning_width_16384"]
    if set(conditioning) != set(EXPECTED_FAMILIES):
        raise ValueError("unexpected conditioning families")
    for family in EXPECTED_FAMILIES:
        family_data = conditioning[family]
        if set(family_data) != {f"{epsilon:g}" for epsilon in EPSILONS}:
            raise ValueError(f"unexpected epsilon grid for {family}")
        for epsilon in EPSILONS:
            cell = family_data[f"{epsilon:g}"]
            if cell.get("runs") != 5:
                raise ValueError(f"expected five paired seeds for {family}, epsilon={epsilon:g}")
            gap = float(cell.get("median_relative_gap", 0.0))
            condition = float(cell.get("median_initial_condition_number", 0.0))
            ratio = float(cell.get("median_final_to_initial_loss_ratio", -1.0))
            if not (
                np.isfinite(gap)
                and np.isfinite(condition)
                and np.isfinite(ratio)
                and gap > 0.0
                and condition > 0.0
                and ratio >= 0.0
            ):
                raise ValueError(f"invalid conditioning aggregate for {family}, epsilon={epsilon:g}")
            if not np.isclose(gap * condition, 1.0, rtol=1.0e-10, atol=1.0e-12):
                raise ValueError(f"gap and condition number disagree for {family}, epsilon={epsilon:g}")

    supporting = record.get("supporting_counts", {})
    if set(supporting) != {"positive_definite", "compatible_singular"}:
        raise ValueError("unexpected supporting-count groups")
    positive = supporting["positive_definite"]
    compatible = supporting["compatible_singular"]
    if not (
        positive.get("runs") == 960
        and positive.get("loss_reduced") == 959
        and positive.get("manufactured_runs") == 480
        and positive.get("manufactured_loss_reduced_at_least_10x") == 480
        and positive.get("width_ge_256_runs") == 720
        and positive.get("width_ge_256_loss_reduced") == 720
        and positive.get("width_16384_runs") == 240
        and positive.get("width_16384_loss_reduced") == 240
    ):
        raise ValueError("invalid positive-definite supporting counts")
    if not (
        compatible.get("runs") == 120
        and compatible.get("loss_reduced_at_least_100x") == 120
        and compatible.get("final_loss_below_1e_minus_12") == 118
        and compatible.get("width_ge_256_runs") == 90
        and compatible.get("width_ge_256_final_loss_below_1e_minus_12") == 90
    ):
        raise ValueError("invalid compatible-singular supporting counts")
    return record


def _load_joint_trajectories(path: Path) -> dict[str, Any]:
    """Load and validate the tracked three-row active/nullspace trajectories."""

    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("unsupported joint-nullspace schema")
    configuration = record.get("configuration", {})
    expected_steps = configuration.get("checkpoint_steps")
    total_steps = configuration.get("steps")
    if (
        not isinstance(expected_steps, list)
        or not expected_steps
        or expected_steps[0] != 0
        or expected_steps[-1] != total_steps
        or expected_steps != sorted(set(expected_steps))
    ):
        raise ValueError("invalid joint-nullspace checkpoint schedule")

    sample_sizes = record.get("sample_sizes", {})
    trajectories = record.get("gd_pair_records", [])
    if not (
        sample_sizes.get("paired_gd_settings") == 18
        and sample_sizes.get("gd_trajectories") == 36
        and len(trajectories) == 18
    ):
        raise ValueError("expected 18 paired joint-nullspace settings")
    floor = float(record.get("summary", {}).get("incompatible_gd_predicted_floor", -1.0))
    if not np.isclose(floor, 0.25**2, rtol=0.0, atol=1.0e-15):
        raise ValueError("unexpected incompatible-target loss floor")

    required = {
        "step",
        "shared_active_subspace_loss",
        "compatible_nullspace_loss",
        "compatible_total_loss",
        "incompatible_nullspace_loss",
        "incompatible_total_loss",
        "paired_loss_offset_error",
    }
    maximum_offset_error = 0.0
    for trajectory in trajectories:
        history = trajectory.get("paired_history", [])
        if [checkpoint.get("step") for checkpoint in history] != expected_steps:
            raise ValueError("incomplete joint-nullspace trajectory")
        for checkpoint in history:
            if set(checkpoint) != required:
                raise ValueError("unexpected joint-nullspace checkpoint fields")
            active = float(checkpoint["shared_active_subspace_loss"])
            compatible_null = float(checkpoint["compatible_nullspace_loss"])
            compatible_total = float(checkpoint["compatible_total_loss"])
            incompatible_null = float(checkpoint["incompatible_nullspace_loss"])
            incompatible_total = float(checkpoint["incompatible_total_loss"])
            offset_error = float(checkpoint["paired_loss_offset_error"])
            if not all(
                np.isfinite(value)
                for value in (
                    active,
                    compatible_null,
                    compatible_total,
                    incompatible_null,
                    incompatible_total,
                    offset_error,
                )
            ):
                raise ValueError("non-finite joint-nullspace checkpoint")
            if active < 0.0 or compatible_total < 0.0 or incompatible_total < 0.0:
                raise ValueError("negative joint-nullspace loss")
            if compatible_null != 0.0 or compatible_total != active:
                raise ValueError("compatible loss does not equal its active component")
            if not np.isclose(incompatible_null, floor, rtol=0.0, atol=1.0e-15):
                raise ValueError("incompatible nullspace component changed")
            if not np.isclose(
                incompatible_total - active,
                floor,
                rtol=0.0,
                atol=max(1.0e-14, 4.0 * offset_error),
            ):
                raise ValueError("paired total losses do not have the predicted offset")
            maximum_offset_error = max(maximum_offset_error, offset_error)
    if maximum_offset_error > 1.0e-12:
        raise ValueError("joint-nullspace checkpoint offset error is too large")
    return record


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid Boolean value {value!r}")
    return normalized == "true"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    required = {
        "family",
        "row_design",
        "epsilon",
        "target_mode",
        "width",
        "seed",
        "exact_structural_rank",
        "target_functionally_compatible",
        "lambda_min",
        "lambda_max",
        "numerical_rank",
        "output_kernel_lambda_min",
        "initial_loss",
        "final_loss",
        "final_to_initial_loss_ratio",
    }
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required CSV columns: {sorted(missing)}")
        raw_rows = list(reader)

    if len(raw_rows) != 1080:
        raise ValueError(f"expected 1080 compatible controls, found {len(raw_rows)}")
    if any(not _as_bool(row["target_functionally_compatible"]) for row in raw_rows):
        raise ValueError("refusing to plot: an input target is not functionally compatible")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        epsilon_text = raw["epsilon"].strip()
        rows.append(
            {
                **raw,
                "epsilon": None if epsilon_text == "" else float(epsilon_text),
                "width": int(raw["width"]),
                "seed": int(raw["seed"]),
                "exact_structural_rank": int(raw["exact_structural_rank"]),
                "numerical_rank": int(raw["numerical_rank"]),
                "lambda_min": float(raw["lambda_min"]),
                "lambda_max": float(raw["lambda_max"]),
                "output_kernel_lambda_min": float(raw["output_kernel_lambda_min"]),
                "initial_loss": float(raw["initial_loss"]),
                "final_loss": float(raw["final_loss"]),
                "final_to_initial_loss_ratio": float(raw["final_to_initial_loss_ratio"]),
                "target_functionally_compatible": True,
            }
        )

    numeric_fields = (
        "lambda_min",
        "lambda_max",
        "output_kernel_lambda_min",
        "initial_loss",
        "final_loss",
        "final_to_initial_loss_ratio",
    )
    if any(
        not np.isfinite(row[field])
        for row in rows
        for field in numeric_fields
    ):
        raise ValueError("non-finite numerical value in compatible controls")
    if any(
        row["lambda_min"] < 0
        or row["lambda_max"] <= 0
        or row["output_kernel_lambda_min"] < 0
        or row["initial_loss"] <= 0
        or row["final_loss"] < 0
        or row["final_to_initial_loss_ratio"] < 0
        for row in rows
    ):
        raise ValueError("invalid eigenvalue or loss value in compatible controls")

    families = tuple(dict.fromkeys(row["family"] for row in rows))
    if families != EXPECTED_FAMILIES:
        raise ValueError(f"unexpected family order/content: {families}")
    if sorted({row["width"] for row in rows}) != list(WIDTHS):
        raise ValueError("unexpected width grid")
    if {row["exact_structural_rank"] for row in rows} != {1, 2}:
        raise ValueError("expected only structural ranks one and two")
    if any(row["exact_structural_rank"] != row["numerical_rank"] for row in rows):
        raise ValueError("structural and numerical ranks do not agree")

    rank_two = [row for row in rows if row["exact_structural_rank"] == 2]
    rank_one = [row for row in rows if row["exact_structural_rank"] == 1]
    if len(rank_two) != 960 or len(rank_one) != 120:
        raise ValueError("unexpected rank-control counts")
    if any(row["output_kernel_lambda_min"] <= 0 for row in rank_two):
        raise ValueError("an independent-row control lacks a positive output-kernel gap")
    if any(row["output_kernel_lambda_min"] != 0 for row in rank_one):
        raise ValueError("an exact-duplicate control has a nonzero output-kernel gap")

    expected_designs = {
        "independent": (None, ("manufactured", "contrast")),
        "epsilon_1": (1.0, ("manufactured", "contrast")),
        "epsilon_0.1": (0.1, ("manufactured", "contrast")),
        "epsilon_0.01": (0.01, ("manufactured", "contrast")),
        "epsilon_0": (0.0, ("manufactured",)),
    }
    observed_keys: set[tuple[str, str, str, int, int]] = set()
    for row in rows:
        design = row["row_design"]
        if design not in expected_designs:
            raise ValueError(f"unexpected row design {design!r}")
        expected_epsilon, target_modes = expected_designs[design]
        if row["epsilon"] != expected_epsilon or row["target_mode"] not in target_modes:
            raise ValueError(f"invalid design/epsilon/target combination for {design}")
        key = (
            row["family"],
            design,
            row["target_mode"],
            row["width"],
            row["seed"],
        )
        if key in observed_keys:
            raise ValueError(f"duplicate compatible-control cell {key}")
        observed_keys.add(key)
    expected_keys = {
        (family, design, target_mode, width, seed)
        for family in EXPECTED_FAMILIES
        for design, (_, target_modes) in expected_designs.items()
        for target_mode in target_modes
        for width in WIDTHS
        for seed in range(5)
    }
    missing_keys = expected_keys.difference(observed_keys)
    extra_keys = observed_keys.difference(expected_keys)
    if missing_keys or extra_keys:
        raise ValueError(
            "incomplete compatible-control grid: "
            f"{len(missing_keys)} missing and {len(extra_keys)} extra cells"
        )

    singular = [
        row
        for row in rows
        if row["exact_structural_rank"] == 1
        and row["target_mode"] == "manufactured"
        and row["epsilon"] == 0
    ]
    if len(singular) != 120:
        raise ValueError("expected 120 compatible exact-duplicate controls")
    for width in WIDTHS:
        if sum(row["width"] == width for row in singular) != 30:
            raise ValueError(f"incomplete singular-compatible group at width {width}")

    contrast = [
        row
        for row in rows
        if row["target_mode"] == "contrast"
        and row["width"] == 16384
        and row["epsilon"] in EPSILONS
    ]
    if len(contrast) != 90:
        raise ValueError("expected 90 width-16384 compatible contrast controls")
    for family in EXPECTED_FAMILIES:
        for epsilon in EPSILONS:
            group = [
                row
                for row in contrast
                if row["family"] == family and row["epsilon"] == epsilon
            ]
            if len(group) != 5 or {row["seed"] for row in group} != set(range(5)):
                raise ValueError(f"incomplete paired group for {family}, epsilon={epsilon}")
    return rows


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _distinct_kernel_instances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse target modes that share an identical initialized kernel."""
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["family"], row["row_design"], row["width"], row["seed"])
        groups[key].append(row)
    if len(groups) != 600:
        raise ValueError(f"expected 600 distinct kernel instances, found {len(groups)}")

    kernel_fields = (
        "epsilon",
        "exact_structural_rank",
        "numerical_rank",
        "lambda_min",
        "lambda_max",
        "output_kernel_lambda_min",
    )
    instances: list[dict[str, Any]] = []
    for key, group in groups.items():
        reference = group[0]
        for row in group[1:]:
            if any(row[field] != reference[field] for field in kernel_fields):
                raise ValueError(f"target modes disagree on shared kernel fields for {key}")
        instances.append(reference)

    rank_two = [row for row in instances if row["exact_structural_rank"] == 2]
    rank_one = [row for row in instances if row["exact_structural_rank"] == 1]
    if len(rank_two) != 480 or len(rank_one) != 120:
        raise ValueError("unexpected deduplicated rank-control counts")
    return instances


def _plot_rank(ax: plt.Axes, rows: list[dict[str, Any]]) -> dict[str, Any]:
    instances = _distinct_kernel_instances(rows)
    ranks = (1, 2)
    rank_two = [row for row in instances if row["exact_structural_rank"] == 2]
    rank_one = [row for row in instances if row["exact_structural_rank"] == 1]
    minimum_positive_gap = min(row["output_kernel_lambda_min"] for row in rank_two)
    matrix = np.asarray(
        [
            [
                sum(
                    row["exact_structural_rank"] == predicted
                    and row["numerical_rank"] == observed
                    for row in instances
                )
                for observed in ranks
            ]
            for predicted in ranks
        ],
        dtype=int,
    )
    counts = (len(rank_one), len(rank_two))
    matches = (int(matrix[0, 0]), int(matrix[1, 1]))
    percentages = [100.0 * match / count for match, count in zip(matches, counts)]
    ys = np.arange(2)
    bars = ax.barh(
        ys,
        percentages,
        height=0.62,
        color=("#56B4E9", "#0072B2"),
        edgecolor="none",
    )
    gap_labels = (
        "$\\lambda_{\\min}(\\mathbf{K}_{\\mathrm{out},(0)})=0$",
        "$\\lambda_{\\min}(\\mathbf{K}_{\\mathrm{out},(0)})>0$",
    )
    for bar, match, count, gap_label in zip(bars, matches, counts, gap_labels):
        ax.text(
            50,
            bar.get_y() + bar.get_height() / 2,
            f"{match}/{count}\n{gap_label}",
            ha="center",
            va="center",
            color="white",
            fontsize=7.4,
            fontweight="semibold",
        )
    ax.set(
        title="(a) Row dependence predicts singularity",
        xlabel="Kernels matching the prediction (%)",
        yticks=ys,
        yticklabels=("Duplicate\n(singular)", "Independent\n(nonsingular)"),
        xlim=(0, 104),
    )
    ax.set_xticks((0, 50, 100))
    ax.grid(True, axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    return {
        "confusion_matrix_rows_predicted_columns_observed": matrix.tolist(),
        "rank_two_positive_output_gap": sum(
            row["output_kernel_lambda_min"] > 0 for row in rank_two
        ),
        "rank_two_count": len(rank_two),
        "rank_one_zero_output_gap": sum(
            row["output_kernel_lambda_min"] == 0 for row in rank_one
        ),
        "rank_one_count": len(rank_one),
        "minimum_positive_output_gap": minimum_positive_gap,
    }


def _plot_singular(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    singular = [
        row
        for row in rows
        if row["exact_structural_rank"] == 1
        and row["target_mode"] == "manufactured"
        and row["epsilon"] == 0
    ]
    summary: dict[str, Any] = {}
    reductions: list[int] = []
    threshold_counts: list[int] = []
    for width in WIDTHS:
        group = [row for row in singular if row["width"] == width]
        reduced = sum(row["final_to_initial_loss_ratio"] <= 0.01 for row in group)
        below = sum(row["final_loss"] < LOSS_THRESHOLD for row in group)
        reductions.append(reduced)
        threshold_counts.append(below)
        summary[str(width)] = {
            "runs": len(group),
            "loss_reduced_at_least_100x": reduced,
            "final_loss_below_1e_minus_12": below,
            "median_final_loss": _median([row["final_loss"] for row in group]),
            "maximum_final_loss": max(row["final_loss"] for row in group),
        }
    matrix = np.asarray([reductions, threshold_counts], dtype=float) / 30.0
    ax.imshow(matrix, cmap="Blues", vmin=0.85, vmax=1.0, aspect="auto")
    for row_index, counts in enumerate((reductions, threshold_counts)):
        for column_index, count in enumerate(counts):
            ax.text(
                column_index,
                row_index,
                f"{count}/30",
                ha="center",
                va="center",
                color="white" if count == 30 else "#102A43",
                fontsize=8.5,
                fontweight="semibold",
            )
    ax.set(
        title="(b) Compatible singular targets",
        xlabel="Network width $d_1$",
        xticks=np.arange(len(WIDTHS)),
        xticklabels=("16", "256", "4,096", "16,384"),
        yticks=(0, 1),
        yticklabels=(
            "$L_{3000}/L_0\\leq10^{-2}$\n(120/120 total)",
            "$L_{3000}<10^{-12}$\n(118/120 total)",
        ),
    )
    ax.set_xticks(np.arange(-0.5, len(WIDTHS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    return summary


def _plot_conditioning(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    family_colors: dict[str, Any],
) -> dict[str, Any]:
    contrast = [
        row
        for row in rows
        if row["target_mode"] == "contrast"
        and row["width"] == 16384
        and row["epsilon"] in EPSILONS
    ]
    summary: dict[str, Any] = {}
    for family in EXPECTED_FAMILIES:
        family_summary: dict[str, Any] = {}
        xs: list[float] = []
        ys: list[float] = []
        for epsilon in EPSILONS:
            group = [
                row
                for row in contrast
                if row["family"] == family and row["epsilon"] == epsilon
            ]
            relative_gaps = [row["lambda_min"] / row["lambda_max"] for row in group]
            condition_numbers = [1.0 / gap for gap in relative_gaps]
            loss_ratios = [row["final_to_initial_loss_ratio"] for row in group]
            x = _median(condition_numbers)
            y = _median(loss_ratios)
            xs.append(x)
            ys.append(max(y, DISPLAY_FLOOR))
            ax.scatter(
                condition_numbers,
                np.maximum(loss_ratios, DISPLAY_FLOOR),
                marker=EPSILON_MARKERS[epsilon],
                s=17,
                color=family_colors[family],
                alpha=0.32,
                linewidth=0,
                zorder=2,
            )
            family_summary[f"{epsilon:g}"] = {
                "runs": len(group),
                "median_relative_gap": _median(relative_gaps),
                "median_initial_condition_number": x,
                "median_final_to_initial_loss_ratio": y,
            }
        ax.plot(
            xs,
            ys,
            color=family_colors[family],
            linestyle=FAMILY_LINESTYLES[family],
            linewidth=1.7,
            alpha=0.95,
        )
        for epsilon, x, y in zip(EPSILONS, xs, ys):
            ax.scatter(
                x,
                y,
                marker=EPSILON_MARKERS[epsilon],
                s=45,
                color=family_colors[family],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
        summary[family] = family_summary

    ax.set(
        title="(c) Ill-conditioning slows finite-horizon GD",
        xlabel=("Initial condition number "
               "$\\kappa_0=\\lambda_{\\max}(\\mathbf{K}_{(0)})/"
               "\\lambda_{\\min}(\\mathbf{K}_{(0)})$"),
        ylabel="Final / initial loss",
        xscale="log",
        yscale="log",
        ylim=(5e-17, 2),
    )
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.07)
    ax.axhline(
        DISPLAY_FLOOR,
        color="0.35",
        linestyle=(0, (2, 2)),
        linewidth=0.8,
        zorder=1,
    )
    ax.text(
        0.01,
        DISPLAY_FLOOR,
        "$10^{-16}$ display floor",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        color="0.35",
        fontsize=7.0,
    )
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=EPSILON_MARKERS[epsilon],
            color="0.35",
            linestyle="none",
            markerfacecolor="0.35",
            markeredgecolor="white",
            markersize=7,
            label=f"$\\varepsilon={epsilon:g}$",
        )
        for epsilon in EPSILONS
    ]
    return {"families": summary, "marker_handles": marker_handles}


def _optimization_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the finite-horizon counts quoted in the manuscript."""
    positive = [row for row in rows if row["exact_structural_rank"] == 2]
    manufactured_positive = [
        row for row in positive if row["target_mode"] == "manufactured"
    ]
    positive_width_ge_256 = [row for row in positive if row["width"] >= 256]
    positive_width_16384 = [row for row in positive if row["width"] == 16384]
    singular = [row for row in rows if row["exact_structural_rank"] == 1]
    if not (
        len(positive) == 960
        and len(manufactured_positive) == 480
        and len(positive_width_ge_256) == 720
        and len(positive_width_16384) == 240
        and len(singular) == 120
    ):
        raise ValueError("unexpected optimization-summary counts")
    return {
        "positive_definite": {
            "runs": len(positive),
            "loss_reduced": sum(
                row["final_loss"] < row["initial_loss"] for row in positive
            ),
            "manufactured_runs": len(manufactured_positive),
            "manufactured_loss_reduced_at_least_10x": sum(
                row["final_to_initial_loss_ratio"] <= 0.1
                for row in manufactured_positive
            ),
            "width_ge_256_runs": len(positive_width_ge_256),
            "width_ge_256_loss_reduced": sum(
                row["final_loss"] < row["initial_loss"]
                for row in positive_width_ge_256
            ),
            "width_16384_runs": len(positive_width_16384),
            "width_16384_loss_reduced": sum(
                row["final_loss"] < row["initial_loss"]
                for row in positive_width_16384
            ),
            "width_16384_median_final_to_initial_loss_ratio": _median(
                [row["final_to_initial_loss_ratio"] for row in positive_width_16384]
            ),
        },
        "compatible_singular": {
            "runs": len(singular),
            "loss_reduced_at_least_100x": sum(
                row["final_to_initial_loss_ratio"] <= 0.01 for row in singular
            ),
            "final_loss_below_1e_minus_12": sum(
                row["final_loss"] < LOSS_THRESHOLD for row in singular
            ),
            "width_ge_256_runs": sum(row["width"] >= 256 for row in singular),
            "width_ge_256_final_loss_below_1e_minus_12": sum(
                row["width"] >= 256 and row["final_loss"] < LOSS_THRESHOLD
                for row in singular
            ),
        },
    }


def _plot_width_diagnostics_aggregate(
    ax: plt.Axes,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    positions = np.arange(len(WIDTHS), dtype=float)
    series = (
        (
            "median_initial_relative_kernel_error",
            "Initial kernel error",
            "#0072B2",
            "o",
            "-",
        ),
        (
            "maximum_relative_kernel_drift",
            "Maximum kernel drift",
            "#D55E00",
            "s",
            "--",
        ),
        (
            "final_frozen_trajectory_discrepancy",
            "Frozen/nonlinear discrepancy",
            "#009E73",
            "^",
            "-.",
        ),
    )
    for key, label, color, marker, linestyle in series:
        values = [float(diagnostics[str(width)][key]) for width in WIDTHS]
        ax.plot(
            positions,
            values,
            label=label,
            color=color,
            marker=marker,
            markersize=4.2,
            linewidth=1.45,
            linestyle=linestyle,
        )

    retained = [
        diagnostics[str(width)]["gap_retained_at_saved_checkpoints"]
        / diagnostics[str(width)]["gap_retained_denominator"]
        for width in WIDTHS
    ]
    ax.plot(
        positions,
        retained,
        label="Gap retained (fraction)",
        color="0.3",
        marker="D",
        markersize=3.8,
        linewidth=1.25,
        linestyle=":",
    )
    ax.set(
        title="(a) Width stabilizes the training kernel",
        xlabel="Network width $d_1$",
        ylabel="Relative diagnostic",
        yscale="log",
        xlim=(-0.2, len(WIDTHS) - 0.8),
        ylim=(5.0e-4, 2.2),
    )
    ax.set_xticks(
        positions,
        ("16\n17/110", "256\n110/110", "4,096\n110/110", "16,384\n110/110"),
    )
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.06)
    ax.legend(
        loc="lower left",
        ncol=2,
        fontsize=6.5,
        frameon=False,
        handlelength=2.0,
        labelspacing=0.25,
    )
    return diagnostics


def _plot_singular_aggregate(
    ax: plt.Axes,
    singular: dict[str, Any],
) -> dict[str, Any]:
    reductions = [
        singular[str(width)]["loss_reduced_at_least_100x"] for width in WIDTHS
    ]
    threshold_counts = [
        singular[str(width)]["final_loss_below_1e_minus_12"] for width in WIDTHS
    ]
    matrix = np.asarray([reductions, threshold_counts], dtype=float) / 30.0
    ax.imshow(matrix, cmap="Blues", vmin=0.85, vmax=1.0, aspect="auto")
    for row_index, counts in enumerate((reductions, threshold_counts)):
        for column_index, count in enumerate(counts):
            ax.text(
                column_index,
                row_index,
                f"{count}/30",
                ha="center",
                va="center",
                color="white" if count == 30 else "#102A43",
                fontsize=8.5,
                fontweight="semibold",
            )
    ax.set(
        title="(b) Compatible singular targets",
        xlabel="Network width $d_1$",
        xticks=np.arange(len(WIDTHS)),
        xticklabels=("16", "256", "4,096", "16,384"),
        yticks=(0, 1),
        yticklabels=(
            "$L_{3000}/L_0\\leq10^{-2}$\n(120/120 total)",
            "$L_{3000}<10^{-12}$\n(118/120 total)",
        ),
    )
    ax.set_xticks(np.arange(-0.5, len(WIDTHS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    return singular


def _plot_nullspace_trajectories(
    ax: plt.Axes,
    joint: dict[str, Any],
) -> dict[str, Any]:
    """Plot the exact active/nullspace loss decomposition for paired targets."""

    trajectories = joint["gd_pair_records"]
    steps = np.asarray(joint["configuration"]["checkpoint_steps"], dtype=float)
    active = np.asarray(
        [
            [checkpoint["shared_active_subspace_loss"] for checkpoint in row["paired_history"]]
            for row in trajectories
        ],
        dtype=float,
    )
    incompatible_total = np.asarray(
        [
            [checkpoint["incompatible_total_loss"] for checkpoint in row["paired_history"]]
            for row in trajectories
        ],
        dtype=float,
    )
    active_median = np.median(active, axis=0)
    active_q25, active_q75 = np.quantile(active, (0.25, 0.75), axis=0)
    incompatible_median = np.median(incompatible_total, axis=0)
    floor = float(joint["summary"]["incompatible_gd_predicted_floor"])

    ax.fill_between(
        steps,
        np.maximum(active_q25, TRAJECTORY_DISPLAY_FLOOR),
        np.maximum(active_q75, TRAJECTORY_DISPLAY_FLOOR),
        color="#0072B2",
        alpha=0.18,
        linewidth=0.0,
    )
    ax.plot(
        steps,
        np.maximum(active_median, TRAJECTORY_DISPLAY_FLOOR),
        color="#0072B2",
        linewidth=1.6,
        label=r"Active component $\|\Pi_{\mathcal{U}}e_k\|^2$" + "\n(= compatible total)",
    )
    ax.plot(
        steps,
        incompatible_median,
        color="#D55E00",
        linewidth=1.5,
        linestyle="--",
        label=r"Incompatible total $\|e_k\|^2$",
    )
    ax.axhline(
        floor,
        color="0.25",
        linewidth=1.25,
        linestyle=":",
        label=r"Null component $\|\Pi_{\mathcal{N}}e_k\|^2=0.0625$",
    )
    ax.text(
        0.98,
        0.06,
        r"Compatible target: $\|\Pi_{\mathcal{N}}e_k\|^2=0$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="0.3",
    )
    ax.set(
        title="(b) Exact nullspace loss floor",
        xlabel="GD step $k$",
        ylabel="Loss component",
        xscale="symlog",
        yscale="log",
        xlim=(0.0, float(steps[-1])),
        ylim=(TRAJECTORY_DISPLAY_FLOOR, 2.0),
    )
    ax.set_xticks((0, 10, 100, 1000), ("0", "10", "$10^2$", "$10^3$"))
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", axis="y", alpha=0.06)
    ax.legend(
        loc="upper right",
        fontsize=7.0,
        frameon=False,
        handlelength=2.0,
        labelspacing=0.25,
    )
    return {
        "paired_settings": len(trajectories),
        "gd_trajectories": 2 * len(trajectories),
        "checkpoint_steps": steps.astype(int).tolist(),
        "predicted_incompatible_loss_floor": floor,
        "maximum_checkpoint_loss_offset_error": float(
            joint["summary"]["paired_gd_max_checkpoint_loss_offset_error"]
        ),
        "median_shared_active_subspace_loss": active_median.tolist(),
        "active_subspace_loss_q25": active_q25.tolist(),
        "active_subspace_loss_q75": active_q75.tolist(),
        "median_incompatible_total_loss": incompatible_median.tolist(),
    }


def _plot_conditioning_aggregate(
    ax: plt.Axes,
    conditioning: dict[str, Any],
    family_colors: dict[str, Any],
) -> dict[str, Any]:
    for family in EXPECTED_FAMILIES:
        xs: list[float] = []
        ys: list[float] = []
        for epsilon in EPSILONS:
            cell = conditioning[family][f"{epsilon:g}"]
            xs.append(float(cell["median_initial_condition_number"]))
            ys.append(max(float(cell["median_final_to_initial_loss_ratio"]), DISPLAY_FLOOR))
        ax.plot(
            xs,
            ys,
            color=family_colors[family],
            linestyle=FAMILY_LINESTYLES[family],
            linewidth=1.7,
            alpha=0.95,
        )
        for epsilon, x, y in zip(EPSILONS, xs, ys):
            ax.scatter(
                x,
                y,
                marker=EPSILON_MARKERS[epsilon],
                s=45,
                color=family_colors[family],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )

    ax.set(
        title="(b) Ill-conditioning slows finite-horizon GD",
        xlabel=("Median initial condition number "
               "$\\kappa_0=\\lambda_{\\max}(\\mathbf{K}_{(0)})/"
               "\\lambda_{\\min}(\\mathbf{K}_{(0)})$"),
        ylabel="Median final / initial loss",
        xscale="log",
        yscale="log",
        ylim=(5e-17, 2),
    )
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.07)
    ax.axhline(
        DISPLAY_FLOOR,
        color="0.35",
        linestyle=(0, (2, 2)),
        linewidth=0.8,
        zorder=1,
    )
    ax.text(
        0.01,
        DISPLAY_FLOOR,
        "$10^{-16}$ display floor",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        color="0.35",
        fontsize=7.0,
    )
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=EPSILON_MARKERS[epsilon],
            color="0.35",
            linestyle="none",
            markerfacecolor="0.35",
            markeredgecolor="white",
            markersize=7,
            label=f"$\\varepsilon={epsilon:g}$",
        )
        for epsilon in EPSILONS
    ]
    return {"families": conditioning, "marker_handles": marker_handles}


def make_figure(aggregate: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    family_colors = FAMILY_COLORS

    fig = plt.figure(figsize=(5.5, 4.1))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.86, 1.14),
        left=0.14,
        right=0.98,
        top=0.95,
        bottom=0.23,
        hspace=0.62,
    )
    width_ax = fig.add_subplot(grid[0, 0])
    conditioning_ax = fig.add_subplot(grid[1, 0])
    panels = aggregate["panels"]
    width_summary = _plot_width_diagnostics_aggregate(
        width_ax,
        panels["width_diagnostics"],
    )
    conditioning_result = _plot_conditioning_aggregate(
        conditioning_ax,
        panels["conditioning_width_16384"],
        family_colors,
    )
    conditioning_summary = conditioning_result["families"]

    family_handles = [
        Line2D(
            [0],
            [0],
            color=family_colors[family],
            linestyle=FAMILY_LINESTYLES[family],
            linewidth=1.8,
            label=FAMILY_LABELS[family],
        )
        for family in EXPECTED_FAMILIES
    ]
    family_legend = fig.legend(
        handles=family_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=3,
        title="Constraint family (color and line style)",
        fontsize=7.0,
        title_fontsize=7.4,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.0,
    )
    fig.add_artist(family_legend)
    conditioning_ax.legend(
        handles=conditioning_result["marker_handles"],
        loc="lower right",
        ncol=3,
        title=(
            r"Prescribed row mixing $\varepsilon$ (marker)"
            + "\n"
            + r"x-axis: measured $\kappa_0$; points: five-seed medians"
        ),
        fontsize=6.5,
        title_fontsize=6.7,
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    return fig, {
        "width_diagnostics": width_summary,
        "conditioning_width_16384": conditioning_summary,
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    default_input = (
        project_root
        / "experiments/theorem_audit/data"
        / "compatible_rank_convergence_conditioning.json"
    )
    parser = argparse.ArgumentParser(
        description="Generate the manuscript figure from its tracked aggregate data."
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=project_root / "figures")
    parser.add_argument(
        "--stem", default="compatible_rank_convergence_conditioning"
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = _load_frozen_aggregate(input_path)
    fig, summary = make_figure(aggregate)
    metadata = {
        "Creator": "experiments/theorem_audit/plot_supportive_evidence.py",
        "Title": "Finite-width kernel stability and conditioning",
        "Subject": f"Compatible-control SHA256: {_sha256(input_path)}",
        "CreationDate": None,
        "ModDate": None,
    }
    pdf_path = output_dir / f"{args.stem}.pdf"
    png_path = output_dir / f"{args.stem}.png"
    temporary_pdf = output_dir / f".{args.stem}.tmp.pdf"
    temporary_png = output_dir / f".{args.stem}.tmp.png"
    figure_inches = [float(value) for value in fig.get_size_inches()]
    fig.savefig(temporary_pdf, format="pdf", metadata=metadata)
    fig.savefig(temporary_png, format="png", dpi=220)
    plt.close(fig)
    temporary_pdf.replace(pdf_path)
    temporary_png.replace(png_path)

    record = {
        "schema_version": 5,
        "aggregate_input": input_path.relative_to(project_root).as_posix(),
        "aggregate_input_sha256": _sha256(input_path),
        "raw_source": aggregate["raw_source"],
        "analysis_script": Path(__file__).resolve().relative_to(project_root).as_posix(),
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "figure_size_inches": figure_inches,
        "compatible_rows": aggregate["raw_source"]["rows"],
        "all_targets_functionally_compatible": True,
        "plotted_statistics": "validated counts and five-seed medians",
        "outputs": {
            "pdf": {
                "path": pdf_path.relative_to(project_root).as_posix(),
                "bytes": pdf_path.stat().st_size,
                "sha256": _sha256(pdf_path),
            },
            "png": {
                "path": png_path.relative_to(project_root).as_posix(),
                "bytes": png_path.stat().st_size,
                "sha256": _sha256(png_path),
            },
        },
        "summary": summary,
        "display_floors": {"conditioning_loss_ratio": DISPLAY_FLOOR},
    }
    summary_path = output_dir / f"{args.stem}_summary.json"
    temporary_summary = output_dir / f".{args.stem}_summary.tmp.json"
    temporary_summary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary_summary.replace(summary_path)
    print(
        "validated frozen counts and medians derived from "
        f"{aggregate['raw_source']['rows']} compatible controls"
    )
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
