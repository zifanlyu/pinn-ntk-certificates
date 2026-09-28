"""Validate, tabulate and plot only the paired rank-control artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .rank_controls import atomic_json, config_digest, load_config, output_dir, target_is_compatible, task_grid


LABELS = {
    "pointwise_poisson": "Pointwise Poisson: -u'' and u at the same point",
    "vector_stokes": "Vector Stokes: co-located momentum components",
    "weak_poisson": "Weak Poisson: two Legendre test functionals",
    "nonlocal_diffusion": "Nonlocal diffusion: two signed-measure rows",
    "weak_boundary": "Weak Poisson: boundary and weak residual",
    "nonlocal_boundary": "Nonlocal diffusion: boundary and residual measure",
}


def _flatten(run: dict[str, Any]) -> dict[str, Any]:
    task, initial, summary = run["task"], run["initial_spectrum"], run["summary"]
    floor = summary["exact_structural_loss_floor"]
    start_excess = summary["initial_loss"] - floor
    end_excess = summary["final_loss"] - floor
    steps = run["optimization"]["steps"]
    # Never fit rounded zero/negative excess losses; report the actual endpoint
    # rate otherwise. This is a finite-time rate, not an asymptotic theorem fit.
    rate = -math.log(end_excess / start_excess) / steps if steps > 0 and min(start_excess, end_excess) > 1e-13 * max(1, summary["initial_loss"]) else None
    return {
        "task_index": run["task_index"], **{k: task[k] for k in ("family", "row_design", "epsilon", "target_mode", "width", "seed", "exact_structural_rank", "target_functionally_compatible")},
        "lambda_min": initial["lambda_min"], "lambda_max": initial["lambda_max"],
        "numerical_rank": initial["numerical_rank"], "rank_tolerance": initial["rank_tolerance"],
        "output_kernel_lambda_min": run["output_kernel_spectrum"]["lambda_min"],
        "hidden_kernel_lambda_min": run["hidden_kernel_min_eigenvalue"],
        "learning_rate": run["optimization"]["learning_rate"],
        "scaled_gap_2_eta_lambda_min": run["optimization"]["scaled_initial_gap_2_eta_lambda_min"],
        "observed_endpoint_log_excess_loss_rate": rate,
        **summary, "elapsed_seconds": run["elapsed_seconds"],
        "source_sha256": run["source_sha256"],
    }


def _save(fig: Any, directory: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")


def _history_band(ax: Any, runs: list[dict], color: Any, label: str, *, frozen: bool = False) -> None:
    if not runs:
        return
    steps = np.asarray([row["step"] for row in runs[0]["history"]])
    key = "frozen_kernel_loss" if frozen else "loss"
    values = np.asarray([[row[key] for row in run["history"]] for run in runs])
    values = np.maximum(values, 1e-16)
    ax.plot(steps, np.median(values, axis=0), linestyle="--" if frozen else "-", color=color, label=label)
    if not frozen and len(runs) > 1:
        ax.fill_between(steps, np.quantile(values, 0.25, axis=0), np.quantile(values, 0.75, axis=0), color=color, alpha=0.13)


def _figures(runs: list[dict], rows: list[dict], directory: Path, config: dict) -> list[str]:
    if any(not run["task"]["target_functionally_compatible"] for run in runs):
        raise ValueError("Figures accept compatible targets only")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7, "figure.constrained_layout.use": True})
    families = list(config["families"])
    ncols = min(2, len(families))
    nrows = math.ceil(len(families) / ncols)
    figures = []

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.3 * nrows), squeeze=False)
    for ax, family in zip(axes.flat, families):
        selected = [r for r in runs if r["task"]["family"] == family and r["task"]["epsilon"] == 0 and r["task"]["target_mode"] == "manufactured"]
        if not selected:
            ax.set_visible(False)
            continue
        width = max(r["task"]["width"] for r in selected)
        selected = [r for r in selected if r["task"]["width"] == width]
        _history_band(ax, selected, "tab:blue", f"Compatible manufactured targets, nonlinear GD (n={len(selected)})")
        _history_band(ax, selected, "tab:blue", "Same targets, frozen K₀", frozen=True)
        ax.set(title=f"{LABELS[family]}\nExact duplicate rows; width {width}", xlabel="Full-batch GD step", ylabel="Normalized squared constraint loss", yscale="log")
        ax.grid(alpha=.2)
        ax.legend()
    for ax in list(axes.flat)[len(families):]:
        ax.set_visible(False)
    fig.suptitle("Compatible manufactured targets converge with a singular NTK; bands = seed interquartile range")
    _save(fig, directory, "singular_target_compatibility")
    plt.close(fig)
    figures.append("singular_target_compatibility")

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.3 * nrows), squeeze=False)
    colors = plt.get_cmap("viridis")(np.linspace(.08, .88, len(config["epsilons"])))
    for ax, family in zip(axes.flat, families):
        selected = [r for r in runs if r["task"]["family"] == family and r["task"]["target_mode"] == "contrast"]
        if not selected:
            ax.set_visible(False)
            continue
        width = max(r["task"]["width"] for r in selected)
        for epsilon, color in zip(config["epsilons"], colors):
            subset = [r for r in selected if r["task"]["width"] == width and r["task"]["epsilon"] == epsilon]
            if not subset:
                continue
            label = "Independent pair" if epsilon is None else f"ε={epsilon:g}"
            gap = np.median([r["initial_spectrum"]["lambda_min"] for r in subset])
            _history_band(ax, subset, color, f"{label}; median λmin(K₀)={gap:.2g}, n={len(subset)}")
            _history_band(ax, subset, color, "", frozen=True)
        ax.set(title=f"{LABELS[family]}\nCompatible contrast targets; width {width}", xlabel="Full-batch GD step", ylabel="Normalized squared constraint loss", yscale="log")
        ax.grid(alpha=.2)
        ax.legend()
    for ax in list(axes.flat)[len(families):]:
        ax.set_visible(False)
    fig.suptitle("Near-dependent rows slow GD: solid = nonlinear; dashed = frozen K₀; bands = seed interquartile range")
    _save(fig, directory, "conditioning_training_dynamics")
    plt.close(fig)
    figures.append("conditioning_training_dynamics")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    family_colors = dict(zip(families, plt.get_cmap("tab10")(np.arange(len(families)))))
    for family in families:
        family_rows = [r for r in rows if r["family"] == family and r["target_mode"] == "contrast" and r["epsilon"] is not None and r["epsilon"] > 0]
        if not family_rows:
            continue
        width = max(r["width"] for r in family_rows)
        group = defaultdict(list)
        for row in family_rows:
            if row["width"] == width:
                group[row["epsilon"]].append(row["lambda_min"])
        xs = np.asarray(sorted(group))
        ys = np.asarray([np.median(group[x]) for x in xs])
        axes[0].loglog(xs, ys, "o-", color=family_colors[family], label=f"{family}, m={width}")
        if len(xs) > 1:
            axes[0].loglog(xs, ys[0] * (xs / xs[0]) ** 2, ":", color=family_colors[family], alpha=.6)
        good = [r for r in family_rows if r["observed_endpoint_log_excess_loss_rate"] is not None and r["observed_endpoint_log_excess_loss_rate"] > 0 and 0 < r["scaled_gap_2_eta_lambda_min"] < 1]
        if good:
            expected = [-2 * math.log1p(-r["scaled_gap_2_eta_lambda_min"]) for r in good]
            axes[1].loglog(expected, [r["observed_endpoint_log_excess_loss_rate"] for r in good], ".", alpha=.45, color=family_colors[family], label=f"{family} ({len(good)} runs)")
    axes[0].set(title="Initial spectral gap versus row independence\nDotted curves: ε² scaling, anchored at smallest ε", xlabel="Mixing parameter ε", ylabel="Median λmin(K₀), float64")
    axes[1].set(title="Observed endpoint decay versus slow frozen-mode rate\nAll widths/seeds; roundoff-limited losses excluded", xlabel="-2 log(1 - 2η λmin(K₀))", ylabel="-log[Lfinal/Linitial] / steps")
    limits = axes[1].get_xlim()
    if limits[0] > 0:
        axes[1].plot(limits, limits, "k--", alpha=.6, label="Equal rates (guide, not theorem equality)")
    for ax in axes:
        ax.grid(alpha=.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    _save(fig, directory, "conditioning_gap_and_rate")
    plt.close(fig)
    figures.append("conditioning_gap_and_rate")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for family in families:
        group = defaultdict(list)
        for row in rows:
            if row["family"] == family and row["epsilon"] is None and row["target_mode"] == "manufactured":
                group[row["width"]].append(row)
        if not group:
            continue
        widths = sorted(group)
        for ax, key in zip(axes, ("output_kernel_lambda_min", "final_relative_kernel_drift_operator")):
            med = [np.median([r[key] for r in group[w]]) for w in widths]
            low = [np.quantile([r[key] for r in group[w]], .25) for w in widths]
            high = [np.quantile([r[key] for r in group[w]], .75) for w in widths]
            ax.loglog(widths, np.maximum(med, 1e-16), "o-", color=family_colors[family], label=family)
            ax.fill_between(widths, np.maximum(low, 1e-16), np.maximum(high, 1e-16), alpha=.15, color=family_colors[family])
    axes[0].set(title="Independent rows: output-weight contribution\nMedian and seed interquartile range", xlabel="Network width m", ylabel="λmin(Kout,0), displayed floor 10⁻¹⁶")
    axes[1].set(title="Independent rows: finite-width NTK movement\nAll parameters trained; manufactured targets", xlabel="Network width m", ylabel="‖Kfinal - K₀‖op / ‖K₀‖op")
    for ax in axes:
        ax.grid(alpha=.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    _save(fig, directory, "rank_control_width_diagnostics")
    plt.close(fig)
    figures.append("rank_control_width_diagnostics")
    return figures


def aggregate(config: dict[str, Any]) -> dict[str, Any]:
    directory = output_dir(config)
    runs = []
    missing = []
    tasks = task_grid(config)
    digest = config_digest(config)
    for index, expected in enumerate(tasks):
        path = directory / "runs" / f"task_{index:05d}.json"
        if not path.exists():
            missing.append(index)
            continue
        run = json.loads(path.read_text())
        actual = run["task"]
        actual_task = tuple(actual[k] for k in ("family", "epsilon", "target_mode", "width", "seed"))
        if run.get("config_sha256") != digest or run.get("status") != "complete" or actual_task != expected or run["task_index"] != index:
            raise ValueError(f"inconsistent artifact {path}")
        array_path = path.parent / run["arrays_file"]
        if not array_path.exists() or hashlib.sha256(array_path.read_bytes()).hexdigest() != run["arrays_sha256"]:
            raise ValueError(f"missing/corrupt array artifact {array_path}")
        runs.append(run)
    if not runs:
        raise ValueError("no completed rank-control runs")
    hashes = sorted({run["source_sha256"] for run in runs})
    if len(hashes) > 1:
        raise ValueError("mixed source fingerprints; aggregate separate output directories")
    if any(not run["task"]["target_functionally_compatible"] for run in runs):
        raise ValueError("Use the compatible reporting overlay for archived full-grid results")
    return _write_aggregate(runs, config, directory, len(tasks), missing, hashes[0])


def _write_aggregate(runs: list[dict], config: dict, directory: Path, expected_runs: int,
                     missing: list[int], source_sha256: str, *, selection: dict | None = None) -> dict[str, Any]:
    rows = [_flatten(run) for run in runs]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["row_design"], row["target_mode"], row["width"])].append(row)
    grouped = []
    for (family, design, mode, width), group in groups.items():
        record = dict(family=family, row_design=design, target_mode=mode, width=width, count=len(group))
        for key in ("lambda_min", "lambda_max", "output_kernel_lambda_min", "final_loss", "final_to_initial_loss_ratio", "final_loss_above_structural_floor", "final_relative_kernel_drift_operator", "observed_endpoint_log_excess_loss_rate"):
            values = [row[key] for row in group if row[key] is not None and math.isfinite(row[key])]
            record[key] = {"median": float(np.median(values)), "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)), "count": len(values)} if values else None
        grouped.append(record)
    figures_dir = directory / "figures"
    figures_dir.mkdir(exist_ok=True)
    figures = _figures(runs, rows, figures_dir, config)
    result = {"schema_version": 1, "expected_runs": expected_runs, "completed_runs": len(runs), "missing_task_indices": missing,
        "coverage_complete": not missing and len(runs) == expected_runs, "config_sha256": config_digest(config), "source_sha256": source_sha256, "groups": grouped,
        "figures": figures,
        "target_functionally_compatible_only": True,
        "interpretation": "Two-row functional controls isolate positivity/conditioning. These do not measure held-out PDE accuracy. Positive-definite cases can converge slowly when lambda_min is small. Singular compatible targets can converge. All presented targets are functionally compatible."}
    if selection is not None:
        result["selection"] = selection
    atomic_json(directory / "aggregate.json", result)
    (figures_dir / "figure_captions.md").write_text(
        f"Coverage: {len(runs)}/{expected_runs} compatible runs. Missing tasks are listed in aggregate.json; plots use available completed runs. All plotted targets are functionally compatible.\n\n"
        "All networks are the paper's biased shallow tanh architecture with shared hidden weights, independent uniform [-1,1] initialization and all parameters trained by full-batch GD in float64. The loss is the mean of two squared functional residuals. Widths use nested initialization prefixes, and each seed is paired across row and target controls. Solid curves are medians over available seeds; shading is the 25th–75th percentile, not a confidence interval. Dashed dynamics freeze the actual initial empirical NTK.\n\n"
        "**singular_target_compatibility:** Both rows and their manufactured targets are identical at epsilon=0. The initial residual lies in the range of the rank-one NTK, allowing convergence despite singularity. Only compatible manufactured targets at the largest available width are shown. Losses below 1e-16 are displayed at 1e-16 to make the logarithmic plot readable; this is a display cutoff, not a fitted loss floor.\n\n"
        "**conditioning_training_dynamics:** The independent pair is compared with ell1 and (ell1+epsilon ell2)/sqrt(1+epsilon²). Coefficients of each base functional are normalized before mixing. Contrast targets excite near-null directions; for positive epsilon they are functionally compatible, although they need not come from the original manufactured PDE solution. The finite initial gap is a diagnostic, not a proof of trajectory-wide positive definiteness.\n\n"
        "**conditioning_gap_and_rate:** The smallest initial eigenvalue scales quadratically with epsilon in the small-epsilon regime. The observed finite-time endpoint log decay is compared with the slowest frozen eigenmode loss decay. Equality is not required: residual spectral support, nonlinear kernel movement, finite-time transients and roundoff affect the observed rate. Values below the stated roundoff cutoff are excluded from rate estimation.\n\n"
        "**rank_control_width_diagnostics:** Kout contains only output-weight Jacobian columns; Kfull-Kout is positive semidefinite up to rounding. The left panel records its finite-width minimum eigenvalue, while the right panel records actual nonlinear kernel movement. Both use independent rows and manufactured targets. Weak and nonlocal rows are the exact stored quadrature functionals; there is no claim about undiscretized integral identities or the limiting kernel.\n"
    )
    _write_findings(directory, rows, result, config)
    print(f"[rank-control aggregate] {len(runs)}/{expected_runs} compatible runs; {directory / 'aggregate.json'}", flush=True)
    return result


def _write_findings(directory: Path, rows: list[dict], result: dict, config: dict) -> None:
    manufactured = [row for row in rows if row["target_mode"] == "manufactured"]
    contrast = [row for row in rows if row["target_mode"] == "contrast"]
    singular = [row for row in manufactured if row["epsilon"] == 0]
    singular_small_loss = sum(row["final_loss"] < 1e-12 for row in singular)
    largest_width = max(row["width"] for row in rows)
    family_count = len({row["family"] for row in rows})
    steps = int(config["steps"])
    conditioning = []
    for epsilon in config["epsilons"]:
        if epsilon == 0:
            continue
        subset = [row for row in contrast if row["width"] == largest_width and row["epsilon"] == epsilon]
        if subset:
            conditioning.append(("Independent" if epsilon is None else f"{epsilon:g}", len(subset),
                                 float(np.median([row["lambda_min"] for row in subset])),
                                 float(np.median([row["final_to_initial_loss_ratio"] for row in subset]))))
    md = ["# Compatible-target rank and conditioning results", "",
          f"Completed **{len(rows)}/{result['expected_runs']} compatible controls**: {len(manufactured)} manufactured-target and {len(contrast)} compatible contrast-target runs. Every reported target is functionally compatible. No training was rerun for this reporting overlay.", "",
          f"Among {len(singular)} exact-duplicate manufactured-target runs, {singular_small_loss} reached loss below 1e-12. The duplicate functionals have structural rank one, and their equal targets permit convergence despite a singular NTK.", "",
          f"The table pools compatible contrast runs across {family_count} functional families at width {largest_width}. It reports medians after {steps:,} full-batch GD steps; family-specific plots should be used for scientific comparisons. Pooling is descriptive and is not a common spectral constant or a statistical confidence interval.", "",
          "| Row mixing | Runs | Median initial minimum eigenvalue | Median final/initial loss |",
          "|---|---:|---:|---:|"]
    md.extend(f"| {label} | {count} | {gap:.6g} | {ratio:.6g} |" for label, count, gap, ratio in conditioning)
    md.extend(["", "Smaller positive eigenvalues permit much slower convergence on these compatible targets. Frozen-kernel rates are comparisons, and neither an initial positive gap nor finite numerical training verifies the theorem's existential width and step-size constants.", "",
               "Compatible contrast targets use independent rows (the unmodified pair or positive mixing parameter). They need not match the original manufactured PDE solution. These two-row controls measure constraint optimization, not held-out PDE solution accuracy.", "",
               "[Singular compatible targets](figures/singular_target_compatibility.png) · [Conditioning and GD](figures/conditioning_training_dynamics.png) · [Gap and decay rate](figures/conditioning_gap_and_rate.png) · [Width diagnostics](figures/rank_control_width_diagnostics.png)", "",
               "[Detailed figure captions](figures/figure_captions.md). The selection and original artifact hashes are recorded in `aggregate.json` and `selection_provenance.json`; `runs.csv` contains only retained compatible runs.", ""])
    (directory / "findings.md").write_text("\n".join(md))
    table = "\n".join(f"{label} & {count} & {gap:.6g} & {ratio:.6g} \\\\" for label, count, gap, ratio in conditioning)
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,booktabs,hyperref}}
\begin{{document}}
\section*{{Compatible-target rank and conditioning results}}
Completed {len(rows)}/{result['expected_runs']} compatible controls: {len(manufactured)} manufactured-target runs and {len(contrast)} compatible contrast-target runs. Every reported target is functionally compatible. This reporting overlay reuses the original completed trajectories.

For duplicate rows $\ell_1=\ell_2$ with equal targets, the normalized residual lies in the range of the rank-one kernel. All {len(singular)} retained duplicate-row runs use manufactured targets; {singular_small_loss} reached $L_T<10^{{-12}}$. Singularity therefore does not preclude convergence for these compatible targets.

The loss is $L(\theta)=\sum_{{i=1}}^2(\ell_i[u_\theta]-y_i)^2$. Full-batch gradient descent uses $\theta_{{t+1}}=\theta_t-2\eta J_t^\top e_t$, with $\eta={config['learning_rate_factor']}/\lambda_{{\max}}(K_0)$. Frozen dynamics satisfy $e_{{t+1}}=(I-2\eta K_0)e_t$. A frozen eigenmode with eigenvalue $\lambda>0$ has squared-loss factor $(1-2\eta\lambda)^2$; smaller gaps permit slower convergence.

\begin{{center}}
\begin{{tabular}}{{lrrr}}
\toprule
Row mixing & Runs & Median $\lambda_{{\min}}(K_0)$ & Median $L_T/L_0$\\
\midrule
{table}
\bottomrule
\end{{tabular}}
\end{{center}}
The table pools compatible contrast targets across {family_count} functional families at width {largest_width}, after {steps:,} full-batch GD steps. Pooled medians are descriptive; they are not a common spectral constant or confidence interval. Use the family-specific figures and detailed captions for comparisons. Positive mixing parameters give independent rows and admit the contrast targets, which need not be targets of the original manufactured PDE solution.

These two-row controls measure constraint optimization, not held-out PDE accuracy. Initial positive definiteness and finite numerical training do not verify the theorem's existential width or step-size constants. The compatible selection and source-artifact hashes are recorded in the accompanying JSON provenance; all plotted and tabulated targets are compatible.
\end{{document}}
"""
    (directory / "findings.tex").write_text(tex)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_history(run: dict, array_path: Path, config: dict) -> None:
    """Require complete checkpoints and exact JSON/NPZ agreement before selection."""
    steps = int(config["steps"])
    expected_steps = list(range(0, steps + 1, int(config["checkpoint_every"])))
    if expected_steps[-1] != steps:
        expected_steps.append(steps)
    history = run.get("history", [])
    if run.get("optimization", {}).get("steps") != steps or [row.get("step") for row in history] != expected_steps:
        raise ValueError(f"incomplete checkpoint history: {array_path}")
    with np.load(array_path, allow_pickle=False) as arrays:
        for array_name, json_name in (("history_step", "step"), ("history_loss", "loss"),
                                     ("history_frozen_kernel_loss", "frozen_kernel_loss")):
            expected = np.asarray([row[json_name] for row in history])
            if not np.isfinite(expected).all() or not np.array_equal(arrays[array_name], expected):
                raise ValueError(f"inconsistent/nonfinite {array_name}: {array_path}")
        for array_name in ("history_loss", "history_frozen_kernel_loss"):
            if np.any(arrays[array_name] < 0):
                raise ValueError(f"negative squared loss: {array_path}")
    for summary_key, row, history_key in (("initial_loss", history[0], "loss"),
                                          ("final_loss", history[-1], "loss"),
                                          ("final_frozen_kernel_loss", history[-1], "frozen_kernel_loss")):
        if run["summary"][summary_key] != row[history_key]:
            raise ValueError(f"inconsistent endpoint summary: {array_path}")


def aggregate_compatible(source_config: str | Path, destination: str | Path) -> dict[str, Any]:
    """Validate a frozen source campaign, then write a compatible-only overlay.

    The source snapshot, config, run manifests and arrays remain byte-for-byte
    unchanged. All source tasks, including excluded ones, must validate first.
    """
    config_path = Path(source_config).resolve()
    snapshot_root = config_path.parent
    snapshot_path = snapshot_root / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    destination = Path(destination).resolve()
    for name, expected_hash in snapshot["files"].items():
        path = (snapshot_root / name).resolve()
        if not path.is_relative_to(snapshot_root) or not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"frozen snapshot changed or missing: {name}")
        if path.is_relative_to(destination):
            raise ValueError("output directory must not contain frozen source files")
    if str(config_path.relative_to(snapshot_root)) not in snapshot["files"]:
        raise ValueError("source config is not covered by the snapshot manifest")
    config = load_config(config_path)
    source_directory = Path(config["output_dir"])
    if not source_directory.is_absolute():
        source_directory = snapshot_root / source_directory
    source_directory = source_directory.resolve()
    if destination == source_directory or destination.is_relative_to(source_directory):
        raise ValueError("output directory must be separate from source run artifacts")
    source_digest = hashlib.sha256()
    for name in ("rank_controls.py", "constraints.py", "problems.py", "features.py", "engine.py"):
        source_digest.update(name.encode())
        source_digest.update((snapshot_root / "experiments" / "theorem_audit" / name).read_bytes())
    expected_source_hash = source_digest.hexdigest()
    expected_config_hash = config_digest(config)
    # Original snapshots predate the filtering flag and used a full grid.
    archive_includes_incompatible = config.get("include_incompatible_targets", True)
    tasks = task_grid(config, include_incompatible=archive_includes_incompatible)
    amplitude = float(config.get("contrast_amplitude", 0.5))
    expected_included = sum(target_is_compatible(epsilon, mode, amplitude) for _, epsilon, mode, _, _ in tasks)
    runs, records = [], []
    for index, expected_task in enumerate(tasks):
        path = source_directory / "runs" / f"task_{index:05d}.json"
        run = json.loads(path.read_text())
        task = run["task"]
        actual_task = tuple(task[key] for key in ("family", "epsilon", "target_mode", "width", "seed"))
        if (run.get("schema_version") != 1 or run.get("artifact_type") != "rank_conditioning_control"
                or run.get("config_sha256") != expected_config_hash or run.get("source_sha256") != expected_source_hash
                or run.get("status") != "complete" or run.get("task_index") != index or actual_task != expected_task):
            raise ValueError(f"inconsistent original task/config/source: {path}")
        compatible = target_is_compatible(task["epsilon"], task["target_mode"], amplitude)
        if task.get("contrast_amplitude") != amplitude or task.get("target_functionally_compatible") is not compatible:
            raise ValueError(f"inconsistent target compatibility: {path}")
        expected_rank = 1 if task["epsilon"] == 0 else 2
        expected_floor = 0.0 if compatible else amplitude ** 2
        if task.get("exact_structural_rank") != expected_rank or task.get("exact_structural_loss_floor") != expected_floor or run["summary"].get("exact_structural_loss_floor") != expected_floor:
            raise ValueError(f"inconsistent structural metadata: {path}")
        array_path = path.with_suffix(".npz")
        if run.get("arrays_file") != array_path.name or not array_path.is_file() or _sha256(array_path) != run.get("arrays_sha256"):
            raise ValueError(f"missing/corrupt original arrays: {array_path}")
        _validate_history(run, array_path, config)
        runs.append(run)
        records.append({"task_index": index, "json_file": str(path), "json_sha256": _sha256(path),
                        "arrays_file": str(array_path), "arrays_sha256": run["arrays_sha256"],
                        "included": compatible})
    # Selection only occurs after every original run has passed validation.
    included = [run for run in runs if run["task"]["target_functionally_compatible"]]
    excluded = [{"task_index": run["task_index"], "reason": "duplicate rows with nonzero antisymmetric target shift; target_functionally_compatible=false"}
                for run in runs if not run["task"]["target_functionally_compatible"]]
    if len(included) != expected_included:
        raise ValueError("included count disagrees with the original task grid")
    selection = {"rule": "retain target_functionally_compatible=true, cross-checked against row mixing and amplitude",
                 "source_config": str(config_path), "source_config_sha256": expected_config_hash,
                 "source_config_file_sha256": _sha256(config_path), "source_snapshot": str(snapshot_path),
                 "source_snapshot_sha256": _sha256(snapshot_path), "source_artifact_directory": str(source_directory),
                 "source_expected_runs": len(tasks), "source_validated_runs": len(runs),
                 "expected_included_runs": expected_included, "included_runs": len(included),
                 "included_task_indices": [run["task_index"] for run in included],
                 "excluded_runs": len(excluded), "excluded_tasks": excluded,
                 "source_grid_includes_incompatible": archive_includes_incompatible,
                 "validation": "entire frozen snapshot; config/source/task identity; all NPZ hashes; full checkpoints; JSON/NPZ losses and endpoints; target compatibility",
                 "analysis_sha256": _sha256(Path(__file__)),
                 "source_artifacts_modified": False}
    result = _write_aggregate(included, config, destination, expected_included, [], expected_source_hash, selection=selection)
    atomic_json(destination / "selection_provenance.json", {**selection, "source_artifacts": records})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a frozen campaign and report only functionally compatible rank controls.")
    parser.add_argument("--source-config", required=True, help="Original controls.json beside the frozen snapshot.json")
    parser.add_argument("--output-dir", required=True, help="Separate directory for compatible plots, tables and provenance")
    args = parser.parse_args()
    aggregate_compatible(args.source_config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
