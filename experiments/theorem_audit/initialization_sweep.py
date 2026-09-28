"""Many-seed finite-width concentration audit, independent of training seeds.

Run ``python -m experiments.theorem_audit.initialization_sweep --config FILE
--task-index 0 --seeds 100`` once per configured PDE, then replace the optional
action with ``aggregate`` to write all-seed CSVs and publication figures.
Every width uses a prefix of the same IID bank within a seed. No optimization
is performed and no initialization is selected based on its eigengap.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.theorem_audit.aggregate import (
    DISPLAY_NAMES, _group_width_metric, _plot_setup, _save_figure, _write_csv,
)
from experiments.theorem_audit.engine import evaluate_finite, make_initialization_bank
from experiments.theorem_audit.features import compile_constraints
from experiments.theorem_audit.problems import get_problem
from experiments.theorem_audit.run import (
    _atomic_json, _common_manifest, _configure_runtime, _kernel_concentration,
    _load_reference, _sha256_file, _source_fingerprint, _strict_json_load, load_config,
)


def sweep_specification(config: dict[str, Any], seeds: int, seed_offset: int) -> dict[str, Any]:
    if seeds < 2 or seed_offset < 0:
        raise ValueError("at least two seeds and a non-negative seed offset are required")
    seed_values = list(range(seed_offset, seed_offset + seeds))
    if set(seed_values).intersection(config["training"]["seeds"]):
        raise ValueError("initialization-only seeds must be disjoint from training seeds")
    return {
        "seeds": seed_values,
        "widths": list(config["training"]["widths"]),
        "nested_prefix": True,
        "relative_rank_tolerance": float(config["reference"]["relative_tolerance"]),
        "sweep_source_sha256": _sha256_file(Path(__file__)),
    }


def run_task(config: dict[str, Any], task_index: int, spec: dict[str, Any]) -> Path:
    if not 0 <= task_index < len(config["problems"]):
        raise ValueError("task-index must be a configured problem index")
    device = _configure_runtime(config)
    problem = get_problem(config["problems"][task_index])
    _, reference, reference_path = _load_reference(config, problem, device)
    compiled = compile_constraints(problem, device=device)
    rows = []
    for seed in spec["seeds"]:
        bank = make_initialization_bank(
            max(spec["widths"]), problem.input_dim, problem.output_dim, seed, device=device,
        )
        for width in spec["widths"]:
            evaluation = evaluate_finite(bank.prefix(width), compiled)
            concentration = _kernel_concentration(
                evaluation.kernel, reference, spec["relative_rank_tolerance"],
            )
            spectrum = concentration["initial_spectrum"]
            ref_spectrum = concentration["reference_spectrum_recomputed"]
            reference_resolved = ref_spectrum["lambda_min"] > ref_spectrum["rank_tolerance"]
            rows.append({
                "problem": problem.name, "status": "complete", "width": width, "seed": seed,
                "constraint_count": compiled.count,
                "parameter_count": width * (problem.input_dim + 1 + problem.output_dim),
                "initial_loss": float(evaluation.loss.cpu()),
                "lambda_min": spectrum["lambda_min"], "lambda_max": spectrum["lambda_max"],
                "numerical_rank": spectrum["numerical_rank"],
                "rank_tolerance": spectrum["rank_tolerance"],
                "numerical_positive": spectrum["lambda_min"] > spectrum["rank_tolerance"],
                "eigenvalues": spectrum["eigenvalues"],
                "relative_operator_error": concentration["relative_operator_error"],
                "relative_frobenius_error": concentration["relative_frobenius_error"],
                "lambda_min_ratio_to_reference": concentration["lambda_min_ratio_to_reference"],
                "generalized_lambda_min": concentration["generalized_lambda_min"],
                "loewner_three_quarters": concentration["loewner_K0_ge_three_quarters_Kinf"],
                "scalar_gap_three_quarters": (
                    spectrum["lambda_min"] >= 0.75 * ref_spectrum["lambda_min"]
                    if reference_resolved else None
                ),
                "operator_quarter_gap_event": (
                    concentration["absolute_operator_error"] <= 0.25 * ref_spectrum["lambda_min"]
                    if reference_resolved else None
                ),
            })
        if (seed - spec["seeds"][0] + 1) % 10 == 0:
            print(f"[initialization] {problem.name}: {seed - spec['seeds'][0] + 1}/{len(spec['seeds'])} seeds", flush=True)
    path = Path(config["_resolved_output_dir"]) / "initialization_sweep" / f"{problem.name}.json"
    _atomic_json(path, {
        **_common_manifest(config, device, _source_fingerprint()),
        "artifact_type": "independent_initialization_sweep", "status": "complete",
        "problem_name": problem.name, "initialization_specification": spec,
        "reference_npz_sha256": _sha256_file(reference_path), "rows": rows,
    })
    print(f"[initialization] wrote {len(rows)} width/seed measurements to {path}", flush=True)
    return path


def aggregate(config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["_resolved_output_dir"])
    source_hash = _source_fingerprint()["sha256"]
    device = _configure_runtime(config)
    rows: list[dict[str, Any]] = []
    issues = []
    for problem_name in config["problems"]:
        path = output / "initialization_sweep" / f"{problem_name}.json"
        try:
            artifact = _strict_json_load(path)
            expected = {(width, seed) for width in spec["widths"] for seed in spec["seeds"]}
            data = artifact.get("rows", [])
            if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
                raise ValueError("malformed measurement rows")
            observed = {(row.get("width"), row.get("seed")) for row in data}
            if (
                artifact.get("status") != "complete"
                or artifact.get("config_hash") != config["_config_hash"]
                or artifact.get("source", {}).get("sha256") != source_hash
                or artifact.get("initialization_specification") != spec
                or artifact.get("problem_name") != problem_name
                or len(data) != len(expected) or observed != expected
                or any(row.get("problem") != problem_name or row.get("status") != "complete" for row in data)
            ):
                raise ValueError("incomplete grid or incompatible configuration/source/specification")
            _, _, reference_path = _load_reference(config, get_problem(problem_name), device)
            if artifact.get("reference_npz_sha256") != _sha256_file(reference_path):
                raise ValueError("reference artifact differs from the one measured")
            for row in data:
                if not all(
                    isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                    for key in ("lambda_min", "lambda_max", "initial_loss", "relative_operator_error")
                ):
                    raise ValueError("non-finite or missing measurement")
            rows.extend(data)
        except (OSError, ValueError, RuntimeError) as error:
            issues.append(f"{problem_name}: {error}")

    _write_csv(output / "tables" / "initialization_seeds.csv", rows, ("problem", "width", "seed"))
    fields = (
        ("relative_operator_error", False), ("relative_frobenius_error", False),
        ("numerical_positive", True), ("loewner_three_quarters", True),
        ("scalar_gap_three_quarters", True), ("operator_quarter_gap_event", True),
    )
    summaries = []
    numeric_rows = [dict(row) for row in rows]
    for row in numeric_rows:
        for key, probability in fields:
            if probability and row.get(key) is not None:
                row[key] = float(row[key])
    for problem in config["problems"]:
        for field, probability in fields:
            for width, center, lower, upper, count in _group_width_metric(
                numeric_rows, problem, field, probability=probability,
            ):
                summaries.append({
                    "problem": problem, "width": width, "metric": field,
                    "center": center, "lower": lower, "upper": upper, "seed_count": count,
                    "interval": "pointwise Wilson 95%" if probability else "interquartile range",
                })
    _write_csv(output / "tables" / "initialization_summary.csv", summaries, ("problem", "width", "metric"))
    plt = _plot_setup()
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))
    panels = (
        ("relative_operator_error", "(a) Finite-width NTK concentration", r"$\|K_0-\widehat K^\infty\|_2/\|\widehat K^\infty\|_2$"),
        ("numerical_positive", "(b) Numerically resolved positive definiteness", "fraction with minimum eigenvalue above tolerance"),
        ("loewner_three_quarters", "(c) Generalized spectral initialization event", r"fraction with $K_0\succeq0.75\widehat K^\infty$"),
        ("scalar_gap_three_quarters", "(d) Minimum-eigenvalue initialization event", r"fraction with $\lambda_{\min}(K_0)\geq0.75\widehat\lambda_{\min}$"),
    )
    cmap = plt.get_cmap("tab20")
    for index, (field, title, ylabel) in enumerate(panels):
        axis = axes.flat[index]
        for problem_index, problem in enumerate(config["problems"]):
            group = sorted((row for row in summaries if row["problem"] == problem and row["metric"] == field), key=lambda row: row["width"])
            if not group:
                continue
            x = np.asarray([row["width"] for row in group])
            center = np.asarray([row["center"] for row in group])
            lower = np.asarray([row["lower"] for row in group])
            upper = np.asarray([row["upper"] for row in group])
            axis.plot(x, center, marker="o", markersize=3, color=cmap(problem_index), label=DISPLAY_NAMES.get(problem, problem))
            axis.fill_between(x, lower, upper, color=cmap(problem_index), alpha=0.10)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("network width m")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if index == 0:
            axis.set_yscale("log")
        else:
            axis.set_ylim(-0.04, 1.04)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    figure.suptitle(f"Initialization-only audit: {len(spec['seeds'])} independent seeds per PDE, paired widths")
    figure.subplots_adjust(hspace=0.35, wspace=0.30, bottom=0.20)
    figures = _save_figure(figure, output / "figures" / "initialization_probability")
    plt.close(figure)
    caption = (
        "Independent initialization-only seed banks, disjoint from training seeds; all widths use nested prefixes. "
        "Panel (a): median relative operator error with interquartile bands. Panels (b-d): observed fractions "
        "with pointwise Wilson 95% intervals across seeds (not simultaneous intervals across PDEs or widths). "
        f"Panel (b) requires lambda_min(K0) > {spec['relative_rank_tolerance']:g} lambda_max(K0); "
        "a failure means numerical positivity was unresolved, not a proof of singularity. "
        "Panels (c,d) require a numerically resolved reference gap. The Loewner event is stronger than the "
        "scalar minimum-eigenvalue event; neither curve establishes the theorem's sufficient width constant. "
        "The additional sufficient operator quarter-gap event is stored in initialization_summary.csv. "
        "Reference integration error is not included in Wilson intervals. Missing or rejected PDE artifacts "
        "are listed in initialization_aggregate.json. No initialization was filtered by loss or eigengap.\n"
    )
    caption_path = output / "figures" / "initialization_probability_caption.md"
    caption_path.write_text(caption, encoding="utf-8")
    result = {
        "config_hash": config["_config_hash"], "source_hash": source_hash,
        "initialization_specification": spec,
        "expected_measurements": len(config["problems"]) * len(spec["widths"]) * len(spec["seeds"]),
        "accepted_measurements": len(rows), "issues": issues,
        "summary": summaries, "figures": figures, "figure_caption": str(caption_path),
    }
    _atomic_json(output / "initialization_aggregate.json", result)
    print(f"[initialization] accepted {len(rows)}/{result['expected_measurements']} measurements; {len(issues)} issues", flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run", "aggregate"), default="run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=10000)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    spec = sweep_specification(config, args.seeds, args.seed_offset)
    if args.action == "aggregate":
        return int(bool(aggregate(config, spec)["issues"]))
    if args.task_index is None:
        parser.error("run requires --task-index (configured PDE index)")
    run_task(config, args.task_index, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
