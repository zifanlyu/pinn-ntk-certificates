"""Write evidence-only campaign findings from explicitly configured run cells.

Run from the project root (including a frozen campaign snapshot):
    python -m experiments.theorem_audit.report_campaign --config /path/broad.json

No result globbing or cached aggregate is used. Missing, rejected, and skipped
cells remain in the coverage denominator. This report does not submit jobs.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.theorem_audit import aggregate as audit
else:
    from . import aggregate as audit


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _finite(row.get(key))) is not None]
    return float(np.median(values)) if values else None


def _fmt(value: Any) -> str:
    value = _finite(value)
    return "--" if value is None else f"{value:.3g}"


def _tex(value: Any) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(character, character) for character in str(value))


def _project_source_hash(project_root: Path) -> str:
    """Hash the selected project's SOURCE_FILES without executing snapshot code."""
    runner = project_root / "experiments/theorem_audit/run.py"
    syntax = ast.parse(runner.read_text(encoding="utf-8"))
    declarations = [node for node in syntax.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "SOURCE_FILES" for target in node.targets)]
    if len(declarations) != 1:
        raise ValueError(f"cannot identify unique SOURCE_FILES in {runner}")
    names = ast.literal_eval(declarations[0].value)
    if not isinstance(names, (tuple, list)) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError("SOURCE_FILES must be a nonempty literal sequence of paths")
    snapshot_path = project_root / "snapshot.json"
    snapshot = audit._load_json(snapshot_path) if snapshot_path.is_file() else None
    combined = hashlib.sha256()
    for relative in names:
        data = (project_root / relative).read_bytes()
        if snapshot is not None and snapshot.get("files", {}).get(relative) != hashlib.sha256(data).hexdigest():
            raise ValueError(f"frozen source changed or unregistered: {relative}")
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(data)
        combined.update(b"\0")
    return combined.hexdigest()


def collect(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    """Reuse strict artifact readers, enumerating only the active config cells."""
    config = audit._load_json(config_path)
    problems = config.get("problems")
    training = config.get("training", {})
    widths, seeds = training.get("widths"), training.get("seeds")
    if not isinstance(problems, list) or not problems or not all(isinstance(p, str) for p in problems):
        raise ValueError("config.problems must be a nonempty list of names")
    if not isinstance(widths, list) or not widths or not all(type(w) is int and w > 0 for w in widths):
        raise ValueError("config.training.widths must be positive integers")
    if not isinstance(seeds, list) or not seeds or not all(type(s) is int for s in seeds):
        raise ValueError("config.training.seeds must be integers")
    if any(len(set(items)) != len(items) for items in (problems, widths, seeds)):
        raise ValueError("duplicate problems, widths, or seeds would duplicate expected cells")
    steps = training.get("steps")
    if type(steps) is not int or steps < 0:
        raise ValueError("config.training.steps must be a nonnegative integer")
    setting = config.get("output_dir")
    if not isinstance(setting, str) or not setting:
        raise ValueError("config.output_dir is required")
    if project_root is None:
        # Frozen campaign configs live beside snapshot.json at their project
        # root. They can be analyzed by a newer reporting-only script without
        # altering the frozen training source or resolving into the live repo.
        project_root = config_path.parent if (config_path.parent / "snapshot.json").is_file() else audit.REPO_ROOT
    output_dir = Path(setting)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()
    source_hash = _project_source_hash(project_root.resolve())
    config_hash, _ = audit._resolve_manifest(config, output_dir)
    references, runs, issues = [], [], []
    for problem in problems:
        path, error = audit._locate_unique(audit._reference_candidates(output_dir, problem))
        reference, _, issue = audit._reference_row(problem, path, config_hash, error, expected_source_hash=source_hash)
        references.append(reference)
        if issue:
            issues.append(issue)
        for width in widths:
            for seed in seeds:
                path, error = audit._locate_unique(audit._run_candidates(output_dir, problem, width, seed))
                row, history, issue = audit._run_row(
                    problem, width, seed, path, config_hash, error, reference,
                    expected_steps=steps, expected_source_hash=source_hash
                )
                if issue:
                    issues.append(issue)
                if history and row["status"] in audit.COMPLETE_STATUSES:
                    first, last = history[0], history[-1]
                    l0 = _finite(first.get("loss"))
                    lmin, lmax = _finite(first.get("lambda_min")), _finite(first.get("lambda_max"))
                    row["initial_relative_gap"] = lmin / lmax if lmin is not None and lmax and lmax > 0 else None
                    frozen = _finite(last.get("frozen_kernel_loss"))
                    row["frozen_final_over_initial_loss"] = frozen / l0 if frozen is not None and l0 and l0 > 0 else None
                    row["completed_steps"] = last["step"]
                    reference_gap = _finite(reference.get("lambda_min"))
                    row["scalar_three_quarter_gap_passed"] = (
                        lmin >= 0.75 * reference_gap
                        if lmin is not None and reference_gap and reference_gap > 0 else None
                    )
                    ratio, frozen_ratio = row.get("final_over_initial_loss"), row["frozen_final_over_initial_loss"]
                    row["actual_over_frozen_final_loss"] = (
                        ratio / frozen_ratio if ratio is not None and frozen_ratio and frozen_ratio > 0 else None
                    )
                runs.append(row)
    return {"config": config, "config_path": str(config_path), "config_hash": config_hash,
            "source_hash": source_hash, "project_root": str(project_root.resolve()),
            "reporting_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "output_dir": str(output_dir), "references": references, "runs": runs,
            "issues": issues, "created_utc": datetime.now(timezone.utc).isoformat()}


def _summaries(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config, runs = report["config"], report["runs"]
    groups, widest = [], []
    for problem in config["problems"]:
        problem_groups = []
        for width in config["training"]["widths"]:
            expected = [r for r in runs if r["problem"] == problem and r["width"] == width]
            complete = [r for r in expected if r["status"] in audit.COMPLETE_STATUSES]
            summary = {"problem": problem, "width": width, "completed": len(complete),
                       "expected": len(expected), "status_counts": dict(Counter(r["status"] for r in expected))}
            for key in ("final_over_initial_loss", "frozen_final_over_initial_loss", "initial_relative_gap",
                        "max_kernel_drift_operator", "initial_reference_error_operator",
                        "max_sqrt_width_per_neuron_drift", "actual_over_frozen_final_loss"):
                summary[f"median_{key}"] = _median(complete, key)
            scalar = [r["scalar_three_quarter_gap_passed"] for r in complete
                      if r.get("scalar_three_quarter_gap_passed") is not None]
            summary["scalar_initial_gap_passes"] = sum(scalar)
            summary["scalar_initial_gap_evaluable"] = len(scalar)
            groups.append(summary)
            problem_groups.append(summary)
        available = [g for g in problem_groups if g["completed"] > 0]
        widest.append(max(available or problem_groups, key=lambda g: g["width"]))
    return groups, widest


def plot_conditioning(report: dict[str, Any], output_dir: Path) -> list[str]:
    """PDE facets avoid interpreting pooled between-PDE differences as causality."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    problems = report["config"]["problems"]
    complete = [r for r in report["runs"] if r["status"] in audit.COMPLETE_STATUSES]
    fig, axes = plt.subplots(math.ceil(len(problems) / 3), 3, figsize=(15, 3.5 * math.ceil(len(problems) / 3)), squeeze=False)
    drifts = [_finite(r.get("max_kernel_drift_operator")) for r in complete]
    positive = [d for d in drifts if d is not None and d > 0]
    vmin = max(min(positive, default=1e-6), 1e-12)
    vmax = max(max(positive, default=1.0), vmin * 10)
    norm, cmap = LogNorm(vmin=vmin, vmax=vmax), plt.get_cmap("viridis")
    widths = sorted(report["config"]["training"]["widths"])
    sizes = {w: 25 + 70 * index / max(1, len(widths) - 1) for index, w in enumerate(widths)}
    plotted = omitted = 0
    for ax, problem in zip(axes.flat, problems):
        local, local_omitted = [r for r in complete if r["problem"] == problem], 0
        for row in local:
            gap = _finite(row.get("initial_relative_gap"))
            actual, frozen = _finite(row.get("final_over_initial_loss")), _finite(row.get("frozen_final_over_initial_loss"))
            drift = _finite(row.get("max_kernel_drift_operator"))
            if gap is None or gap <= 0 or actual is None or frozen is None or drift is None:
                local_omitted += 1
                continue
            color, size = cmap(norm(max(drift, vmin))), sizes[row["width"]]
            actual, frozen = max(actual, 1e-300), max(frozen, 1e-300)
            ax.plot([gap, gap], [actual, frozen], color=color, alpha=.45, linewidth=.8)
            ax.scatter(gap, actual, s=size, color=[color], marker="o", alpha=.8, edgecolors="black", linewidths=.25)
            ax.scatter(gap, frozen, s=size, color=[color], marker="+", linewidths=1.2)
            plotted += 1
        omitted += local_omitted
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{audit.DISPLAY_NAMES.get(problem, problem)} (n={len(local)}, omitted={local_omitted})", fontsize=10)
        ax.set_xlabel(r"Initial inverse condition number $\lambda_{\min}/\lambda_{\max}$")
        ax.set_ylabel(r"Final loss / initial loss $L_T/L_0$")
        ax.grid(True, which="both", alpha=.18)
        if not local:
            ax.text(.5, .5, "No accepted completed runs", ha="center", transform=ax.transAxes)
    for ax in list(axes.flat)[len(problems):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color="black", marker="o", linestyle="None", label="Nonlinear GD"),
               Line2D([], [], color="black", marker="+", linestyle="None", label="Frozen initial kernel")]
    handles.extend(Line2D([], [], color="gray", marker="o", linestyle="None", markersize=math.sqrt(sizes[w]), label=f"width {w}") for w in widths)
    fig.legend(handles=handles, loc="upper center", ncol=min(5, len(handles)), bbox_to_anchor=(.46, .97), fontsize=9)
    fig.suptitle("Conditioning and observed loss reduction, paired with the frozen-kernel prediction", y=.999, fontsize=14)
    fig.subplots_adjust(top=.9, bottom=.075, hspace=.45, wspace=.35, right=.88)
    colorbar_ax = fig.add_axes([.91, .17, .018, .62])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    colorbar.set_label(r"Maximum checkpoint kernel drift $\|K_k-K_0\|_{op}/\|K_0\|_{op}$")
    training = report["config"]["training"]
    fig.text(.015, .012, f"Biased two-layer tanh; all parameters trained; float64; loss = ||P-y||². "
             f"T={training['steps']}; eta={training['learning_rate_factor']}/lambda_max(K0).\n"
             f"Each marker is one seed; vertical line pairs actual/frozen losses; smaller loss ratio is better. "
             f"{plotted} plotted; {omitted} omitted for unavailable/nonpositive spectral data. No pooled fit.", fontsize=9)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "conditioning_loss_relationship.pdf", output_dir / "conditioning_loss_relationship.png"]
    for path in paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return [str(path) for path in paths]


def write_report(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    report = collect(config_path.resolve(), project_root)
    output_dir = Path(report["output_dir"])
    config = report["config"]
    groups, widest = _summaries(report)
    complete = [r for r in report["runs"] if r["status"] in audit.COMPLETE_STATUSES]
    statuses = dict(Counter(r["status"] for r in report["runs"]))
    references_complete = sum(r["artifact_status"] in audit.COMPLETE_STATUSES for r in report["references"])
    expected = len(report["runs"])
    heading = (f"Completed {len(complete)}/{expected} configured training runs and "
               f"{references_complete}/{len(report['references'])} reference kernels.")
    setup = (f"Configuration {config.get('name', 'unnamed')}: {len(config['problems'])} PDEs, widths "
             f"{config['training']['widths']}, {len(config['training']['seeds'])} seeds, "
             f"{config['training']['steps']} full-batch GD steps. All parameters of the biased "
             f"two-layer tanh network are trained in float64; loss is ||P-y||² and "
             f"eta={config['training']['learning_rate_factor']}/lambda_max(K0).")
    limitations = [
        "These summaries describe completed numerical runs. Unfinished cells remain in the coverage denominator; missing work is not a failed convergence experiment.",
        "All registered PDE designs have analytic independence routes under the paper's assumptions. QMC estimates the gap; an unresolved or nonpositive numerical eigenvalue does not refute limiting positive definiteness.",
        "The spectral learning-rate rule and tested widths do not verify the theorem's deterministic derivative bounds, sufficient width inequality, or all-iteration guarantee.",
        "Positive definiteness alone does not predict finite-budget loss reduction: the relative gap, initial modal energy, finite width, and kernel drift also matter. The plot is descriptive and does not fit a pooled causal relationship across PDEs.",
        "The largest width reported for a PDE is the largest width with accepted completed data; its seed count can be incomplete. Full per-PDE/per-width coverage is in findings_by_problem_width.csv.",
        "Kernel drift and gap preservation are checked at recorded checkpoints. Kernel drift is normalized by the largest eigenvalue and can be small while exceeding the smallest eigenvalue.",
        "Weak, nonlocal, and Caputo experiments optimize their stored finite quadrature constraints. Training loss does not establish continuum solution accuracy or generalization.",
        "The existing main-paper conditioning run uses a different output-bias and loss convention. Its numerical values are not combined with this campaign.",
    ]
    md = [f"# Campaign findings: {config.get('name', 'unnamed')}", "", heading, "", setup, "",
          f"Status counts: `{json.dumps(statuses, sort_keys=True)}`.", "",
          "Values below are seed medians at the widest completed width for each PDE. Loss columns are final/initial loss; lower is better. Drift is the maximum recorded operator-norm change relative to the initial kernel, computed per seed before taking the median.", "",
          "| PDE | Width | Seeds done/expected | Initial relative gap | GD loss ratio | Frozen loss ratio | Kernel drift |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    latex_rows = []
    for row in widest:
        name = audit.DISPLAY_NAMES.get(row["problem"], row["problem"])
        cells = [name, str(row["width"]) if row["completed"] else "--", f"{row['completed']}/{row['expected']}",
                 _fmt(row["median_initial_relative_gap"]), _fmt(row["median_final_over_initial_loss"]),
                 _fmt(row["median_frozen_final_over_initial_loss"]), _fmt(row["median_max_kernel_drift_operator"])]
        md.append("| " + " | ".join(cells) + " |")
        latex_rows.append(" & ".join(_tex(cell) for cell in cells) + r" \\")
    improved = sum(r.get("final_over_initial_loss") is not None and r["final_over_initial_loss"] < 1 for r in complete)
    tenfold = sum(r.get("final_over_initial_loss") is not None and r["final_over_initial_loss"] <= .1 for r in complete)
    outcomes = (f"Of the {len(complete)} completed runs, {improved} reduce training loss and {tenfold} "
                "reduce it by at least a factor of ten within the configured budget. These counts include all PDEs and widths and are descriptive, not a test of a common convergence rate.")
    md.extend(["", outcomes, "", "![Conditioning versus loss reduction](figures/conditioning_loss_relationship.png)", "", "Interpretation limits:", ""])
    md.extend(f"- {item}" for item in limitations)
    md.extend(["", f"Configuration SHA-256: `{report['config_hash']}`.",
               f"Report generated: {report['created_utc']}. See findings.json for every expected cell and issue.", ""])
    figures = plot_conditioning(report, output_dir / "figures")
    tex = [r"\documentclass[10pt]{article}", r"\usepackage[margin=0.7in]{geometry}",
           r"\usepackage{booktabs,graphicx,amsmath,hyperref}", r"\begin{document}",
           r"\section*{Campaign findings: " + _tex(config.get("name", "unnamed")) + "}", _tex(heading), "",
           _tex(setup.replace("²", "^2")), "", r"\paragraph{Widest completed width per PDE.}",
           r"Seed medians; loss columns are $L_T/L_0$. Drift is the maximum recorded relative operator-norm change.",
           r"\begin{center}\small\begin{tabular}{lrrrrrr}\toprule",
           r"PDE & Width & Seeds & Relative gap & GD loss & Frozen loss & Drift \\", r"\midrule",
           *latex_rows, r"\bottomrule\end{tabular}\end{center}", _tex(outcomes), "",
           r"\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{\detokenize{" + str(Path(figures[0]).relative_to(output_dir)) + "}}",
           r"\caption{Conditioning and final loss, separated by PDE. Circles show nonlinear GD, plus signs show the initial frozen kernel, marker size denotes width, and color denotes checkpoint kernel drift. Each seed is shown; no pooled fit is made.}\end{figure}",
           r"\paragraph{Interpretation limits.}\begin{itemize}",
           *[r"\item " + _tex(item) for item in limitations], r"\end{itemize}",
           r"\noindent Configuration SHA-256: \texttt{" + _tex(report["config_hash"]) + "}.",
           r"\end{document}", ""]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "findings.md").write_text("\n".join(md), encoding="utf-8")
    (output_dir / "findings.tex").write_text("\n".join(tex), encoding="utf-8")
    report.update({"coverage": {"completed": len(complete), "expected": expected, "status_counts": statuses},
                   "by_problem_width": groups, "widest_completed": widest, "figures": figures,
                   "interpretation_limits": limitations})
    audit._write_json(output_dir / "findings.json", report)
    audit._write_csv(output_dir / "findings_by_problem_width.csv", groups, ("problem", "width", "completed", "expected"))
    return {"findings": str(output_dir / "findings.md"), "coverage": report["coverage"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, help="Root for relative output_dir; campaign snapshots are detected automatically")
    args = parser.parse_args()
    print(json.dumps(write_report(args.config, args.project_root), indent=2))


if __name__ == "__main__":
    main()
