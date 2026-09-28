"""Plot the appendix active/nullspace loss decomposition.

Run from the repository root with the locked environment::

    python -m experiments.theorem_audit.plot_joint_nullspace_evidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_supportive_evidence import (
    TRAJECTORY_DISPLAY_FLOOR,
    _load_joint_trajectories,
    _plot_nullspace_trajectories,
    _sha256,
)


def make_figure(joint: dict) -> tuple[plt.Figure, dict]:
    """Return the self-contained appendix figure and its plotted statistics."""

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
    figure = plt.figure(figsize=(5.5, 2.55))
    axes = figure.add_subplot(1, 1, 1)
    summary = _plot_nullspace_trajectories(axes, joint)
    axes.set_title(
        "Exact PDE-row relation: active loss decays; nullspace loss persists\n"
        "18 paired settings; curves are medians and the band is the interquartile range"
    )
    axes.set_xlim(0.0, 6000.0)
    axes.set_xticks(
        (0, 10, 100, 1000, 5000),
        ("0", "10", "$10^2$", "$10^3$", "$5\\times10^3$"),
    )
    figure.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.21)
    return figure, summary


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    default_input = (
        project_root
        / "experiments/theorem_audit/data"
        / "joint_nullspace_controls.json"
    )
    parser = argparse.ArgumentParser(
        description="Generate the appendix joint-nullspace loss figure."
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=project_root / "figures")
    parser.add_argument("--stem", default="joint_nullspace_loss_decomposition")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    joint = _load_joint_trajectories(input_path)
    figure, summary = make_figure(joint)
    script_path = Path(__file__).resolve()
    helper_path = script_path.with_name("plot_supportive_evidence.py")
    metadata = {
        "Creator": "experiments/theorem_audit/plot_joint_nullspace_evidence.py",
        "Title": "Active- and nullspace loss for exact PDE-row relations",
        "Subject": f"Joint-nullspace input SHA256: {_sha256(input_path)}",
        "CreationDate": None,
        "ModDate": None,
    }
    pdf_path = output_dir / f"{args.stem}.pdf"
    png_path = output_dir / f"{args.stem}.png"
    temporary_pdf = output_dir / f".{args.stem}.tmp.pdf"
    temporary_png = output_dir / f".{args.stem}.tmp.png"
    figure_inches = [float(value) for value in figure.get_size_inches()]
    figure.savefig(temporary_pdf, format="pdf", metadata=metadata)
    figure.savefig(temporary_png, format="png", dpi=220)
    plt.close(figure)
    temporary_pdf.replace(pdf_path)
    temporary_png.replace(png_path)

    record = {
        "schema_version": 1,
        "joint_nullspace_input": input_path.relative_to(project_root).as_posix(),
        "joint_nullspace_input_sha256": _sha256(input_path),
        "analysis_script": script_path.relative_to(project_root).as_posix(),
        "analysis_script_sha256": _sha256(script_path),
        "plot_helper": helper_path.relative_to(project_root).as_posix(),
        "plot_helper_sha256": _sha256(helper_path),
        "figure_size_inches": figure_inches,
        "paired_settings": joint["sample_sizes"]["paired_gd_settings"],
        "gd_trajectories": joint["sample_sizes"]["gd_trajectories"],
        "plotted_statistics": "median and interquartile range over paired settings",
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
        "display_floors": {"active_subspace_loss": TRAJECTORY_DISPLAY_FLOOR},
    }
    summary_path = output_dir / f"{args.stem}_summary.json"
    temporary_summary = output_dir / f".{args.stem}_summary.tmp.json"
    temporary_summary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary_summary.replace(summary_path)
    print(
        "validated and plotted "
        f"{joint['sample_sizes']['gd_trajectories']} joint-nullspace trajectories"
    )
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
