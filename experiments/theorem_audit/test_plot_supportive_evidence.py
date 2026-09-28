"""Regression checks for the manuscript's finite-width GD evidence figure.

Run from the repository root with

    python -m unittest experiments.theorem_audit.test_plot_supportive_evidence
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt

from .plot_supportive_evidence import (
    _load_frozen_aggregate,
    make_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("plot_supportive_evidence.py")
INPUT = (
    PROJECT_ROOT
    / "experiments/theorem_audit/data"
    / "compatible_rank_convergence_conditioning.json"
)
SUMMARY = PROJECT_ROOT / "figures/compatible_rank_convergence_conditioning_summary.json"
PDF = PROJECT_ROOT / "figures/compatible_rank_convergence_conditioning.pdf"
PNG = PROJECT_ROOT / "figures/compatible_rank_convergence_conditioning.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SupportiveEvidencePlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aggregate = _load_frozen_aggregate(INPUT)

    def test_plotted_counts_match_manuscript_claims(self) -> None:
        figure, summary = make_figure(self.aggregate)
        try:
            self.assertEqual(list(figure.get_size_inches()), [5.5, 4.1])
            self.assertEqual(len(figure.axes), 2)
            width_diagnostics = summary["width_diagnostics"]
            self.assertAlmostEqual(
                width_diagnostics["16"]["median_initial_relative_kernel_error"],
                2.23e-1,
            )
            self.assertAlmostEqual(
                width_diagnostics["16384"]["maximum_relative_kernel_drift"],
                8.62e-4,
            )
            self.assertAlmostEqual(
                width_diagnostics["16384"]["final_frozen_trajectory_discrepancy"],
                6.91e-3,
            )
            self.assertEqual(
                [
                    width_diagnostics[str(width)]["gap_retained_at_saved_checkpoints"]
                    for width in (16, 256, 4096, 16384)
                ],
                [17, 110, 110, 110],
            )

            pointwise = summary["conditioning_width_16384"]["pointwise_poisson"]
            self.assertAlmostEqual(
                pointwise["0.1"]["median_relative_gap"],
                0.0024158522567237738,
            )
            self.assertAlmostEqual(
                pointwise["0.01"]["median_final_to_initial_loss_ratio"],
                0.17312564280154474,
            )
        finally:
            plt.close(figure)

    def test_figure_remains_legible_at_manuscript_width(self) -> None:
        figure, _ = make_figure(self.aggregate)
        try:
            visible_text = [
                item
                for axes in figure.axes
                for item in (
                    *axes.texts,
                    *axes.get_xticklabels(),
                    *axes.get_yticklabels(),
                    axes.title,
                    axes.xaxis.label,
                    axes.yaxis.label,
                )
                if item.get_text().strip()
            ]
            self.assertGreater(len(visible_text), 25)
            self.assertGreaterEqual(min(item.get_fontsize() for item in visible_text), 7.0)
            self.assertEqual(len(figure.legends), 1)
            self.assertIsNotNone(figure.axes[1].get_legend())
            with tempfile.TemporaryDirectory() as temporary_directory:
                output = Path(temporary_directory) / "supportive_evidence.png"
                figure.savefig(output, dpi=120)
                self.assertGreater(output.stat().st_size, 10_000)
        finally:
            plt.close(figure)

    def test_committed_artifacts_match_provenance_record(self) -> None:
        record = json.loads(SUMMARY.read_text())
        self.assertEqual(record["schema_version"], 5)
        self.assertEqual(
            record["aggregate_input"],
            "experiments/theorem_audit/data/compatible_rank_convergence_conditioning.json",
        )
        self.assertEqual(record["aggregate_input_sha256"], _sha256(INPUT))
        self.assertEqual(record["analysis_script_sha256"], _sha256(SCRIPT))
        self.assertFalse(Path(record["analysis_script"]).is_absolute())
        for name, path in (("pdf", PDF), ("png", PNG)):
            self.assertEqual(record["outputs"][name]["sha256"], _sha256(path))
            self.assertEqual(record["outputs"][name]["bytes"], path.stat().st_size)
            self.assertFalse(Path(record["outputs"][name]["path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
