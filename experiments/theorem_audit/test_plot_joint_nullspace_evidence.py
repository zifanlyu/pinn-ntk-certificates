"""Regression checks for the appendix joint-nullspace loss figure."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt

from .plot_joint_nullspace_evidence import make_figure
from .plot_supportive_evidence import _load_joint_trajectories


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("plot_joint_nullspace_evidence.py")
HELPER = Path(__file__).with_name("plot_supportive_evidence.py")
INPUT = (
    PROJECT_ROOT
    / "experiments/theorem_audit/data"
    / "joint_nullspace_controls.json"
)
SUMMARY = PROJECT_ROOT / "figures/joint_nullspace_loss_decomposition_summary.json"
PDF = PROJECT_ROOT / "figures/joint_nullspace_loss_decomposition.pdf"
PNG = PROJECT_ROOT / "figures/joint_nullspace_loss_decomposition.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JointNullspacePlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.joint = _load_joint_trajectories(INPUT)

    def test_plotted_decomposition_matches_audit(self) -> None:
        figure, summary = make_figure(self.joint)
        try:
            self.assertEqual(list(figure.get_size_inches()), [5.5, 2.55])
            self.assertEqual(len(figure.axes), 1)
            self.assertEqual(summary["paired_settings"], 18)
            self.assertEqual(summary["gd_trajectories"], 36)
            self.assertAlmostEqual(
                summary["predicted_incompatible_loss_floor"], 0.0625
            )
            self.assertLess(summary["maximum_checkpoint_loss_offset_error"], 1e-12)
            self.assertGreater(summary["median_shared_active_subspace_loss"][0], 1.0)
            self.assertLess(summary["median_shared_active_subspace_loss"][-1], 1e-28)
        finally:
            plt.close(figure)

    def test_appendix_layout_is_legible(self) -> None:
        figure, _ = make_figure(self.joint)
        try:
            axes = figure.axes[0]
            visible_text = [
                item
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
            self.assertGreaterEqual(min(item.get_fontsize() for item in visible_text), 7.0)
            self.assertIsNotNone(axes.get_legend())
            with tempfile.TemporaryDirectory() as temporary_directory:
                output = Path(temporary_directory) / "joint_nullspace.png"
                figure.savefig(output, dpi=120)
                self.assertGreater(output.stat().st_size, 10_000)
        finally:
            plt.close(figure)

    def test_committed_artifacts_match_provenance_record(self) -> None:
        record = json.loads(SUMMARY.read_text())
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(
            record["joint_nullspace_input"],
            "experiments/theorem_audit/data/joint_nullspace_controls.json",
        )
        self.assertEqual(record["joint_nullspace_input_sha256"], _sha256(INPUT))
        self.assertEqual(record["analysis_script_sha256"], _sha256(SCRIPT))
        self.assertEqual(record["plot_helper_sha256"], _sha256(HELPER))
        for name, path in (("pdf", PDF), ("png", PNG)):
            self.assertEqual(record["outputs"][name]["sha256"], _sha256(path))
            self.assertEqual(record["outputs"][name]["bytes"], path.stat().st_size)
            self.assertFalse(Path(record["outputs"][name]["path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
