"""Regression checks for selection and complete archival artifact validation."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .rank_controls import config_digest, task_grid
from .rank_controls_aggregate import aggregate_compatible


class CompatibleControlTests(unittest.TestCase):
    def _fixture(self, root: Path):
        source = root / "snapshot"
        code = source / "experiments" / "theorem_audit"
        code.mkdir(parents=True)
        source_digest = hashlib.sha256()
        names = ("rank_controls.py", "constraints.py", "problems.py", "features.py", "engine.py")
        for name in names:
            content = f"# Frozen fixture: {name}\n".encode()
            (code / name).write_bytes(content)
            source_digest.update(name.encode())
            source_digest.update(content)
        config = {"schema_version": 1, "output_dir": "results/controls", "families": ["pointwise_poisson"],
                  "epsilons": [None, 0.1, 0.0], "target_modes": ["manufactured", "contrast"],
                  "widths": [16], "seeds": [0], "steps": 3, "checkpoint_every": 2,
                  "learning_rate_factor": 0.2, "contrast_amplitude": 0.5}
        config_path = source / "controls.json"
        config_path.write_text(json.dumps(config))
        files = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in [config_path, *(code / name for name in names)]}
        (source / "snapshot.json").write_text(json.dumps({"files": files}))
        runs = source / "results" / "controls" / "runs"
        runs.mkdir(parents=True)
        for index, (family, epsilon, mode, width, seed) in enumerate(task_grid(config, include_incompatible=True)):
            compatible = index != 5
            floor = 0.0 if compatible else 0.25
            history = [{"step": step, "loss": loss, "frozen_kernel_loss": loss}
                       for step, loss in zip([0, 2, 3], [1.0, 0.7, 0.5])]
            arrays_path = runs / f"task_{index:05d}.npz"
            np.savez(arrays_path, history_step=[0, 2, 3], history_loss=[1.0, 0.7, 0.5],
                     history_frozen_kernel_loss=[1.0, 0.7, 0.5])
            run = {"schema_version": 1, "artifact_type": "rank_conditioning_control", "status": "complete",
                   "task_index": index, "config_sha256": config_digest(config), "source_sha256": source_digest.hexdigest(),
                   "task": {"family": family, "epsilon": epsilon, "target_mode": mode, "width": width, "seed": seed,
                            "contrast_amplitude": 0.5, "target_functionally_compatible": compatible,
                            "exact_structural_rank": 1 if epsilon == 0 else 2, "exact_structural_loss_floor": floor},
                   "optimization": {"steps": 3}, "history": history,
                   "summary": {"initial_loss": 1.0, "final_loss": 0.5, "final_frozen_kernel_loss": 0.5,
                               "exact_structural_loss_floor": floor},
                   "arrays_file": arrays_path.name, "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest()}
            arrays_path.with_suffix(".json").write_text(json.dumps(run))
        return config, config_path, runs

    def test_default_grid_preserves_all_compatible_modes_and_original_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self._fixture(Path(tmp))
            original_digest = config_digest(config)
            compatible = task_grid(config)
            self.assertEqual(len(compatible), 5)
            self.assertIn(("pointwise_poisson", None, "contrast", 16, 0), compatible)
            self.assertIn(("pointwise_poisson", 0.1, "contrast", 16, 0), compatible)
            self.assertEqual(len(task_grid(config, include_incompatible=True)), 6)
            self.assertEqual(config_digest(config), original_digest)
            self.assertEqual(len(task_grid({**config, "contrast_amplitude": 0.0})), 6)

    def test_overlay_retains_original_task_indices_and_does_not_modify_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config_path, _ = self._fixture(root)
            before = {str(p): p.read_bytes() for p in config_path.parent.rglob("*") if p.is_file()}
            destination = root / "compatible"
            destination.mkdir()
            with patch("experiments.theorem_audit.rank_controls_aggregate._write_aggregate", return_value={}) as writer:
                aggregate_compatible(config_path, destination)
            selected = writer.call_args.args[0]
            self.assertEqual([run["task_index"] for run in selected], [0, 1, 2, 3, 4])
            provenance = json.loads((destination / "selection_provenance.json").read_text())
            self.assertEqual(provenance["source_validated_runs"], 6)
            self.assertEqual(provenance["expected_included_runs"], 5)
            self.assertEqual(provenance["excluded_tasks"][0]["task_index"], 5)
            self.assertEqual(before, {str(p): p.read_bytes() for p in config_path.parent.rglob("*") if p.is_file()})

    def test_excluded_artifacts_must_validate_before_filtering(self):
        for corruption in ("npz", "history", "source", "config", "compatibility"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, config_path, runs = self._fixture(root)
                path = runs / "task_00005.json"
                run = json.loads(path.read_text())
                if corruption == "npz":
                    path.with_suffix(".npz").write_bytes(b"corrupt")
                elif corruption == "history":
                    run["history"] = run["history"][:-1]
                elif corruption == "source":
                    run["source_sha256"] = "incorrect"
                elif corruption == "config":
                    run["config_sha256"] = "incorrect"
                else:
                    run["task"]["target_functionally_compatible"] = True
                path.write_text(json.dumps(run))
                with patch("experiments.theorem_audit.rank_controls_aggregate._write_aggregate") as writer:
                    with self.assertRaises(ValueError):
                        aggregate_compatible(config_path, root / "compatible")
                    writer.assert_not_called()

    def test_snapshot_change_and_output_collision_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config_path, runs = self._fixture(root)
            with self.assertRaises(ValueError):
                aggregate_compatible(config_path, runs.parent)
            code_path = config_path.parent / "experiments/theorem_audit/engine.py"
            code_path.write_text("changed source")
            with self.assertRaises(ValueError):
                aggregate_compatible(config_path, root / "compatible")


if __name__ == "__main__":
    unittest.main()
