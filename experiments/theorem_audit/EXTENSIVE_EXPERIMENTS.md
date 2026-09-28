# Extensive empirical validation

**The reported campaign is complete.** Read
[CAMPAIGN_STATUS.md](CAMPAIGN_STATUS.md) for coverage and the distinction
between the bundled aggregate and raw records that require regeneration.
The validation protocol (`empirical_validation_protocol.tex` in the paper
source) maps each experiment to the paper's results.

The tracked
[`data/compatible_rank_convergence_conditioning.json`](data/compatible_rank_convergence_conditioning.json)
contains the counts and five-seed medians used by the manuscript figure. It
also records the SHA256 of the validated 1080-row compatible-control source
table; it does not purport to contain individual histories.
The separate tracked
[`data/joint_nullspace_controls.json`](data/joint_nullspace_controls.json)
contains the 36 paired compatible/incompatible trajectories and their saved
active/nullspace loss decomposition used by the appendix figure.

| Included experiment | Design | Completed |
|---|---|---:|
| PDE training | 11 PDEs x 7 widths (4--16384) x 10 seeds; 3000 GD steps | 770 |
| Compatible controls | 6 families; independent/near-dependent/duplicate rows; 4 widths; 5 seeds | 1080 |
| Initialization concentration | 11 PDEs x 7 widths x 100 disjoint seeds | 7700 |

All use the paper's biased two-layer tanh architecture, train every parameter,
initialize iid Uniform[-1,1], and use float64 full-batch GD on `||P-y||^2`.
Each loss group is normalized by its row count; the fixed learning rate is
`0.2/lambda_max(K_initial)`. Widths use nested initialization prefixes.
References use 8 independently scrambled Sobol replicates of 32768 samples.

Controls include 600 manufactured-target runs for all designs and 480 compatible
contrast-target runs for independent designs. Exact duplicate rows use manufactured
targets only. The 1080-run two-row table excludes every incompatible target; the
separate three-row audit supplies the paired incompatible trajectories in the
appendix. Two-row controls isolate rank/conditioning; the eleven PDE systems provide
sampled-loss benchmarks.

## Reproduce the manuscript figure

Run from the repository root with the locked environment:

```bash
uv sync
uv run python experiments/theorem_audit/plot_supportive_evidence.py
uv run python -m unittest experiments.theorem_audit.test_plot_supportive_evidence
```

This validates the compatible-control aggregate and regenerates the main-text
PDF/PNG and its provenance JSON. Paths written to that JSON are repository-relative.
Since summary statistics cannot reconstruct the original 1080 compatible-control
trajectories, regenerating that full report requires a newly executed campaign:

```bash
CAMPAIGN=experiments/theorem_audit/results/campaign_compatible_new
uv run python -m experiments.theorem_audit.rank_controls_aggregate \
  --source-config "$CAMPAIGN/controls.json" \
  --output-dir "$CAMPAIGN/results/controls_compatible"
```

## Run a new campaign on a Slurm cluster

Run **`campaign.py`** with a fresh output directory:

```bash
CAMPAIGN=experiments/theorem_audit/results/campaign_compatible_new
uv run python experiments/theorem_audit/campaign.py prepare \
  --directory "$CAMPAIGN" \
  --config experiments/theorem_audit/configs/extensive.json \
  --controls-config experiments/theorem_audit/configs/rank_controls_paper.json
uv run python experiments/theorem_audit/campaign.py submit \
  --directory "$CAMPAIGN" --parallel 24 --batch-size 2 --initialization-seeds
```

The current control grid excludes incompatible cases before scheduling. Each
worker verifies frozen source hashes. `submission.json` records job IDs;
`snapshot.json` and `environment.freeze.txt` record provenance. Arrays request one
CPU and 4 GB per task; aggregation reports missing/failed cells. Existing snapshots
are not overwritten. The smaller PDE pilot remains separate from these results.

The tests compare derivatives/GD with autograd and verify rank, compatibility and
artifact validation. Checkpoint diagnostics and numerical reference gaps do not
certify the theorem's conservative sufficient width/step bounds. Small training
loss does not establish PDE generalization; Caputo results concern the finite
quadrature loss.
