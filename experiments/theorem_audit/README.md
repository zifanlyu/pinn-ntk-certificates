# Finite-width theorem audit

The expanded 2026-09-07 campaign adds widths through 16384, ten training seeds,
100 independent initialization seeds, and exact/near-dependent functional
controls. Run `campaign.py` using
[`EXTENSIVE_EXPERIMENTS.md`](EXTENSIVE_EXPERIMENTS.md); the original smaller
configurations below remain available.

This directory runs a reproducible audit of the paper's optimization theorem on
eleven linear PDE systems.  It is a mechanistic experiment: it measures positivity,
finite-width kernel concentration, kernel stability, parameter motion, and loss
decay.  The stored rank artifacts certify the design-side positivity conditions
under the paper's theorems; QMC spectra and training trajectories are empirical
checks, not independent proofs of the theorems.

Each theorem-backed rank procedure certifies the output-weight contribution
`K_out` alone. Because `K_full = K_out + K_hidden` and both contributions are
positive semidefinite, `K_out` positive definite already certifies `K_full`
positive definite; `K_hidden` need not pass a separate test. The code stores
`K_hidden` only as a decomposition diagnostic. A future case registered with
the numerical route is gated directly on `K_full`, never on both blocks.

## Exact model and optimization setup

Every case uses the model studied in the paper,

```text
u(x) = W2 tanh(W1 x + b1) / sqrt(width),  gamma = 1,
```

with one hidden layer, no global output bias, and all of `W1`, `b1`, and `W2`
trainable.  Every coordinate is initialized independently from `Uniform[-1,1]`;
in particular, the output weights are conditionally centered.  Computation is in
float64.  Each constraint group is scaled by the inverse square root of its
number of rows, the loss is `||prediction - target||_2^2` (without a factor
`1/2`), and optimization is deterministic full-batch gradient descent with
`eta = learning_rate_factor / lambda_max(K_0)`.

Widths in one seed use prefixes of a single maximum-width initialization bank.
This pairing makes width comparisons substantially less noisy while preserving
the prescribed i.i.d. law at every width.

## Cases and positivity route

| Case | Constraint class | Positivity treatment |
|---|---|---|
| `poisson_1d` | strong, pointwise | automatic local-operator certificate |
| `variable_elliptic_2d` | strong, pointwise | automatic local-operator certificate |
| `heat_1d` | strong, pointwise space-time | automatic local-operator certificate |
| `transport_1d` | strong, pointwise space-time | automatic local-operator certificate |
| `wave_1d` | second-order space-time, repeated initial rows | automatic local-operator certificate |
| `biharmonic_2d` | fourth-order, clamped boundary | automatic local-operator certificate |
| `stokes_2d` | vector-valued system with shared locations | automatic local-operator certificate |
| `weak_poisson_1d` | four energy-orthonormal Legendre weak rows plus Dirichlet data | automatic weak-functional certificate |
| `nonlocal_diffusion_1d` | three symmetric value-only nonlocal rows plus Dirichlet data | automatic signed-measure certificate |
| `integro_diff_1d` | integro-differential rows plus Dirichlet data | automatic weak-functional certificate |
| `caputo_diffusion_1d` | quadrature-discretized fractional-in-time rows | automatic weak-functional certificate for the exact discrete loss |

All eleven cases have theorem-backed positivity routes.  The first seven use the
pointwise DNTK result and verified local operator-coefficient ranks.  Weak
Poisson, the integro-differential problem, and the stored Caputo quadrature use
the weak-functional theorem and a full-row-rank matrix over the
loss-scaled jet-evaluation atoms in the exact discrete loss.  The value-only
nonlocal diffusion problem uses the signed-measure theorem and joint
residual--boundary measure independence.  Replicated QMC is therefore a
quantitative gap estimate, not the logical positivity test.  If a future case is
registered with the `numerical` route, only that case is gated before training;
the prespecified design note `theorem_audit_experiment.tex` in the paper
source describes the gate.
Operationally, each reference task calls the registry-selected certificate
routine before training, writes the loss-scaled coefficient matrix, rank, and
pass/fail result under `positivity_certificate` in its JSON manifest, and sets
`positivity_classification.allows_training` from that result. An automatic case
whose certificate fails is rejected during problem validation and cannot be
trained. The
floating-point matrix rank stored by the implementation is an audit of the
case-specific analytic rank witness documented in the paper; it is not presented
as an independent rigorous certificate.
The QMC lower diagnostic uses the two-sided 95% Student-t multiplier for the
configured replicate count: 12.706 (smoke, 2 replicates), 3.182 (local, 4),
and 2.365 (paper, 8).

The Caputo certificate is deliberately scoped to the finite quadrature loss
that is actually trained.  Automatic positivity for the continuum fractional
operator would require a separate boundedness, density, and Banach-valued
analyticity argument on an appropriate function space.

## Run locally

Install the locked environment once from the repository root:

```powershell
uv sync
```

First verify all eleven definitions and code paths:

```powershell
uv run python experiments/theorem_audit/run.py all --config experiments/theorem_audit/configs/smoke.json
```

Run the moderate local audit:

```powershell
uv run python experiments/theorem_audit/run.py all --config experiments/theorem_audit/configs/local.json
uv run python experiments/theorem_audit/run.py aggregate --config experiments/theorem_audit/configs/local.json
```

The paper configuration is intentionally larger:

```powershell
uv run python experiments/theorem_audit/run.py all --config experiments/theorem_audit/configs/paper.json
uv run python experiments/theorem_audit/run.py aggregate --config experiments/theorem_audit/configs/paper.json
```

Restrict a command without changing the configuration:

```powershell
uv run python experiments/theorem_audit/run.py reference --config experiments/theorem_audit/configs/paper.json --problem weak_poisson_1d
uv run python experiments/theorem_audit/run.py train --config experiments/theorem_audit/configs/paper.json --problem poisson_1d --width 1024 --seed 0
```

`reference` estimates the limiting Gram matrix.  `train` evaluates finite-width
initialization and full-batch GD.  `all` runs reference jobs before training,
and `aggregate` validates manifests and creates tables/figures from completed
runs.  Outputs are isolated by configuration under
`experiments/theorem_audit/results/`; reruns do not glob results from another
configuration.

## Three-row joint-nullspace controls

The independent three-row audit tests exact and near dependencies beyond
duplicate pairs for a co-located two-output design, a joint weak--boundary
design, and a joint finite-measure nonlocal--boundary design. It covers 72
initial kernels and 36 paired compatible/incompatible GD trajectories. Run it
from the repository root with:

```powershell
uv run python -m experiments.theorem_audit.joint_nullspace_controls
uv run python -m experiments.theorem_audit.plot_joint_nullspace_evidence
uv run python -m unittest experiments.theorem_audit.test_joint_nullspace_controls
uv run python -m unittest experiments.theorem_audit.test_plot_joint_nullspace_evidence
```

The first command deterministically regenerates
`data/joint_nullspace_controls.json` and `joint_nullspace_controls.tex`,
including configuration and source hashes. The weak and nonlocal controls use
fixed quadrature and therefore audit the exact stored discrete prediction maps;
they do not estimate continuum discretization error.
The second command validates that aggregate and writes the appendix figure
`figures/joint_nullspace_loss_decomposition.{pdf,png}` together with a JSON
record of plotted values and source, script, helper, and output hashes.

## Slurm arrays

The supplied job files target `paper.json`.  From the repository root, create
the log directory and submit the reference array:

```bash
mkdir -p experiments/theorem_audit/logs
REF_JOB=$(sbatch --parsable experiments/theorem_audit/slurm/reference_array.sh)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${REF_JOB} experiments/theorem_audit/slurm/training_array.sh)
sbatch --dependency=afterany:${TRAIN_JOB} experiments/theorem_audit/slurm/aggregate.sh
```

The reference array has 11 tasks, one per ordered problem.  The training array
has `11 problems x 3 widths x 5 seeds = 165` tasks, with index
`((problem_index * 3) + width_index) * 5 + seed_index`.  Array tasks are
idempotent and write disjoint run directories.  Set `CONFIG` to use another
configuration, and adjust the Slurm array bounds consistently.  Set
`PYTHON_BIN` (for example, to an environment's Python executable) if `uv` is
not available on the compute node.

The aggregation job uses `afterany` deliberately: it reports missing, failed,
or positivity-gated runs instead of being suppressed by one failed task.

## Extended campaign and findings

The standalone validation protocol (`empirical_validation_protocol.tex` in the
paper source) maps the exact rank/nullspace, joint functional independence,
compatible singular loss, conditioning, and finite-width GD results to distinct
empirical tests.
`configs/extensive.json` registers 770 broad training runs: eleven PDEs, widths
`[4,16,64,256,1024,4096,16384]`, ten seeds, and 3000 GD updates. Its reference
kernels use eight independently scrambled replicates with 32768 fine samples
each. `configs/campaign_pilot.json` registers 99 training runs at widths
`[16,256,4096]`, three seeds, and 1000 updates. These counts describe configured
work, not completed results. Rank-infeasible narrow widths are retained.

After execution, create an evidence-only report from the exact campaign config:

```bash
python -m experiments.theorem_audit.report_campaign --config /absolute/path/to/campaign/broad.json
```

`report_campaign.py` automatically detects frozen campaign roots from
`snapshot.json`; `--project-root /path/to/root` handles other layouts. It reads
the configured manifest cells, validates their identities and completion,
and writes `findings.md`, `findings.tex`, `findings.json`, a complete table by
PDE and width, and a plot comparing actual/frozen-kernel loss reduction against
the initial inverse condition number. The plot separates PDEs, colors kernel
drift, and identifies widths and individual seeds. Missing or rejected runs
remain in the coverage denominator. The report can be regenerated during a
campaign and labels incomplete coverage explicitly.

Compile the generated report from its output directory with `latexmk -pdf
findings.tex`. The reports describe empirical optimization behavior; they do
not claim that measured widths and the spectral learning rate certify the
theorem's deterministic sufficient bounds.

Regenerate the two-panel main-text figure with:

```bash
uv run python experiments/theorem_audit/plot_supportive_evidence.py
```

The plotting command validates the tracked aggregate in
`data/compatible_rank_convergence_conditioning.json`, then writes the PDF, PNG,
and a JSON provenance record to
`figures/compatible_rank_convergence_conditioning.*`. The aggregate contains
the displayed counts and five-seed medians and records the SHA256 of the
validated 1080-row source table. Panel (a) gives the frozen width diagnostics
from 110 runs per width across eleven PDE systems: initialization error, maximum
kernel drift, frozen-trajectory discrepancy, and saved-checkpoint gap retention.
Panel (b) plots five-seed medians on log axes, encoding constraint family by
color and line style and the prescribed row-mixing parameter $\varepsilon$ by
marker shape. The horizontal coordinate is the measured condition number
$\kappa_0$, not $\varepsilon$.

The compatible-control aggregate cannot reconstruct individual histories. Rerun `campaign.py` to
regenerate raw histories and the full compatible-control source table.

Regression-test the plotted counts, medians, layout, and minimum text size with:

```bash
uv run python -m unittest experiments.theorem_audit.test_plot_supportive_evidence
```
