# Completed distributed campaign

The numerical campaign completed on **7 September 2026**. All configured jobs
completed; the reported control analysis includes compatible targets only.

| Included results | Completed | Coverage |
|---|---:|---|
| PDE training | 770 / 770 | 11 PDEs, 7 widths, 10 seeds, 3000 GD steps |
| Compatible-target controls | 1080 / 1080 | 600 manufactured targets + 480 compatible contrast targets |
| Initialization measurements | 7700 / 7700 | 100 independent seeds per PDE and width |
| Limiting-kernel references | 11 / 11 | 8 scrambled QMC replicates per PDE |

All 770 PDE runs reduced loss; 767 reduced it by at least a factor of ten.
Across widths 16 to 16384, pooled median relative kernel drift decreased from
0.611 to 0.000862. Every PDE had lower median drift at the larger width.
These numerical diagnostics do not verify the theorem's sufficient width bound.

This repository includes the manuscript figure, its machine-readable
provenance record, and a compact aggregate containing every count and median
shown in that figure:

- [Frozen figure aggregate](data/compatible_rank_convergence_conditioning.json)
- [Reproduction instructions](EXTENSIVE_EXPERIMENTS.md)
- [Generated provenance record](../../figures/compatible_rank_convergence_conditioning_summary.json)

The compatible-control report excludes all 120 structurally incompatible cases
from tables, figures and conclusions. The remaining contrast cases use independent
functionals, so their targets are attainable in the trial space; they are reported
separately from manufactured-PDE targets. Selection uses compatibility, never
observed loss or convergence.

The aggregate records the SHA256 of the validated 1080-row source table. Raw
histories, per-run manifests, and scheduler records are not bundled and cannot
be reconstructed from summary statistics; rerun the documented campaign to
regenerate them. This limitation does not affect one-command reproduction of
the plotted counts and five-seed medians.
