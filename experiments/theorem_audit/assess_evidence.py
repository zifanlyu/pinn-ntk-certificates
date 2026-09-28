#!/usr/bin/env python3
"""Summarize operator-class GD, stability, and compatible-control evidence.

Trajectory extrema are computed over stored checkpoints. They are descriptive
finite-horizon diagnostics, not all-iteration certificates.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

OPERATOR_CLASSES = {
    'classical_local': (
        'poisson_1d', 'variable_elliptic_2d', 'heat_1d', 'transport_1d',
        'wave_1d', 'biharmonic_2d', 'stokes_2d',
    ),
    'weak_variational': ('weak_poisson_1d',),
    'integral_nonlocal': (
        'nonlocal_diffusion_1d', 'integro_diff_1d', 'caputo_diffusion_1d',
    ),
}

WIDTH_METRICS = (
    'initial_reference_error_operator',
    'max_kernel_drift_operator',
    'final_relative_error_to_frozen_dynamics',
    'final_over_initial_loss',
    'max_sqrt_width_per_neuron_drift',
    'max_jacobian_drift_over_initial_sqrt_gap',
)

RANK_SENSITIVITY_TOLERANCES = (1e-14, 1e-12, 1e-10, 1e-8)

def median(rows, key):
    return statistics.median(float(x[key]) for x in rows if x.get(key) is not None)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', default=str(ROOT/'experiments/theorem_audit/results/campaign_20260907'))
    args = parser.parse_args()
    campaign = Path(args.campaign).resolve()
    base = campaign/'results'
    inputs = [base/'broad/findings.json', base/'broad/initialization_aggregate.json', base/'controls_compatible/aggregate.json', base/'controls_compatible/runs.csv']
    broad, init, controls = [json.loads(p.read_text()) for p in inputs[:3]]
    rows = list(csv.DictReader(inputs[3].open()))
    runs = broad['runs']
    assert len(runs) == 770 and all(r['status']=='complete' for r in runs)
    assert not broad['issues'] and not init['issues'] and init['accepted_measurements']==7700
    assert controls['coverage_complete'] and controls['target_functionally_compatible_only']
    assert len(rows)==1080 and all(r['target_functionally_compatible']=='True' for r in rows)
    widths = broad['config']['training']['widths']
    problems = broad['config']['problems']
    width_rows = []
    for width in widths:
        group = [r for r in runs if r['width']==width]
        width_rows.append(dict(width=width, runs=len(group), **{k:median(group,k) for k in WIDTH_METRICS},
            gap_preserved=sum(r['gap_preserved_through_training'] is True for r in group),
            informative_jacobian_bound=sum((r['min_jacobian_perturbation_gap_lower_bound'] or 0)>0 for r in group)))
    assert set().union(*map(set, OPERATOR_CLASSES.values())) == set(problems)
    operator_class_rows = []
    operator_class_width_rows = []
    training_seeds = broad['config']['training']['seeds']
    for operator_class, class_problems in OPERATOR_CLASSES.items():
        class_runs = [r for r in runs if r['problem'] in class_problems]
        expected = len(class_problems) * len(widths) * len(training_seeds)
        assert len(class_runs) == expected
        operator_class_rows.append(dict(
            operator_class=operator_class,
            problems=';'.join(class_problems),
            runs=len(class_runs),
            endpoint_loss_reduced=sum(r['final_over_initial_loss'] < 1 for r in class_runs),
            endpoint_loss_reduced_10x=sum(r['final_over_initial_loss'] <= .1 for r in class_runs),
        ))
        for width in widths:
            group = [r for r in class_runs if r['width'] == width]
            assert len(group) == len(class_problems) * len(training_seeds)
            operator_class_width_rows.append(dict(
                operator_class=operator_class,
                width=width,
                runs=len(group),
                **{k: median(group, k) for k in WIDTH_METRICS},
                gap_preserved_at_saved_checkpoints=sum(
                    r['gap_preserved_through_training'] is True for r in group
                ),
                informative_jacobian_bound_at_saved_checkpoints=sum(
                    (r['min_jacobian_perturbation_gap_lower_bound'] or 0) > 0
                    for r in group
                ),
            ))
    kernel_groups = {}
    kernel_fields = ('epsilon', 'exact_structural_rank', 'numerical_rank',
                     'lambda_min', 'lambda_max', 'output_kernel_lambda_min')
    for row in rows:
        key = (row['family'], row['row_design'], int(row['width']), int(row['seed']))
        if key in kernel_groups:
            assert all(row[field] == kernel_groups[key][field] for field in kernel_fields)
        else:
            kernel_groups[key] = row
    kernel_instances = list(kernel_groups.values())
    assert len(kernel_instances) == 600
    independent_kernels = [r for r in kernel_instances if int(r['exact_structural_rank'])==2]
    singular_kernels = [r for r in kernel_instances if int(r['exact_structural_rank'])==1]
    rank_sensitivity = []
    for tolerance in RANK_SENSITIVITY_TOLERANCES:
        rank_sensitivity.append(dict(
            relative_tolerance=tolerance,
            independent_nonsingular=sum(
                float(r['lambda_min']) > tolerance * float(r['lambda_max'])
                for r in independent_kernels
            ),
            independent_total=len(independent_kernels),
            duplicate_singular=sum(
                float(r['lambda_min']) <= tolerance * float(r['lambda_max'])
                for r in singular_kernels
            ),
            duplicate_total=len(singular_kernels),
        ))
    independent = [r for r in rows if int(r['exact_structural_rank'])==2]
    singular = [r for r in rows if int(r['exact_structural_rank'])==1]
    manufactured_independent = [r for r in independent if r['target_mode']=='manufactured']
    wide_independent = [r for r in independent if int(r['width'])>=256]
    widest_independent = [r for r in independent if int(r['width'])==16384]
    groups = []
    for family in sorted({r['family'] for r in rows}):
        for width in sorted({int(r['width']) for r in rows}):
            subset = [r for r in rows if r['family']==family and int(r['width'])==width and r['target_mode']=='contrast' and r['epsilon']!='']
            levels=[]
            for epsilon in [1., .1, .01]:
                level=[r for r in subset if float(r['epsilon'])==epsilon]
                assert len(level)==5
                levels.append(dict(epsilon=epsilon,relative_gap=statistics.median(float(r['lambda_min'])/float(r['lambda_max']) for r in level),loss_ratio=median(level,'final_to_initial_loss_ratio')))
            groups.append(dict(family=family,width=width,levels=levels,
                gap_ordered=all(a['relative_gap']>b['relative_gap'] for a,b in zip(levels,levels[1:])),
                loss_ordered=all(a['loss_ratio']<b['loss_ratio'] for a,b in zip(levels,levels[1:]))))
    per_problem=[]
    for problem in problems:
        narrow=[r for r in runs if r['problem']==problem and r['width']==16]
        wide=[r for r in runs if r['problem']==problem and r['width']==16384]
        concentration=[r for r in init['summary'] if r['problem']==problem and r['metric']=='relative_operator_error' and r['width']>=64]
        slope=float(np.polyfit(np.log([r['width'] for r in concentration]),np.log([r['center'] for r in concentration]),1)[0])
        motion_widths=[w for w in widths if w>=256]
        motion=[median([r for r in runs if r['problem']==problem and r['width']==w],'max_sqrt_width_per_neuron_drift')/math.sqrt(w) for w in motion_widths]
        motion_slope=float(np.polyfit(np.log(motion_widths),np.log(motion),1)[0])
        per_problem.append(dict(problem=problem,loss_ratio_width16=median(narrow,'final_over_initial_loss'),loss_ratio_width16384=median(wide,'final_over_initial_loss'),
            solution_grid_relative_l2_width16384=median(wide,'final_solution_relative_l2'),concentration_slope_width_ge64=slope,neuron_motion_slope_width_ge256=motion_slope))
    widest=[r for r in runs if r['width']==16384]
    comparison=[]
    for family in ['pointwise_poisson','nonlocal_diffusion']:
        subset=[r for r in rows if r['family']==family and int(r['width'])==16384 and r['epsilon']=='0.01']
        manufactured=sorted([r for r in subset if r['target_mode']=='manufactured'],key=lambda r:int(r['seed']))
        contrast=sorted([r for r in subset if r['target_mode']=='contrast'],key=lambda r:int(r['seed']))
        assert len(manufactured)==len(contrast)==5
        comparison.append(dict(family=family,paired_initial_eigenvalues_identical=all(a['lambda_min']==b['lambda_min'] and a['lambda_max']==b['lambda_max'] for a,b in zip(manufactured,contrast)),
            lambda_min=median(manufactured,'lambda_min'),manufactured_loss_ratio=median(manufactured,'final_to_initial_loss_ratio'),compatible_contrast_loss_ratio=median(contrast,'final_to_initial_loss_ratio')))
    result=dict(scope='Completed runs only; all control targets functionally compatible; descriptive post-hoc analysis.',
        input_sha256={str(p.relative_to(campaign)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        coverage=dict(pde_runs=len(runs),compatible_controls=len(rows),initializations=init['accepted_measurements']),
        rank=dict(distinct_kernel_instances=len(kernel_instances),
            independent_kernel_instances=len(independent_kernels),
            independent_rank2=sum(int(r['numerical_rank'])==2 for r in independent_kernels),
            independent_Kout_positive=sum(float(r['output_kernel_lambda_min'])>0 for r in independent_kernels),
            singular_kernel_instances=len(singular_kernels),
            singular_rank1=sum(int(r['numerical_rank'])==1 for r in singular_kernels)),
        gd=dict(pde_loss_reduced=sum(r['final_over_initial_loss']<1 for r in runs),
            pde_loss_reduced_10x=sum(r['final_over_initial_loss']<=.1 for r in runs),
            positive_initial_ntk_runs=len(independent),
            positive_initial_ntk_loss_reduced=sum(float(r['final_loss'])<float(r['initial_loss']) for r in independent),
            manufactured_positive_initial_ntk_runs=len(manufactured_independent),
            manufactured_positive_initial_ntk_loss_reduced_10x=sum(float(r['final_to_initial_loss_ratio'])<=.1 for r in manufactured_independent),
            width_ge256_positive_initial_ntk_runs=len(wide_independent),
            width_ge256_positive_initial_ntk_loss_reduced=sum(float(r['final_loss'])<float(r['initial_loss']) for r in wide_independent),
            width16384_positive_initial_ntk_runs=len(widest_independent),
            width16384_positive_initial_ntk_loss_reduced=sum(float(r['final_loss'])<float(r['initial_loss']) for r in widest_independent),
            width16384_positive_initial_ntk_median_loss_ratio=median(widest_independent,'final_to_initial_loss_ratio'),
            compatible_singular_runs=len(singular),
            compatible_singular_loss_reduced_100x=sum(float(r['final_to_initial_loss_ratio'])<=.01 for r in singular),
            compatible_singular_loss_below_1e_minus12=sum(float(r['final_loss'])<1e-12 for r in singular),
            width_ge256_compatible_singular_runs=sum(int(r['width'])>=256 for r in singular),
            width_ge256_compatible_singular_loss_below_1e_minus12=sum(int(r['width'])>=256 and float(r['final_loss'])<1e-12 for r in singular)),
        width_summary=width_rows,operator_class=operator_class_rows,
        operator_class_width_summary=operator_class_width_rows,
        rank_tolerance_sensitivity=rank_sensitivity,
        checkpoint_scope='Gap, Jacobian-drift, kernel-drift, and neuron-motion maxima are over saved checkpoints, not every GD iterate.',
        per_problem=per_problem,conditioning_groups=groups,
        gap_ordered_groups=sum(g['gap_ordered'] for g in groups),loss_ordered_groups=sum(g['loss_ordered'] for g in groups),same_kernel_targets=comparison,
        widest_kernel_norm_bound_positive=sum(r['max_kernel_drift_over_initial_gap']<1 for r in widest),
        inconclusive_qmc=[r['problem'] for r in broad['references'] if r['qmc_status']=='inconclusive'])
    out=base/'evidence_assessment';out.mkdir(exist_ok=True)
    (out/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,data in [
        ('width_summary',width_rows),
        ('operator_class',operator_class_rows),
        ('operator_class_width_summary',operator_class_width_rows),
        ('rank_tolerance_sensitivity',rank_sensitivity),
        ('per_problem',per_problem),
    ]:
        with (out/f'{name}.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps({k:result[k] for k in [
        'coverage', 'rank', 'rank_tolerance_sensitivity', 'gd',
        'operator_class', 'gap_ordered_groups', 'loss_ordered_groups',
        'same_kernel_targets', 'widest_kernel_norm_bound_positive',
        'inconclusive_qmc',
    ]},indent=2))
    print(out)

if __name__=='__main__':main()
