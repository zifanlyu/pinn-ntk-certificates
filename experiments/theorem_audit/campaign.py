#!/usr/bin/env python3
"""Freeze, submit, and aggregate the empirical theorem-validation campaign."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    destination = Path(args.directory).resolve()
    if destination.exists():
        raise SystemExit(f'Refusing to replace existing snapshot: {destination}')
    destination.mkdir(parents=True)
    source = ROOT / 'experiments/theorem_audit'
    target = destination / 'experiments/theorem_audit'
    target.mkdir(parents=True)
    for path in sorted(source.glob('*.py')):
        shutil.copy2(path, target / path.name)
    shutil.copytree(source / 'configs', target / 'configs')
    for name in ('empirical_validation_protocol.tex', 'theorem_audit_experiment.tex',
                 'pinn_ntk_iclr2027.tex', 'requirements.txt', 'pyproject.toml', 'uv.lock'):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, destination / name)
    for pattern in ('*.md','*.tex'):
        for path in source.glob(pattern):
            shutil.copy2(path, target / path.name)
    configs = {}
    for kind, config_path in (('broad', args.config), ('controls', args.controls_config)):
        if config_path:
            config = json.loads(Path(config_path).read_text())
            config['output_dir'] = f'results/{kind}'
            save(destination / f'{kind}.json', config)
            configs[kind] = f'{kind}.json'
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], text=True, capture_output=True, check=True)
    (destination / 'environment.freeze.txt').write_text(freeze.stdout)
    git = subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, text=True, capture_output=True)
    diff = subprocess.run(['git','diff','--binary','HEAD'], cwd=ROOT, capture_output=True)
    (destination / 'source_changes.patch').write_bytes(diff.stdout)
    files = {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    save(destination / 'snapshot.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_root=str(ROOT), git_commit=git.stdout.strip(), python=sys.executable,
        configs=configs, files=files))
    (destination / 'logs').mkdir()
    print(destination)


def verify_snapshot(directory):
    snapshot = json.loads((directory / 'snapshot.json').read_text())
    for name, expected in snapshot['files'].items():
        path = directory / name
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f'Snapshot changed or missing: {name}')
    return snapshot


def task_count(kind, config):
    if kind == 'broad':
        return len(config['problems']) * len(config['training']['widths']) * len(config['training']['seeds'])
    from experiments.theorem_audit.rank_controls import task_grid
    # Controls supplies its own canonical task ordering.
    return len(task_grid(config))


def record_failure(directory, kind, config_path, index, error):
    """Retain a configuration-matched failure instead of silently losing a cell."""
    if kind == 'broad':
        from experiments.theorem_audit import run
        config = run.load_config(str(config_path))
        name, width, seed = run._all_training_tasks(config)[index]
        problem = run.get_problem(name)
        path, _ = run._run_paths(config, name, width, seed)
        payload = {**run._common_manifest(config, run._configure_runtime(config), run._source_fingerprint()),
                   'artifact_type':'finite_width_training','status':'failed',
                   'problem':run._problem_manifest(problem),
                   'task':{'width':width,'seed':seed},
                   'failure':{'type':type(error).__name__,'message':str(error)}}
        run._atomic_json(path,payload)
    else:
        save(directory/'logs'/f'failed_{kind}_cell_{index}.json',
             {'task_index':index,'error_type':type(error).__name__,'message':str(error)})


def worker(args):
    directory = Path(args.directory).resolve()
    verify_snapshot(directory)
    os.chdir(directory)
    sys.path.insert(0, str(directory))
    failures = []
    if args.kind == 'reference':
        from experiments.theorem_audit.run import main
        return main(['reference', '--config', str(directory/'broad.json'), '--task-index', str(args.task_index)])
    if args.kind == 'initialization':
        from experiments.theorem_audit.initialization_sweep import main
        return main(['--config',str(directory/'broad.json'),'--task-index',str(args.task_index),'--seeds','100'])
    if args.kind == 'aggregate':
        from experiments.theorem_audit.run import main
        def attempt(label, callback):
            try:
                failures.append(callback())
            except (Exception, SystemExit) as error:
                import traceback
                traceback.print_exc()
                failures.append(1)
                save(directory/'logs'/f'aggregation_error_{label}.json',
                     {'type':type(error).__name__,'message':str(error)})
        attempt('broad',lambda: main(['aggregate','--config',str(directory/'broad.json')]))
        submission_path = directory / 'submission.json'
        initialization_requested = submission_path.exists() and 'initialization' in json.loads(submission_path.read_text()).get('jobs', {})
        broad_output = Path(json.loads((directory/'broad.json').read_text())['output_dir'])
        if not broad_output.is_absolute():
            broad_output = directory / broad_output
        # Local workers produce the same artifacts without a Slurm submission
        # manifest. Include those measurements instead of silently omitting them.
        initialization_present = any((broad_output/'initialization_sweep').glob('*.json'))
        if initialization_requested or initialization_present:
            from experiments.theorem_audit.initialization_sweep import main as initialization_main
            attempt('initialization',lambda: initialization_main(['aggregate','--config',str(directory/'broad.json'),'--seeds','100']))
        if (directory/'controls.json').exists():
            from experiments.theorem_audit.rank_controls import main as control_main
            attempt('controls',lambda: control_main(['aggregate','--config',str(directory/'controls.json')]))
        report = directory / 'experiments/theorem_audit/report_campaign.py'
        if report.exists():
            result = subprocess.run([sys.executable,str(report),'--config',str(directory/'broad.json')])
            failures.append(result.returncode)
        return int(any(failures))
    path = directory / f'{args.kind}.json'
    config = json.loads(path.read_text())
    if args.kind == 'broad':
        from experiments.theorem_audit.run import main
        command = 'train'
    else:
        from experiments.theorem_audit.rank_controls import main
        command = 'run'
    count = task_count(args.kind,config)
    begin = args.task_index * args.batch_size
    for index in range(begin,min(begin+args.batch_size,count)):
        try:
            code = main([command,'--config',str(path),'--task-index',str(index)])
            if code:
                failures.append(index)
                record_failure(directory,args.kind,path,index,RuntimeError(f'exit code {code}'))
        except (Exception, SystemExit) as error:
            import traceback
            traceback.print_exc()
            failures.append(index)
            record_failure(directory,args.kind,path,index,error)
    if failures:
        save(directory/'logs'/f'failed_{args.kind}_{args.task_index}.json', {'task_indices':failures})
    return int(bool(failures))


def submit(args):
    directory = Path(args.directory).resolve()
    snapshot = verify_snapshot(directory)
    if (directory/'submission.json').exists():
        raise SystemExit('Submission already recorded; inspect it before scheduling duplicates.')
    python = snapshot['python']
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(), 'jobs':{}, 'counts':{}}
    def launch(kind, count=None, dependency=None, batch_size=1):
        log = str(directory/'logs'/f'{kind}_%A_%a.out')
        argv = ['sbatch','--parsable','--job-name',f'ntk-{kind}',
                '--chdir',str(directory),'--time',args.time,'--cpus-per-task','1',
                '--mem-per-cpu','4G','--output',log,'--error',log]
        if count:
            argv += ['--array',f'0-{count-1}%{args.parallel}']
        if dependency:
            argv += ['--dependency',dependency]
        command = [python,str(directory/'experiments/theorem_audit/campaign.py'),'worker',
                   '--directory',str(directory),'--kind',kind,'--batch-size',str(batch_size)]
        shell = '#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0\n'
        shell += 'export MPLCONFIGDIR="${TMPDIR:-/tmp}/ntk-mpl-${SLURM_JOB_ID}"\n'
        shell += shlex.join(command) + ' --task-index "${SLURM_ARRAY_TASK_ID:-0}"\n'
        jobfile = directory/'logs'/f'{kind}.sh'
        jobfile.write_text(shell)
        argv.append(str(jobfile))
        result = subprocess.run(argv,text=True,capture_output=True,timeout=60,check=True)
        jobid = result.stdout.strip().split(';')[0]
        if not jobid.isdigit():
            raise RuntimeError(f'Unrecognized sbatch result: {result.stdout!r}')
        manifest['jobs'][kind]=jobid
        manifest['counts'][kind]=count or 1
        save(directory/'submission.json',manifest)
        print(f'{kind}: job {jobid}, {count or 1} array tasks',flush=True)
        return jobid
    config = json.loads((directory/'broad.json').read_text())
    ref = launch('reference',len(config['problems']))
    n = task_count('broad',config)
    manifest['broad_runs']=n
    broad = launch('broad',(n+args.batch_size-1)//args.batch_size,f'afterany:{ref}',args.batch_size)
    jobs = [broad]
    if args.initialization_seeds:
        manifest['initialization_draws']=len(config['problems'])*len(config['training']['widths'])*100
        jobs.append(launch('initialization',len(config['problems']),f'afterany:{ref}'))
    if 'controls' in snapshot['configs']:
        sys.path.insert(0,str(directory))
        config = json.loads((directory/'controls.json').read_text())
        n = task_count('controls',config)
        manifest['control_runs']=n
        jobs.append(launch('controls',(n+args.batch_size-1)//args.batch_size,None,args.batch_size))
    launch('aggregate',dependency='afterany:'+':'.join(jobs))
    save(directory/'submission.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--directory',required=True);p.add_argument('--config',required=True);p.add_argument('--controls-config')
    p=sub.add_parser('submit');p.add_argument('--directory',required=True);p.add_argument('--parallel',type=int,default=24);p.add_argument('--batch-size',type=int,default=5);p.add_argument('--time',default='24:00:00');p.add_argument('--initialization-seeds',action='store_true',help='Also evaluate 100 disjoint initialization seeds at each width/PDE')
    p=sub.add_parser('worker');p.add_argument('--directory',required=True);p.add_argument('--kind',choices=['reference','broad','controls','aggregate','initialization'],required=True);p.add_argument('--task-index',type=int,default=0);p.add_argument('--batch-size',type=int,default=5)
    args=parser.parse_args()
    if args.command=='prepare':return prepare(args)
    if args.command=='submit':return submit(args)
    return worker(args)

if __name__=='__main__':
    sys.path.insert(0,str(ROOT))
    raise SystemExit(main())
