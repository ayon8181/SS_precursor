import argparse
import glob
import json
from pathlib import Path
import yaml
from .asdf_io import inspect_file
from .pipeline import dump_json, prepare, stack_prepared


def main():
    parser = argparse.ArgumentParser(description='Read-only ASDF inspection and SS precursor baseline')
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect')
    inspect.add_argument('paths', nargs='+', help='ASDF files or glob patterns')
    inspect.add_argument('--stations', type=int, default=3, help='Stations per file; 0 means all')
    inspect.add_argument('--output', type=Path, default=Path('outputs/inspection.json'))
    prep = sub.add_parser('prepare')
    prep.add_argument('path', type=Path)
    prep.add_argument('--config', type=Path, default=Path('example_config.yaml'))
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--limit', type=int, help='Diagnostic station limit; default processes all')
    prep.add_argument('--station', action='append', help='Explicit station ID; repeat to select a diagnostic subset')
    prep.add_argument('--network', help='Exact network code, e.g. SY')
    stack = sub.add_parser('stack')
    stack.add_argument('directory', type=Path)
    validate = sub.add_parser('validate')
    validate.add_argument('--output', type=Path, default=Path('outputs/validation'))
    event_plan = sub.add_parser('plan-events', help='Freeze inputs/configuration and enumerate every event')
    event_plan.add_argument('paths', nargs='+', help='ASDF paths or quoted globs')
    event_plan.add_argument('--config', type=Path, default=Path('sy_config.yaml'))
    event_plan.add_argument('--network', default='SY')
    event_plan.add_argument('--output', type=Path, required=True)
    batch_prep = sub.add_parser('prepare-events', help='Prepare event tasks, locally or in a scheduler array')
    batch_prep.add_argument('--plan', type=Path, required=True)
    batch_prep.add_argument('--root', type=Path, required=True)
    batch_prep.add_argument('--task-index', type=int, help='Zero-based event index; omit to process all tasks')
    batch_prep.add_argument('--workers', type=int, default=1)
    bin_plan = sub.add_parser('plan-bins', help='Index completed caches and select geographic bins')
    bin_plan.add_argument('--plan', type=Path, required=True, help='Event plan')
    bin_plan.add_argument('--root', type=Path, required=True, help='Event cache root')
    bin_plan.add_argument('--output', type=Path, required=True, help='New bin-plan directory')
    bin_plan.add_argument('--allow-incomplete', action='store_true')
    bin_plan.add_argument('--stack-settings', type=Path, help='Optional YAML overrides for bin/stack settings; reuses event caches')
    batch_stack = sub.add_parser('stack-bins', help='Stack eligible reflection bins across all events')
    batch_stack.add_argument('--plan', type=Path, required=True, help='Bin plan.json')
    batch_stack.add_argument('--output', type=Path, required=True)
    batch_stack.add_argument('--task-index', type=int, help='Zero-based bin index; omit to process all bins')
    batch_stack.add_argument('--workers', type=int, default=1)
    batch_stack.add_argument('--plots', action='store_true')
    paper = sub.add_parser('paper-plots', help='Absolute-slowness contour plots and wiggle sections from existing caches')
    paper.add_argument('--plan', type=Path, required=True, help='Existing bin plan.json')
    paper.add_argument('--output', type=Path, required=True, help='New paper-plot output root')
    paper.add_argument('--task-index', type=int, help='One bin index; omit to process all eligible bins')
    paper.add_argument('--workers', type=int, default=1)
    paper.add_argument('--settings', type=Path, help='Paper-plot YAML settings; defaults are built in')
    count = sub.add_parser('task-count', help='Print number of array tasks as an integer')
    count.add_argument('plan', type=Path)
    args = parser.parse_args()
    if args.command == 'inspect':
        paths = sorted({Path(p) for pattern in args.paths for p in glob.glob(pattern)})
        if not paths:
            parser.error('No ASDF input files matched')
        reports = []
        for path in paths:
            try:
                reports.append(inspect_file(path, args.stations or None))
            except Exception as exc:
                reports.append({'path': str(path.resolve()), 'error': f'{type(exc).__name__}: {exc}'})
            print(json.dumps({'inspected': str(path), 'completed_files': len(reports), 'total_files': len(paths)}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dump_json(args.output, reports)
        print(json.dumps({'files': len(paths), 'failed_files': sum('error' in r for r in reports), 'output': str(args.output)}))
    elif args.command == 'prepare':
        cfg = yaml.safe_load(args.config.read_text())
        result = prepare(args.path, cfg, args.output, args.limit, args.station, args.network)
        print(json.dumps(result))
        if result['accepted'] == 0:
            raise SystemExit(1)
    elif args.command == 'stack':
        print(json.dumps(stack_prepared(args.directory)))
    elif args.command == 'validate':
        from .validation import controlled_validation
        result = controlled_validation(args.output)
        print(json.dumps(result))
        if not result['passed']:
            raise SystemExit(1)
    elif args.command == 'plan-events':
        from .batch import make_event_plan
        print(json.dumps(make_event_plan(args.paths, yaml.safe_load(args.config.read_text()), args.output, args.network)))
    elif args.command == 'plan-bins':
        from .batch import make_bin_plan
        settings = yaml.safe_load(args.stack_settings.read_text()) if args.stack_settings else None
        print(json.dumps(make_bin_plan(args.plan, args.root, args.output, args.allow_incomplete, settings)))
    elif args.command in ('prepare-events', 'stack-bins'):
        from .batch import read_plan, run_tasks, prepare_event, stack_bin
        is_event = args.command == 'prepare-events'
        plan = read_plan(args.plan, 'events-v1' if is_event else 'bins-v1')
        if args.task_index is not None and not 0 <= args.task_index < len(plan['tasks']):
            parser.error('Task index out of range')
        if args.task_index is not None and args.workers != 1:
            parser.error('Use one worker per array task; omit --task-index for local parallelism')
        indices = [args.task_index] if args.task_index is not None else range(len(plan['tasks']))
        if is_event:
            run_tasks(prepare_event, [(args.plan.resolve(), args.root.resolve(), i) for i in indices], args.workers)
        else:
            run_tasks(stack_bin, [(args.plan.resolve(), args.output.resolve(), i, args.plots) for i in indices], args.workers)
    elif args.command == 'task-count':
        from .batch import read_plan
        kind = json.loads(args.plan.read_text())['kind']
        if kind not in ('events-v1', 'bins-v1'):
            parser.error('Unsupported task plan')
        print(len(read_plan(args.plan, kind)['tasks']))
    elif args.command == 'paper-plots':
        from .batch import read_plan, run_tasks
        from .paper_plots import paper_plots, settings_with_defaults
        plan = read_plan(args.plan, 'bins-v1')
        settings = settings_with_defaults(yaml.safe_load(args.settings.read_text()) if args.settings else None)
        if args.task_index is not None and not 0 <= args.task_index < len(plan['tasks']):
            parser.error('Task index out of range')
        if args.task_index is not None and args.workers != 1:
            parser.error('Use one worker for an explicitly selected task')
        indices = [args.task_index] if args.task_index is not None else range(len(plan['tasks']))
        run_tasks(paper_plots, [(args.plan.resolve(), args.output.resolve(), i, settings) for i in indices], args.workers)
