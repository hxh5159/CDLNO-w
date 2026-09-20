"""Loop command planner: reuse native shell argv and real data-free parsers."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
TASKS=('airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car')
ENTRIES=dict(airfoil='exp_airfoil.py',darcy='exp_darcy.py',elasticity='exp_elas.py',
             pipe='exp_pipe.py',ns='exp_ns.py',plasticity='exp_plas.py')


def project(task):
    return ROOT/({'airfrans':'Airfoil-Design-AirfRANS','car':'Car-Design-ShapeNetCar'}.get(task,'PDE-Solving-StandardBenchmark'))


def native_parse(task,action,entry,tokens):
    """Execute only parser construction statements; never import a data entry."""
    folder=project(task);nodes=[]
    for node in ast.parse((folder/entry).read_text()).body:
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='parser':nodes.append(node)
        elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='parser.add_argument':nodes.append(node)
    scope={'argparse':argparse}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<native parser only>','exec'),scope)
    route=folder/('models/cdlno_run.py' if task=='car' else 'cdlno_entry.py')
    spec=importlib.util.spec_from_file_location('_loop_launcher_'+task,route)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parser=scope['parser']
    if task in ENTRIES:args=module.parse_args(parser,task,tokens)
    else:args=module.parse_args(parser,evaluation=action=='eval',argv=tokens)
    if getattr(args,'linearno_family',None)!='linearno_loop':raise ValueError('this launcher requires family=linearno_loop; old families use their own scripts')
    return args


def gpu_environment(tokens,environment):
    probe=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    probe.add_argument('--gpu',type=int,default=0)
    args,remaining=probe.parse_known_args(tokens)
    if args.gpu<0:raise ValueError('gpu must be a nonnegative index')
    env=dict(environment);visible=env.get('CUDA_VISIBLE_DEVICES')
    if visible not in (None,''):
        devices=visible.split(',')
        if any(not x.strip() for x in devices) or args.gpu>=len(devices):raise ValueError('--gpu is outside CUDA_VISIBLE_DEVICES')
        selected=devices[args.gpu].strip()
        if selected=='-1':raise ValueError('CUDA_VISIBLE_DEVICES=-1 disables GPUs')
    elif visible=='':selected=''  # Explicit CPU-only environment remains CPU-only.
    else:selected=str(args.gpu)
    env['CUDA_VISIBLE_DEVICES']=selected
    # Standard entries overwrite this variable with args.gpu. Pass the selected
    # physical identifier there; industrial adapters use the masked logical 0.
    return remaining,env,dict(requested_index=args.gpu,inherited_visible=visible,selected_visible=selected)


def plan(task,action,tokens,environment=None):
    if task not in TASKS or action not in ('train','resume','eval'):raise ValueError('expected TASK train|resume|eval')
    if any(any(ord(c)<32 for c in token) for token in tokens):raise ValueError('control characters are not supported in launcher arguments')
    banned={'--model','--cfd_model','--eval','--resume','--run_dir','--cdlno-run-dir'}
    if any(t.split('=')[0] in banned for t in tokens):raise ValueError('family/action are fixed; use train|resume|eval and --experiment-dir')
    remaining,env,gpu=gpu_environment(tokens,os.environ if environment is None else environment)
    env['CDLNO_REPO_ROOT']=str(ROOT)
    # No CUDA use during parser-only inspection.
    action_file='eval' if action=='eval' else 'train'
    extra=['--resume'] if action=='resume' else []
    # Standard writes CUDA_VISIBLE_DEVICES=str(args.gpu) and uses .cuda();
    # industrial entries use cuda:args.gpu under the selected mask.
    selected=gpu['selected_visible']
    if task in ENTRIES and selected and not selected.isdecimal():
        raise ValueError('Standard native --gpu requires numeric CUDA device IDs')
    native_gpu=selected if task in ENTRIES and selected else '0'
    launch_tokens=[*remaining,'--gpu',native_gpu,*extra]
    original=ROOT/f'tran_evaluate/linearno/{task}_{action_file}.sh'
    preview=subprocess.run(['bash',str(original),'--dry-run',*launch_tokens],env=env,text=True,capture_output=True)
    if preview.returncode:raise ValueError(preview.stderr or preview.stdout)
    lines=preview.stdout.splitlines()
    command=shlex.split(next(l[len('Command:'):].strip() for l in lines if l.startswith('Command:')))
    cwd=Path(next(l[len('Working directory:'):].strip() for l in lines if l.startswith('Working directory:')))
    entry=command[2];argv=command[3:]
    if cwd!=project(task) or entry!=(ENTRIES.get(task) or ('main_evaluation.py' if action=='eval' else 'main.py')):
        raise ValueError('unexpected native launcher target')
    if action!='train' and not any(t.split('=')[0] in ('--experiment-dir','--linearno-run-dir') for t in argv):
        raise ValueError('eval/resume requires the exact RUN; no latest guessing')
    prior=os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    try:args=native_parse(task,action,entry,argv)
    finally:
        if prior is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=prior
    run=str(args.linearno_run_dir)
    if action=='train' and not any(t.split('=')[0] in ('--experiment-dir','--linearno-run-dir') for t in argv):
        argv.extend(['--experiment-dir',run])
    config=args._linearno_loop_config
    return dict(task=task,action=action,cwd=str(cwd),entry=entry,argv=argv,run=run,
        config=config,gpu=gpu,python=command[0],environment=env,
        original_launcher=str(original),dry_run_scope='native parser and metadata only; no model/data/torch.load')


def execute(value):
    command=[value['python'],'-u','-B',str(Path(__file__).with_name('entry.py')),
             value['task'],value['action'],value['entry'],*value['argv']]
    print('Run: '+value['run'],flush=True)
    print('GPU mapping: '+json.dumps(value['gpu']),flush=True)
    print('Working directory: '+value['cwd'],flush=True)
    print('Command: '+shlex.join(command),flush=True)
    return subprocess.run(command,cwd=value['cwd'],env=value['environment']).returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('task',choices=TASKS);parser.add_argument('action',choices=('train','resume','eval'))
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--print-run-dir',action='store_true')
    parser.add_argument('--plan-json',action='store_true');parser.add_argument('--then-eval',action='store_true')
    args,tokens=parser.parse_known_args()
    try:
        if args.then_eval and args.action=='eval':raise ValueError('--then-eval is only for train/resume')
        value=plan(args.task,args.action,tokens)
        if args.print_run_dir:print(value['run']);return 0
        if args.plan_json:
            print(json.dumps({k:v for k,v in value.items() if k!='environment'}));return 0
        if args.dry_run:
            print('Run: '+value['run']);print('GPU mapping: '+json.dumps(value['gpu']))
            print('Working directory: '+value['cwd'])
            print('Native command: '+shlex.join([value['python'],'-u',value['entry'],*value['argv']]))
            print('DRY RUN: real parser/metadata validated; no model, data, weights or directory created.')
            if args.then_eval:print('Then eval same RUN only after successful training; future metadata validation is deferred.')
            return 0
        code=execute(value)
        if code or not args.then_eval:return code
        # The trained metadata supplies all architecture/profile/seed settings.
        follow=['--experiment-dir',value['run'],'--gpu','0']
        path_flags={'--data_path','--my_path','--data_dir','--save_dir','--device'}
        i=0
        while i<len(value['argv']):
            token=value['argv'][i];flag=token.split('=')[0]
            if flag in path_flags:
                follow.append(token)
                if '=' not in token:i+=1;follow.append(value['argv'][i])
            i+=1
        evaluation=plan(args.task,'eval',follow,value['environment'])
        return execute(evaluation)
    except (ValueError,OSError,StopIteration) as error:parser.error(str(error))


if __name__=='__main__':
    sys.path.insert(0,str(ROOT))
    raise SystemExit(main())
