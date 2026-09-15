"""No-data check of 16 remote launch commands and original parser definitions."""
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
AUDIT=Path(__file__).resolve().parent
REMOTE_DATA='/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data'
TASKS=('darcy','elasticity','airfoil','pipe','ns','plasticity','car','airfrans')
env={k:v for k,v in os.environ.items() if not k.startswith('CDLNO_')}
env.update(CDLNO_PYTHON=sys.executable,PYTHONPATH=str(ROOT))
PARSER=r'''
import argparse,ast,json,sys
from pathlib import Path
task,entry,*tokens=sys.argv[1:]
nodes=[]
for node in ast.parse(Path(entry).read_text()).body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='parser':
        nodes.append(node)
    elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and ast.unparse(node.value.func.value)=='parser' and node.value.func.attr=='add_argument':
        nodes.append(node)
scope={'argparse':argparse}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<only-original-parser-definitions>','exec'),scope)
if task=='car':
    from models.cdlno_run import parse_args
    args=parse_args(scope['parser'],evaluation=entry=='main_evaluation.py',argv=tokens)
elif task=='airfrans':
    from cdlno_entry import parse_args,resolve_hparams
    import yaml
    args=parse_args(scope['parser'],evaluation=entry=='main_evaluation.py',argv=tokens)
    hp=resolve_hparams(args,yaml.safe_load(Path('params.yaml').read_text())['CDLNO'])
else:
    from cdlno_entry import parse_args
    args=parse_args(scope['parser'],task,tokens)
result=vars(args)
if task=='airfrans': result['resolved_hparams']=hp
print(json.dumps(result,default=str))
'''


def run(command,cwd=ROOT,environment=None,expected=0):
    p=subprocess.run(command,cwd=cwd,env=environment or env,capture_output=True,text=True,timeout=45)
    assert p.returncode==expected,(command,p.returncode,p.stdout,p.stderr)
    return p.stdout


def launch(task,mode,cwd,extra=(),environment=None):
    output=run(['bash',str(ROOT/'tran_evaluate'/f'{task}.sh'),mode,*extra,'--dry-run'],cwd,environment)
    lines=output.splitlines()
    command=shlex.split(next(s[len('Command:'):] for s in lines if s.startswith('Command:')))
    project=Path(next(s[len('Working directory: '):] for s in lines if s.startswith('Working directory: ')))
    assert command[:2]==[sys.executable,'-u']
    parsed=json.loads(run([sys.executable,'-B','-c',PARSER,task,*command[2:]],project,environment))
    return command,project,parsed


records=[]
runs={}
for script in [ROOT/'path.sh',*(ROOT/'tran_evaluate').glob('*.sh')]:
    run(['bash','-n',str(script)])
for path in (ROOT/'tran_evaluate').glob('*.py'):
    ast.parse(path.read_text(),feature_version=(3,10))
with tempfile.TemporaryDirectory(prefix='cdlno remote launch check ') as tmp:
    cwd=Path(tmp)
    for task in TASKS:
        assert '--dry-run' in run(['bash',str(ROOT/'tran_evaluate'/f'{task}.sh'),'help'],cwd)
        for mode in ('train','eval'):
            command,project,args=launch(task,mode,cwd)
            config_task='shapenet_car' if task=='car' else task
            preset=json.loads((project/'configs/CDLNO'/f'{config_task}.json').read_text())
            for key,value in preset['model'].items(): assert args[key]==value,(task,mode,key,args[key],value)
            if task not in ('car','airfrans'):
                for key,value in preset['training'].items(): assert args[key]==value,(task,key)
                assert args['eval']==int(mode=='eval')
                expected_suffix=dict(darcy='fno',elasticity='fno',ns='fno',airfoil='fno/airfoil/naca',pipe='fno/pipe',plasticity='fno/plas_N987_T20.mat')[task]
                assert args['data_path']==REMOTE_DATA+'/'+expected_suffix
                run_path=args['cdlno_run_dir']
                assert run_path==str(ROOT/'runs/CDLNO'/task/'entry')
            elif task=='car':
                assert (args['nb_epochs'],args['weight'],args['fold_id'])==(200,.5,0)
                if mode=='train': assert (args['lr'],args['batch_size'],args['preprocessed'],args['val_iter'])==(.001,1,1,10)
                assert args['data_dir']==REMOTE_DATA+'/mlcfd_data/training_data'
                assert args['save_dir']==REMOTE_DATA+'/mlcfd_data/preprocessed_data'
                run_path=args['run_dir']
            else:
                assert args['resolved_hparams']==preset['initial_training']
                assert args['my_path']==REMOTE_DATA+'/AirfRANS'+('/Dataset' if mode=='train' else '')
                assert args['task']=='full' and args['nmodel']==1
                if mode=='train': assert args['score']==0
                run_path=args['run_dir']
            if task in runs: assert runs[task]==run_path
            runs[task]=run_path
            records.append(dict(task=task,mode=mode,command=command,project=str(project),arguments=args,status='passed'))

        # Through shell dispatch AND parser: spaces, relative paths, --flag=value,
        # last-option overrides for architectures/chunks and independent run paths.
        standard=task not in ('car','airfrans')
        extras=(['--data_path=data path','--cdlno-run-dir','other run','--n-layers=12','--front-blocks','3','--cdpa-source-chunk-size=2'] if standard else
                ['--run_dir','other run','--n_layers=12','--front_blocks','3','--cdpa_source_chunk_size=2'] +
                (['--data_dir=data path','--save_dir','cache path'] if task=='car' else ['--my_path=data path']))
        _,_,custom=launch(task,'eval',cwd,extras)
        assert (custom['n_layers'],custom['front_blocks'],custom['cdpa_source_chunk_size'])==(12,3,2)
        assert custom['cdlno_run_dir' if standard else 'run_dir']==str(cwd/'other run')
        assert custom['data_path' if standard else ('data_dir' if task=='car' else 'my_path')]==str(cwd/'data path')
        if task=='car': assert custom['save_dir']==str(cwd/'cache path')

    assert len(set(runs.values()))==8
    assert list(cwd.iterdir())==[], 'Launch dry-run must never create run/data files'
    override_env={**env,'CDLNO_DATA_ROOT':str(cwd/'alternate data'),'CDLNO_RUNS_ROOT':str(cwd/'output'),'CDLNO_RUN_TAG':'test2'}
    _,_,custom=launch('elasticity','train',cwd,environment=override_env)
    assert custom['data_path']==str(cwd/'alternate data/fno')
    assert custom['cdlno_run_dir']==str(cwd/'output/elasticity/test2')
    # No fake dataset files. An empty temporary directory checks honest missing
    # reporting; NPY header logic below is exercised entirely in memory.
    empty_env={**env,'CDLNO_DATA_ROOT':str(cwd)}
    report=json.loads(run(['bash',str(ROOT/'tran_evaluate/inspect_data.sh')],cwd,empty_env))
    assert report['root_exists'] and not report['discovery']['candidates']
    assert all(not item['exists'] for name,files in report['tasks'].items() if name!='car' for item in files)
    missing_env={**env,'CDLNO_DATA_ROOT':str(cwd/'absent')}
    report=json.loads(run(['bash',str(ROOT/'tran_evaluate/inspect_data.sh'),'--no-search'],cwd,missing_env,expected=2))
    assert not report['root_exists']
    assert list(cwd.iterdir())==[]

spec=importlib.util.spec_from_file_location('cdlno_readonly_inspector',ROOT/'tran_evaluate/inspect_data.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
for version in ((1,0),(2,0),(3,0)):
    header=b"{'descr': '<f4', 'fortran_order': True, 'shape': (972, 2, 1200), }\n"
    content=b'\x93NUMPY'+bytes(version)+struct.pack('<H' if version[0]==1 else '<I',len(header))+header
    result=module.npy_header(io.BytesIO(content))
    assert result['shape']==[972,2,1200] and result['fortran_order'] and not result['values_read']
try: module.npy_header(io.BytesIO(b'wrong!'))
except ValueError: pass
else: raise AssertionError('Invalid NPY magic accepted')

payload=dict(status='passed',launches=records,distinct_task_run_directories=len(runs),
             additional_eight_override_cases='passed',environment_override='passed',
             shell_and_python310_syntax='passed',missing_data_readonly_checks='passed',
             npy_in_memory_header_versions=[1,2,3],real_data_and_real_mat_vtk_manifest_checks='not run',
             model_forward_backward_and_gpu='not run: no model changes in this task')
(AUDIT/'launcher-checks.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print('PASS: 16 default commands + 8 override commands, original AST-only parsers, 8 isolated paths, syntax, empty/missing data and 3 in-memory NPY headers.')
