"""LAA2 compile/freeze/diff evidence, without changing any model or old file."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
STATUS='docs/LOOP_LINEARNO_LATENT_ADAPTER_IMPLEMENTATION_STATUS.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False)


def main():
    snapshot=json.loads((OUT/'start-manifest.json').read_text())
    source=sorted((ROOT/'cdlno/linearno_loop/v3').glob('*.py'))
    tests=sorted((ROOT/'tests/loop_linearno_latent_adapter').glob('*laa2*.py'))
    tests+=sorted((ROOT/'tests/loop_linearno_latent_adapter').glob('primitive_*.py'))
    compiled=source+tests+sorted(OUT.glob('*.py'))
    start=time.perf_counter()
    for path in compiled:compile(path.read_bytes(),str(path),'exec')
    static=dict(python=dict(count=len(compiled),seconds=time.perf_counter()-start,
                           files=[str(p.relative_to(ROOT)) for p in compiled],no_pyc=True))
    shells=sorted(p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p.endswith('.sh'))
    records=[]
    for path in shells:
        start=time.perf_counter();p=subprocess.run(['bash','-n',path],cwd=ROOT,capture_output=True,text=True)
        records.append(dict(path=path,exit_code=p.returncode,seconds=time.perf_counter()-start,output=p.stdout+p.stderr))
        assert p.returncode==0,p.stdout+p.stderr
    static['shell']=dict(count=len(shells),results=records)
    start=time.perf_counter();p=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    static['diff']=dict(exit_code=p.returncode,seconds=time.perf_counter()-start,output=p.stdout+p.stderr)
    assert p.returncode==0
    static['status']='PASS';write('static-checks.json',static)

    freeze={}
    for label,path in [('laa0',OUT.parent/'laa0/start-manifest.json'),('laa2',OUT/'start-manifest.json')]:
        manifest=json.loads(path.read_text());same=0;changed=[];missing=[]
        for name,record in manifest['files'].items():
            f=ROOT/name
            if not f.exists():missing.append(name)
            elif sha(f)==record['sha256']:same+=1
            else:changed.append(dict(path=name,before=record['sha256'],after=sha(f),authorized=name==STATUS))
        assert not missing and all(r['authorized'] for r in changed),(label,changed,missing)
        if label=='laa0':assert not changed
        freeze[label]=dict(compared=len(manifest['files']),unchanged=same,changed=changed,missing=missing)
    laa1=json.loads((OUT.parent/'laa1/end-freeze.json').read_text())['new_code']
    assert all(sha(ROOT/path)==digest for path,digest in laa1.items())
    assert subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True)==snapshot['diff']
    assert subprocess.check_output(['git','diff','--cached','--binary'],cwd=ROOT,text=True)==snapshot['staged']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==snapshot['head']
    new_paths=[]
    for args in (['git','ls-files','-z'],['git','ls-files','--others','--exclude-standard','-z'],['git','ls-files','--others','--ignored','--exclude-standard','-z']):
        for name in subprocess.check_output(args,cwd=ROOT).decode().split('\0'):
            if name and name not in snapshot['files']:new_paths.append(name)
    freeze.update(status='PASS',laa1_code_files_unchanged=len(laa1),user_diff_unchanged=True,
                  staged_unchanged=True,head_unchanged=True,new_files=sorted(set(new_paths)),
                  allowed_existing_changes=[STATUS],git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    write('end-freeze.json',freeze)
    mapping=[]
    for path in source+tests:
        tree=ast.parse(path.read_text());mapping.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),
            lines=len(path.read_text().splitlines()),symbols=[dict(name=n.name,line=n.lineno,kind=type(n).__name__)
                for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]))
    write('source-map.json',mapping)
    patch=''.join(''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(ROOT)))) for p in source+tests)
    patch+=''.join(difflib.unified_diff((OUT/'status-before.md').read_text().splitlines(True),(ROOT/STATUS).read_text().splitlines(True),fromfile=STATUS+' (LAA2 start)',tofile=STATUS))
    with (OUT/'source-diff.patch').open('x') as f:f.write(patch)

    checks={name:json.loads((OUT/(name+'-final.json')).read_text()) for name in ('new','schema','regression')}
    assert all(value['status']=='PASS' for value in checks.values())
    cpu=checks['new']['cpu_error_rows']
    results=dict(stage='LAA2',status='PASS',scope='independent primitives only',head=snapshot['head'],
        test_suites={name:{k:v for k,v in row.items() if k not in ('cpu_error_rows','cuda_cases')} for name,row in checks.items()},
        cuda_cases=checks['new']['cuda_cases'],
        cpu_max_errors={dtype:max(r['max_absolute'] for r in cpu if r['dtype']==dtype) for dtype in {r['dtype'] for r in cpu}},
        test_first=dict(oracle_passed=2,expected_missing_implementation_errors=20,manifest='test-first-manifest.json'),
        intermediate_failure=dict(tests=22,passed=21,failed=1,reason='new test incorrectly demanded bitwise equality for identical heads across GEMM rows; FP64 difference 6.94e-18',
                                  resolution='use the predeclared FP64 mathematical tolerance; no production math or old tolerance/golden change',log='primitives-attempt1.log'),
        static=dict(python=len(compiled),shell=len(shells),status='PASS'),freeze='end-freeze.json',
        parameter_examples=[dict(kind='adapter',dh=3,M=5,r=4,parameters=64),dict(kind='adapter',dh=26,M=32,r=4,parameters=464),
            dict(kind='adapter',dh=16,M=64,r=4,parameters=640),dict(kind='latent',H=6,Dz=7,parameters=109),
            dict(kind='latent',H=104,Dz=704,parameters=147448),dict(kind='latent',H=208,Dz=776,parameters=324216)],
        old_full_regression=dict(rerun=False,source='../laa0/regression-summary.json',passed=644,failed=9,skipped=36),
        not_run=['LAA3 and later','V3 complete attention/core/wrapper/task/checkpoint integration','real datasets and full training',
                 'three-seed convergence/accuracy/SOTA','real latency/epoch-time/memory performance','remote target stack',
                 'full task AMP/scaler resume','compile/distributed'])
    write('results.json',results)
    write('delivery-review.json',dict(status='PASS',reviews=[
        dict(point='independent oracle and contraction axes',status='PASS',evidence=['test-first-manifest.json','test_laa2_oracles','test_laa2_adapter','test_laa2_latent']),
        dict(point='unchanged V2 latent class, keys and initialization',status='PASS',evidence=['test_exact_v2_class_keys_initialization_and_outputs','regression-final.log','end-freeze.json']),
        dict(point='isolated feature RNG and staged gradients',status='PASS',evidence=['test_laa2_rng_amp','test_laa2_adapter','test_laa2_latent']),
        dict(point='AMP dtype and strict state reload',status='PASS',evidence=['new-final.json']),
        dict(point='stage isolation and legacy freeze',status='PASS',evidence=['schema-final.json','regression-final.json','end-freeze.json'])],
        unresolved_semantic_conflicts=[],next_stage_executed=False))
    print(json.dumps(dict(status='PASS',python=len(compiled),shell=len(shells),freeze={k:freeze[k] for k in ('laa0','laa2')},test_counts={k:v['tests'] for k,v in checks.items()}),ensure_ascii=False))


if __name__=='__main__':main()
