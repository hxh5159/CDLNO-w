"""Create-only final LAA4 freeze, syntax, source-map and delivery records."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
STATUS='docs/LOOP_LINEARNO_LATENT_ADAPTER_IMPLEMENTATION_STATUS.md'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False)


def main():
    snapshot=json.loads((OUT/'start-manifest.json').read_text())
    source=[ROOT/'cdlno/linearno_loop/v3/core.py']
    tests=sorted((ROOT/'tests/loop_linearno_latent_adapter').glob('*laa4*.py'))
    tests+=sorted((ROOT/'tests/loop_linearno_latent_adapter').glob('core_*.py'))
    compiled=source+tests+sorted(OUT.glob('*.py'));start=time.perf_counter()
    for p in compiled:compile(p.read_bytes(),str(p),'exec')
    static=dict(python=dict(count=len(compiled),seconds=time.perf_counter()-start,files=[str(p.relative_to(ROOT)) for p in compiled],no_pyc=True))
    records=[]
    for name in sorted(p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p.endswith('.sh')):
        start=time.perf_counter();r=subprocess.run(['bash','-n',name],cwd=ROOT,capture_output=True,text=True)
        records.append(dict(path=name,exit_code=r.returncode,seconds=time.perf_counter()-start,output=r.stdout+r.stderr));assert r.returncode==0
    static['shell']=dict(count=len(records),results=records)
    start=time.perf_counter();r=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    static['diff']=dict(exit_code=r.returncode,seconds=time.perf_counter()-start,output=r.stdout+r.stderr);assert r.returncode==0
    static['status']='PASS';write('static-checks.json',static)
    freeze={}
    for label,path in [('laa0',OUT.parent/'laa0/start-manifest.json'),('laa4',OUT/'start-manifest.json')]:
        manifest=json.loads(path.read_text());same=0;changed=[];missing=[]
        for name,row in manifest['files'].items():
            p=ROOT/name
            if not p.exists():missing.append(name)
            elif sha(p)==row['sha256']:same+=1
            else:changed.append(dict(path=name,before=row['sha256'],after=sha(p),authorized=name==STATUS))
        assert not missing and all(r['authorized'] for r in changed),(changed,missing)
        if label=='laa0':assert not changed
        freeze[label]=dict(compared=len(manifest['files']),unchanged=same,changed=changed,missing=missing)
    assert subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True)==snapshot['diff']
    assert subprocess.check_output(['git','diff','--cached','--binary'],cwd=ROOT,text=True)==snapshot['staged']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==snapshot['head']
    new=[]
    for args in ([],['--others','--exclude-standard'],['--others','--ignored','--exclude-standard']):
        for name in subprocess.check_output(['git','ls-files','-z',*args],cwd=ROOT).decode().split('\0'):
            if name and name not in snapshot['files']:new.append(name)
    freeze.update(status='PASS',user_diff_unchanged=True,staged_unchanged=True,head_unchanged=True,
        allowed_existing_changes=[STATUS],new_files=sorted(set(new)));write('end-freeze.json',freeze)
    mapping=[]
    for p in source+tests:
        tree=ast.parse(p.read_text());mapping.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),
            symbols=[dict(name=n.name,line=n.lineno,kind=type(n).__name__) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]))
    write('source-map.json',mapping)
    patch=''.join(''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(ROOT)))) for p in source+tests)
    patch+=''.join(difflib.unified_diff((OUT/'status-before.md').read_text().splitlines(True),(ROOT/STATUS).read_text().splitlines(True),fromfile=STATUS+' (LAA4 start)',tofile=STATUS))
    with (OUT/'source-diff.patch').open('x') as f:f.write(patch)
    checks={name:json.loads((OUT/(name+'-final.json')).read_text()) for name in ('new','previous','regression')}
    assert all(r['status']=='PASS' for r in checks.values())
    n=checks['new'];trace=json.loads((OUT/'round-traces.json').read_text());assert trace['status']=='PASS'
    cpu={dtype:dict(rows=sum(r['dtype']==dtype for r in n['oracle_rows']),
        output_max_abs=max(r['errors']['output'] for r in n['oracle_rows'] if r['dtype']==dtype),
        all_max_abs=max(max(r['errors'].values()) for r in n['oracle_rows'] if r['dtype']==dtype)) for dtype in ('torch.float64','torch.float32')}
    write('results.json',dict(stage='LAA4',status='PASS',scope='task-independent core only',head=snapshot['head'],
        test_suites={name:{k:v for k,v in row.items() if k not in ('oracle_rows','cost_rows','cuda_rows','formal_profile_rows')} for name,row in checks.items()},
        matrices=dict(cpu_oracle=len(n['oracle_rows']),parameter_mac=len(n['cost_rows']),cuda=len(n['cuda_rows']),formal_profile_core=len(n['formal_profile_rows'])),
        cpu_errors=cpu,cuda_rows=n['cuda_rows'],formal_profile_rows=n['formal_profile_rows'],round_traces='round-traces.json',
        router_cost_note='LAA1 logical router formula includes first RB singleton; actual execution saves 2*B*N*H. Both counts and difference verified; old formula untouched.',
        first_tests=dict(oracle_passed=2,missing_implementation_methods=9,expected_errors=360),
        initial_failure=dict(methods=11,failed_subcases=96,errors=0,reason='new accounting observer missed functional.linear inside adapter',
                             correction='observe actual functional calls within adapter; no formula/tolerance change'),
        static=dict(python=len(compiled),shell=len(records),status='PASS'),freeze='end-freeze.json',
        old_full_regression=dict(rerun=False,source='../laa0/regression-summary.json',passed=644,failed=9,skipped=36),
        not_run=['LAA5 and later','eight-task wrapper/CLI/checkpoint/RNG/DataLoader integration',
                 'real data/full epochs/paired seeds/SOTA','latency/memory/epoch efficiency',
                 'remote target environment','full task AMP/scaler resume','compile/distributed']))
    write('delivery-review.json',dict(status='PASS',reviews=[
        dict(point='full physical block ownership and once-only head',status='PASS',evidence='module/parameter ids, schedules, state keys, qkv/context and head hooks'),
        dict(point='SR/RB/LB math, live graph and inherited dtype helper',status='PASS',evidence='108 CPU oracle rows; round-traces.json; 54 CUDA cases'),
        dict(point='initialization, off equivalence and feature timing',status='PASS',evidence='exact allocation/apply RNG; four ablations; v1 AdamW/dropout; first-round gradient tests'),
        dict(point='strict configuration guard and actual costs',status='PASS',evidence='metadata conflicts before tensor application; 192 cost rows with explicit singleton savings'),
        dict(point='old code freeze and stage scope',status='PASS',evidence='68 previous/72 legacy; end-freeze.json')],unresolved_semantic_conflicts=[],next_stage_executed=False))
    print(json.dumps(dict(status='PASS',python=len(compiled),shell=len(records),freeze=freeze['laa4'],
        tests={k:v['tests'] for k,v in checks.items()},cpu_errors=cpu),ensure_ascii=False))


if __name__=='__main__':main()
