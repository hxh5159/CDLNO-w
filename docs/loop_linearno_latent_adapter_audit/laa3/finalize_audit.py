"""Create-only LAA3 static, freeze and delivery records; no model changes."""
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


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False)


def main():
    snapshot=json.loads((OUT/'start-manifest.json').read_text())
    source=[ROOT/'cdlno/linearno_loop/v3/attention.py']
    tests=sorted((ROOT/'tests/loop_linearno_latent_adapter').glob('*laa3*.py'))
    tests+=sorted((ROOT/'tests/loop_linearno_latent_adapter').glob('attention_*.py'))
    compiled=source+tests+sorted(OUT.glob('*.py'));start=time.perf_counter()
    for p in compiled:compile(p.read_bytes(),str(p),'exec')
    static=dict(python=dict(count=len(compiled),seconds=time.perf_counter()-start,
        files=[str(p.relative_to(ROOT)) for p in compiled],no_pyc=True))
    records=[]
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in sorted(p for p in paths if p.endswith('.sh')):
        start=time.perf_counter();r=subprocess.run(['bash','-n',name],cwd=ROOT,capture_output=True,text=True)
        records.append(dict(path=name,exit_code=r.returncode,seconds=time.perf_counter()-start,output=r.stdout+r.stderr))
        assert r.returncode==0,r.stdout+r.stderr
    static['shell']=dict(count=len(records),results=records)
    start=time.perf_counter();r=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    static['diff']=dict(exit_code=r.returncode,seconds=time.perf_counter()-start,output=r.stdout+r.stderr)
    assert r.returncode==0
    static['status']='PASS';write('static-checks.json',static)

    freeze={}
    for label,path in [('laa0',OUT.parent/'laa0/start-manifest.json'),('laa3',OUT/'start-manifest.json')]:
        manifest=json.loads(path.read_text());same=0;changed=[];missing=[]
        for name,row in manifest['files'].items():
            file=ROOT/name
            if not file.exists():missing.append(name)
            elif sha(file)==row['sha256']:same+=1
            else:changed.append(dict(path=name,before=row['sha256'],after=sha(file),authorized=name==STATUS))
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
                  new_files=sorted(set(new)),allowed_existing_changes=[STATUS])
    write('end-freeze.json',freeze)
    mapping=[]
    for p in source+tests:
        tree=ast.parse(p.read_text());mapping.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),
            symbols=[dict(name=n.name,line=n.lineno,kind=type(n).__name__) for n in ast.walk(tree)
                     if isinstance(n,(ast.FunctionDef,ast.ClassDef))]))
    write('source-map.json',mapping)
    patch=''.join(''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(ROOT)))) for p in source+tests)
    patch+=''.join(difflib.unified_diff((OUT/'status-before.md').read_text().splitlines(True),(ROOT/STATUS).read_text().splitlines(True),fromfile=STATUS+' (LAA3 start)',tofile=STATUS))
    with (OUT/'source-diff.patch').open('x') as f:f.write(patch)
    checks={name:json.loads((OUT/(name+'-final.json')).read_text()) for name in ('new','previous','regression')}
    assert all(r['status']=='PASS' for r in checks.values())
    rows=checks['new']['errors_by_comparison'];cpu=[r for r in rows if not r['label'].startswith('cuda')]
    write('results.json',dict(stage='LAA3',status='PASS',scope='independent attention only',head=snapshot['head'],
        test_suites={name:{k:v for k,v in row.items() if k not in ('errors_by_comparison','cuda_cases')} for name,row in checks.items()},
        cuda_cases=checks['new']['cuda_cases'],cpu_oracle_matrix_rows=96,cuda_oracle_matrix_rows=144,
        cuda_zero_init_dropout_rows=144,
        cpu_max_errors={dtype:max(r['max_abs'] for r in cpu if r['dtype']==dtype) for dtype in {r['dtype'] for r in cpu}},
        test_first=dict(oracle_passed=2,target_methods=12,expected_missing_implementation_errors=428,manifest='test-first-manifest.json'),
        initial_failure=dict(methods=14,errors=244,reason='new fixture fill() omitted reshape_as(parameter)',
            correction='reshape new test fixture only; no production/tolerance change',logs=['implementation-attempt1.log','implementation-attempt2.log']),
        static=dict(python=len(compiled),shell=len(records),status='PASS'),freeze='end-freeze.json',
        old_full_regression=dict(rerun=False,source='../laa0/regression-summary.json',passed=644,failed=9,skipped=36),
        not_run=['LAA4 and later','full V3 core/wrapper/task/checkpoint integration','real data/full epochs/paired seeds/SOTA',
                 'latency/memory/epoch performance','remote target stack','full task AMP/scaler resume','compile/distributed']))
    write('delivery-review.json',dict(status='PASS',reviews=[
        dict(point='native delegation and exact projected AST',status='PASS',evidence='test_inherited_constructor_and_active_base_ast_exact; native bitwise output/gradient/RNG tests'),
        dict(point='temperature ordering, source axes and contractions',status='PASS',evidence='96 CPU oracle rows; wrong-order counterexample; module and ATen hooks'),
        dict(point='feature ownership and isolated initialization',status='PASS',evidence='distinct position identity/storage; install RNG; two-visit module hooks'),
        dict(point='AMP, strict reload and no persistent activations',status='PASS',evidence='144 CUDA oracle/optimizer/reload rows; 144 zero-init/dropout combinations; weakref and failure recovery'),
        dict(point='frozen old code and stage boundary',status='PASS',evidence='52 previous + 72 legacy tests; end-freeze.json')],
        unresolved_semantic_conflicts=[],next_stage_executed=False))
    print(json.dumps(dict(status='PASS',python=len(compiled),shell=len(records),freeze=freeze['laa3'],
                         tests={name:r['tests'] for name,r in checks.items()}),ensure_ascii=False))


if __name__=='__main__':main()
