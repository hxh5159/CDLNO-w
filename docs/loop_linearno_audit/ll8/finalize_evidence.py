"""LL8 actual evidence and complete old-tree preservation, including ignored files."""
from collections import Counter
import ast,difflib,hashlib,importlib.metadata,json,os,platform,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
ALLOWED={'docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md','tests/loop_linearno/test_artifacts.py'}
PREFIX='docs/loop_linearno_audit/ll8/'
NEW={'docs/LOOP_LINEARNO_COMMANDS.md','docs/LOOP_LINEARNO_LL8_LAUNCHERS.md',
     'tests/loop_linearno/test_launchers.py','tests/loop_linearno/launch_record_worker.py'}
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def read(name):return json.loads((OUT/name).read_text())
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def inventory():
    rows={}
    for kind,args in [('tracked',[]),('untracked',['--others','--exclude-standard']),('ignored',['--others','--ignored','--exclude-standard'])]:
        for raw in git('ls-files','-z',*args).split(b'\0'):
            if raw:
                name=os.fsdecode(raw);p=ROOT/name
                if p.is_file():rows[name]=dict(path=name,classification=kind,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return rows

def results():
    matrix=read('dry-run-matrix.json');assert matrix['previews']==184 and matrix['negative_checks']==240
    assert all(r['passed'] for r in matrix['rows']) and matrix['new_run_paths_unique']
    extra=read('additional-conflicts.json');assert len(extra)==144 and all(r['rejected'] for r in extra)
    fair=read('fair-report.json');assert len(fair)==144
    for task in {r['task'] for r in fair}:
        for preset in ('p1_c3_r2_s1','p2_c2_r2_s2'):
            for seed in (0,1,2):
                triple=[r for r in fair if (r['task'],r['preset'],r['seed'])==(task,preset,seed)]
                assert len(triple)==3 and len({r['backbone_hash'] for r in triple})==1
                assert all(r['data_order']==triple[0]['data_order'] for r in triple)
    native=read('recorded-native.json');assert native['completed']==native['expected']==24
    for task in {r['task'] for r in native['rows']}:
        triple=[r for r in native['rows'] if r['task']==task];assert len(triple)==3
        for r in triple:
            assert 'batches' in r['exact_LL7'] and r['batches']==triple[0]['batches']
            for member,values in r['manifest']['members'].items():
                assert values['actual_call_schedule']==r['manifest']['expected_call_schedule']
                assert values['public_backbone_initial_sha256']==triple[0]['manifest']['members'][member]['public_backbone_initial_sha256']
    for name in ('attnres-report.json','body-report.json','sr-parity.json','accounting.json','rb-report.json','lb-report.json','modes-report.json'):
        assert read(name)==json.loads((OUT.parent/'ll7'/name).read_text()),name
    log=(OUT/'loop-tests.log').read_text();assert 'Ran 88 tests in 54.950s' in log and 'OK (skipped=2)' in log
    log=(OUT/'launcher-final.log').read_text();assert 'Ran 6 tests in 5.429s' in log and '\nOK\n' in log
    from cdlno.linearno.air_entry import _provenance
    from cdlno.linearno.car_entry import task_provenance
    from cdlno.linearno_loop.provenance import provenance
    from cdlno.linearno_history.provenance import research_provenance
    from cdlno.linearno_loop.industrial_state import provenance as industrial
    previous=json.loads((OUT.parent/'ll7/preserved-provenance.json').read_text());hashes={}
    for name,fn in [('air',_provenance),('car',task_provenance),('history_air',lambda:research_provenance(_provenance())),('history_car',lambda:research_provenance(task_provenance())),('loop',provenance)]:
        now=fn();hashes[name]={k:now[k] for k in ('source_sha256','normalized_patch_sha256')}
        assert hashes[name]==previous[name],name
    for task in ('car','airfrans'):
        row=next(r for r in native['rows'] if r['task']==task)
        metadata=json.loads((Path(row['artifact'])/'architecture.json').read_text());now=industrial(task)
        hashes['loop_'+task]={k:now[k] for k in ('source_sha256','normalized_patch_sha256')}
        assert all(now[k]==metadata['provenance_spec'][k] for k in hashes['loop_'+task])
    write('preserved-provenance.json',hashes)
    checks=[]
    for p in sorted((ROOT/'tran_evaluate/linearno_loop').glob('*')):
        if p.suffix=='.sh':
            subprocess.run(['bash','-n',str(p)],check=True);checks.append(str(p.relative_to(ROOT)))
        elif p.suffix=='.py':ast.parse(p.read_text(),feature_version=(3,10));compile(p.read_text(),str(p),'exec')
    for path in NEW|ALLOWED:
        if path.endswith('.py'):ast.parse((ROOT/path).read_text(),feature_version=(3,10))
    blocks=re.findall(r'```bash\n(.*?)```',(ROOT/'docs/LOOP_LINEARNO_COMMANDS.md').read_text(),re.S)
    for block in blocks:subprocess.run(['bash','-n'],input=block,text=True,check=True)
    write('environment.json',dict(python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ('torch','numpy','torch-geometric','timm','einops')},device='CPU only',GPU_remote='NOT RUN'))
    write('delivery-review.json',dict(stage='LL8',status='PASS',tests=dict(full_loop_methods=88,pass_=86,skip=2,failures=0,errors=0,seconds=54.950,
        final_targeted_methods=6,final_targeted_pass=6,final_targeted_seconds=5.429,shell_previews=184,negative_cases=384,
        paired_models=144,recorded_native_runs=24,document_bash_blocks=len(blocks),shell_syntax=checks),
        reviewed=dict(launch='original launcher argv, actual parser, explicit bridge to native entry; Car uses LL7 loop contract',
        fairness='same backbone hashes and data order across modes; RNG-neutral observations, native scientific steps exact versus LL7',
        records='actual initial models/counts, first-forward hook schedule, immutable existing manifests, independent Air members',
        compatibility='all production/schema/model/path/old launcher files byte-equal; all legacy/loop source fingerprints equal',
        user_flow='portable remote paths, GPU mask mapping, quoted spaces, errors propagated, evaluation only after success'),
        not_run=['real datasets/training/VTK metrics','GPU/remote Python3.10 torch2.11 cu128','formal144-run matrix','full-width industrial training','LL9+']))

def freeze():
    original=read('start-manifest.json');before={r['path']:r for r in original['files']};now=inventory()
    added={p:r for p,r in now.items() if p not in before}
    allowed_new=lambda p:p in NEW or p.startswith((PREFIX,'tran_evaluate/linearno_loop/'))
    changed=[dict(path=p,before=r,after=now.get(p)) for p,r in before.items() if now.get(p)!=r]
    unexpected=[r for r in changed if r['path'] not in ALLOWED]
    extra=[r for p,r in added.items() if not allowed_new(p)]
    ignored=[r for p,r in added.items() if r['classification']=='ignored' and not(p.startswith(PREFIX) and p.endswith('.log'))]
    diff=[]
    for p in sorted(ALLOWED):
        old=(OUT/'before'/p).read_text();assert hashlib.sha256(old.encode()).hexdigest()==before[p]['sha256']
        diff.extend(difflib.unified_diff(old.splitlines(True),(ROOT/p).read_text().splitlines(True),fromfile='before-LL8/'+p,tofile=p))
    for p in sorted(added):
        if p in NEW or p.startswith('tran_evaluate/linearno_loop/'):
            diff.extend(difflib.unified_diff([], (ROOT/p).read_text().splitlines(True),fromfile='/dev/null',tofile=p))
    (OUT/'source.diff').write_text(''.join(diff));(OUT/'end-status.txt').write_bytes(git('status','--short','--branch','--untracked-files=all'))
    check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    flags=dict(head_unchanged=git('rev-parse','HEAD').decode().strip()==original['head'],tree_unchanged=git('rev-parse','HEAD^{tree}').decode().strip()==original['tree'],
        staged_unchanged=git('diff','--cached','--binary')==(OUT/'start-staged.diff').read_bytes(),diff_check=check.returncode==0,
        all_existing_production_unchanged=not any(r['path'].startswith(('cdlno/','linearno_loop/','PDE-Solving-StandardBenchmark/','Airfoil-Design-AirfRANS/','Car-Design-ShapeNetCar/','tran_evaluate/','monitor/')) or r['path']=='path.sh' for r in changed))
    value=dict(baseline_count=len(before),unchanged=len(before)-len(changed),changed=changed,
        unexpected_changes=unexpected,unexpected_new=extra,unexpected_ignored=ignored,checks=flags,
        head=original['head'],tree=original['tree'],branch=git('branch','--show-current').decode().strip(),remote=git('remote','-v').decode(),
        new_files=[r for p,r in inventory().items() if p not in before and p!=PREFIX+'end-freeze.json'],
        passed=not(unexpected or extra or ignored) and all(flags.values()))
    write('end-freeze.json',value)
    print(json.dumps({k:value[k] for k in ('passed','baseline_count','unchanged','unexpected_changes','unexpected_new','unexpected_ignored','checks')},indent=2))
    assert value['passed']

if __name__=='__main__':results();freeze()
