"""24 paired-mode native synthetic runs with LL8 records, compared to LL7."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from linearno_loop.config import run_directory_id
from tran_evaluate.linearno_loop.launch import project

def main():
    before=[]
    for name in ('native-matrix.json','industrial-matrix.json'):
        before+=json.loads((OUT.parent/'ll7'/name).read_text())['rows']
    cases=[r for r in before if r['preset']=='p1_c3_r2_s1']
    temp=Path(tempfile.mkdtemp(prefix='loop-ll8-native-',dir='/home/hwz/CDLNO-artifacts'))
    env=dict(os.environ,PYTHONPATH=str(ROOT/'tests')+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
    def one(case):
        task,preset,mode=(case[k] for k in ('task','preset','mode'));air=task=='airfrans'
        old_root=Path(case['artifact']);old_run=next((old_root/('full' if task in ('airfrans','car') else 'continuous')).iterdir())
        cfg=json.loads((old_run/'architecture.json').read_text())['resolved_config']
        root=temp/task/mode;root.mkdir(parents=True);run=root/run_directory_id(cfg);report=root/'result.json'
        cmd=[sys.executable,'-B',str(ROOT/'tests/loop_linearno/launch_record_worker.py'),task,'ensemble' if air else 'train',str(run),str(report),preset,mode]
        with (root/'worker.log').open('w') as stream:
            result=subprocess.run(cmd,cwd=project(task),env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        if result.returncode:raise AssertionError(str(root/'worker.log')+'\n'+(root/'worker.log').read_text()[-5500:])
        current=json.loads(report.read_text())
        if task in ('airfrans','car'):previous=case['reports']['full-'+('ensemble' if air else 'train')]
        else:
            candidates=list(old_root.glob('*train*.json'))
            if not candidates:candidates=[old_root/'full-train.json']
            previous=json.loads(candidates[0].read_text())
        # Native reports use different names for Standard state hashes.
        keys=[k for k in ('state_hash','resume_hash','prediction','prediction_hash','batches','time_queries','numpy_next') if k in previous]
        assert keys and 'batches' in keys
        for key in keys:assert current[key]==previous[key],(task,mode,key)
        if task not in ('airfrans','car'):
            from cdlno.linearno_loop.checkpoint import inspect_checkpoint,read_pair
            from cdlno.training_state import _same
            a,wa=read_pair(inspect_checkpoint(old_run,'final')[1]);b,wb=read_pair(inspect_checkpoint(run,'final')[1])
            assert _same(wa,wb),(task,mode,'weights')
            assert a['resume_state']==b['resume_state'],(task,mode,'optimizer_scheduler_rng')
            keys+=['strict_final_weights','optimizer_scheduler_rng']
        manifest=json.loads((run/'loop_run_manifest.json').read_text())
        assert all(r['actual_call_schedule'] for r in manifest['members'].values())
        return dict(task=task,mode=mode,artifact=str(run),exact_LL7=keys,manifest=manifest,batches=current['batches'])
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(one,r) for r in cases]):
            row=future.result();rows.append(row);print(row['task'],row['mode'],'exact',flush=True)
            (OUT/'recorded-native.json').write_text(json.dumps(dict(expected=24,completed=len(rows),artifact_root=str(temp),rows=rows),indent=2)+'\n')
    for task in {r['task'] for r in rows}:
        triple=[r for r in rows if r['task']==task];assert len(triple)==3
        for row in triple[1:]:
            assert row['batches']==triple[0]['batches'],(task,'paired data order')
            for member,v in row['manifest']['members'].items():
                assert v['public_backbone_initial_sha256']==triple[0]['manifest']['members'][member]['public_backbone_initial_sha256']
    print('24 native runs: observation exact versus LL7; paired backbone hashes/data order equal across three modes',flush=True)

if __name__=='__main__':main()
