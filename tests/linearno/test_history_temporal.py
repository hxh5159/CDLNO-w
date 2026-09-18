"""R7 native temporal loops, causal histories and exact mask continuation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from linearno.temporal_worker import ROOT, PROJECT, args_for
from model_dict import get_model
from cdlno.linearno.standard_entry import model_kwargs
from cdlno.linearno_history import checkpoint as ck
from cdlno.training_state import _same

SIGS=('A0K0','A1K0','A0K1','A1K1')

class HistoryTemporal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);cls.rows=[]
    @classmethod
    def tearDownClass(cls):
        if p:=os.environ.get('LINEARNO_R7_REPORT'):Path(p).write_text(json.dumps(cls.rows,indent=2))

    def test_presets_depths_and_launchers(self):
        for task in ('ns','plasticity'):
            for depth in range(4,9):
                for sig in SIGS:
                    args=args_for(task,'--n-layers',depth,'--linearno_latent_attnres',sig[1],'--linearno_history_k_conditioning',sig[3])
                    model=get_model(args).Model(**model_kwargs(args))
                    count=sum(p.numel() for p in model.parameters())
                    if depth==8 and sig=='A0K0':self.assertEqual(count,3377921 if task=='ns' else 1799428)
                    self.assertFalse(any('temperature' in k for k in model.state_dict()))
                    self.rows.append(dict(kind='preset_count',task=task,depth=depth,signature=sig,parameters=count))
            for action in ('train','resume','eval'):
                p=subprocess.run(['bash',str(ROOT/f'tran_evaluate/linearno_history/{task}.sh'),action,'--dry-run','--gpu','1'],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr);self.assertIn('--model LinearNO_Structured_Mesh_2D',p.stdout)

    def test_eight_native_cycles_masks_history_and_fair_queries(self):
        artifacts=Path(os.environ.get('LINEARNO_R7_ARTIFACTS',tempfile.mkdtemp(prefix='history-r7-')));artifacts.mkdir(parents=True,exist_ok=True)
        for task in ('ns','plasticity'):
            order=queries=initial=None
            for sig in SIGS:
                with self.subTest(task=task,signature=sig):
                    name=f'{task}__paper_table8_on_release_model__L4__{sig}__seed17'
                    continuous=artifacts/'continuous'/name;split=artifacts/'split'/name;reports={}
                    for action,directory,label in [('train',continuous,'continuous'),('interrupt',split,'interrupt'),('resume',split,'resume'),('eval',split,'eval')]:
                        report=artifacts/f'{name}-{label}.json';log=report.with_suffix('.log')
                        with log.open('w') as stream:
                            p=subprocess.run([sys.executable,'-B',str(ROOT/'tests/linearno/history_temporal_worker.py'),task,action,str(directory),str(report),sig],cwd=PROJECT,
                                env={**os.environ,'PYTHONPATH':str(ROOT),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},stdout=stream,stderr=subprocess.STDOUT,timeout=480)
                        self.assertEqual(p.returncode,0,log.read_text()[-4500:]);reports[label]=json.loads(report.read_text())
                    mc,pc=ck.inspect_checkpoint(continuous);ms,ps=ck.inspect_checkpoint(split)
                    self.assertTrue(_same(ck._read_pair(pc,mc),ck._read_pair(ps,ms)))
                    for key in ('optimizer','scheduler','rng','dataloader_generators','sampler_state'):self.assertEqual(mc['resume_state'][key],ms['resume_state'][key])
                    for key in ('batches','time_queries','masks'):
                        self.assertEqual(reports['continuous'][key],reports['interrupt'][key]+reports['resume'][key],key)
                    self.assertEqual(reports['continuous']['prediction_hash'],reports['eval']['prediction_hash'])
                    for report in reports.values():
                        seq=report['history_lengths']
                        if sig!='A0K0':
                            self.assertGreater(len(seq),0)
                            # before_block called once for read and once inside append.
                            self.assertEqual(seq,[0,0,1,1,2,2,3,3]*(len(seq)//8))
                        else:self.assertEqual(seq,[])
                    h=json.loads((continuous/'linearno_run_manifest.json').read_text())['public_backbone_initial_sha256']
                    if order is None:order=reports['continuous']['batches'];queries=reports['continuous']['time_queries'];initial=h
                    else:
                        self.assertEqual(order,reports['continuous']['batches']);self.assertEqual(queries,reports['continuous']['time_queries']);self.assertEqual(initial,h)
                    self.rows.append(dict(kind='native_cycle',task=task,signature=sig,depth=4,synthetic=True,exact=True,artifacts=str(split),counters=reports['continuous']['counters'],mask_events=len(reports['continuous']['masks']),status='PASS'))

    def test_fair_collate_isolates_original_time_rng(self):
        from cdlno.linearno_history.fair_run import GeneratorCollate
        from linearno.temporal_worker import native_main, synthetic_values
        # Use the actual unchanged collate function extracted from the entry.
        scope=dict(torch=torch);native_main('plasticity',synthetic_values('plasticity')[0],scope)
        batch=[(torch.zeros(3,2),torch.arange(20.),torch.zeros(3,1),torch.ones(3,4,20))]*2
        g=torch.Generator().manual_seed(77);collate=GeneratorCollate(scope['random_collate_fn'],g)
        torch.manual_seed(15);state=torch.get_rng_state();saved=g.get_state();a=collate(batch)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        torch.rand(100);g.set_state(saved);b=collate(batch)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)

if __name__=='__main__':unittest.main()
