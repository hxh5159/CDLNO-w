"""V1 archive/observer foundations. No task exp/main imports or dataset files."""
from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdlno import CDLNO, CDLNOArchitectureConfig
from cdlno.checkpoint import save_sidecar
from cdlno.training_state import (TrainingArchive, TrainingProtocol, ResumeError,
    capture_rng, restore_rng, isolated_evaluation, ordered_fingerprint, _same)
from cdlno.training_observer import EpochPolicy, EpochObserver

ROOT = Path(__file__).resolve().parents[1]
MODES = ('full', 'no_sa', 'identity')


def seed():
    random.seed(101); np.random.seed(202); torch.manual_seed(303)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(404)


def protocol(schedule='onecycle'):
    return TrainingProtocol(task='ns', total_epochs=4, updates_per_epoch=2,
        scheduler_steps_per_epoch=2 if schedule == 'onecycle' else 1,
        training=dict(optimizer={'name': 'AdamW', 'lr': .001, 'weight_decay': .01},
                      scheduler={'name': schedule, 'total_epochs': 4}, loss='synthetic MSE (not task loss)', batch_size=2),
        data=dict(split='synthetic fixtures only', ordered_fingerprint=ordered_fingerprint([torch.arange(4)])), adapter={})


def build(mode='full', schedule='onecycle', device='cpu'):
    model = CDLNO(CDLNOArchitectureConfig(d_model=8, num_heads=2, M=4, L=3, F=1,
        output_dim=2, front_latent_mode=mode, cdpa_mode='every_block', structured=True, grid_shape=(5, 7))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    scheduler = (torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.001, total_steps=8)
                 if schedule == 'onecycle' else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4))
    return model, optimizer, scheduler


def run_epochs(model, optimizer, scheduler, generator, start, end, schedule, records=None, observer=None, visualize=None):
    records = [] if records is None else records
    device = next(model.parameters()).device
    # Real DataLoader/generator shuffle on immutable synthetic tensors.
    dataset = torch.utils.data.TensorDataset(torch.arange(4))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, generator=generator)
    for ep in range(start, end):
        model.train()
        for (ids,) in loader:
            picks = random.sample(range(20), 3)
            offset = float(np.random.uniform())
            x = torch.randn(2, 35, 8, device=device) + offset + sum(picks)/100
            y = torch.randn(2, 35, 2, device=device)
            optimizer.zero_grad()
            loss = (model(x)-y).square().mean()
            loss.backward(); optimizer.step()
            if schedule == 'onecycle':
                scheduler.step()
            records.append(dict(ids=ids.tolist(), picks=picks, offset=offset, loss=loss.item(), lr=optimizer.param_groups[0]['lr']))
        if schedule != 'onecycle':
            scheduler.step()
        if observer is not None:
            observer.after_epoch(ep+1, model, optimizer, scheduler, global_step=(ep+1)*2,
                scheduler_steps=(ep+1)*(2 if schedule == 'onecycle' else 1),
                normalizers={'mean': torch.tensor([1., 2.]), 'std': np.array([2., 3.], dtype=np.float32)},
                generators={'loader': generator}, history=records, visualize=visualize)
    return records


def worker(action, directory, mode, schedule, device):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(device == 'cpu')
    seed()
    directory = Path(directory)
    model, opt, sched = build(mode, schedule, device)
    generator = torch.Generator().manual_seed(505)
    norms = {'mean': torch.tensor([1., 2.]), 'std': np.array([2., 3.], dtype=np.float32)}
    policy = EpochPolicy('ns', 4, visualize_every=0, checkpoint_every=2)
    if action != 'resume':
        directory.mkdir()
        save_sidecar(directory/'architecture.json', model.config)
    archive = TrainingArchive(directory, protocol(schedule))
    observer = EpochObserver(archive, policy)
    start, history = 0, []
    if action == 'resume':
        # Deliberately disturb all RNG streams after constructor/dataloader setup.
        random.random(); np.random.rand(9); torch.rand(9); torch.rand(9, generator=generator)
        if device == 'cuda':
            torch.rand(9, device=device)
        result = archive.restore(directory/'weights/epoch_0002.pt', model, opt, sched,
                                 normalizers=norms, generators={'loader': generator})
        start, history = result['progress']['next_epoch'], result['history']
    with sdpa_kernel(SDPBackend.MATH):
        records = run_epochs(model, opt, sched, generator, start, 2 if action == 'split' else 4,
                             schedule, records=history, observer=observer)
    result = dict(model={k: v.cpu() for k, v in model.state_dict().items()}, optimizer=opt.state_dict(),
                  scheduler=sched.state_dict(), records=records, rng=capture_rng({'loader': generator}))
    torch.save(result, directory/(action+'-result.pt'))
    print(action, mode, schedule, device, 'passed', len(records), 'updates')


class TrainingStateChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        seed()
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.model, self.opt, self.sched = build()
        save_sidecar(self.path/'architecture.json', self.model.config)
        self.archive = TrainingArchive(self.path, protocol())
        self.generator = torch.Generator().manual_seed(505)
        self.norms = {'mean': torch.tensor([1., 2.]), 'std': np.array([2., 3.], dtype=np.float32)}

    def save_after(self, start=0, end=1):
        with sdpa_kernel(SDPBackend.MATH):
            history = run_epochs(self.model, self.opt, self.sched, self.generator, start, end, 'onecycle')
        return self.archive.save(self.model, self.opt, self.sched, completed_epochs=end, global_step=end*2,
            scheduler_steps=end*2, normalizers=self.norms, generators={'loader': self.generator}, history=history)

    def test_policy_all_eight_completed_epoch_boundaries_and_final(self):
        for task in ('darcy','elasticity','pipe','plasticity','ns','car','airfrans','airfoil'):
            total = 200 if task=='car' else 398 if task=='airfrans' else 500
            policy = EpochPolicy(task, total)
            for ep in (1, 49, 50, 99, 100, 150, 200, 398, 500):
                if ep > total: continue
                with self.subTest(task=task, ep=ep):
                    result = policy.events(ep)
                    self.assertEqual(result['visualize'], ep%50==0 or ep==total)
                    self.assertEqual(result['checkpoint'], ep==total or (task in ('ns','car','airfrans','airfoil') and ep%100==0))
            self.assertTrue(EpochPolicy(task, 3).events(3)['checkpoint'])
        for invalid in (0, -1, True, 2.5):
            with self.assertRaises(ValueError): EpochPolicy('ns', 500).events(invalid)
        self.assertFalse(EpochPolicy('ns', 500, visualize_every=0).events(500)['visualize'])

    def test_cpu_fresh_process_continuation_all_modes_both_schedulers(self):
        for mode in MODES:
            for schedule in ('onecycle', 'cosine'):
                with self.subTest(mode=mode, scheduler=schedule):
                    self.process_comparison(mode, schedule, 'cpu')

    def process_comparison(self, mode, schedule, device):
        with tempfile.TemporaryDirectory() as temp:
            continuous, split = Path(temp)/'continuous', Path(temp)/'split'
            for action, path in (('continuous',continuous),('split',split),('resume',split)):
                result = subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',action,str(path),mode,schedule,device],
                    cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            a = torch.load(continuous/'continuous-result.pt', weights_only=True, map_location='cpu')
            b = torch.load(split/'resume-result.pt', weights_only=True, map_location='cpu')
            if device == 'cpu':
                self.assertTrue(_same(a,b), 'continuous and resumed states must be bitwise equal on CPU')
            else:
                torch.testing.assert_close(a['model'], b['model'], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(a['optimizer'], b['optimizer'], atol=1e-6, rtol=1e-5)
                self.assertTrue(_same(a['rng'],b['rng']))
                self.assertTrue(_same(a['scheduler'],b['scheduler']))
                for x,y in zip(a['records'],b['records']):
                    for key in ('ids','picks','offset','lr'): self.assertEqual(x[key],y[key])
                    self.assertAlmostEqual(x['loss'],y['loss'],places=5)

    @unittest.skipUnless(torch.cuda.is_available(), 'GPU continuation not run: CUDA unavailable')
    def test_gpu_fresh_process_identity_onecycle(self):
        self.process_comparison('identity', 'onecycle', 'cuda')

    def test_roundtrip_weights_pair_sidecar_and_structure_unchanged(self):
        keys = list(self.model.state_dict())
        objects = {name:id(p) for name,p in self.model.named_parameters()}
        sidecar = (self.path/'architecture.json').read_bytes()
        saved = self.save_after()
        weights = self.path/'weights'/saved.name
        pure = torch.load(weights,weights_only=True)
        self.assertEqual(list(pure),keys)
        state = self.archive.read(weights)
        self.assertTrue(_same(pure,state['model']))
        self.archive.restore(saved, self.model,self.opt,self.sched,normalizers=self.norms,generators={'loader':self.generator})
        self.assertEqual({name:id(p) for name,p in self.model.named_parameters()},objects)
        self.assertEqual((self.path/'architecture.json').read_bytes(),sidecar)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(_same(self.archive.read(self.path/'checkpoints/latest.json'),state))

    def test_wrong_protocol_normalizer_groups_plan_and_mode_rejected_before_weights(self):
        saved = self.save_after()
        original = copy.deepcopy(self.model.state_dict())
        variants = [replace(protocol(),total_epochs=5),replace(protocol(),data=dict(split='other',ordered_fingerprint='0'*64)),
                    replace(protocol(),training={**protocol().training,'loss':'changed'})]
        for value in variants:
            with self.assertRaisesRegex(ResumeError,'protocol mismatch'):
                TrainingArchive(self.path,value).restore(saved,self.model,self.opt,self.sched,normalizers=self.norms,generators={'loader':self.generator})
        with self.assertRaisesRegex(ResumeError,'normalizer'):
            self.archive.restore(saved,self.model,self.opt,self.sched,normalizers={},generators={'loader':self.generator})
        wrong_opt=torch.optim.AdamW(reversed(list(self.model.parameters())),lr=.001,weight_decay=.01)
        wrong_sched=torch.optim.lr_scheduler.OneCycleLR(wrong_opt,max_lr=.001,total_steps=8)
        with self.assertRaisesRegex(ResumeError,'parameter group order'):
            self.archive.restore(saved,self.model,wrong_opt,wrong_sched,normalizers=self.norms,generators={'loader':self.generator})
        wrong_sched=torch.optim.lr_scheduler.OneCycleLR(self.opt,max_lr=.001,total_steps=12)
        with self.assertRaisesRegex(ResumeError,'scheduler'):
            self.archive.restore(saved,self.model,self.opt,wrong_sched,normalizers=self.norms,generators={'loader':self.generator})
        other,op,sc=build('identity')
        with self.assertRaisesRegex(ResumeError,'architecture'):
            self.archive.restore(saved,other,op,sc,normalizers=self.norms,generators={'loader':self.generator})
        self.assertTrue(_same(original,self.model.state_dict()))

    def test_missing_state_fields_and_bad_weights_cannot_resume(self):
        saved = self.save_after()
        payload = self.archive.read(saved)
        for field in ('optimizer','scheduler','rng','normalizers','progress','model'):
            damaged=copy.deepcopy(payload);del damaged[field]
            torch.save(damaged,saved)
            self.rehash(saved)
            with self.assertRaisesRegex(ResumeError,'incomplete'):
                self.archive.read(saved)
        torch.save(payload,saved);self.rehash(saved)
        for field in ('step','exp_avg','exp_avg_sq'):
            damaged=copy.deepcopy(payload)
            del next(iter(damaged['optimizer']['state'].values()))[field]
            torch.save(damaged,saved);self.rehash(saved)
            with self.assertRaisesRegex(ResumeError,'Adam moment'):
                self.archive.restore(saved,self.model,self.opt,self.sched,normalizers=self.norms,generators={'loader':self.generator})
        torch.save(payload,saved);self.rehash(saved)
        weights=self.path/'weights'/saved.name
        torch.save({'wrong':torch.zeros(1)},weights)
        with self.assertRaisesRegex(ResumeError,'checksum'):
            self.archive.read(saved)
        with self.assertRaisesRegex(ResumeError,'belong'):
            self.archive.read(Path('/tmp/elsewhere')/saved.name)
        torch.save(self.model.state_dict(),self.path/'model.pt')
        with self.assertRaises(ResumeError):self.archive.read(self.path/'model.pt')

    def rehash(self, checkpoint):
        # Explicit corruption fixture: repair checksum to reach semantic validation.
        manifest=checkpoint.with_suffix('.json');value=json.loads(manifest.read_text())
        value['checkpoint']['sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(value))

    def test_failed_second_file_write_preserves_last_commit_and_retry(self):
        first=self.save_after()
        pointer=(self.path/'checkpoints/latest.json').read_bytes()
        with sdpa_kernel(SDPBackend.MATH):run_epochs(self.model,self.opt,self.sched,self.generator,1,2,'onecycle')
        from cdlno import training_state as state
        original=state._atomic
        def fail_weights(path,*a,**kw):
            if Path(path).parent.name=='weights':raise OSError('simulated disk write failure')
            return original(path,*a,**kw)
        kwargs=dict(completed_epochs=2,global_step=4,scheduler_steps=4,normalizers=self.norms,generators={'loader':self.generator})
        with patch.object(state,'_atomic',side_effect=fail_weights),self.assertRaises(OSError):
            self.archive.save(self.model,self.opt,self.sched,**kwargs)
        self.assertEqual((self.path/'checkpoints/latest.json').read_bytes(),pointer)
        self.assertFalse((self.path/'checkpoints/epoch_0002.pt').exists())
        self.assertFalse((self.path/'weights/epoch_0002.pt').exists())
        self.assertFalse((self.path/'checkpoints/.writer.lock').exists())
        self.assertEqual(self.archive.read(first)['progress']['completed_epochs'],1)
        self.archive.save(self.model,self.opt,self.sched,**kwargs)
        with self.assertRaisesRegex(ResumeError,'older checkpoint'):
            self.archive.restore(first,self.model,self.opt,self.sched,normalizers=self.norms,generators={'loader':self.generator})

    def test_idempotent_final_no_duplicate_or_overwrite(self):
        with sdpa_kernel(SDPBackend.MATH):run_epochs(self.model,self.opt,self.sched,self.generator,0,4,'onecycle')
        kw=dict(completed_epochs=4,global_step=8,scheduler_steps=8,normalizers=self.norms,generators={'loader':self.generator})
        path=self.archive.save(self.model,self.opt,self.sched,**kw)
        before=path.read_bytes();mtime=path.stat().st_mtime_ns
        self.assertEqual(self.archive.save(self.model,self.opt,self.sched,**kw),path)
        self.assertEqual(path.read_bytes(),before);self.assertEqual(path.stat().st_mtime_ns,mtime)
        self.assertEqual(len(list((self.path/'checkpoints').glob('*.pt'))),1)
        self.assertEqual(self.archive.read(self.path/'checkpoints/final.json')['progress']['completed_epochs'],4)
        with torch.no_grad():next(self.model.parameters()).add_(1)
        with self.assertRaisesRegex(ResumeError,'different state'):
            self.archive.save(self.model,self.opt,self.sched,**kw)

    def test_committed_pair_survives_failed_latest_and_prevents_rollback(self):
        first = self.save_after()
        pointer = (self.path/'checkpoints/latest.json').read_bytes()
        with sdpa_kernel(SDPBackend.MATH):
            run_epochs(self.model,self.opt,self.sched,self.generator,1,2,'onecycle')
        from cdlno import training_state as state
        original = state._atomic
        def fail_index(path, *a, **kw):
            if Path(path).name == 'latest.json':
                raise OSError('simulated index failure after durable pair commit')
            return original(path, *a, **kw)
        kwargs = dict(completed_epochs=2,global_step=4,scheduler_steps=4,normalizers=self.norms,generators={'loader':self.generator})
        with patch.object(state,'_atomic',side_effect=fail_index), self.assertRaises(OSError):
            self.archive.save(self.model,self.opt,self.sched,**kwargs)
        second = self.path/'checkpoints/epoch_0002.pt'
        self.assertEqual((self.path/'checkpoints/latest.json').read_bytes(),pointer)
        self.assertEqual(self.archive.read(second)['progress']['completed_epochs'],2)
        with self.assertRaisesRegex(ResumeError,'older checkpoint'):
            self.archive.restore(first,self.model,self.opt,self.sched,normalizers=self.norms,generators={'loader':self.generator})
        before = second.read_bytes()
        self.archive.save(self.model,self.opt,self.sched,**kwargs)
        self.assertEqual(second.read_bytes(),before)
        self.assertEqual(self.archive.read(self.path/'checkpoints/latest.json')['progress']['completed_epochs'],2)

    def test_semantically_damaged_state_rejected_before_model_or_rng_mutation(self):
        saved = self.save_after()
        payload = self.archive.read(saved)
        weights = copy.deepcopy(self.model.state_dict())
        rng = capture_rng({'loader':self.generator})
        mutations = [lambda p: p['scheduler'].update(_last_lr=[]),
                     lambda p: p['optimizer']['param_groups'][0].update(betas=[2., .99]),
                     lambda p: next(iter(p['optimizer']['state'].values())).update(exp_avg=torch.zeros(1, dtype=torch.float64)),
                     lambda p: p.update(rng={'cpu':torch.get_rng_state()})]
        for mutate in mutations:
            damaged = copy.deepcopy(payload); mutate(damaged)
            torch.save(damaged,saved); self.rehash(saved)
            with self.assertRaises(ResumeError):
                self.archive.restore(saved,self.model,self.opt,self.sched,normalizers=self.norms,generators={'loader':self.generator})
            self.assertTrue(_same(weights,self.model.state_dict()))
            self.assertTrue(_same(rng,capture_rng({'loader':self.generator})))
        manifest = saved.with_suffix('.json'); manifest.write_text('{}')
        with self.assertRaisesRegex(ResumeError,'manifest'):
            self.archive.read(saved)

    def test_rng_roundtrip_generators_and_exception_isolation(self):
        self.model.train();self.model.front_blocks[0].eval()
        flags=[m.training for m in self.model.modules()]
        before=capture_rng({'loader':self.generator})
        weights=copy.deepcopy(self.model.state_dict())
        with self.assertRaisesRegex(RuntimeError,'plot failed'):
            with isolated_evaluation(self.model,generators={'loader':self.generator}):
                self.assertFalse(any(m.training for m in self.model.modules()))
                random.random();np.random.rand();torch.rand(3);torch.rand(3,generator=self.generator)
                if torch.cuda.is_available():torch.rand(3,device='cuda')
                raise RuntimeError('plot failed')
        self.assertTrue(_same(before,capture_rng({'loader':self.generator})))
        self.assertEqual(flags,[m.training for m in self.model.modules()])
        self.assertTrue(_same(weights,self.model.state_dict()))
        with self.assertRaisesRegex(ResumeError,'generator names'):
            restore_rng(before,{})

    def test_visualization_enabled_disabled_gives_identical_training(self):
        from cdlno.visualization import ColorScale,render_fields
        outcomes=[]
        for enabled in (False,True):
            seed();model,opt,sched=build('no_sa');gen=torch.Generator().manual_seed(505)
            directory=self.path/str(enabled);directory.mkdir();save_sidecar(directory/'architecture.json',model.config)
            observer=EpochObserver(TrainingArchive(directory,protocol()),EpochPolicy('ns',4,visualize_every=2 if enabled else 0,checkpoint_every=0))
            def draw(ep):
                # Extra model forward and all RNG streams; isolation must undo them.
                random.random();np.random.rand();torch.rand(1,generator=gen)
                h=torch.randn(1,35,8);out=model(h)[0]
                xx,yy=np.meshgrid(np.arange(7),np.arange(5),indexing='xy')
                xy=np.stack((xx.ravel(),yy.ravel()),-1)
                scales=[ColorScale(-1,1,1) for _ in range(2)]
                render_fields(directory/f'viz{ep}',coordinates=xy,truth=np.ones((35,2)),prediction=out,
                    scales=scales,channel_names=['a','b'],task='synthetic',case_id='case0',completed_epoch=ep,grid_shape=(5,7))
            with sdpa_kernel(SDPBackend.MATH):records=run_epochs(model,opt,sched,gen,0,4,'onecycle',observer=observer,visualize=draw)
            outcomes.append(dict(model=copy.deepcopy(model.state_dict()),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                                 rng=capture_rng({'loader':gen}),records=records,grads=[p.grad.clone() for p in model.parameters()]))
        self.assertTrue(_same(outcomes[0],outcomes[1]))

    def test_visualization_failure_still_saves_complete_checkpoint(self):
        self.save_after()
        policy=EpochPolicy('ns',4,visualize_every=1,checkpoint_every=1)
        observer=EpochObserver(self.archive,policy)
        def fail(_):
            random.random();raise RuntimeError('headless renderer failed')
        with warnings.catch_warnings(), self.assertLogs('cdlno.training_observer', level='WARNING') as logged:
            warnings.simplefilter('error')
            result=observer.after_epoch(1,self.model,self.opt,self.sched,global_step=2,scheduler_steps=2,
                normalizers=self.norms,generators={'loader':self.generator},visualize=fail,
                history=self.archive.read(self.path/'checkpoints/latest.json')['history'])
        self.assertIn('headless renderer', logged.output[0])
        self.assertIn('headless renderer failed',result['visualization_error'])
        self.assertTrue(result['checkpoint'].is_file())

    def test_protocol_and_data_order_are_not_model_architecture(self):
        self.assertNotEqual(ordered_fingerprint([torch.tensor([1,2])]),ordered_fingerprint([torch.tensor([2,1])]))
        self.assertNotEqual(ordered_fingerprint(['ab','c']),ordered_fingerprint(['a','bc']))
        with self.assertRaises(ResumeError):replace(protocol(),updates_per_epoch=True)
        with self.assertRaises(ResumeError):replace(protocol(),data={})
        self.assertNotIn('checkpoint_every',self.model.config.to_dict())


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(*sys.argv[2:])
    else:unittest.main()
