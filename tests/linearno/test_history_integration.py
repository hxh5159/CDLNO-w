"""R5 internal-only CPU synthetic matrix, strict metadata and A/K composition."""
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
from unittest.mock import patch

import numpy as np
import torch

from cdlno.linearno.profiles import resolve_config as resolve_profile
from cdlno.linearno import checkpoint as legacy
from cdlno.linearno import schema as legacy_schema
from cdlno.training_state import _same
from cdlno.linearno_history.config import (resolve_config, validate_config, innovation_spec,
                                          DEFAULT_FEATURES, structural_differences)
from cdlno.linearno_history.factory import build_model, class_for
from cdlno.linearno_history import checkpoint as ck
from cdlno.linearno_history.context import RawHistoryContext
from linearno.test_history_core import ROOT, inputs, call, VARIANTS
from linearno.test_schema import sample_metadata
from linearno.attention_support import errors
from linearno.attnres_reference import reference as a_oracle
from linearno.history_k_reference import reference as k_oracle

SIGNATURES = ('A0K0', 'A1K0', 'A0K1', 'A1K1')


def config(sig='A0K0', layers=4, variant='plain', dropout=0., history_dropout=None, **overrides):
    task = 'airfrans' if variant == 'airfrans' else 'car' if variant == 'shapenet' else 'airfoil'
    explicit = {'model.layers': layers, 'model.hidden': 12, 'model.heads': 3,
                'model.linearno_rank': 8, 'model.ref': 3, 'model.dropout': dropout,
                'runtime.seed': 51005, 'runtime.device': 'cpu'}
    if task not in ('airfrans', 'car'):
        explicit.update({'model.H': 3, 'model.W': 5, 'model.space_dim': 2,
            'model.fun_dim': 1, 'model.out_dim': 2, 'model.unified_pos': False,
            'model.time_input': True, 'model.linearno_variant': variant})
    explicit.update(overrides)
    profile = resolve_profile(task, 'official_release', explicit=explicit)
    features = {
        'linearno_latent_attnres': sig[1]=='1',
        'linearno_history_k_conditioning': sig[3]=='1',
        'linearno_attnres_history_dropout_p': history_dropout,
    }
    return resolve_config(profile, family='linearno_history', features=features)


def io(config):
    return inputs(config['base_linearno']['variant'], config['model_spec']['constructor_kwargs'])


def forward(model, config, values, **kwargs):
    return call(model, config['base_linearno']['variant'], values, **kwargs)


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)


def scheduler(opt):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.9)


def step(model, c, values, opt, sched):
    opt.zero_grad(set_to_none=True)
    y = forward(model, c, values)
    loss = (y-.73).square().mean()
    loss.backward(); opt.step(); sched.step()
    return y.detach(), loss.detach()


def metadata(c, model, opt, sched, *, epoch=1, total_epochs=1, generators=None, samplers=None):
    old = sample_metadata()
    sections = {k: old[k] for k in ('data_spec', 'normalizer_spec', 'provenance_spec')}
    sections['data_spec']['split'] = 'R5 synthetic fixture, no real dataset'
    sections['provenance_spec'].update(
        target_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        base_commit='d5abe014ed05ec9286200d677b039bbd68697f96',
        source_sha256=ck.source_hash(), normalized_patch_sha256=ck.source_hash(),
        command=['python','-B','-m','unittest','linearno.test_history_integration'],
        environment={'torch':str(torch.__version__), 'device':'cpu','scope':'R5 synthetic MSE only'})
    sections['resume_state'] = legacy.resume_state(opt, sched, epoch, 1, total_epochs,
        {} if generators is None else generators, {} if samplers is None else samplers)
    sections['ensemble_manifest'] = []
    return ck.make_metadata(c, **sections)


def copy_matching(source, target):
    """Explicit common tensor copy, never cross-architecture checkpoint loading."""
    with torch.no_grad():
        dest = target.state_dict()
        for key, value in source.state_dict().items():
            if key in dest:
                assert value.shape == dest[key].shape
                dest[key].copy_(value)


def gates(model, a=0., k=0., w=0.):
    with torch.no_grad():
        if hasattr(model, 'latent_attnres'):
            for receiver in model.latent_attnres.receivers.values():
                receiver.gamma.fill_(a); receiver.w.fill_(w)
        if hasattr(model, 'history_k'):
            for gate in model.history_k.raw_gates.values():
                gate.fill_(k)


def assert_exact(a, b):
    if not _same(a, b):
        raise AssertionError('recursive exact equality failed')


class HistoryIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.rows = []

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        if path := os.environ.get('LINEARNO_R5_REPORT'):
            Path(path).write_text(json.dumps(dict(scope='R5 CPU synthetic internal factory/checkpoint',
                torch=str(torch.__version__), rows=cls.rows), indent=2))

    def test_20_config_complete_block_matrix_fresh_process_reload(self):
        worker = r'''
import json,sys,torch
from pathlib import Path
from cdlno.linearno_history.checkpoint import load_checkpoint
from linearno.test_history_integration import forward
p=torch.load(sys.argv[1],weights_only=True)
torch.set_num_threads(1)
result=[]
for row in p:
 model,metadata,changes=load_checkpoint(row['directory'],expected=row['config'])
 assert not changes
 with torch.no_grad(): y=forward(model,row['config'],row['inputs'])
 torch.testing.assert_close(y,row['output'],atol=0,rtol=0)
 torch.testing.assert_close(model.state_dict(),row['state'],atol=0,rtol=0)
 result.append(dict(signature=row['config']['features']['feature_signature'],
                    depth=len(model.blocks), max_abs=(y-row['output']).abs().max().item()))
Path(sys.argv[2]).write_text(json.dumps(result))
'''
        with tempfile.TemporaryDirectory(prefix='history-r5-matrix-') as temp:
            jobs = []
            for layers in range(4,9):
                # A single real public backbone initial state for all four cases.
                c0 = config(layers=layers)
                torch.manual_seed(51005); public = build_model(c0)
                common = {k: v.clone() for k,v in public.state_dict().items()}
                rng_after_public = torch.get_rng_state().clone()
                for sig in SIGNATURES:
                    with self.subTest(signature=sig, layers=layers):
                        c = config(sig,layers)
                        torch.manual_seed(51005); model = build_model(c)
                        self.assertTrue(torch.equal(torch.get_rng_state(),rng_after_public))
                        # Explicit same public backbone, independently seeded feature subtrees.
                        with torch.no_grad():
                            for key,value in common.items(): model.state_dict()[key].copy_(value)
                        self.assertEqual(hasattr(model,'latent_attnres'),sig[1]=='1')
                        self.assertEqual(hasattr(model,'history_k'),sig[3]=='1')
                        counts = dict(attention=0, ffn=0, final_norm=0, head=0)
                        def count(key):
                            def hook(*unused): counts[key]+=1
                            return hook
                        handles=[]
                        for block in model.blocks:
                            handles += [block.Attn.to_out.register_forward_hook(count('attention')),
                                        block.mlp.register_forward_hook(count('ffn'))]
                        handles += [model.blocks[-1].ln_3.register_forward_hook(count('final_norm')),
                                    model.blocks[-1].mlp2.register_forward_hook(count('head'))]
                        values = io(c); model.train()
                        opt=optimizer(model); sched=scheduler(opt)
                        _,loss=step(model,c,values,opt,sched)
                        self.assertEqual(counts,dict(attention=layers,ffn=layers,final_norm=1,head=1))
                        for handle in handles: handle.remove()
                        self.assertTrue(torch.isfinite(loss))
                        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                        meta=metadata(c,model,opt,sched)
                        directory=Path(temp)/c['run_signature']; directory.mkdir()
                        rng=torch.get_rng_state().clone()
                        ck.save_checkpoint(directory,model,meta)
                        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                        if sig=='A0K0':
                            self.assertFalse(ck.EXTRAS & meta.keys())
                            manifest=json.loads((directory/'checkpoints/epoch_0001.json').read_text())
                            self.assertEqual(manifest['format'],'linearno-epoch-pair-v1')
                            # Actual original loader, not just the new dispatch.
                            _, state=legacy.read_pair(directory/'checkpoints/epoch_0001.json',type(model))
                            assert_exact(state,model.state_dict())
                        model.eval()
                        with torch.no_grad(): out=forward(model,c,values)
                        jobs.append(dict(directory=str(directory),config=c,inputs={k:v.detach() for k,v in values.items()},
                                         output=out,state=model.state_dict()))
                        self.rows.append(dict(kind='20-matrix',signature=sig,depth=layers,counts=counts,
                            parameters=sum(p.numel() for p in model.parameters()),loss=loss.item(),status='PASS'))
            source=Path(temp)/'worker-input.pt'; torch.save(jobs,source)
            out=Path(temp)/'worker-result.json'
            subprocess.run([sys.executable,'-B','-c',worker,str(source),str(out)],check=True,
                           cwd='/tmp',env={**os.environ,'PYTHONPATH':str(ROOT)+':'+str(ROOT/'tests')},timeout=100)
            rows=json.loads(out.read_text()); self.assertEqual(len(rows),20)
            self.assertTrue(all(r['max_abs']==0 for r in rows))
            self.rows.append(dict(kind='fresh-process',rows=rows,status='PASS'))

    def test_omitted_flags_explicit_false_exact_legacy_path_rng(self):
        for variant in VARIANTS:
            for train in (False,True):
                with self.subTest(variant=variant,train=train):
                    c=config(variant=variant,dropout=.2)
                    omitted=resolve_config(c['profile_spec'])
                    explicit=resolve_config(c['profile_spec'],features=DEFAULT_FEATURES)
                    self.assertEqual(c,omitted); self.assertEqual(c,explicit)
                    values=io(c); copies=[]
                    for cfg in (omitted,explicit):
                        torch.manual_seed(1123); model=build_model(cfg).train(train)
                        self.assertIs(type(model),class_for(cfg['model_spec']['class_path']))
                        self.assertFalse(any(hasattr(model,k) for k in ('history_k','latent_attnres','_history_model_spec')))
                        opt=optimizer(model); sched=scheduler(opt)
                        v={k:t.detach().clone().requires_grad_() for k,t in values.items()}
                        y,loss=step(model,cfg,v,opt,sched)
                        m=metadata(cfg,model,opt,sched)
                        copies.append((model,opt,sched,m,y,loss,{k:t.grad for k,t in v.items()},
                            {k:p.grad for k,p in model.named_parameters()},torch.get_rng_state().clone()))
                    a,b=copies
                    for x,y in zip(a[3:],b[3:]): assert_exact(x,y)
                    assert_exact(a[0].state_dict(),b[0].state_dict())
                    assert_exact(a[1].state_dict(),b[1].state_dict()); assert_exact(a[2].state_dict(),b[2].state_dict())
                    with tempfile.TemporaryDirectory() as tmp:
                        paths=[]
                        for index,item in enumerate(copies):
                            directory=Path(tmp)/str(index);directory.mkdir()
                            with patch.object(legacy,'save_pair',wraps=legacy.save_pair) as save:
                                path=ck.save_checkpoint(directory,item[0],item[3]);self.assertEqual(save.call_count,1)
                            paths.append(path)
                        self.assertEqual(json.loads(paths[0].read_text()),json.loads(paths[1].read_text()))
                    self.rows.append(dict(kind='baseline-exact',variant=variant,train=train,max_abs=0,status='PASS'))

    def test_joint_gate_decomposition_all_variants_and_both_gradients(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                configs={sig:config(sig,variant=variant) for sig in SIGNATURES}
                models={sig:build_model(c).double().eval() for sig,c in configs.items()}
                for model in models.values(): copy_matching(models['A0K0'],model)
                for sig in ('A1K0','A0K1'): copy_matching(models['A1K1'],models[sig])
                values={k:v.double() for k,v in io(configs['A0K0']).items()}
                joint=models['A1K1']; c=configs['A1K1']
                for a,k,reference in [(0.,0.,'A0K0'),(.7,0.,'A1K0'),(0.,.6,'A0K1')]:
                    gates(joint,a,k,.2);gates(models[reference],a,k,.2)
                    actual=forward(joint,c,values);expected=forward(models[reference],configs[reference],values)
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                    self.rows.append(dict(kind='decomposition',variant=variant,A=a,K=k,errors=errors(actual,expected)))
                gates(joint,.7,.6,.2)
                result=forward(joint,c,values)
                result.square().sum().backward()
                for name,p in joint.named_parameters():
                    if name.startswith(('latent_attnres.','history_k.')):
                        self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
                a_ids={id(p) for p in joint.latent_attnres.parameters()}
                k_ids={id(p) for p in joint.history_k.parameters()}
                self.assertFalse(a_ids & k_ids)
                self.assertFalse(hasattr(models['A0K1'],'latent_attnres'))
                self.assertFalse(hasattr(models['A1K0'],'history_k'))
                self.rows.append(dict(kind='joint-active-gradients',variant=variant,
                    A_gradient_norm=sum(p.grad.norm().item() for p in joint.latent_attnres.parameters()),
                    K_gradient_norm=sum(p.grad.norm().item() for p in joint.history_k.parameters()),
                    all_finite=True))

    def test_joint_order_raw_once_residual_once_independent_oracles(self):
        for variant in VARIANTS:
            c=config('A1K1',variant=variant)
            model=build_model(c).double().eval();gates(model,.8,.7,.2)
            values={k:v.double() for k,v in io(c).items()}
            events=[]; traces=[]; records={}; handles=[]
            def k_pre(module,args):
                index,Z,base,W,history=args;events.append(('K',index));records.setdefault(index,{})['K_history']=history
            def a_pre(module,args):
                index,current,history=args;events.append(('A',index));records[index]['A_history']=history
            handles += [model.history_k.register_forward_pre_hook(k_pre),model.latent_attnres.register_forward_pre_hook(a_pre)]
            for index,block in enumerate(model.blocks):
                handles.append(block.ln_1.register_forward_pre_hook(
                    lambda m,a,i=index: records.setdefault(i,{}).update(block_input=a[0])))
                for label,module in [('attention',block.Attn.to_out),('FFN',block.mlp)]:
                    def save_output(m,a,y,i=index,l=label):
                        events.append((l,i)); records[i][l]=y
                    handles.append(module.register_forward_hook(save_output))
            def observed(trace):
                traces.append(trace);events.append(('block_complete',trace.index))
            original=RawHistoryContext.after_block
            def append(context,index,raw):
                events.append(('append',index));self.assertIs(raw,traces[index].factors.C_raw)
                return original(context,index,raw)
            with patch.object(RawHistoryContext,'after_block',append):forward(model,c,values,observe=observed)
            for h in handles:h.remove()
            self.assertEqual(events,[(label,i) for i in range(4) for label in ('K','A','attention','FFN','block_complete','append')])
            metrics=[]
            for trace in traces:
                i,f=trace.index,trace.factors
                self.assertIs(records[i]['K_history'],records[i]['A_history'])
                self.assertEqual(len(trace.history),i)
                for s,raw in enumerate(trace.history): self.assertIs(raw,traces[s].factors.C_raw)
                expected_k,_=k_oracle(i,f.Z,f.base_k_logits,model.blocks[i].Attn.to_k.weight,
                                     trace.history,model.history_k.state_dict())
                expected_c,_=a_oracle(f.C_raw,trace.history,model.latent_attnres.state_dict(),i)
                torch.testing.assert_close(f.combined_k_logits,expected_k,atol=1e-12,rtol=1e-10)
                torch.testing.assert_close(trace.C_tilde,expected_c,atol=1e-12,rtol=1e-10)
                metrics.append(dict(layer=i,K_combined=errors(f.combined_k_logits,expected_k),
                                    A_fused=errors(trace.C_tilde,expected_c)))
                block=model.blocks[i]; attn=block.Attn
                q_logits=f.base_q_logits
                if variant in ('temp','conv_temp'):
                    expected_k=expected_k/attn.temperature_k.clamp(.01,1.)
                    q_logits=q_logits/attn.temperature_q.clamp(.01,1.)
                elif variant=='shapenet':
                    expected_k=expected_k/attn.tempreature_k.clamp(.1,2.)
                    q_logits=q_logits/attn.tempreature_q.clamp(.1,2.)
                torch.testing.assert_close(f.K,expected_k.softmax(-2),atol=1e-12,rtol=1e-10)
                torch.testing.assert_close(f.Q,q_logits.softmax(-1),atol=0,rtol=0)
                torch.testing.assert_close(f.C_raw,f.K.transpose(-1,-2)@f.V,atol=1e-12,rtol=1e-10)
                readout=(f.Q@trace.C_tilde).transpose(1,2).reshape(f.Z.shape[0],f.Z.shape[2],-1)
                torch.testing.assert_close(records[i]['attention'],attn.to_out(readout),atol=0,rtol=0)
                x,update,ffn=records[i]['block_input'],records[i]['attention'],records[i]['FFN']
                after_attention=x+update if variant=='shapenet' else update+x
                torch.testing.assert_close(ffn,block.mlp(block.ln_2(after_attention)),atol=0,rtol=0)
                expected_output=after_attention+ffn if variant=='shapenet' else ffn+after_attention
                if block.last_layer:expected_output=block.mlp2(block.ln_3(expected_output))
                torch.testing.assert_close(trace.output,expected_output,atol=0,rtol=0)
            self.rows.append(dict(kind='joint-order-oracle',variant=variant,
                                 raw_appends=len(traces),metrics=metrics,status='PASS'))

    def test_train_dropout_rng_distinction_and_K_only_has_no_A_calls(self):
        c0=config();ca=config('A1K0');ckc=config('A0K1');cak=config('A1K1')
        models=[build_model(c).train() for c in (c0,ca,ckc,cak)]
        for m in models:copy_matching(models[0],m)
        values=io(c0); states=[]
        for m,c in zip(models,(c0,ca,ckc,cak)):
            torch.manual_seed(3321); forward(m,c,values);states.append(torch.get_rng_state())
        self.assertTrue(torch.equal(states[0],states[2]))
        self.assertFalse(torch.equal(states[0],states[1]))
        self.assertTrue(torch.equal(states[1],states[3]))
        with patch('cdlno.linearno_history.attnres.LatentSummaryAttnRes.forward',side_effect=AssertionError('A called')):
            forward(models[2],ckc,values)
        for c in (ca,cak):self.assertEqual(c['features']['linearno_attnres_history_dropout_p'],.1)

    def test_A1K0_no_history_dropout_is_explicit_and_rng_stable(self):
        normal = config('A1K0')
        nodrop = config('A1K0', history_dropout=0.0)
        self.assertNotEqual(normal['run_signature'], nodrop['run_signature'])
        self.assertIn('__nodrop__', nodrop['run_signature'])
        self.assertNotIn('attnres_history_dropout_p', normal['model_spec']['constructor_kwargs'])
        self.assertEqual(nodrop['model_spec']['constructor_kwargs']['attnres_history_dropout_p'], 0.0)
        normal_model = build_model(normal).train()
        nodrop_model = build_model(nodrop).train()
        copy_matching(normal_model, nodrop_model)
        self.assertEqual(normal_model.latent_attnres.dropout_p, 0.1)
        self.assertEqual(nodrop_model.latent_attnres.dropout_p, 0.0)
        values = io(normal)
        torch.manual_seed(8821); before = torch.get_rng_state(); forward(nodrop_model, nodrop, values); after = torch.get_rng_state()
        torch.manual_seed(8821); expected_before = torch.get_rng_state()
        self.assertTrue(torch.equal(before, expected_before))
        self.assertTrue(torch.equal(after, expected_before))
        with self.assertRaises(ValueError): config('A1K1', history_dropout=0.0)

        # The ablation is a distinct research checkpoint shape/protocol and
        # must round-trip through the metadata-first strict loader.
        with tempfile.TemporaryDirectory() as tmp:
            opt = optimizer(nodrop_model); sched = scheduler(opt)
            step(nodrop_model, nodrop, values, opt, sched)
            saved = metadata(nodrop, nodrop_model, opt, sched)
            ck.save_checkpoint(Path(tmp), nodrop_model, saved)
            restored, loaded, _ = ck.load_checkpoint(Path(tmp), expected=nodrop, strict=True)
            self.assertEqual(restored.latent_attnres.dropout_p, 0.0)
            self.assertEqual(loaded['innovation_spec']['attnres']['dropout']['p'], 0.0)
            with self.assertRaises(ValueError):
                ck.load_checkpoint(Path(tmp), expected=normal, strict=True)

    def test_config_validation_no_models_created_on_invalid_fields(self):
        c=config()
        bad_flags=[{'linearno_latent_attnres':'false'},{'linearno_history_k_conditioning':1},
            {'linearno_latent_attnres':True,'linearno_history_k_conditioning':True,'linearno_attnres_history_dropout_p':0},
            {'linearno_attnres_history_dropout_p':.1},{'feature_signature':'A1K1'}]
        for flags in bad_flags:
            with self.assertRaises(ValueError):resolve_config(c['profile_spec'],family='linearno_history',features=flags)
        for family in ('transolver','cdlno','kcdno','msar_lno','linearno'):
            with self.assertRaises(ValueError):resolve_config(c['profile_spec'],family=family,features={'linearno_latent_attnres':True})
        for layers in (0,3,9):
            with self.assertRaises(ValueError):config('A1K1',layers)
        for path,value in [('family','transolver'),('config_schema_version',2),('config_schema_version',True)]:
            bad=copy.deepcopy(c);bad[path]=value
            with self.assertRaises(ValueError):validate_config(bad)
        # Config parsing does not import torch or task models in a fresh process.
        subprocess.run([sys.executable,'-B','-c',
            "import sys; import cdlno.linearno_history.config; assert 'torch' not in sys.modules"],check=True)

    def test_metadata_negative_matrix_before_weight_access(self):
        configs={sig:config(sig) for sig in SIGNATURES}; saved={}
        with tempfile.TemporaryDirectory() as tmp:
            for sig,c in configs.items():
                model=build_model(c);opt=optimizer(model);sched=scheduler(opt)
                step(model,c,io(c),opt,sched)
                meta=metadata(c,model,opt,sched);directory=Path(tmp)/sig;directory.mkdir()
                ck.save_checkpoint(directory,model,meta);saved[sig]=(meta,directory)
            rejected=0
            for source,(meta,directory) in saved.items():
                for target,c in configs.items():
                    if source==target:continue
                    with self.subTest(source=source,target=target),patch.object(torch,'load',side_effect=AssertionError('weight read')), \
                         self.assertRaisesRegex(ValueError,'structural'):
                        ck.load_checkpoint(directory,expected=c)
                    rejected+=1
            meta,directory=saved['A1K1']
            mismatches=[config('A1K1',5),config('A1K1',variant='temp'),
                config('A1K1',**{'model.linearno_rank':12}),config('A1K1',**{'model.heads':2})]
            for c in mismatches:
                with patch.object(torch,'load',side_effect=AssertionError('weight read')),self.assertRaisesRegex(ValueError,'structural'):
                    ck.load_checkpoint(directory,expected=c)
            mutations=[lambda m:m.pop('innovation_spec'),lambda m:m.update(schema_version=2),
                lambda m:m['innovation_spec'].update(schema_version=2),
                lambda m:m['innovation_spec']['base_linearno'].update(d_h=8),
                lambda m:m['model_spec']['constructor_kwargs'].update(unknown=1),
                lambda m:m['innovation_spec']['code_schema'].update(implementation_version='r1-contract-only'),
                lambda m:m['resume_state'].pop('rng'),lambda m:m['normalizer_spec']['records'].clear()]
            for mutate in mutations:
                bad=copy.deepcopy(meta);mutate(bad);ck.rehash(bad)
                with self.assertRaises(ValueError):ck.validate_metadata(bad)
                # Also drive the actual disk reader. Recompute every checksum so
                # failures establish semantic validation, not just file corruption.
                sidecar=directory/'checkpoints/epoch_0001.metadata.json'
                manifest_path=directory/'checkpoints/epoch_0001.json'
                pointer=directory/'checkpoints/final.json'
                originals=[p.read_bytes() for p in (sidecar,manifest_path,pointer)]
                try:
                    sidecar.write_text(json.dumps(bad))
                    manifest=json.loads(manifest_path.read_text())
                    manifest['metadata']['sha256']=legacy.sha256(sidecar)
                    manifest_path.write_text(json.dumps(manifest))
                    pointer.write_text(json.dumps(dict(manifest=manifest_path.name,sha256=legacy.sha256(manifest_path))))
                    with patch.object(torch,'load',side_effect=AssertionError('weight read')),self.assertRaises(ValueError):
                        ck.load_checkpoint(directory)
                finally:
                    for path,content in zip((sidecar,manifest_path,pointer),originals):path.write_bytes(content)
            with self.assertRaisesRegex(ValueError,'strict=True'):ck.load_checkpoint(directory,strict=False)
            self.rows.append(dict(kind='negative-matrix',cross_feature_rejections=rejected,structural_rejections=4,
                                  malformed_metadata=len(mutations),status='PASS'))

    def test_protocol_changes_reported_and_resume_rejected(self):
        c=config('A1K1');model=build_model(c);opt=optimizer(model);sched=scheduler(opt)
        step(model,c,io(c),opt,sched);meta=metadata(c,model,opt,sched)
        with tempfile.TemporaryDirectory() as tmp:
            ck.save_checkpoint(tmp,model,meta)
            runtime=config('A1K1',**{'runtime.device':'cuda:1'})
            new,saved,changes=ck.load_checkpoint(tmp,expected=runtime)
            self.assertEqual(set(changes),{'runtime'});assert_exact(saved,meta)
            altered=config('A1K1',**{'training.lr':.004})
            _,saved,changes=ck.load_checkpoint(tmp,expected=altered)
            self.assertEqual(set(changes),{'training'});assert_exact(saved,meta)
            with patch.object(torch,'load',side_effect=AssertionError('weight read')),self.assertRaisesRegex(ValueError,'training protocol'):
                ck.resume_checkpoint(tmp,expected=altered,optimizer_factory=None,scheduler_factory=None,generators={})
            with self.assertRaisesRegex(ValueError,'sampler/generator'):
                ck.resume_checkpoint(tmp,optimizer_factory=None,scheduler_factory=None,generators={'unexpected':torch.Generator()})
            with self.assertRaisesRegex(ValueError,'overwrite'):
                with torch.no_grad():next(model.parameters()).add_(.1)
                ck.save_checkpoint(tmp,model,meta)

    def test_wrong_state_keys_shapes_rejected_before_load_state_dict(self):
        models={sig:build_model(config(sig)) for sig in SIGNATURES}
        for src in SIGNATURES:
            for dst in SIGNATURES:
                if src==dst:continue
                with patch.object(torch.nn.Module,'load_state_dict',side_effect=AssertionError('must precheck')), \
                     self.assertRaisesRegex(ValueError,'keys differ'):
                    legacy.strict_load(models[dst],models[src].state_dict())
        model=models['A1K1'];state=dict(model.state_dict());first=next(iter(state))
        state[first]=torch.zeros(1)
        with patch.object(torch.nn.Module,'load_state_dict',side_effect=AssertionError('must precheck')), \
             self.assertRaisesRegex(ValueError,'shape/dtype'):
            legacy.strict_load(model,state)

    def test_resume_restores_dropout_optimizer_scheduler_and_all_rng(self):
        for sig in SIGNATURES:
            c=config(sig,dropout=.2);model=build_model(c).train();gates(model,.7,.4,.2)
            values=io(c);opt=optimizer(model);sched=scheduler(opt)
            generator=torch.Generator().manual_seed(c['initialization']['dataloader_generator_seed'])
            random.seed(662);np.random.seed(663);torch.manual_seed(664)
            step(model,c,values,opt,sched)
            meta=metadata(c,model,opt,sched,epoch=1,total_epochs=3,generators={'train':generator})
            with tempfile.TemporaryDirectory() as tmp:
                ck.save_checkpoint(tmp,model,meta)
                order=torch.randperm(13,generator=generator)
                draws=(random.random(),np.random.permutation(20).tolist(),torch.rand(7))
                out,loss=step(model,c,values,opt,sched)
                next_rng=torch.get_rng_state().clone()
                restored_generator=torch.Generator()
                new,o,s,read,changes=ck.resume_checkpoint(tmp,expected=c,
                    optimizer_factory=lambda p:torch.optim.AdamW(p,lr=.001,weight_decay=.01),
                    scheduler_factory=scheduler,generators={'train':restored_generator})
                self.assertTrue(new.training);self.assertFalse(changes);assert_exact(read,meta)
                assert_exact(order,torch.randperm(13,generator=restored_generator))
                assert_exact(draws,(random.random(),np.random.permutation(20).tolist(),torch.rand(7)))
                actual,actual_loss=step(new,c,values,o,s)
                assert_exact(out,actual);assert_exact(loss,actual_loss)
                assert_exact(model.state_dict(),new.state_dict());assert_exact(opt.state_dict(),o.state_dict())
                assert_exact(sched.state_dict(),s.state_dict());assert_exact(next_rng,torch.get_rng_state())
                self.rows.append(dict(kind='resume-exact',signature=sig,max_abs=0,status='PASS'))

    def test_industrial_joint_strict_native_roundtrip(self):
        for variant in ('airfrans','shapenet'):
            c=config('A1K1',variant=variant);model=build_model(c);values=io(c)
            gates(model,.8,.4,.3);opt=optimizer(model);sched=scheduler(opt)
            step(model,c,values,opt,sched);meta=metadata(c,model,opt,sched)
            model.eval()
            with tempfile.TemporaryDirectory() as tmp:
                ck.save_checkpoint(tmp,model,meta);new,read,_=ck.load_checkpoint(tmp,expected=c)
                assert_exact(forward(model,c,values),forward(new,c,values));assert_exact(meta,read)
            # Exercise the industrial adapter's metadata path for the new
            # A1K0 p=0 constructor field as well as the generic checkpoint path.
            nodrop = config('A1K0', variant=variant, history_dropout=0.0)
            nodrop_model = build_model(nodrop); nodrop_values = io(nodrop)
            nodrop_opt = optimizer(nodrop_model); nodrop_sched = scheduler(nodrop_opt)
            step(nodrop_model, nodrop, nodrop_values, nodrop_opt, nodrop_sched)
            nodrop_meta = metadata(nodrop, nodrop_model, nodrop_opt, nodrop_sched)
            from cdlno.linearno_history.industrial import make_metadata as industrial_make_metadata
            adapted = industrial_make_metadata(
                profile_spec=nodrop['profile_spec'], model_spec=nodrop['model_spec'],
                **{key: nodrop_meta[key] for key in (
                    'data_spec','objective_spec','evaluation_spec','provenance_spec',
                    'normalizer_spec','resume_state','ensemble_manifest')})
            self.assertEqual(adapted['innovation_spec']['attnres']['dropout']['p'], 0.0)
            self.rows.append(dict(kind='industrial-joint-object-reload',variant=variant,status='PASS'))

if __name__=='__main__':unittest.main()
