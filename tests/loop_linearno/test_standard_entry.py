"""LL6 public routing, metadata-before-load, strict continuation and isolation."""
import ast
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from cdlno.linearno_loop.standard_entry import LoopStandardRun
from cdlno.linearno_loop import checkpoint
from cdlno.linearno_loop.provenance import REPLACEMENTS,legacy_source
from cdlno.linearno.standard_entry import start,finish
from cdlno.linearno.schema import pack_state,unpack_state
from linearno_loop.config import run_directory_id,resolve_config
from linearno_loop.contracts import PRESETS,RESIDUAL_MODES,digest
from loop_linearno.native_worker import fixture_config
from linearno.static_worker import ROOT,PROJECT,parser_for
from linearno.temporal_worker import parser_for as temporal_parser
from cdlno_entry import parse_args
from model_dict import get_model
from linearno_entry import StandardRun,model_kwargs

TASKS=('airfoil','darcy','elasticity','pipe','ns','plasticity')


def parse(task,tokens):
    return parse_args((temporal_parser if task in ('ns','plasticity') else parser_for)(task),task,list(map(str,tokens)))


def flags(task,preset='p1_c3_r2_s1',mode='lb_attnres_1_over_r'):
    return ['--model','LinearNO_Irregular_Mesh' if task=='elasticity' else 'LinearNO_Structured_Mesh_2D',
        '--linearno-loop','1','--linearno-loop-topology',preset,'--linearno-loop-residual-mode',mode]


class EntryTests(unittest.TestCase):
    def test_six_task_profiles_and_modes_resolve_without_legacy_defaults(self):
        for task in TASKS:
            for preset in PRESETS:
                for mode in RESIDUAL_MODES:
                    for profile in ('paper_table8_on_release_model','official_release','transolver_matched'):
                        a=parse(task,flags(task,preset,mode)+['--linearno-profile',profile])
                        c=resolve_config(task,profile,options=dict(topology_preset=preset,residual_mode=mode))
                        self.assertEqual(a._linearno_loop_config,c)
                        self.assertEqual(a.linearno_rank,2*a._linearno_config['values']['model']['linearno_rank'])
                        self.assertEqual(model_kwargs(a),c['model_spec']['constructor_kwargs'])
                        self.assertIn(run_directory_id(c),a.linearno_run_dir.name)
        a=parse('ns',flags('ns'));self.assertEqual((a.ref,a.mlp_ratio,a.n_hidden),(10,2,256))
        custom=flags('darcy','custom')+['--linearno-loop-prefix-blocks',0,'--linearno-loop-core-blocks',2,
            '--linearno-loop-repeats',3,'--linearno-loop-suffix-blocks',1,'--linearno-loop-rank-multiplier',1]
        a=parse('darcy',custom);self.assertEqual((a.n_layers,a.linearno_rank),(3,64))

    def test_illegal_flags_fail_before_model_load_rng_or_directory(self):
        base=flags('darcy')
        cases=[base+['--linearno_latent_attnres',0],base+['--linearno-history-k-conditioning',0],
            base+['--linearno_attnres_history_dropout_p',0],base+['--linearno-fair-run',1],
            base+['--linearno-rank',64,'--linearno-loop-rank-multiplier',1],
            base+['--linearno-loop-prefix-blocks',1],base+['--n-layers',8],base+['--slice_num',64],
            base+['--linearno-loop-repeats',0],base+['--linearno-loop-rank-multiplier',3],
            base+['--linearno-loop-rank-multiplier','NaN'],base+['--linearno-loop','0'],
            base+['--linearno-loop-residual-mode','unknown'],base+['--model','Transolver_Structured_Mesh_2D'],
            base+['--linearno-profile','official_release','--epochs',3],
            base[:2]+['--linearno-loop',1],base[:2]+base[4:],base+['--resume'],
            base+['--linearno-loop-topology','custom','--linearno-loop-core-blocks',1]]
        from cdlno.linearno_loop import construction
        for tokens in cases:
            rng=torch.get_rng_state().clone()
            with self.subTest(tokens=tokens),patch('torch.load',side_effect=AssertionError('early weight read')),patch.object(
                    construction,'build_from_config',side_effect=AssertionError('early model')):
                with self.assertRaises(SystemExit):parse('darcy',tokens)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_legacy_parser_factory_no_fields_rng_and_exact_source_projection(self):
        from cdlno.linearno_loop.standard_entry import intercept
        for task in TASKS:
            commands=[[],['--model','Transolver_Irregular_Mesh'],['--model','CDLNO'],
                ['--model','LinearNO_Irregular_Mesh' if task=='elasticity' else 'LinearNO_Structured_Mesh_2D'],
                ['--model','LinearNO_Irregular_Mesh' if task=='elasticity' else 'LinearNO_Structured_Mesh_2D',
                 '--linearno_latent_attnres',1,'--linearno_history_k_conditioning',0]]
            for tokens in commands:
                rng=torch.get_rng_state().clone();a=parse(task,tokens);after=torch.get_rng_state().clone()
                torch.set_rng_state(rng)
                with patch('cdlno.linearno_loop.standard_entry.intercept',return_value=None):b=parse(task,tokens)
                # Timestamp/uuid run names are intentionally fresh on each parse.
                va={k:v for k,v in vars(a).items() if k not in ('linearno_run_dir',)}
                vb={k:v for k,v in vars(b).items() if k not in ('linearno_run_dir',)}
                self.assertEqual(va,vb);self.assertTrue(torch.equal(after,torch.get_rng_state()))
                self.assertFalse(hasattr(a,'_linearno_loop_config'))
                if tokens:
                    if hasattr(a,'_linearno_history_config'):
                        self.assertEqual(get_model(a).Model.__module__,'cdlno.linearno_history.standard_entry')
                    else:self.assertIs(get_model(a),get_model(b))
        out=ROOT/'docs/loop_linearno_audit/ll6'
        for name in REPLACEMENTS:
            self.assertEqual(legacy_source(name,(ROOT/name).read_text()),(out/'before'/name).read_text())
        old=json.loads((out/'pre-edit.json').read_text())
        from cdlno.linearno.standard_entry import provenance as pure
        from cdlno.linearno_history.standard_entry import provenance as history
        from cdlno.linearno_history.checkpoint import source_hash
        for name,fn in [('pure',pure),('history',history)]:
            for key in ('source_sha256','normalized_patch_sha256'):self.assertEqual(fn()[key],old[name][key])
        self.assertEqual(source_hash(),old['history_source'])

    def test_real_pair_negative_shapes_mode_conflicts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=fixture_config('airfoil','p1_c3_r2_s1','lb_attnres_1_over_r');directory=Path(temp)/run_directory_id(cfg)
            tokens=flags('airfoil')+['--experiment-dir',directory,'--epochs',3,'--n-hidden',8,'--n-heads',2,
                '--linearno-rank',4,'--batch-size',2,'--seed',17,'--dropout',.1]
            a=parse('airfoil',tokens);a._linearno_data=dict(split=cfg['profile_spec']['values']['data']['split'],
                sampling=cfg['profile_spec']['values']['data']['sampling'],checksums={'SYNTHETIC':digest('test fixture')},scope='SYNTHETIC')
            start(a,'airfoil')
            try:
                m=get_model(a).Model(**model_kwargs(a));run=StandardRun(a,m);self.assertIsInstance(run,LoopStandardRun)
                x=torch.randn(4,221*51,2);train=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x),batch_size=2,shuffle=True)
                test=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[:2]),batch_size=2)
                opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.weight_decay)
                sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,epochs=3,steps_per_epoch=2)
                run.prepare(opt,sch,train,test)
                for (batch,) in train:
                    opt.zero_grad();m(batch,None).square().mean().backward();opt.step();sch.step()
                run.complete_epoch(1);path=run.save(m)
            finally:finish(a)
            metadata,weights=checkpoint.read_pair(path,expected=cfg)
            self.assertEqual(metadata['family'],'linearno_loop')
            self.assertFalse(any('round2' in k for k in weights))
            checkpoint.strict_load(m,weights)
            with self.assertRaises(SystemExit):parse('airfoil',tokens)
            for more in (['--linearno-loop-residual-mode','sr_1_over_r'],['--linearno-rank',8],
                         ['--linearno-loop-topology','p2_c2_r2_s2'],['--n-heads',4],['--linearno_latent_attnres',0]):
                with patch('torch.load',side_effect=AssertionError('early read')):
                    with self.assertRaises(SystemExit):parse('airfoil',['--resume','--experiment-dir',directory,*more])
            a2=parse('airfoil',['--resume','--experiment-dir',directory]);self.assertEqual(a2._linearno_loop_config,cfg)
            bad=copy.deepcopy(weights);bad.pop(next(k for k in bad if k.startswith('loop.lb_')))
            with self.assertRaises(ValueError):checkpoint.strict_load(m,bad)
            state=unpack_state(metadata['resume_state']['optimizer'])
            for kind in ('groups','shape','ids'):
                wrong=copy.deepcopy(state)
                if kind=='groups':wrong['param_groups'].append(copy.deepcopy(wrong['param_groups'][0]))
                if kind=='shape':next(iter(wrong['state'].values()))['exp_avg']=torch.zeros(1)
                if kind=='ids':wrong['param_groups'][0]['params'][1]=wrong['param_groups'][0]['params'][0]
                with self.assertRaises(ValueError):checkpoint.validate_optimizer_state(wrong,opt)
            checkpoint.validate_optimizer_state(state,opt)
            initial=(directory/'architecture.json').read_bytes()
            self.assertEqual(run.save(m),path)
            with torch.no_grad():next(m.parameters()).add_(1)
            with self.assertRaises(ValueError):run.save(m)
            self.assertEqual((directory/'architecture.json').read_bytes(),initial)


if __name__=='__main__':unittest.main()
