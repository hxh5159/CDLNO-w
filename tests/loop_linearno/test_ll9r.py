"""LL9R boundary precision, isolation and mutation-sensitive freeze acceptance."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from cdlno.linearno_loop.attnres import PointDepthAttnRes
from cdlno.linearno_loop.core import LinearNOLoopCore
from cdlno.linearno_loop.construction import build_from_config
from linearno_loop.config import resolve_config
from tools.linearno_loop_support import TASKS, PRESETS, MODES, configuration, inputs

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT/'docs/loop_linearno_audit/ll9r'


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mixed_boundary_matches_explicit_cast_with_live_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            for count in (2, 3):
                with self.subTest(dtype=dtype, count=count):
                    g=torch.Generator().manual_seed(104)
                    anchor=torch.randn(2,7,8,generator=g,requires_grad=True)
                    raw=[torch.randn(2,7,8,generator=g).to(dtype).requires_grad_() for _ in range(count-1)]
                    sources=(anchor,*raw);saved=[v.detach().clone() for v in sources]
                    receiver=PointDepthAttnRes(8)
                    with torch.no_grad():receiver.query.copy_(torch.linspace(-.1,.2,8))
                    reference=copy.deepcopy(receiver)
                    ref_sources=[v.detach().clone().requires_grad_() for v in sources]
                    result=LinearNOLoopCore._rb_receive(receiver,sources,anchor)
                    expected=reference(tuple(v.to(anchor.dtype) for v in ref_sources))
                    torch.testing.assert_close(result,expected,atol=0,rtol=0)
                    loss=result.square().mean();loss.backward();expected.square().mean().backward()
                    self.assertTrue(torch.isfinite(loss))
                    for value,old,ref in zip(sources,saved,ref_sources):
                        self.assertEqual(value.dtype,old.dtype);self.assertTrue(torch.equal(value,old))
                        self.assertTrue(torch.isfinite(value.grad).all());self.assertGreater(value.grad.abs().sum(),0)
                        torch.testing.assert_close(value.grad,ref.grad,atol=0,rtol=0)
                    for p,q in zip(receiver.parameters(),reference.parameters()):
                        self.assertTrue(torch.isfinite(p.grad).all())
                        torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
                    # Canonical dtype is supplied by anchor, never inferred from list order.
                    reordered=LinearNOLoopCore._rb_receive(receiver,(*raw,anchor),anchor)
                    self.assertEqual(reordered.dtype,torch.float32)
                    with self.assertRaisesRegex(TypeError,'dtype must match'):
                        receiver(sources)  # Shared primitive contract remains strict.

    def test_homogeneous_source_objects_and_no_cast(self):
        anchor=torch.randn(2,3,8);source=torch.randn_like(anchor)
        receiver=PointDepthAttnRes(8);seen=[]
        h=receiver.register_forward_pre_hook(lambda m,a:seen.append(a[0]))
        try:
            out=LinearNOLoopCore._rb_receive(receiver,(anchor,source),anchor)
            self.assertIs(seen[0][0],anchor);self.assertIs(seen[0][1],source)
            torch.testing.assert_close(out,receiver((anchor,source)),atol=0,rtol=0)
        finally:h.remove()

    def test_raw_cache_boundary_all_presets_and_custom(self):
        real=LinearNOLoopCore._rb_receive
        configs=[configuration('airfoil',p,'rb_attnres',small=True) for p in PRESETS]
        for repeats in (1,3):
            configs.append(resolve_config('airfoil',options=dict(topology_preset='custom',prefix_blocks=0,
                recurrent_core_blocks=2,loop_repeats=repeats,suffix_blocks=1,residual_mode='rb_attnres',linearno_rank=8),
                profile_overrides={'model.hidden':8,'model.heads':2,'model.H':5,'model.W':7}))
        for c in configs:
            model=build_from_config(c);calls=[];saved=[]
            def observed(receiver,sources,anchor):
                old=[(id(v),v.dtype,v.detach().clone()) for v in sources]
                # Keep the original cache references alive for verification.
                saved.extend(sources[1:]);calls.append(len(sources))
                out=real(receiver,sources,anchor)
                for v,(ident,dtype,value) in zip(sources,old):
                    self.assertEqual(id(v),ident);self.assertEqual(v.dtype,dtype)
                    self.assertTrue(torch.equal(v,value))
                return out
            # Local oneDNN lacks BF16 convolution backward on this CPU. Keep
            # autocast BF16, using PyTorch's native CPU convolution for this test.
            with torch.backends.mkldnn.flags(enabled=False), patch.object(LinearNOLoopCore,'_rb_receive',staticmethod(observed)):
                with torch.autocast('cpu',dtype=torch.bfloat16):out=model(*inputs(c))
                out.float().square().mean().backward()
            C=model.loop.recurrent_core_blocks;R=model.loop.loop_repeats
            expected=[s for r in range(R) for s in [r+1]+[r+2]*(2*C-1)]+[R+1]
            self.assertEqual(calls,expected)
            self.assertTrue(saved);self.assertTrue(all(v.dtype==torch.bfloat16 and v.grad_fn is not None for v in saved))
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            clone=build_from_config(c);clone.load_state_dict(model.state_dict(),strict=True)
            self.assertEqual(list(model.state_dict()),list(clone.state_dict()))

    def test_sr_lb_never_enter_rb_boundary_and_keep_state(self):
        for task in TASKS:
            for preset in PRESETS:
                for mode in (MODES[0],MODES[2]):
                    c=configuration(task,preset,mode,small=True);m=build_from_config(c)
                    with patch.object(LinearNOLoopCore,'_rb_receive',side_effect=AssertionError('SR/LB entered RB cast')):
                        y=m(*inputs(c));y.square().mean().backward()
                    self.assertTrue(torch.isfinite(y).all())

    def test_shared_primitive_and_sr_lb_source_ast_frozen(self):
        snapshot=Path(json.loads((AUDIT/'start-manifest.json').read_text())['snapshot'])/'source'
        for name in ('attnres.py','body.py','standard.py','airfrans.py','shapenet.py','checkpoint.py'):
            relative='cdlno/linearno_loop/'+name
            self.assertEqual((ROOT/relative).read_bytes(),(snapshot/relative).read_bytes())
        def methods(path):
            tree=ast.parse(path.read_text())
            return {n.name:ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
        old=methods(snapshot/'cdlno/linearno_loop/core.py');new=methods(ROOT/'cdlno/linearno_loop/core.py')
        for name in old:
            if name!='_rb_forward':self.assertEqual(old[name],new[name],name)

    def test_legacy_import_parser_namespace_and_missing_loop_dependency(self):
        code=r'''
import ast,argparse,importlib.abc,importlib.util,json,sys,subprocess
from pathlib import Path
root=Path(sys.argv[1]);kind=sys.argv[2];mode=sys.argv[3]
project=root/({'standard':'PDE-Solving-StandardBenchmark','air':'Airfoil-Design-AirfRANS','car':'Car-Design-ShapeNetCar'}[kind])
entry=project/('models/cdlno_run.py' if kind=='car' else 'cdlno_entry.py')
tree=ast.parse((project/('exp_darcy.py' if kind=='standard' else 'main.py')).read_text())
nodes=[n for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='parser' or isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='parser.add_argument']
def parser():
 s={'argparse':argparse,'Path':Path};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<parser>','exec'),s);return s['parser']
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='cdlno' or fullname.startswith('cdlno.'):
   raise ModuleNotFoundError('intentional missing shared package',name=fullname)
sys.meta_path.insert(0,Block())
spec=importlib.util.spec_from_file_location('entry',entry);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
tokens=['--cfd_model' if kind=='car' else '--model', 'Transolver_Structured_Mesh_2D' if kind=='standard' else 'Transolver']
if mode=='loop':tokens += ['--linearno-loop','1']
def parse(mod,p):return mod.parse_args(p,'darcy',argv=tokens) if kind=='standard' else mod.parse_args(p,argv=tokens)
if mode=='loop':
 try:parse(module,parser())
 except SystemExit as e:assert e.code==2
 else:raise AssertionError('missing loop dependency accepted')
else:
 p=parser();args=parse(module,p)
 old_source=subprocess.check_output(['git','show','5b991226c5354af3332b2f7306b370aef0950c79:'+str(entry.relative_to(root))],cwd=root,text=True)
 old=type('Entry',(),{})();scope={'__file__':str(entry)};exec(compile(old_source,str(entry),'exec'),scope);old.parse_args=scope['parse_args']
 q=parser();expected=parse(old,q);assert vars(args)==vars(expected)
 def actions(p):return [(a.option_strings,a.dest,a.default,a.required,a.nargs,a.choices,a.type) for a in p._actions]
 assert actions(p)==actions(q)
 assert not any(k=='cdlno' or k.startswith('cdlno.') for k in sys.modules)
print('PASS',kind,mode)
'''
        for kind in ('standard','air','car'):
            for mode in ('legacy','loop'):
                result=subprocess.run([sys.executable,'-I','-B','-c',code,str(ROOT),kind,mode],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                if mode=='loop':self.assertIn('linearno_loop selected but its shared cdlno loop package is unavailable',result.stderr)

    def test_pure_and_history_do_not_load_loop_adapters(self):
        code=r'''
import sys
sys.path[:0]=[sys.argv[1]+'/tests',sys.argv[1]]
from linearno.test_static_integration import args_for
for flags in ((),('--linearno_latent_attnres','1','--linearno_history_k_conditioning','0')):
 args=args_for('darcy',*flags)
 assert args.linearno_family in ('linearno','linearno_history')
 assert 'cdlno.linearno_loop.standard_entry' not in sys.modules
 assert 'cdlno.linearno_loop.industrial_entry' not in sys.modules
'''
        p=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)

    def test_air_projection_detects_real_training_and_eval_mutations(self):
        from output_recording_projection import strip_recording
        from msar_entry_projection import strip_msar
        from visualization_projection import strip_visualization
        for file in ('main.py','main_evaluation.py','train.py'):
            original=ast.parse((ROOT/'Airfoil-Design-AirfRANS'/file).read_text())
            for projection in (strip_recording,strip_msar,lambda t:strip_visualization(strip_msar(t))):
                baseline=projection(copy.deepcopy(original));mutated=copy.deepcopy(original)
                # Alter a surviving native calculation/call, not any new-family guard.
                target=next(n for n in ast.walk(baseline) if isinstance(n,ast.Call) and
                            ast.unparse(n.func) in ('train.main','Results_test','metrics.Results_test','optimizer.step','lr_scheduler.step'))
                needle=ast.dump(target)
                candidates=[n for n in ast.walk(mutated) if isinstance(n,ast.Call) and ast.dump(n)==needle]
                if not candidates:
                    # A native call may have observational kwargs projected out;
                    # mutate the same function's native callee instead.
                    candidates=[n for n in ast.walk(mutated) if isinstance(n,ast.Call) and ast.unparse(n.func)==ast.unparse(target.func)]
                self.assertTrue(candidates)
                candidates[-1].func=ast.Name(id='BROKEN_NATIVE_CALCULATION',ctx=ast.Load())
                self.assertNotEqual(ast.dump(baseline),ast.dump(projection(mutated)),file)

    def test_exact_repair_projections_reject_source_mutation(self):
        from cdlno.linearno_loop.ll9r_projection import project,REPLACEMENTS
        from ll9r_projection import strip_repair
        snapshot=Path(json.loads((AUDIT/'start-manifest.json').read_text())['snapshot'])/'source'
        for name in REPLACEMENTS:
            current=(ROOT/name).read_text();old=(snapshot/name).read_text()
            self.assertEqual(project(name,current),old,name)
            if name.endswith('/core.py'):
                with self.assertRaises(ValueError):project(name,current.replace('canonical_dtype = anchor.dtype','canonical_dtype = sources[-1].dtype'))
                self.assertNotEqual(project(name,current.replace('partial+raw','partial-raw')),old)
        for name in json.loads((AUDIT/'routing-replacements.json').read_text()):
            current=ast.parse((ROOT/name).read_text())
            self.assertEqual(ast.dump(strip_repair(current)),ast.dump(ast.parse((snapshot/name).read_text())))

    def test_frozen_document_revisions_exact_committed_bytes_and_old_content(self):
        from frozen_revisions import ROWS,expected_hash
        for name,row in ROWS.items():
            raw=(ROOT/name).read_bytes()
            self.assertEqual(raw,subprocess.check_output(['git','show',row['commit']+':'+name],cwd=ROOT))
            self.assertEqual(hashlib.sha256(raw).hexdigest(),expected_hash(name,row['old_sha256']))
            self.assertNotEqual(hashlib.sha256(raw+b'changed').hexdigest(),expected_hash(name,row['old_sha256']))
            with self.assertRaises(AssertionError):expected_hash(name,'0'*64)

    def test_freeze_keeps_untracked_source_and_rejects_real_py_change(self):
        from ll9r_test_source_projection import project_test_source
        rows=json.loads((ROOT/'docs/loop_linearno_audit/ll1/start-manifest.json').read_text())['files']
        eligible=[r for r in rows if '__pycache__' not in Path(r['path']).parts and not r['path'].endswith(('.pyc','.pyo'))]
        self.assertTrue(any(r['classification']=='untracked' for r in eligible))
        row=next(r for r in eligible if r['path']=='cdlno/linearno/attention.py')
        raw=project_test_source(row['path'],(ROOT/row['path']).read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])
        self.assertNotEqual(hashlib.sha256(project_test_source(row['path'],raw+b'\n# mutation\n')).hexdigest(),row['sha256'])


if __name__=='__main__':unittest.main()
