"""Synthetic AirfRANS interfaces; original data/metrics entrypoints never run."""
import ast
from output_recording_projection import strip_recording
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import yaml

from cdlno.airfrans import AirfRANSModel
from cdlno.checkpoint import SidecarMismatch

ROOT = Path(__file__).resolve().parents[1]
AIR = ROOT / 'Airfoil-Design-AirfRANS'
BASE = '75e0f67643806a81cd1d3f6adc88dd8c02416fe7'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


# Do not collide with standard/Car projects' local modules in the combined suite.
entry = module(AIR / 'cdlno_entry.py', 'air_cdlno_entry_for_test')
HAS_PYG = importlib.util.find_spec('torch_geometric') is not None
if HAS_PYG:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader


def original(filename):
    return subprocess.check_output(['git', 'show', BASE + ':Airfoil-Design-AirfRANS/' + filename],
                                   cwd=ROOT, text=True)


def parser(evaluation=False, baseline=False):
    file = 'main_evaluation.py' if evaluation else 'main.py'
    tree = ast.parse(original(file) if baseline else (AIR / file).read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'parser':
            nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'parser.add_argument':
            nodes.append(node)
    namespace = {'argparse': __import__('argparse')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<parser-only>', 'exec'), namespace)
    return namespace['parser']


def args(extra=(), evaluation=False):
    return entry.parse_args(parser(evaluation), evaluation=evaluation, argv=['--model', 'CDLNO', *extra])


def hparams():
    return yaml.safe_load((AIR / 'params.yaml').read_text())['CDLNO']


def run(a, evaluation=False):
    with redirect_stdout(io.StringIO()):
        return entry.AirRun(a, hparams(), device='cpu', evaluation=evaluation)


SMALL = ['--n_hidden', '16', '--n_heads', '4', '--slice_num', '4']


class AirWrapperChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(110)

    def test_defaults_modes_gradients_placeholder_initialization(self):
        a = args()
        self.assertEqual((a.n_hidden, a.n_heads, a.slice_num, a.n_layers, a.front_blocks), (256, 8, 64, 8, 2))
        self.assertEqual(entry.resolve_hparams(a, hparams())['nb_epochs'], 398)
        for mode in ('off', 'entry', 'every_block'):
            m = AirfRANSModel(**entry.model_kwargs(args(['--cdpa_mode', mode])))
            self.assertEqual(m.preprocess[0].in_features, 71)
            self.assertFalse(hasattr(m, 'time_fc'))
            self.assertFalse(any(isinstance(n, nn.Conv2d) for n in m.modules()))
            self.assertTrue(((m.placeholder >= 0) & (m.placeholder < 1/256)).all())
            x = torch.randn(11, 7, requires_grad=True)
            pos = torch.randn(11, 2, requires_grad=True)
            out = m(SimpleNamespace(x=x, pos=pos))
            self.assertEqual(out.shape, (11, 4))
            out.square().mean().backward()
            for name, p in m.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(x.grad.abs().max().item(), 0)
            self.assertGreater(pos.grad.abs().max().item(), 0)
            for block in m.core.front_blocks:
                q = block.down.latent_queries.reshape(64, 256)
                torch.testing.assert_close(q @ q.T, torch.eye(64), atol=1e-6, rtol=1e-6)
            for cdpa in m.core.cdpa_at.values():
                self.assertEqual(torch.count_nonzero(cdpa.w).item(), 0)

    def test_reference_matches_original_get_grid_and_lifting(self):
        original_model = module(AIR / 'models/Transolver.py', 'air_original_model_for_test')
        tree = ast.parse((AIR / 'models/Transolver.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Transolver')
        grid_func = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_grid'))
        class CPUGrid(ast.NodeTransformer):
            count = 0
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                    self.count += 1
                    return self.visit(node.func.value)
                return self.generic_visit(node)
        cpu = CPUGrid()
        grid_func = cpu.visit(grid_func)
        self.assertEqual(cpu.count, 1)
        namespace = dict(np=np, torch=torch)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[grid_func], type_ignores=[])), '<original-grid-cpu-device-only>', 'exec'), namespace)
        m = AirfRANSModel(**entry.model_kwargs(args(SMALL)))
        pos = torch.cat((torch.tensor([[-2., -1.5], [4., 1.5]]), torch.randn(15, 2)))
        x = torch.randn(17, 7)
        expected_distance = namespace['get_grid'](SimpleNamespace(ref=8), pos[None])
        old_stem = original_model.MLP(71, 32, 16, n_layers=0, res=False, act='gelu')
        old_stem.linear_pre[0].load_state_dict(m.preprocess[0].state_dict())
        old_stem.linear_post.load_state_dict(m.preprocess[2].state_dict())
        seen = []
        handle = m.core.register_forward_pre_hook(lambda mod, values: seen.append(values[0]))
        m(SimpleNamespace(x=x, pos=pos))
        handle.remove()
        expected = old_stem(torch.cat((x[None], expected_distance), dim=-1)) + m.placeholder[None, None]
        torch.testing.assert_close(seen[0], expected, atol=0, rtol=0)
        self.assertEqual(expected_distance[0, 0, 0].item(), 0)
        self.assertEqual(expected_distance[0, 1, -1].item(), 0)
        self.assertEqual(m.adapter_architecture()['reference_domain'], [[-2., 4.], [-1.5, 1.5]])

    def test_cli_yaml_inheritance_and_explicit_overrides(self):
        current = yaml.safe_load((AIR / 'params.yaml').read_text())
        base = yaml.safe_load(original('params.yaml'))
        self.assertEqual({k:v for k,v in current.items() if k != 'CDLNO'}, base)
        self.assertEqual(current['CDLNO'], current['Transolver'])
        self.assertEqual(current['Transolver']['nb_epochs'], 398)
        a = args(['--n-layers', '12', '--front-blocks', '7', '--nb_epochs', '400'])
        self.assertEqual(AirfRANSModel(**entry.model_kwargs(a)).config.P, 5)
        self.assertEqual(entry.resolve_hparams(a, hparams())['nb_epochs'], 400)
        # A legitimate future YAML edit wins over parser defaults, without guessing.
        self.assertEqual(entry.resolve_hparams(args(), dict(hparams(), nb_epochs=401))['nb_epochs'], 401)
        for evaluation in (False, True):
            old = entry.parse_args(parser(evaluation), evaluation=evaluation, argv=[] if evaluation else ['--model', 'Transolver'])
            base_args = parser(evaluation, True).parse_args([] if evaluation else ['--model', 'Transolver'])
            self.assertEqual({k:getattr(old,k) for k in vars(base_args)}, vars(base_args))
            if evaluation:
                self.assertEqual((old.model, old.task), ('Transolver', 'full'))
        with self.assertRaises(ValueError):
            entry.resolve_hparams(args(['--batch_size', '2']), hparams())
        with self.assertRaises(ValueError):
            args(['--front_blocks', '8'])
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            args([], True)

    def test_sidecar_read_first_member_paths_and_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            flags = [*SMALL, '--run_dir', str(Path(temp) / 'run'), '--nmodel', '2']
            training = run(args(flags))
            models = [AirfRANSModel(**entry.model_kwargs(args(SMALL))) for _ in range(2)]
            torch.save(models, training.checkpoint)
            for index, m in enumerate(models):
                Path(training.member_dir(index)).mkdir()
                torch.save(m, Path(training.member_dir(index)) / 'model')
            before = training.sidecar.read_bytes()
            evaluation = run(args([*flags, '--cdpa_source_chunk_size', '1'], True), True)
            with patch('torch.load', wraps=torch.load) as loading:
                loaded = evaluation.load()
                self.assertEqual(loading.call_count, 1)
                self.assertEqual(loading.call_args.kwargs, dict(map_location='cpu', weights_only=False))
            self.assertEqual(len(loaded), 2)
            for index in range(2):
                x = SimpleNamespace(x=torch.randn(7, 7), pos=torch.randn(7, 2))
                torch.testing.assert_close(evaluation.load(member=index)(x), models[index](x), atol=1e-5, rtol=3e-4)
                self.assertEqual(loaded[index].core.source_chunk_size, 1)
            for flags2 in (['--task', 'scarce'], ['--slice_num', '5'], ['--nmodel', '1'], ['--nb_epochs', '399']):
                with patch('torch.load') as forbidden:
                    with self.assertRaises(SidecarMismatch):
                        run(args([*flags, *flags2], True), True)
                    forbidden.assert_not_called()
            self.assertEqual(training.sidecar.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                run(args(flags))
            self.assertNotEqual(evaluation.result_dir, run(args(flags, True), True).result_dir)
            self.assertNotEqual(training.member_dir(0), training.member_dir(1))
            malformed = json.loads(before)
            malformed['metadata']['wrapper_architecture']['stem_width'] = 65
            training.sidecar.write_text(json.dumps(malformed))
            with patch('torch.load') as forbidden, self.assertRaises(SidecarMismatch):
                evaluation.load()
            forbidden.assert_not_called()
            training.sidecar.unlink()
            with patch('torch.load') as forbidden, self.assertRaises(FileNotFoundError):
                evaluation.load()
            forbidden.assert_not_called()
            self.assertFalse(training.sidecar.exists())
            for n in (0,1):
                self.assertTrue((Path(training.member_dir(n))/'model').exists())
            automatic = args([*SMALL, '--save_path', temp])
            self.assertNotEqual(run(automatic).directory, run(automatic).directory)

    def test_checkpoint_list_type_length_and_strict_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            flags = [*SMALL, '--run_dir', str(Path(temp) / 'run')]
            training = run(args(flags))
            evaluation = run(args(flags, True), True)
            m = AirfRANSModel(**entry.model_kwargs(args(SMALL)))
            for saved in (m, [], [nn.Linear(71, 4)]):
                torch.save(saved, training.checkpoint)
                with self.assertRaises(SidecarMismatch):
                    evaluation.load()
            del m.placeholder
            torch.save([m], training.checkpoint)
            with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                evaluation.load()

    def test_whole_model_and_list_fresh_processes_in_air_cwd(self):
        with tempfile.TemporaryDirectory() as temp:
            # Execute the actual entry's model branch and save expressions,
            # without the data/epoch/score parts of either entry.
            calls = []
            for filename in ('train.py', 'main.py'):
                tree = ast.parse((AIR / filename).read_text())
                calls.append(ast.unparse(next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.save')))
            train_tree = ast.parse((AIR / 'main.py').read_text())
            selection = ast.unparse(next(n for n in ast.walk(train_tree) if isinstance(n, ast.If)
                                       and ast.unparse(n.test) == "args.model == 'Transolver'"))
            eval_tree = ast.parse((AIR / 'main_evaluation.py').read_text())
            loading = ast.unparse(next(n for n in ast.walk(eval_tree) if isinstance(n, ast.If)
                                     and ast.unparse(n.test) == "model == 'CDLNO'"))
            setup = '''
import torch, sys, os.path as osp
from pathlib import Path
from argparse import ArgumentParser
from types import SimpleNamespace
import yaml
from cdlno_entry import parse_args, model_kwargs, resolve_hparams, AirRun
from models.CDLNO import Model
torch.set_num_threads(1)
p=ArgumentParser(); p.add_argument('--model'); p.add_argument('--task',default='full'); p.add_argument('--nmodel',type=int,default=2); p.add_argument('--weight',type=float,default=1.); p.add_argument('--save_path',default=sys.argv[1])
a=parse_args(p,argv=['--model','CDLNO','--n_hidden','16','--n_heads','4','--slice_num','4','--run_dir',sys.argv[1]])
h=yaml.safe_load(Path('params.yaml').read_text())['CDLNO']
args=a; cdlno_model_kwargs=model_kwargs; device='cpu'
'''
            writer = setup + '''
r=AirRun(a,h,device='cpu'); cdlno_run=r; models=[]
x=torch.randn(13,7); pos=torch.randn(13,2); outputs=[]
for i in range(2):
'''
            writer += '\n'.join(' ' + line for line in selection.splitlines()) + '\n'
            writer += ' model.eval(); path=r.member_dir(i); Path(path).mkdir()\n'
            writer += ' ' + calls[0] + '\n'
            writer += '''
 models.append(model); outputs.append(model(SimpleNamespace(x=x,pos=pos)).detach())
'''
            writer += calls[1] + '\n'
            writer += '''
torch.save(dict(x=x,pos=pos,outputs=outputs),r.directory/'expected.pt')
'''
            reader = setup + '''
a.cdpa_source_chunk_size=1
before=(Path(a.run_dir)/'architecture.json').read_bytes()
model='CDLNO'
'''
            reader += loading + '\n'
            reader += '''
r=cdlno_run; loaded=mod; data=torch.load(r.directory/'expected.pt',weights_only=True)
for i,m in enumerate(loaded):
 assert type(m).__module__=='cdlno.airfrans'
 assert m.core.source_chunk_size==1
 for candidate in (m,r.load(member=i)):
  torch.testing.assert_close(candidate(SimpleNamespace(x=data['x'],pos=data['pos'])),data['outputs'][i],atol=1e-5,rtol=3e-4)
assert r.sidecar.read_bytes()==before
print('fresh AirfRANS cwd: whole model/list and sidecar passed')
'''
            for code in (writer, reader):
                result = subprocess.run([sys.executable, '-B', '-c', code, str(Path(temp)/'run')],
                    cwd=AIR, env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(HAS_PYG, 'real torch_geometric absent: AirfRANS PyG integration NOT completed')
class AirRealPyG(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    @staticmethod
    def data(n):
        return Data(x=torch.randn(n,7), pos=torch.randn(n,2), y=torch.randn(n,4),
                    surf=torch.arange(n)>=n//2, edge_index=torch.tensor([[0,1],[1,0]]))

    def test_variable_nodes_batch_and_immutable_fields_no_y_leak(self):
        m = AirfRANSModel(**entry.model_kwargs(args(SMALL))).eval()
        for n in (9,31):
            raw = self.data(n)
            for data in (raw, Batch.from_data_list([raw])):
                before = {k:(v.clone(),v.data_ptr()) for k,v in data if isinstance(v,torch.Tensor)}
                result = m(data)
                self.assertEqual(result.shape,(n,4))
                for k,(v,ptr) in before.items():
                    torch.testing.assert_close(v,data[k],rtol=0,atol=0)
                    self.assertEqual(ptr,data[k].data_ptr())
            expected=m(raw)
            changed=raw.clone(); changed.y=torch.randn_like(raw.y)*1000; changed.surf=~raw.surf
            torch.testing.assert_close(m(changed),expected,atol=0,rtol=0)
            del changed.y
            torch.testing.assert_close(m(changed),expected,atol=0,rtol=0)
            order=torch.randperm(n)
            torch.testing.assert_close(m(Data(x=raw.x[order],pos=raw.pos[order])),expected[order],atol=2e-6,rtol=2e-5)

    def test_sampled_single_graph_ptr_and_multiple_graph_rejection(self):
        m=AirfRANSModel(**entry.model_kwargs(args(SMALL)))
        data=Batch.from_data_list([self.data(29)])
        sampled=data.clone(); idx=torch.tensor([27,0,5,9,3,12,20])
        # Original Infer_test assigns exactly these attributes; ptr remains [0,29].
        for name in ('pos','x','y','surf','batch'):
            setattr(sampled,name,getattr(sampled,name)[idx])
        self.assertEqual(sampled.ptr.tolist(),[0,29])
        torch.testing.assert_close(m(sampled),m(Data(x=sampled.x,pos=sampled.pos)),atol=0,rtol=0)
        self.assertEqual(sampled.ptr.tolist(),[0,29])
        multiple=Batch.from_data_list([self.data(5),self.data(8)])
        with self.assertRaisesRegex(ValueError,'one physical graph'):
            m(multiple)
        for fields in (dict(ptr=torch.tensor([0,3,7])),dict(batch=torch.tensor([0,0,0,1,1,1,1])),
                       dict(ptr=torch.tensor([0,7,7]),batch=torch.zeros(7,dtype=torch.long))):
            with self.assertRaisesRegex(ValueError,'one physical graph'):
                m(Data(x=torch.randn(7,7),pos=torch.randn(7,2),**fields))
        with self.assertRaises(ValueError):
            m(Data(x=torch.randn(7,7),pos=torch.randn(7,2),ptr=torch.tensor([0,29])))

    def test_original_weighted_loss_train_and_test(self):
        training=module(AIR/'train.py','air_train_for_test')  # functions only, no epoch loop called.
        m=AirfRANSModel(**entry.model_kwargs(args(SMALL)))
        reference=copy.deepcopy(m); data=self.data(23); out=reference(data)
        criterion=nn.MSELoss(reduction='none'); loss=criterion(out,data.y)
        surface=loss[data.surf].mean(); volume=loss[~data.surf].mean()
        (volume+1.7*surface).backward()
        opt=torch.optim.Adam(m.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.001,total_steps=3)
        before=m.placeholder.detach().clone(); initial_step=scheduler.last_epoch
        loader=DataLoader([data],batch_size=1)
        result=training.train('cpu',m,loader,opt,scheduler,criterion='MSE_weighted',reg=1.7)
        self.assertAlmostEqual(float(result[0]),loss.mean().item(),places=6)
        self.assertAlmostEqual(float(result[4]),surface.item(),places=6)
        self.assertAlmostEqual(float(result[5]),volume.item(),places=6)
        self.assertEqual(scheduler.last_epoch,initial_step+1)
        self.assertFalse(torch.equal(m.placeholder,before))
        for (name,p),(_,q) in zip(m.named_parameters(),reference.named_parameters()):
            self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)
            torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
        self.assertTrue(np.isfinite(training.test('cpu',m,loader)[0]))

    def test_32000_synthetic_points_small_width_backward(self):
        m=AirfRANSModel(n_hidden=8,n_head=2,slice_num=4)
        data=self.data(32000); data.x.requires_grad_()
        out=m(data); self.assertEqual(out.shape,(32000,4))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(data.x.grad).all())


class AirFrozenChecks(unittest.TestCase):
    def test_complete_entry_ast_preserves_original_protocol(self):
        class Project(ast.NodeTransformer):
            def visit_ImportFrom(self,node):
                return None if node.module=='cdlno_entry' else node
            def visit_Assign(self,node):
                name=ast.unparse(node.targets[0])
                if name in ('cdlno_run','score_path','score_array_path'):
                    return None
                if name=='args':
                    node.value=ast.parse('parser.parse_args()',mode='eval').body
                if name=='tasks': node.value=ast.parse("['full']",mode='eval').body
                if name=='model_names': node.value=ast.parse("['Transolver']",mode='eval').body
                if name in ('log_path','results_dir') and isinstance(node.value,ast.IfExp):
                    node.value=node.value.orelse
                return self.generic_visit(node)
            def visit_If(self,node):
                if ast.unparse(node.test) in ("args.model == 'CDLNO'", "model == 'CDLNO'"):
                    return [self.visit(n) for n in node.orelse]
                return self.generic_visit(node)
            def visit_IfExp(self,node):
                if ast.unparse(node.test)=='cdlno_run is not None':
                    return self.visit(node.orelse)
                return self.generic_visit(node)
            def visit_Call(self,node):
                if ast.unparse(node.func)=='torch.load':
                    node.keywords=[k for k in node.keywords if k.arg not in ('weights_only','map_location')]
                if ast.unparse(node.func)=='osp.join' and node.args and ast.unparse(node.args[0])=='score_array_path':
                    node.args=[ast.Constant(value='scores'),ast.parse('args.task',mode='eval').body,*node.args[1:]]
                return self.generic_visit(node)
            def visit_Name(self,node):
                if node.id=='score_path': return ast.Constant(value='scores')
                return node
        for filename in ('main.py','main_evaluation.py'):
            projected=Project().visit(strip_recording(ast.parse((AIR/filename).read_text())))
            self.assertEqual(ast.dump(projected),ast.dump(ast.parse(original(filename))),filename)

    def test_frozen_training_sampling_metrics_and_original_models(self):
        # New CDLNO files can be committed after the audit; freeze only the
        # original file inventory from BASE, including any later deletions.
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'--','Airfoil-Design-AirfRANS'],cwd=ROOT,text=True).splitlines()
        count=0
        for name in names:
            if Path(name).name in ('main.py','main_evaluation.py','params.yaml'): continue
            expected=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
            if Path(name).name == 'train.py':
                self.assertEqual(ast.dump(strip_recording(ast.parse((ROOT/name).read_text()))), ast.dump(ast.parse(expected)), name)
            else:
                self.assertEqual((ROOT/name).read_bytes(),expected,name)
            count+=1
        self.assertTrue((AIR/'params.yaml').read_text().startswith(original('params.yaml')))
        print('AirfRANS original frozen files byte-identical:',count)

    # Argument-only launch check; actual sidecar/load tests do not mock this.
    @patch('cdlno.checkpoint.resolve_front_latent_mode', return_value='full')
    def test_scripts_configuration_and_evaluation_model_selection(self, _mode):
        for evaluation,filename in ((False,'CDLNO.sh'),(True,'CDLNO_Evaluation.sh')):
            script=AIR/'scripts'/filename
            subprocess.run(['bash','-n',str(script)],check=True)
            extras=['--my_path','/custom/Dataset' if not evaluation else '/custom','--run_dir','/tmp/new-or-existing',
                    '--n_layers','12','--front_blocks','7','--task','aoa','--nb_epochs','400']
            capture='python() { printf "%s\\0" "$@"; }; export -f python; bash "$@"'
            result=subprocess.run(['bash','-c',capture,'capture',str(script),*extras],cwd=AIR,check=True,capture_output=True)
            tokens=result.stdout.decode().rstrip('\0').split('\0')
            a=entry.parse_args(parser(evaluation),evaluation=evaluation,argv=tokens[1:])
            self.assertEqual((a.model,a.task,a.n_layers,a.front_blocks),('CDLNO','aoa',12,7))
            self.assertEqual(a.my_path,'/custom' if evaluation else '/custom/Dataset')
            self.assertEqual(entry.resolve_hparams(a,hparams())['nb_epochs'],400)
            if not evaluation: self.assertEqual(a.score,0)
        tree=ast.parse((AIR/'main_evaluation.py').read_text())
        selected=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='model_names')
        self.assertEqual(eval(compile(ast.Expression(selected.value),'<selection>','eval'),{'args':args(['--run_dir','/tmp/r'],True)}),['CDLNO'])
        for filename in ('main.py','main_evaluation.py','cdlno_entry.py','models/CDLNO.py'):
            ast.parse((AIR/filename).read_text(),feature_version=(3,10))
        ast.parse((ROOT/'cdlno/airfrans.py').read_text(),feature_version=(3,10))


if __name__=='__main__': unittest.main()
