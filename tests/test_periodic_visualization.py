"""Actual adapters on synthetic fields; no task entry import or real data."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset
from cdlno.kcdno.config import KCDNOArchitectureConfig
from cdlno.kcdno.standard import StandardModel
from cdlno.kcdno.matched_config import MatchedLRSAConfig
from cdlno.periodic_visualization import PeriodicFields, due
from cdlno.training_state import capture_rng, restore_rng
from visualization_projection import strip_visualization
from msar_entry_projection import strip_msar

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'PDE-Solving-StandardBenchmark'))
from utils.normalizer import UnitTransformer


def config(point=False, matched=False):
    cls=MatchedLRSAConfig if matched else KCDNOArchitectureConfig
    return cls(L=2,d=8,h=2,M=3,point_module='point_ffn' if point else 'conv_ffn')


def standard(task):
    shape={'elasticity':None,'ns':(64,64),'plasticity':(101,31)}.get(task,(5,7))
    model=StandardModel(config=config(shape is None),task_name=task,
                        **({} if shape is None else dict(H=shape[0],W=shape[1])))
    n=19 if shape is None else shape[0]*shape[1]
    coords=torch.rand(2,n,2)
    if task=='ns':
        dataset=TensorDataset(coords,torch.randn(2,n,10),torch.randn(2,n,10))
    elif task=='plasticity':
        dataset=TensorDataset(coords,torch.linspace(0,1,20).repeat(2,1),torch.randn(2,n,1),torch.randn(2,n,4,20))
    else:
        dataset=TensorDataset(coords,torch.randn(2,n),torch.randn(2,n)+3)
    return model,dataset,shape


class PeriodicChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(13)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)

    def assertTree(self,a,b):
        if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
        elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assertTree(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assertTree(x,y)
        else:self.assertEqual(a,b)

    def test_frequency_static_real_wrappers_and_original_decode(self):
        self.assertEqual([e for e in range(1,104) if due(e,103)],[50,100,103])
        self.assertTrue(due(398,398));self.assertFalse(due(0,500))
        for task in ('darcy','elasticity','airfoil','pipe'):
            with self.subTest(task=task):
                m,data,shape=standard(task);viz=PeriodicFields(self.path/task,task,'KCDNO',seed=1)
                n=data.tensors[0].shape[1]
                yn=UnitTransformer(torch.randn(4,n)+7) if task!='airfoil' else None
                xn=UnitTransformer(torch.randn(4,n,2)*3+5) if task=='pipe' else None
                options=dict(grid_shape=shape,output_normalizer=yn,coordinate_normalizer=xn)
                with patch('cdlno.periodic_visualization.render_fields') as render:
                    self.assertIsNone(viz.after_epoch(m,49,500,dataset=data,**options));render.assert_not_called()
                    event=viz.after_epoch(m,50,500,dataset=data,**options)
                    self.assertEqual(event['status'],'completed')
                    field=next(c.kwargs for c in render.call_args_list if c.args[0].name=='field')
                    with torch.no_grad():
                        expected=m(data.tensors[0][:1],data.tensors[1][:1,:,None] if task=='darcy' else None).squeeze(-1)
                        if yn is not None:expected=yn.decode(expected)
                    np.testing.assert_allclose(field['prediction'][:,0],expected[0].numpy(),rtol=1e-6)
                    np.testing.assert_array_equal(field['truth'][:,0],data.tensors[2][0].numpy())
                    coords=data.tensors[0][:1] if xn is None else xn.decode(data.tensors[0][:1])
                    np.testing.assert_allclose(field['coordinates'],coords[0].numpy())
                    self.assertEqual(field['grid_shape'],shape)
                    scales={p.name:p.read_bytes() for p in (viz.directory/'scales').glob('*')}
                    viz.after_epoch(m,100,500,dataset=data,**options)
                    self.assertEqual(scales,{p.name:p.read_bytes() for p in (viz.directory/'scales').glob('*')})

    def test_NS_prediction_feedback_and_plasticity_twenty_separate_conditions(self):
        for task in ('ns','plasticity'):
            m,data,shape=standard(task);viz=PeriodicFields(self.path/task,task,'KCDNO')
            inputs=[];outputs=[]
            def before(module,args,kwargs):inputs.append(([v.detach().clone() for v in args],{k:v.clone() for k,v in kwargs.items()}))
            h=m.register_forward_pre_hook(before,with_kwargs=True)
            h2=m.register_forward_hook(lambda mod,args,out:outputs.append(out.detach().clone()))
            try:
                with patch('cdlno.periodic_visualization.render_fields'),patch('cdlno.periodic_visualization.render_curves',side_effect=lambda directory,*a,**k:directory.mkdir()):
                    result=viz.after_epoch(m,50,500,dataset=data,grid_shape=shape)
                self.assertEqual(result['status'],'completed')
                length=10 if task=='ns' else 20
                self.assertEqual(len(inputs),2*length)
                for case in range(2):
                    for t in range(length):
                        args,kwargs=inputs[case*length+t]
                        if task=='ns':
                            expected=data.tensors[1][case:case+1] if t==0 else torch.cat((inputs[case*length+t-1][0][1][...,1:],outputs[case*length+t-1]),-1)
                            self.assertTree(args[1],expected)
                        else:
                            self.assertTree(args[1],data.tensors[2][case:case+1])
                            self.assertTree(kwargs['T'],data.tensors[1][case:case+1,t:t+1])
                    saved=np.load(viz.directory/f'epoch_0050/case_{case:03d}/trajectory.npz')
                    self.assertEqual(saved['prediction'].shape,(shape[0]*shape[1],1 if task=='ns' else 4,length))
            finally:h.remove();h2.remove()

    def test_observer_exact_training_rng_optimizer_and_failure_isolation(self):
        m,data,shape=standard('darcy');m.train()
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1)
        def step(model,optimizer,sched):
            optimizer.zero_grad();out=model(data.tensors[0],data.tensors[1][...,None]);loss=(out[...,0]-data.tensors[2]).square().mean()
            loss.backward();optimizer.step();sched.step()
        step(m,opt,scheduler)
        control=copy.deepcopy(m);oc=torch.optim.AdamW(control.parameters(),lr=.001);oc.load_state_dict(copy.deepcopy(opt.state_dict()))
        sc=torch.optim.lr_scheduler.StepLR(oc,step_size=1);sc.load_state_dict(copy.deepcopy(scheduler.state_dict()))
        before=copy.deepcopy(m.state_dict());grads=[p.grad.clone() for p in m.parameters()]
        rng=capture_rng();flags=[p.training for p in m.modules()]
        def consume(*a,**kw):
            random.random();np.random.rand();torch.rand(7)
        viz=PeriodicFields(self.path,'darcy','KCDNO')
        with patch('cdlno.periodic_visualization.render_fields',side_effect=consume):
            event=viz.after_epoch(m,50,500,dataset=data,grid_shape=shape)
        self.assertEqual(event['status'],'completed');self.assertTree(capture_rng(),rng)
        self.assertTree(m.state_dict(),before);self.assertEqual(flags,[p.training for p in m.modules()])
        for p,g in zip(m.parameters(),grads):self.assertTree(p.grad,g)
        with patch('cdlno.periodic_visualization.render_fields',side_effect=RuntimeError('render failure')):
            with self.assertLogs('cdlno.periodic_visualization',level='WARNING'):
                event=viz.after_epoch(m,100,500,dataset=data,grid_shape=shape)
        self.assertEqual(event['status'],'failed');self.assertTree(capture_rng(),rng)
        step(m,opt,scheduler);restore_rng(rng);step(control,oc,sc)
        self.assertTree(m.state_dict(),control.state_dict());self.assertTree(opt.state_dict(),oc.state_dict());self.assertTree(scheduler.state_dict(),sc.state_dict())

    def test_real_PyG_car_and_air_decoding_masks_inputs(self):
        try:from torch_geometric.data import Data
        except ImportError:self.skipTest('real PyG absent')
        sys.path.insert(0,str(ROOT/'Car-Design-ShapeNetCar'))
        from models.KCDNO import Model
        from cdlno.kcdno.airfrans import AirfRANSModel
        for task in ('car','airfrans'):
            m=Model(config=config(True)) if task=='car' else AirfRANSModel(config=config(True))
            g=Data(x=torch.randn(23,7),pos=torch.rand(23,3 if task=='car' else 2),y=torch.randn(23,4),surf=torch.arange(23)%2==0)
            before=g.clone();geom=torch.rand(8,3);viz=PeriodicFields(self.path/task,task,'KCDNO')
            norm=(np.zeros(7),np.ones(7),np.arange(4),np.arange(4)+1)
            if task=='car':
                dataset=[(g,geom)]
            else:
                # No fake radius_graph: explicitly exercise only already-sampled
                # real PyG inference/visualization when torch_cluster is absent.
                viz.cases=[(g.clone(),torch.arange(23),23)];dataset=[g]
            with patch('cdlno.periodic_visualization.render_fields') as render:
                event=viz.after_epoch(m,50,500,dataset=dataset,coef_norm=norm)
            self.assertEqual(event['status'],'completed')
            for k in before.to_dict():self.assertTree(g[k],before[k])
            saved=np.load(viz.directory/'epoch_0050/case_000/case.npz')
            np.testing.assert_allclose(saved['truth'],g.y.numpy()*(norm[3]+1e-8)+norm[2])
            if task=='car':
                surface=next(c.kwargs for c in render.call_args_list if c.args[0].name=='surface_pressure')
                np.testing.assert_array_equal(surface['display_mask'],g.surf.numpy())
                self.assertEqual(surface['coordinates'].shape,(23,3))
                volume=next(c.kwargs for c in render.call_args_list if c.args[0].name=='volume_velocity')
                self.assertFalse(np.any(volume['display_mask'] & g.surf.numpy()))
                self.assertEqual(volume['coordinate_labels'],('$x$','$z$'))
            else:
                self.assertEqual(render.call_args_list[0].kwargs['prediction'].shape,(23,4))
                self.assertIn('not the official',render.call_args_list[0].kwargs['metadata']['caption_note'])

    def test_air_real_sampling_graph_dependency(self):
        try:
            import torch_cluster
            from torch_geometric.data import Data
        except ImportError:self.skipTest('real torch_cluster missing: AirfRANS sampling/graph integration not executed')
        from cdlno.kcdno.airfrans import AirfRANSModel
        model=AirfRANSModel(config=config(True));g=Data(x=torch.randn(29,7),pos=torch.rand(29,2),y=torch.randn(29,4),surf=torch.arange(29)%2==0)
        viz=PeriodicFields(self.path,'airfrans','KCDNO')
        with patch('cdlno.periodic_visualization.render_fields'):
            event=viz.after_epoch(model,50,500,dataset=[g],hparams=dict(subsampling=19,r=.4,max_neighbors=12))
        self.assertEqual(event['status'],'completed');self.assertEqual(viz.cases[0][0].x.shape,(19,7))

    def test_production_hooks_preserve_complete_AST(self):
        snapshot=Path(os.environ.get('CDLNO_VISUALIZATION_BASELINE',
                      '/home/hwz/CDLNO-artifacts/visualization-before-133lyhbo/source'))
        if not snapshot.is_dir():
            self.skipTest('pre-visualization source snapshot absent; set CDLNO_VISUALIZATION_BASELINE')
        files=[f'PDE-Solving-StandardBenchmark/exp_{s}.py' for s in ('darcy','elas','airfoil','pipe','ns','plas')]
        files+=['Car-Design-ShapeNetCar/train.py','Airfoil-Design-AirfRANS/train.py','Airfoil-Design-AirfRANS/main.py']
        for file in files:
            with self.subTest(file=file):
                source=ast.parse((ROOT/file).read_text());old=ast.parse((snapshot/file).read_text())
                self.assertEqual(ast.dump(strip_visualization(strip_msar(source))),ast.dump(old))

    def test_real_export_caption_fonts_and_color_saturation(self):
        import matplotlib as mpl
        m,data,shape=standard('airfoil');viz=PeriodicFields(self.path,'airfoil','KCDNO',seed=2)
        style=dict(mpl.rcParams)
        event=viz.after_epoch(m,50,500,dataset=TensorDataset(*(t[:1] for t in data.tensors)),grid_shape=shape)
        self.assertEqual(event['status'],'completed');self.assertEqual(style,dict(mpl.rcParams))
        path=viz.directory/'epoch_0050/case_000/field'
        info=json.loads((path/'metadata.json').read_text())
        self.assertEqual(info['png_dpi'],600);self.assertEqual(info['typography']['pdf.fonttype'],42)
        self.assertIn('Training seed: 2.',(path/'caption.txt').read_text())
        self.assertTrue((path/'caption.tex').read_text().startswith('\\caption{'))
        self.assertTrue((path/'fields.pdf').read_bytes().startswith(b'%PDF'))
        values=np.load(path/'fields.npz');np.testing.assert_allclose(values['signed_error'],values['prediction']-values['truth'])

    def test_CDLNO_modes_and_matched_share_observer_without_math_changes(self):
        from test_front_task_modes import parse,build
        for mode in ('full','no_sa','identity','matched'):
            if mode=='matched':m=StandardModel(config=config(matched=True),task_name='darcy',H=5,W=7)
            else:m=build('darcy',parse('darcy',['--n-hidden','8','--n-heads','2','--slice_num','3','--front-latent-mode',mode]))
            n=m.H*m.W;data=TensorDataset(torch.rand(1,n,2),torch.rand(1,n),torch.rand(1,n))
            before=copy.deepcopy(m.state_dict());viz=PeriodicFields(self.path/mode,'darcy',mode)
            with patch('cdlno.periodic_visualization.render_fields'):
                event=viz.after_epoch(m,50,500,dataset=data,grid_shape=(m.H,m.W))
            self.assertEqual(event['status'],'completed');self.assertTree(before,m.state_dict())


if __name__=='__main__':unittest.main(verbosity=2)
