"""M9 accounting/measurement contract and hand examples, no real datasets."""
import copy
import math
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

import torch

from cdlno.msar_lno.modules import LearnedQueryDown,QueryAlignedUpCross,PairwiseAttnResFusion,CoverageFloorLoss
from cdlno.msar_lno.config import MSARTrainingConfig
from cdlno.msar_lno.standard import StaticModel
from cdlno.msar_lno.profiles import resolve_profile
from tools.cdlno_perf.models import Case,synthetic_inputs
from tools.cdlno_perf.measure import benchmark,contexts,precision_settings
from tools.cdlno_perf.msar import (build_msar,parameters,matrix_formula,scalar_and_coverage_ledger,
    live_audit,coverage_path_audit,synthetic_loss)
from tools.msar_benchmark import parser,run


class MSARPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_hand_down_up_pair_coverage_without_shared_reference(self):
        # Two orthogonal raw values, Q/K identity, RMS epsilon included by hand.
        down=LearnedQueryDown(2,1,2).double();up=QueryAlignedUpCross(2,1).double()
        with torch.no_grad():
            down.latent_queries.copy_(torch.eye(2))
            for module in (down,up):
                for name in ('to_q','to_k','to_v','to_out'):
                    if hasattr(module,name):
                        getattr(module,name).weight.copy_(torch.eye(2))
                        if getattr(module,name).bias is not None:getattr(module,name).bias.zero_()
                module.q_norm.weight.fill_(1);module.k_norm.weight.fill_(1)
        x=torch.eye(2,dtype=torch.float64)[None]
        p=1/(1+math.exp(-1/((.5+1e-6)*math.sqrt(2))))
        expected=torch.tensor([[[p,1-p],[1-p,p]]],dtype=torch.float64)
        torch.testing.assert_close(down(x),expected,atol=2e-15,rtol=2e-15)
        torch.testing.assert_close(up(x,x),expected,atol=2e-15,rtol=2e-15)
        torch.testing.assert_close(down(x,valid_mask=torch.tensor([[True,False]])),
                                   torch.tensor([[[1.,0.],[1.,0.]]],dtype=torch.float64),atol=0,rtol=0)
        fusion=PairwiseAttnResFusion(2).double()
        with torch.no_grad():fusion.w[0]=math.log(3)*math.sqrt(.5+1e-6)
        torch.testing.assert_close(fusion(x[:,:1],x[:,1:]),torch.tensor([[[1.5,.5]]],dtype=torch.float64),atol=1e-15,rtol=1e-15)
        a=torch.tensor([[[[.9,.1,0],[.7,.3,0]],[[.5,.5,0],[.3,.7,0]]]])
        # p=(.6,.4,0), kappa*mu=(.2,.2,.2): one source deficit .2.
        loss=CoverageFloorLoss(MSARTrainingConfig(coverage_kappa=.6))(a)
        self.assertAlmostEqual(loss.item(),.04/(1/3+1e-6),places=7)
        self.assertEqual(CoverageFloorLoss(MSARTrainingConfig(coverage_mode='off'))(None).item(),0)

    def test_parameters_live_matrix_and_payload_complete(self):
        case=Case.preset('elasticity','task',B=2,N=35);args,target=synthetic_inputs(case)
        for profile in ('light','full'):
            model=build_msar(case,'msar_'+profile+'_off')
            cost=live_audit(model,args,target,contexts(torch.device('cpu'),'fp32','math'),2,35)
            p=parameters(model)
            self.assertEqual(p['total'],1710169 if profile=='light' else 6811825)
            self.assertEqual(p['by_component']['pair3'],3*model.config.d)
            self.assertEqual(p['coverage_parameters'],0)
            m=model.config.num_latents;h=model.config.heads
            payload=4*2*(h[0]*m[0]*35+h[1]*m[1]*m[0]+h[2]*m[2]*m[1]+h[3]*m[3]*m[2])
            self.assertEqual(cost['separate_nonmatrix']['coverage_training_only']['summed_fp32_A_payload_bytes'],payload)
            self.assertEqual(len(cost['live_sdpa']),20)  # four Down, twelve SA, four Up
            self.assertFalse(any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.modules()))

    def test_n_slope_counts_both_N_projections_down_up_and_stem_head(self):
        case=Case.preset('elasticity','task',B=2,N=35)
        model=build_msar(case,'msar_light_off');d=96;m1=512
        # Per added point: stem 2->192->96; Down K/V; final Up Q/O;
        # both QK+AV cross reads; scalar output. No N-point FFN/SA.
        hand_per_point=2*(2*192+192*96+4*d*d+4*m1*d+d)
        first=matrix_formula(model,2,35);second=matrix_formula(model,2,36)
        self.assertEqual(second['matrix_macs']-first['matrix_macs'],hand_per_point)

    def test_coverage_callback_actual_objective_and_off_default_unchanged(self):
        config=resolve_profile('light',d=8,num_latents=[7,5,3,2],heads=[2,2,4,4])
        model=StaticModel(config=config,task_name='elasticity',training_config=MSARTrainingConfig(coverage_kappa=1))
        args=(torch.randn(1,11,2),None);target=torch.randn(1,11,1)
        ctx=contexts(torch.device('cpu'),'fp32','math')
        enabled=coverage_path_audit(model,args,target,ctx)
        self.assertGreater(enabled['objective']['coverage_raw'],0)
        self.assertEqual(len(enabled['attention']),4)
        self.assertAlmostEqual(enabled['objective']['total'],enabled['objective']['pde']+.01*enabled['objective']['coverage_raw'],places=6)
        calls=[]
        def closure(m,a,t):calls.append(True);return synthetic_loss(m,a,t)
        result=benchmark(model,args,target,ctx,torch.device('cpu'),warmup=1,iterations=2,loss_closure=closure)
        self.assertEqual(len(calls),4)  # warmup + 2 samples + untimed profiler
        self.assertEqual(result['optimizer']['successful_steps_per_active_parameter'],[3])
        self.assertEqual(len(result['train_step']['samples_ms']),2)
        model.training_config=MSARTrainingConfig(coverage_mode='off')
        twin=copy.deepcopy(model)
        off=coverage_path_audit(model,args,target,ctx)
        self.assertEqual(off['attention'],[])
        self.assertEqual(off['objective']['pde'],off['objective']['total'])
        benchmark(model,args,target,ctx,torch.device('cpu'),warmup=1,iterations=2)
        benchmark(twin,args,target,ctx,torch.device('cpu'),warmup=1,iterations=2,loss_closure=synthetic_loss)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,twin.state_dict()[key],atol=0,rtol=0)
        with self.assertRaises(ValueError):benchmark(model,args,target,ctx,torch.device('cpu'),compile_model=True,loss_closure=synthetic_loss)

    def test_cli_refuses_overwrite_and_keeps_profiles_separate(self):
        p=parser();args=p.parse_args(['--task','pipe','--audit-only','--output','unused.json'])
        self.assertEqual(args.models[:4],['msar_light_off','msar_light_floor','msar_full_off','msar_full_floor'])
        case=Case.preset('pipe','task',B=1,grid=(5,7))
        self.assertEqual(case.M,32)
        model=build_msar(case,'msar_full_floor')
        self.assertEqual(model.config.num_latents,(1024,512,256,128))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'evidence.json';path.write_text('original')
            args=p.parse_args(['--output',str(path)])
            with self.assertRaises(FileExistsError):run(args)
            self.assertEqual(path.read_text(),'original')

    def test_eight_task_four_variants_real_train_parsers_and_eval_previews(self):
        # Reuse safe parser extractors; never import a top-level exp/main.
        import test_msar_static as static
        import test_msar_temporal as temporal
        import test_msar_industrial as industrial
        root=Path(__file__).resolve().parents[1]
        for task in (*static.TASKS,*temporal.TEMPORAL,*industrial.PROJECTS):
            arguments=(static.arguments if task in static.TASKS else
                       temporal.arguments if task in temporal.TEMPORAL else industrial.arguments)
            script=root/'tran_evaluate/msar_lno'/f'{task}.sh'
            for profile in ('light','full'):
                for mode in ('off','floor'):
                    flags=['--profile',profile,'--coverage-mode',mode,'--coverage-weight','0' if mode=='off' else '.01']
                    out=subprocess.check_output(['bash',str(script),'train',*flags,'--dry-run'],text=True)
                    command=shlex.split(next(line[9:] for line in out.splitlines() if line.startswith('Command: ')))
                    actual=arguments(task,flags=command[3:])
                    self.assertEqual(actual.msar_architecture['d'],96 if profile=='light' else 192)
                    self.assertEqual(actual.msar_training['coverage_mode'],mode)
                    self.assertEqual(actual.msar_training['coverage_weight'],0 if mode=='off' else .01)
                    self.assertFalse(actual.msar_run_dir.exists())
                    evaluation=subprocess.check_output(['bash',str(script),'eval','--msar-run-dir',str(actual.msar_run_dir),*flags,'--dry-run'],text=True)
                    self.assertIn('DRY RUN',evaluation)
                    # Actual eval read-first/negative checkpoint parsing is covered
                    # by unchanged M5-M8 tests with real saved sidecars, not this preview.


if __name__=='__main__':unittest.main(verbosity=2)
