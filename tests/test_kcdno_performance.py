"""Complete matrix cost vs independent closed forms; existing timing reused."""
import unittest
import torch
from tools.cdlno_perf.models import Case,build,synthetic_inputs,KERNEL_MODELS
from tools.cdlno_perf.costs import audit
from tools.cdlno_perf.measure import contexts
from tools.cdlno_perf.kernel_costs import formulas,check_cost

class KernelCosts(unittest.TestCase):
    def test_live_point_conv_full_cost_and_counts(self):
        torch.set_num_threads(1)
        for task in ('elasticity','darcy'):
            case=Case.preset(task,B=2,d=16,h=4,M=4,kernel_rank=5,**({'N':35} if task=='elasticity' else {'grid':(5,7)}))
            for name in KERNEL_MODELS:
                model,_,_=build(case,name);args,target=synthetic_inputs(case)
                result=audit(model,args,target,contexts(torch.device('cpu')))
                check_cost(model,case,result)
                self.assertFalse(result['parameters']['missing_grad_names'])
                self.assertFalse(result['parameters']['nonfinite_grad_names'])
                if name=='kcdno_all':
                    self.assertEqual([result['counts'][k] for k in ('kernel_writers','kernel_queries','kernel_reads','kernel_reader_calls')],[7,7,28,7])
                    self.assertTrue(any('.reader' in key for key in result['temporary_materializations']))
                self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))

    def test_default_principal_checks_actual_core_parameters(self):
        f=formulas(1,7225,128,64,8,16,True)
        self.assertEqual(f['principal_parameter_reduction'],495616)
        reduction=1-f['kernel_core_main_mac']/f['matched_core_main_mac']
        self.assertAlmostEqual(reduction*100,.196,places=3)
        for conv in (False,True):
            from cdlno.kcdno.config import KCDNOArchitectureConfig
            from cdlno.kcdno.core import KCDNO
            from cdlno.kcdno.matched_config import MatchedLRSAConfig
            from cdlno.kcdno.matched import MatchedLRSA
            k=KCDNO(KCDNOArchitectureConfig(point_module='conv_ffn' if conv else 'point_ffn'))
            m=MatchedLRSA(MatchedLRSAConfig(point_module='conv_ffn' if conv else 'point_ffn'))
            # Exact norm/bias/scorer/gamma correction to leading d²/dr terms.
            correction=2*8*128+2*8*(128//8)-4*(8-1)*128-(8-1)
            self.assertEqual(sum(p.numel() for p in m.parameters())-sum(p.numel() for p in k.parameters()),495616+correction)

if __name__=='__main__':unittest.main(verbosity=2)
