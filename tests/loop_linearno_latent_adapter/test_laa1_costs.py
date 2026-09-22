import json
from pathlib import Path
import re
import unittest
from .support import config
from .oracle import count
from linearno_loop.v3.costs import analytic_cost,source_counts
from linearno_loop.v3.profiles import width_for
from linearno_loop.v3.contracts import TASKS,RESIDUAL_MODES,ADAPTER_MODES

ROOT=Path(__file__).resolve().parents[2]
PROMPT=ROOT/'PLAN_Looped_LinearNO/Looped_LinearNO_LatentFFN_BilateralAdapter_CostProfiles_Codex_Staged_Prompts.md'

class CostTests(unittest.TestCase):
    def test_all_64_widths_and_exact_preimplementation_costs(self):
        text=PROMPT.read_text();tables=[]
        for line in text.splitlines():
            cells=[x.strip() for x in line.strip('|').split('|')]
            if len(cells)==9 and re.fullmatch(r'\d+ → \d+',cells[0]) and all(re.fullmatch(r'\d+/\d+',x) for x in cells[1:]):
                tables.append((int(cells[0].split(' → ')[1]),[tuple(map(int,x.split('/'))) for x in cells[1:]]))
        self.assertEqual(len(tables),8)
        old=json.loads((ROOT/'docs/loop_linearno_latent_adapter_audit/laa0/cost-recalculation.json').read_text())
        for idx,(depth,widths) in enumerate(tables):
            profile='matched_v1' if idx<4 else 'efficient_v1'
            for task,(H,Dz) in zip(TASKS,widths):
                with self.subTest(task=task,profile=profile,depth=depth):
                    self.assertEqual(width_for(profile,'d'+str(depth),task),dict(hidden_width=H,latent_width=Dz))
                    row=next(r for r in old['rows'] if r['task']==task and r['cost_profile']==profile and r['executed_depth']==depth)
                    cost=analytic_cost(config(task,cost_profile=profile,executed_depth=depth))
                    self.assertEqual(cost['total_parameters'],row['parameters'])
                    self.assertEqual(cost['matrix_macs'],row['matrix_macs'])
                    self.assertEqual(cost['matrix_flops'],2*row['matrix_macs'])
    def test_independent_tensor_inventory_all_modes_features_tasks(self):
        for task in TASKS:
            for mode in RESIDUAL_MODES:
                for latent in (False,True):
                    for adapter in (False,True):
                        c=config(task,residual_mode=mode,latent_enabled=latent,adapter_mode=ADAPTER_MODES[1] if adapter else 'none')
                        s=c['loop_spec'];m=c['profile_spec']['values']['model'];B,N=2,13
                        channel=(m['ref']**2 if m['unified_pos'] else m['space_dim'])+m['fun_dim']
                        if task=='airfrans' and m['unified_pos']:channel+=m['space_dim']
                        o=count(task,s['hidden_width'],s['latent_width'],s['actual_M'],s['heads'],2,4,2,2,m['ffn_ratio'],B,N,
                                channel,m['out_dim'],s['variant'],m['time_input'],latent,adapter,residual=mode)
                        actual=analytic_cost(c,batch=B,points=N)
                        self.assertEqual((actual['total_parameters'],actual['matrix_macs'],actual['router_contraction_macs']),
                                         (o['parameters'],o['matrix_macs'],o['router_contraction_macs']))
                        self.assertEqual(actual['router_source_counts'],o['source_counts'])
                        self.assertFalse(actual['profile_match_claim_applies'])
                        self.assertEqual(analytic_cost(c)['profile_match_claim_applies'],mode=='sr_1_over_r' and latent and adapter)
    def test_physical_parameters_versus_executed_calls_and_custom_R(self):
        outputs=[]
        for R in (1,3):
            c=config('car',cost_profile='custom',hidden_width=200,latent_width=19,actual_M=7,topology_preset='custom',
                     prefix_blocks=0,recurrent_core_blocks=2,loop_repeats=R,suffix_blocks=1,adapter_mode='none',residual_mode='lb_attnres_1_over_r')
            a=analytic_cost(c,batch=1,points=17);outputs.append(a)
            self.assertEqual(a['unique_depth'],3);self.assertEqual(a['executed_depth'],2*R+1)
            self.assertEqual(a['parameter_groups']['router'],2*200*R)
            self.assertEqual(a['matrix_mac_groups']['adapter'],0)
            self.assertEqual(a['head_visits'],1)
            self.assertFalse(a['profile_match_claim_applies'])
        self.assertEqual(outputs[0]['parameter_groups']['shared_core'],outputs[1]['parameter_groups']['shared_core'])
        self.assertEqual(outputs[1]['matrix_mac_groups']['shared_core'],3*outputs[0]['matrix_mac_groups']['shared_core'])
    def test_source_counts_and_scalar_scope(self):
        self.assertEqual(source_counts(3,2,'rb_attnres'),[1,2,2,2,2,2,2,3,3,3,3,3,3])
        c=analytic_cost(config())
        self.assertFalse(c['non_matrix']['included_in_matrix_macs']);self.assertIsNone(c['latency'])
        for v in (True,0,-1,'32',1.5):
            with self.assertRaises(ValueError):analytic_cost(config(),points=v)
    def test_table_claim_excludes_changed_base_protocol_or_representative_shape(self):
        from linearno_loop.v3.config import resolve_config
        from linearno_loop.v3.contracts import ARCHITECTURE_SELECTOR
        altered=resolve_config('pipe','transolver_matched',options={'architecture':ARCHITECTURE_SELECTOR})
        self.assertFalse(analytic_cost(altered)['profile_match_claim_applies'])
        self.assertFalse(analytic_cost(config(),batch=2,points=13)['profile_match_claim_applies'])
