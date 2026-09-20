import copy
import json
from pathlib import Path
import unittest

from cdlno.linearno.profiles import CATALOG, TASKS, PROFILES
from linearno_loop.config import resolve_config, validate_config, run_directory_id, route_intent
from linearno_loop.contracts import LoopSchemaError, PRESETS, RESIDUAL_MODES, TOPOLOGY_FIELDS, seal
from linearno_loop.matrix import configuration_matrix


def options(**changes):
    return {'topology_preset':'p1_c3_r2_s1','residual_mode':'sr_1_over_r',**changes}


class ConfigTests(unittest.TestCase):
    def test_all_current_task_profiles_and_immutable_baseline(self):
        before=copy.deepcopy(CATALOG)
        ll0=json.loads((Path(__file__).resolve().parents[2]/'docs/loop_linearno_audit/ll0/resolved-profiles.json').read_text())
        for row in ll0:
            c=resolve_config(row['task'],row['profile'],options=options())
            self.assertEqual(c['profile_spec'],row['current'])
            s=c['loop_spec'];self.assertEqual(s['base_rank'],row['proposed_loop']['base_rank'])
            self.assertEqual(s['resolved_rank'],2*s['base_rank'])
            self.assertEqual(validate_config(json.loads(json.dumps(c))),c)
        self.assertEqual(CATALOG,before)

    def test_preset_derived_counts_and_parameter_formula(self):
        for preset, (P,C,R,S) in PRESETS.items():
            for mode in RESIDUAL_MODES:
                c=resolve_config('darcy',options={'topology_preset':preset,'residual_mode':mode});s=c['loop_spec']
                self.assertEqual(s['unique_depth'],5 if C==3 else 6);self.assertEqual(s['executed_depth'],8)
                count={'sr_1_over_r':0,'rb_attnres':2*C*R+1,'lb_attnres_1_over_r':R}[mode]
                visits={'sr_1_over_r':0,'rb_attnres':31 if C==3 else 21,'lb_attnres_1_over_r':5}[mode]
                self.assertEqual(s['attnres']['receiver_count'],count)
                self.assertEqual(s['attnres']['router_parameter_count'],2*128*count)
                self.assertEqual(s['attnres']['source_visits'],visits)
                self.assertTrue(s['point_domain_attnres']);self.assertFalse(s['feature_timestep_encoding'])
                self.assertNotIn('n_layers',c['model_spec']['constructor_kwargs'])

    def test_custom_edges_and_strict_integer_validation(self):
        custom=dict(topology_preset='custom',residual_mode='sr_1_over_r',prefix_blocks=0,
                    recurrent_core_blocks=1,loop_repeats=1,suffix_blocks=1)
        self.assertEqual(resolve_config('elasticity',options=custom)['loop_spec']['executed_depth'],2)
        for field in TOPOLOGY_FIELDS:
            for invalid in (True,False,1.,'2',None,-1,float('nan'),float('inf')):
                with self.subTest(field=field,value=invalid),self.assertRaises(LoopSchemaError):
                    resolve_config('darcy',options={**custom,field:invalid})
            if field!='prefix_blocks':
                with self.assertRaises(LoopSchemaError):resolve_config('darcy',options={**custom,field:0})
            with self.assertRaises(LoopSchemaError):resolve_config('darcy',options={k:v for k,v in custom.items() if k!=field})
            with self.assertRaises(LoopSchemaError):resolve_config('darcy',options={**options(),field:1})

    def test_missing_and_unknown_choices_fail_closed(self):
        for bad in ({},None,[],dict(topology_preset='custom'),dict(residual_mode='rb_attnres'),
                    {**options(),'surprise':1},{**options(),'unique_depth':5},
                    {**options(),'residual_mode':'sr_unscaled'},{**options(),'topology_preset':'other'},
                    {**options(),'topology_preset':True}):
            with self.subTest(bad=bad),self.assertRaises(LoopSchemaError):resolve_config('darcy',options=bad)
        for key in ('model.layers','model.linearno_rank','model.qk_dim'):
            with self.assertRaises(LoopSchemaError):resolve_config('darcy',options=options(),profile_overrides={key:8})

    def test_rank_actual_multiplier_and_car_divisibility(self):
        for task in TASKS:
            c=resolve_config(task,options=options(rank_multiplier=1));s=c['loop_spec']
            self.assertEqual(s['resolved_rank'],s['base_rank'])
            for value in (True,0,3,'2',2.,None):
                with self.subTest(task=task,value=value),self.assertRaises(LoopSchemaError):
                    resolve_config(task,options=options(rank_multiplier=value))
        c=resolve_config('darcy',options=options(linearno_rank=47));s=c['loop_spec']
        self.assertEqual((s['rank_policy'],s['resolved_rank'],s['rank_multiplier']),('explicit_actual',47,None))
        self.assertEqual(c['profile_spec']['values']['model']['linearno_rank'],64)
        for actual in (0,True,'64',64.,None):
            with self.assertRaises(LoopSchemaError):resolve_config('car',options=options(linearno_rank=actual))
        with self.assertRaises(LoopSchemaError):resolve_config('car',options=options(linearno_rank=48))
        self.assertEqual(resolve_config('car',options=options(linearno_rank=96))['loop_spec']['operator_contract']['rank_mapping']['original_value'],3)
        with self.assertRaises(LoopSchemaError):resolve_config('car',options=options(linearno_rank=64,rank_multiplier=2))
        with self.assertRaises(LoopSchemaError):resolve_config('darcy',options=options(),profile_overrides={'model.hidden':127})

    def test_source_labels_mutability_and_resealed_tampering(self):
        c=resolve_config('darcy',options=options(),profile_overrides={'model.hidden':64})
        self.assertEqual(c['field_sources']['loop_spec']['hidden'],'cli_explicit')
        self.assertEqual(c['field_sources']['loop_spec']['rank_multiplier'],'family_default')
        self.assertEqual(c['field_sources']['loop_spec']['prefix_blocks'],'preset')
        for k,v in [('executed_depth',9),('schema_version',True),('head_dim',9),('resolved_rank',64),
                    ('hidden','64'),('point_domain_attnres',1),('feature_timestep_encoding',True)]:
            bad=copy.deepcopy(c);bad['loop_spec'][k]=v
            with self.subTest(k=k),self.assertRaisesRegex(LoopSchemaError,k):validate_config(seal(bad,'config_hash'))
        bad=copy.deepcopy(c);bad['model_spec']['constructor_kwargs']['profile']='official_release'
        with self.assertRaisesRegex(LoopSchemaError,'profile'):validate_config(seal(bad,'config_hash'))
        bad=copy.deepcopy(c);bad['field_sources']['loop_spec']['hidden']='profile'
        with self.assertRaisesRegex(LoopSchemaError,'field_sources'):validate_config(seal(bad,'config_hash'))
        validate_config(c)['loop_spec']['sharing']['core']='changed'
        self.assertEqual(c['loop_spec']['sharing']['core'],'same_physical_module_each_round')

    def test_native_constructor_constraints_and_frozen_protocols(self):
        cases=[('plasticity',{'model.hidden':15,'model.heads':3}),
               ('airfrans',{'model.space_dim':3}),('airfrans',{'model.out_dim':1}),
               ('car',{'model.fun_dim':3}),('car',{'model.unified_pos':True}),
               ('darcy',{'objective.type':'other'}),('darcy',{'evaluation.type':'other'}),
               ('darcy',{'data.split':'test'}),('darcy',{'model.heads':True}),
               ('darcy',{'model.hidden':'128'}),('darcy',{'model.dropout':float('nan')})]
        for task,overrides in cases:
            with self.subTest(task=task,overrides=overrides),self.assertRaises(LoopSchemaError):
                resolve_config(task,options=options(),profile_overrides=overrides)

    def test_routing_is_not_installed_and_false_history_still_conflicts(self):
        for family in ('linearno','linearno_history','transolver','cdlno'):
            self.assertIsNone(route_intent(family,{}))
            self.assertIsNone(route_intent(family,{'linearno_rank':64}))
        for name,value in [('linearno_latent_attnres',False),('linearno_history_k_conditioning',0),
                           ('linearno_attnres_history_dropout_p',None)]:
            with self.assertRaises(LoopSchemaError):route_intent('linearno',{'linearno_loop':True,name:value})
        for family in ('transolver','cdlno','linearno_history'):
            with self.assertRaises(LoopSchemaError):route_intent(family,{'linearno_loop':True})
        for value in (0,1,'1',None):
            with self.assertRaises(LoopSchemaError):route_intent('linearno',{'linearno_loop':value})
        self.assertIsNone(route_intent('linearno',{'linearno_loop':False}))
        self.assertEqual(route_intent('linearno',{'linearno_loop':True}), 'linearno_loop')
        with self.assertRaises(LoopSchemaError):route_intent('linearno',{'topology_preset':'custom'})

    def test_formal_matrix_pairing_and_unique_run_identifiers(self):
        m=configuration_matrix();self.assertEqual(len(m['runs']),288);self.assertEqual(len(m['custom_controls']),24)
        ids=set();pairs={}
        for row in m['runs']:
            c=row['config'];s=c['loop_spec'];seed=c['profile_spec']['values']['runtime']['seed']
            self.assertEqual(validate_config(c),c);ids.add(run_directory_id(c))
            key=(row['group'],s['task'],s['topology_preset'],seed)
            if key in pairs:self.assertEqual(pairs[key],c['fair_comparison'])
            else:pairs[key]=c['fair_comparison']
            self.assertEqual(row['future_acceptance']['forward'],'NOT RUN')
            self.assertEqual(row['preview']['status'],'PLANNED_NOT_RUNNABLE_LL1')
        self.assertEqual(len(ids),288)
        self.assertEqual(m['paired_seeds'],[0,1,2])


if __name__=='__main__':unittest.main()
