"""User-confirmed Car MSE default; independent of paper evaluation axes."""
import json
import unittest
import torch
from torch_geometric.data import Data
from cdlno.linearno.profiles import (PROFILES, CAR_OBJECTIVE_CONTRACT, resolve_config,
                                    require_resolved_objective, validate_resolved)
from cdlno.linearno.car_entry import CarRun


class CarObjectiveDecision(unittest.TestCase):
    def test_three_profiles_default_to_release_MSE_without_changing_model_or_budget(self):
        for profile in PROFILES:
            config=resolve_config('car',profile)
            old=resolve_config('car',profile,contract='car_l7')
            self.assertEqual(config['integration_contract'],CAR_OBJECTIVE_CONTRACT)
            require_resolved_objective(config)
            self.assertEqual(validate_resolved(json.loads(json.dumps(config))),config)
            for key in ('model','training','data','runtime','evaluation'):
                self.assertEqual(config['values'][key],old['values'][key])
            spec=config['values']['objective']
            self.assertEqual((spec['kind'],spec['space'],spec['volume_region'],spec['surface_weight']),
                             ('MSE','normalized','all points',.5))
            self.assertEqual(spec['volume_channels'],[0,1,2]);self.assertEqual(spec['surface_channels'],[3])
            # Hand example includes large surface velocities to distinguish all
            # points from volume-only velocity, and ignores nonsurface pressure.
            data=Data(y=torch.zeros(3,4),surf=torch.tensor([False,True,True]))
            out=torch.tensor([[1.,2.,3.,100.],[4.,5.,6.,2.],[7.,8.,9.,4.]],requires_grad=True)
            run=object.__new__(CarRun);run.config=config
            loss,p,v=run.objective(out,data)
            expected=sum(i*i for i in range(1,10))/9 + .5*(4+16)/2
            self.assertAlmostEqual(loss.item(),expected,places=5)
            loss.backward()
            self.assertEqual(out.grad[0,3].item(),0.)
            self.assertTrue(torch.all(out.grad[:,:3]>0))

    def test_historical_paper_underdetermination_and_explicit_override_remain_visible(self):
        for contract in (None,'car_l7'):
            historical=resolve_config('car',contract=contract)
            self.assertEqual(historical['values']['objective']['kind'],'relative_L2')
            with self.assertRaisesRegex(ValueError,'reviewed decisions'):require_resolved_objective(historical)
            self.assertEqual(validate_resolved(historical),historical)
        config=resolve_config('car',explicit={'objective.surface_weight':.25})
        self.assertEqual(config['field_sources']['objective.surface_weight'],'cli_explicit')
        self.assertEqual(config['values']['objective']['surface_weight'],.25)
        self.assertEqual(validate_resolved(config),config)
        with self.assertRaisesRegex(ValueError,'inapplicable'):
            resolve_config('airfrans',contract=CAR_OBJECTIVE_CONTRACT)
