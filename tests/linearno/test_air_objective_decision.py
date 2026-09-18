"""Reviewed AirfRANS objective; preserve historical profiles and other tasks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from cdlno.linearno.profiles import (AIRFRANS_OBJECTIVE_CONTRACT, PROFILES, TASKS,
    digest, require_resolved_objective, resolve_config, validate_resolved)


class AirObjectiveDecision(unittest.TestCase):
    def test_default_all_profiles_use_reviewed_MSE_and_keep_other_axes(self):
        expected = dict(kind='MSE', volume_channels=[0, 1, 2, 3],
            surface_channels=[0, 1, 2, 3], volume_region='~surf', surface_region='surf',
            surface_weight=1, space='normalized',
            reduction='per-region point mean per channel then4-channel mean; batch1', epsilon=None)
        for profile in PROFILES:
            with self.subTest(profile=profile):
                current = resolve_config('airfrans', profile)
                historical = resolve_config('airfrans', profile, contract=None)
                require_resolved_objective(current)
                self.assertEqual(current['values']['objective'], expected)
                self.assertEqual(current['integration_contract'], AIRFRANS_OBJECTIVE_CONTRACT)
                self.assertEqual(validate_resolved(json.loads(json.dumps(current))), current)
                for axis in ('model', 'training', 'evaluation', 'data', 'runtime'):
                    self.assertEqual(current['values'][axis], historical['values'][axis])
                for key in expected:
                    self.assertEqual(current['field_sources']['objective.' + key], 'integration_contract')
        # Matching objectives does not silently equalize epochs or metric labels.
        self.assertEqual(resolve_config('airfrans')['values']['training']['epochs'], 400)
        self.assertEqual(resolve_config('airfrans', 'transolver_matched')['values']['training']['epochs'], 398)
        self.assertIn('rL2', resolve_config('airfrans')['values']['evaluation']['field_metric']['primary'])
        self.assertIn('MSE', resolve_config('airfrans', 'official_release')['values']['evaluation']['field_metric']['primary'])

    def test_source_facts_and_prior_metadata_resolution_are_not_rewritten(self):
        old = resolve_config('airfrans', contract=None)
        self.assertNotIn('integration_contract', old)
        self.assertEqual(old['values']['objective']['surface_weight'], .5)
        self.assertEqual(old['values']['objective']['kind'], 'relative_L2')
        with self.assertRaisesRegex(ValueError, 'reviewed decisions'):
            require_resolved_objective(old)
        self.assertEqual(validate_resolved(json.loads(json.dumps(old))), old)
        modified = copy.deepcopy(old)
        modified['integration_contract'] = AIRFRANS_OBJECTIVE_CONTRACT
        modified['config_hash'] = digest({k: v for k, v in modified.items() if k != 'config_hash'})
        with self.assertRaisesRegex(ValueError, 'versioned profile'):
            validate_resolved(modified)
        for task in TASKS:
            if task not in ('airfrans', 'car'):
                for profile in PROFILES:
                    self.assertEqual(resolve_config(task, profile), resolve_config(task, profile, contract=None))
                with self.assertRaisesRegex(ValueError, 'inapplicable'):
                    resolve_config(task, contract=AIRFRANS_OBJECTIVE_CONTRACT)
        with self.assertRaisesRegex(ValueError, 'reviewed decisions'):
            require_resolved_objective(resolve_config('car', contract=None))

    def test_explicit_configuration_still_has_priority_and_is_visible(self):
        config = resolve_config('airfrans', explicit={'objective.surface_weight': .75, 'model.linearno_rank': 16})
        self.assertEqual(config['values']['objective']['surface_weight'], .75)
        self.assertEqual(config['field_sources']['objective.surface_weight'], 'cli_explicit')
        self.assertEqual(config['values']['model']['linearno_rank'], 16)
        self.assertEqual(validate_resolved(config), config)

    def test_actual_model_metadata_roundtrip_and_objective_separation(self):
        # Exercise the actual Air class signature. No fake model or external pickle.
        from cdlno.linearno.airfrans import AirfRANSLinearNO
        from cdlno.linearno.schema import make_metadata, read_metadata, write_metadata, compare_metadata
        from linearno.test_schema import sample_metadata
        metadata = sample_metadata()
        config = resolve_config('airfrans')
        kwargs = dict(space_dim=7, n_layers=8, n_hidden=256, dropout=0., n_head=8,
                      act='gelu', mlp_ratio=2, fun_dim=0, out_dim=4, ref=8,
                      unified_pos=True, linearno_rank=32, linear=True)
        sections = {k: copy.deepcopy(v) for k,v in metadata.items() if k.endswith('_spec') and k not in ('model_spec','profile_spec','objective_spec','evaluation_spec')}
        sections.update(resume_state=metadata['resume_state'], ensemble_manifest=[],
            model_spec=dict(class_path='cdlno.linearno.airfrans.AirfRANSLinearNO', constructor_kwargs=kwargs),
            objective_spec=config['values']['objective'], evaluation_spec=config['values']['evaluation'])
        saved = make_metadata(profile_spec=config, constructor=AirfRANSLinearNO, **sections)
        with tempfile.TemporaryDirectory(prefix='linearno-air-objective-') as temp:
            path = Path(temp)/'metadata.json'
            write_metadata(path, saved, constructor=AirfRANSLinearNO)
            before = path.read_bytes()
            self.assertEqual(read_metadata(path, constructor=AirfRANSLinearNO), saved)
            self.assertEqual(path.read_bytes(), before)
        alternate = resolve_config('airfrans', explicit={'objective.surface_weight': .75})
        sections['objective_spec'] = alternate['values']['objective']
        requested = make_metadata(profile_spec=alternate, constructor=AirfRANSLinearNO, **sections)
        self.assertIn('objective_spec', compare_metadata(saved, requested, constructor=AirfRANSLinearNO))
        with self.assertRaisesRegex(ValueError, 'resume protocol'):
            compare_metadata(saved, requested, constructor=AirfRANSLinearNO, resume=True)
