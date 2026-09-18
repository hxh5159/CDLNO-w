import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(os.environ.get("LINEARNO_REFERENCE_ROOT", "/home/hwz/LinearNO"))


class ConverterChecks(unittest.TestCase):
    def test_standard_strict_identity_and_module_prefix(self):
        sys.path.insert(0, str(ROOT / "PDE-Solving-StandardBenchmark"))
        from model.LinearNO import Model
        from cdlno.linearno.converter import (ConversionError, add_module_prefix,
                                               convert_state_dict, load_standard_state_dict)
        model = Model(space_dim=2, n_layers=2, n_hidden=12, n_head=3, mlp_ratio=2,
                      fun_dim=1, out_dim=2, linearno_variant="temp", linearno_rank=4,
                      H=3, W=5)
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        converted = convert_state_dict("standard", add_module_prefix(state), model)
        self.assertEqual(list(converted), list(state))
        for key in state:
            torch.testing.assert_close(converted[key], state[key], atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "standard.pt"
            torch.save(state, path)
            loaded = load_standard_state_dict(path)
            self.assertEqual(list(loaded), list(state))
        for bad in (dict(state, unexpected=torch.ones(1)),
                    {key: value for key, value in list(state.items())[1:]},
                    dict(state, **{"placeholder": torch.zeros(2)})):
            with self.assertRaises(ConversionError):
                convert_state_dict("standard", bad, model)
        with self.assertRaises(ConversionError):
            convert_state_dict("airfrans", state, model)

    def _official_object(self, task, path):
        subdir = "AirfRANS" if task == "airfrans" else "ShapeNetCar"
        script = r'''
import sys, torch
from models.LinearAttnNeuralOperator import LinearAttentionNeuralOperator
task, output = sys.argv[1:]
if task == "airfrans":
    model = LinearAttentionNeuralOperator(space_dim=7, n_layers=2, n_hidden=16,
        dropout=0., n_head=4, act="gelu", mlp_ratio=2, fun_dim=0, out_dim=4,
        slice_num=8, ref=8, unified_pos=True, linear=True)
else:
    model = LinearAttentionNeuralOperator(space_dim=3, n_layers=2, n_hidden=16,
        dropout=0., n_head=4, Time_Input=False, act="gelu", mlp_ratio=2, fun_dim=4,
        out_dim=4, key_ratio=2, ref=8, unified_pos=False, H=85, W=85, isregular=False)
torch.save(model, output)
'''
        result = subprocess.run([sys.executable, "-B", "-c", script, task, str(path)],
                                cwd=REFERENCE / subdir,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_trusted_objects_are_task_explicit_and_strict(self):
        from cdlno.linearno.airfrans import AirfRANSLinearNO
        from cdlno.linearno.shapenet import ShapeNetLinearNO
        from cdlno.linearno.converter import (ConversionError, convert_trusted_object,
                                              extract_trusted_object)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            air_path, car_path = temp / "air.pt", temp / "car.pt"
            self._official_object("airfrans", air_path)
            self._official_object("shapenet", car_path)
            with self.assertRaises(ConversionError):
                extract_trusted_object(air_path, "airfrans", reference_root=REFERENCE)
            air = AirfRANSLinearNO(space_dim=7, n_layers=2, n_hidden=16, n_head=4,
                                   mlp_ratio=2, fun_dim=0, out_dim=4, linearno_rank=8,
                                   ref=8, unified_pos=True, linear=True)
            converted, provenance = convert_trusted_object(
                air_path, "airfrans", air, reference_root=REFERENCE, trusted=True)
            self.assertEqual(provenance["source_task"], "airfrans")
            self.assertIn("blocks.0.Attn.temperature", converted)
            air.load_state_dict(converted, strict=True)
            car = ShapeNetLinearNO(space_dim=3, n_layers=2, n_hidden=16, n_head=4,
                                   mlp_ratio=2, fun_dim=4, out_dim=4, linearno_rank=8,
                                   ref=8, unified_pos=False, H=85, W=85, isregular=False)
            converted, provenance = convert_trusted_object(
                car_path, "shapenet", car, reference_root=REFERENCE, trusted=True)
            self.assertEqual(provenance["source_task"], "shapenet")
            self.assertIn("blocks.0.Attn.tempreature_q", converted)
            car.load_state_dict(converted, strict=True)
            with self.assertRaises(ConversionError):
                convert_trusted_object(air_path, "shapenet", car,
                                       reference_root=REFERENCE, trusted=True)


if __name__ == "__main__":
    unittest.main()
