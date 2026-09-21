import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from cdlno.linearno_loop.v2.checkpoint import read_pair, save_pair
from cdlno.linearno_loop.v2.construction import build_from_config
from cdlno.linearno.checkpoint import strict_load
from linearno_loop.contracts import seal
from linearno_loop.v2.config import resolve_config
from linearno_loop.v2.schema import write_metadata
from support import metadata


TASK_VARIANTS = (("ns","plain"),("elasticity","temp"),("plasticity","conv"),
                 ("darcy","conv_temp"),("airfrans","airfrans"),("car","shapenet"))


def small_config(task, mode="round_specific", residual="sr_1_over_r", topology="p1_c3_r2_s1", multiplier=1):
    overrides={"runtime.seed":19,"model.hidden":8,"model.heads":2,"model.ffn_ratio":1,"model.ref":2}
    if task not in ("airfrans","car"):
        overrides.update({"model.H":2,"model.W":3})
    return resolve_config(task,options={"topology_preset":topology,"residual_mode":residual,
        "core_ffn_mode":mode,"linearno_rank":4*multiplier},profile_overrides=overrides)


def inputs(task, dtype=torch.float32):
    if task=="airfrans":
        return (SimpleNamespace(x=torch.randn(5,7,dtype=dtype),pos=torch.randn(5,2,dtype=dtype),
            batch=torch.zeros(5,dtype=torch.long),ptr=torch.tensor([0,5])),)
    if task=="car":
        data=SimpleNamespace(x=torch.randn(5,7,dtype=dtype),batch=torch.zeros(5,dtype=torch.long),ptr=torch.tensor([0,5]))
        return ((data,None),)
    x=torch.randn(2,6,2 if task!="elasticity" else 2,dtype=dtype)
    config=small_config(task);fun=config["profile_spec"]["values"]["model"]["fun_dim"]
    fx=None if fun==0 else torch.randn(2,6,fun,dtype=dtype)
    T=torch.randn(2,1,dtype=dtype) if config["profile_spec"]["values"]["model"]["time_input"] else None
    return (x,fx,T)


class LF4WrapperTests(unittest.TestCase):
    def test_common_initialization_modes_and_residuals(self):
        for task,_ in TASK_VARIANTS:
            configs=[small_config(task,mode,residual) for mode in ("round_specific","round_specific_latent")
                     for residual in ("sr_1_over_r","rb_attnres","lb_attnres_1_over_r")]
            states=[]
            for config in configs:
                before=torch.random.get_rng_state().clone();model=build_from_config(config)
                self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
                states.append({k:v for k,v in model.state_dict().items() if not k.startswith(("loop.latent_ffns.","loop.rb_","loop.lb_"))})
                latent=config["loop_spec"]["core_ffn_mode"].endswith("latent")
                self.assertEqual(any(k.startswith("loop.latent_ffns.") for k in model.state_dict()),latent)
                if latent:
                    for module in model.loop.latent_ffns:
                        self.assertEqual(module.linear2.weight.count_nonzero(),0)
                        self.assertEqual(module.linear2.bias.count_nonzero(),0)
            for state in states[1:]:
                self.assertEqual(states[0].keys(),state.keys())
                self.assertTrue(all(torch.equal(states[0][key],state[key]) for key in state),task)

    def test_six_variants_two_modes_three_residuals_forward_backward_step(self):
        for task,variant in TASK_VARIANTS:
            for mode in ("round_specific","round_specific_latent"):
                for residual in ("sr_1_over_r","rb_attnres","lb_attnres_1_over_r"):
                    with self.subTest(task=task,mode=mode,residual=residual):
                        config=small_config(task,mode,residual)
                        model=build_from_config(config)
                        args=inputs(task)
                        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
                        output=model(*args);self.assertTrue(torch.isfinite(output).all())
                        output.square().mean().backward();optimizer.step()
                        self.assertEqual(model.loop.executed_depth,8)

    def test_preset_custom_rank_and_state_ownership(self):
        configs=[small_config("darcy",topology="p1_c3_r2_s1"),small_config("darcy",topology="p2_c2_r2_s2")]
        custom=resolve_config("darcy",options={"topology_preset":"custom","prefix_blocks":0,
            "recurrent_core_blocks":2,"loop_repeats":3,"suffix_blocks":1,"residual_mode":"sr_1_over_r",
            "core_ffn_mode":"round_specific_latent","linearno_rank":4},profile_overrides={
            "runtime.seed":19,"model.hidden":8,"model.heads":2,"model.ffn_ratio":1,"model.H":2,"model.W":3})
        configs.append(custom)
        for config in configs:
            model=build_from_config(config);loop=config["loop_spec"]
            self.assertEqual(len(model.loop.core_operators),loop["recurrent_core_blocks"])
            self.assertEqual(sum(len(row) for row in model.loop.core_ffns),loop["recurrent_core_blocks"]*loop["loop_repeats"])
            self.assertFalse(any("core_operators" in key and ".1." in key.split("core_operators",1)[1] for key in ()))

    def test_v2_checkpoint_pair_strict_and_version_rejection(self):
        config=small_config("elasticity",mode="round_specific_latent",residual="lb_attnres_1_over_r")
        model=build_from_config(config);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        args=inputs("elasticity");model(*args).square().mean().backward();optimizer.step()
        architecture=metadata(config)
        checkpoint=copy.deepcopy(architecture);checkpoint["resume_state"]["epoch"]=1
        checkpoint["resume_state"]["global_step"]=1;checkpoint=seal(checkpoint,"metadata_hash")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_metadata(root/"architecture.json",architecture)
            manifest=save_pair(root,model,checkpoint)
            saved,state=read_pair(manifest,expected=config);self.assertEqual(saved,checkpoint)
            clone=build_from_config(config);strict_load(clone,state)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),clone.state_dict().values())))
            from cdlno.linearno_loop.checkpoint import inspect_checkpoint as inspect_v1
            with self.assertRaisesRegex(ValueError,"format"):
                inspect_v1(root,"latest")


if __name__=="__main__":unittest.main()
