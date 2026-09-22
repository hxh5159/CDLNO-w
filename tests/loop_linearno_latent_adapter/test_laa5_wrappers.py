import unittest
from unittest.mock import patch

import torch

from .wrapper_support import (ABLATIONS, MODES, TASK_VARIANTS, WRAPPER_ROWS,
                              config, construct, example, finite_step, invoke, public_state)


class WrapperTests(unittest.TestCase):
    def test_feature_off_full_wrapper_matches_v1_initialization_output_and_gradients(self):
        from cdlno.linearno_loop.construction import build_from_config as build_v1
        from linearno_loop.config import resolve_config as resolve_v1
        for task in (name for name, _ in TASK_VARIANTS):
            actual_m = 8 if task == "car" else 5
            c = config(task, latent=False, adapter=False)
            if actual_m != c["loop_spec"]["actual_M"]:
                c = config(task, latent=False, adapter=False)
                base = c["base_profile_spec"]
                options = dict(architecture="operator_latent_adapter_v3", cost_profile="custom",
                    hidden_width=8, latent_width=11, heads=2, actual_M=actual_m,
                    topology_preset="d12", residual_mode="sr_1_over_r",
                    latent_enabled=False, adapter_mode="none", adapter_rank=2,
                    adapter_alpha=3.0)
                from linearno_loop.v3.config import resolve_config as resolve_v3
                with patch("linearno_loop.v3.config._profile", return_value=base):
                    c = resolve_v3(task, options=options)
            profile_overrides = {"model.hidden": 8, "model.heads": 2,
                                 "model.H": 2, "model.W": 3,
                                 "runtime.seed": c["profile_spec"]["values"]["runtime"]["seed"]}
            v1_options = dict(topology_preset="custom", prefix_blocks=2,
                recurrent_core_blocks=4, loop_repeats=2, suffix_blocks=2,
                residual_mode="sr_1_over_r", linearno_rank=actual_m)
            v1 = resolve_v1(task, "paper_table8_on_release_model",
                            options=v1_options, profile_overrides=profile_overrides)
            new, old = construct(c), build_v1(v1)
            self.assertEqual(new.state_dict().keys(), old.state_dict().keys())
            for key in new.state_dict():
                torch.testing.assert_close(new.state_dict()[key], old.state_dict()[key],
                                           atol=0, rtol=0)
            args = example(c, batch=1); old_args = args
            new_args = tuple(item.detach().clone().requires_grad_(item.is_floating_point())
                             if isinstance(item, torch.Tensor) else item for item in args)
            if task == "airfrans":
                new_args = example(c, points=6); old_args = (new_args[0].clone(),)
            elif task == "car":
                new_args = example(c, points=6); data, geom = new_args[0]
                old_args = ((data.clone(), geom.clone()),)
            y = invoke(new, c, new_args); z = invoke(old, c, old_args)
            torch.testing.assert_close(y, z, atol=0, rtol=0)
            y.square().mean().backward(); z.square().mean().backward()
            for left, right in zip(new.parameters(), old.parameters()):
                self.assertEqual(left.grad is None, right.grad is None)
                if left.grad is not None:
                    torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)

    def test_full_formal_d12_matrix_forward_backward_adamw(self):
        # Six variants x two cost profiles x three residuals x four ablations.
        # Formal H/Dz/M/head values are retained; only B/N/grid are synthetic.
        index = 0
        for task, variant in TASK_VARIANTS:
            for cost in ("matched_v1", "efficient_v1"):
                for mode in MODES:
                    for latent, adapter in ABLATIONS:
                        index += 1
                        with self.subTest(cost=cost, mode=mode, latent=latent,
                                          adapter=adapter, task=task):
                            c = config(task, cost=cost, mode=mode, latent=latent,
                                       adapter=adapter, formal_width=True)
                            model = construct(c)
                            output, _ = finite_step(model, c)
                            expected = ((6, 4) if task in ("airfrans", "car") else
                                        (1, 6, c["profile_spec"]["values"]["model"]["out_dim"]))
                            self.assertEqual(tuple(output.shape), expected)
                            self.assertEqual(c["loop_spec"]["variant"], variant)
                            WRAPPER_ROWS.append(dict(task=task, variant=variant,
                                cost_profile=cost, residual=mode, latent=latent,
                                adapter=adapter, H=c["loop_spec"]["hidden_width"],
                                Dz=c["loop_spec"]["latent_width"], M=c["loop_spec"]["actual_M"],
                                parameters=sum(p.numel() for p in model.parameters()), status="PASS"))
        self.assertEqual(index, 144)
        self.assertEqual({row["variant"] for row in WRAPPER_ROWS},
                         {variant for _, variant in TASK_VARIANTS})

    def test_standard_native_irregular_structured_temporal_contracts(self):
        for task in ("elasticity", "airfoil", "plasticity", "ns"):
            c = config(task)
            model = construct(c).eval()
            args = example(c)
            output = invoke(model, c, args)
            self.assertEqual(output.shape[:2], args[0].shape[:2])
            if task == "plasticity":
                self.assertTrue(model.Time_Input)
                with self.assertRaises(ValueError):
                    model(args[0], args[1], torch.ones(args[0].shape[0], 2))
            if c["loop_spec"]["variant"] in ("conv", "conv_temp"):
                with self.assertRaises(ValueError):
                    model(args[0][:, :-1], None if args[1] is None else args[1][:, :-1],
                          args[2])

    def test_airfrans_single_graph_reference_and_member_isolation(self):
        c = config("airfrans")
        first, second = construct(c, member_seed=101), construct(c, member_seed=102)
        self.assertFalse({id(p) for p in first.parameters()} & {id(p) for p in second.parameters()})
        self.assertFalse(torch.equal(first.preprocess.linear_pre[0].weight,
                                     second.preprocess.linear_pre[0].weight))
        data = example(c, points=9)[0]
        self.assertEqual(first(data).shape, (9, 4))
        sampled = data.clone(); sampled.x = sampled.x[:5]; sampled.pos = sampled.pos[:5]
        sampled.batch = sampled.batch[:5]
        self.assertEqual(first(sampled).shape, (5, 4))
        bad = data.clone(); bad.batch[-1] = 1
        with self.assertRaises(ValueError): first(bad)
        self.assertNotIn("reference", first.state_dict())

    def test_car_tuple_single_graph_and_h208_m32_without_old_constraint(self):
        c = config("car", cost="matched_v1", formal_width=True)
        self.assertEqual((c["loop_spec"]["hidden_width"], c["loop_spec"]["head_dim"],
                          c["loop_spec"]["actual_M"]), (208, 26, 32))
        model = construct(c).eval(); data, geom = example(c, points=9)[0]
        self.assertEqual(model((data, geom)).shape, (9, 4))
        with self.assertRaises(ValueError): model(data)
        bad = data.clone(); bad.ptr = torch.tensor([0, 8])
        with self.assertRaises(ValueError): model((bad, geom))
        # Old wrapper keeps its historical M % d_h restriction.
        from cdlno.linearno.shapenet import ShapeNetLinearNO
        with self.assertRaises(ValueError):
            ShapeNetLinearNO(n_hidden=208, n_head=8, linearno_rank=32)

    def test_initialization_order_public_pairing_and_feature_keys(self):
        for task in (name for name, _ in TASK_VARIANTS):
            reference = None
            for latent, adapter in ABLATIONS:
                c = config(task, latent=latent, adapter=adapter)
                before = torch.get_rng_state().clone(); model = construct(c)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                state = public_state(model)
                if reference is None: reference = state
                self.assertEqual(state.keys(), reference.keys())
                for key in state:
                    torch.testing.assert_close(state[key], reference[key], atol=0, rtol=0)
                keys = tuple(model.state_dict())
                self.assertEqual(any(".latent_processor." in key for key in keys), latent)
                self.assertEqual(any(".adapter." in key for key in keys), adapter)


if __name__ == "__main__": unittest.main()
