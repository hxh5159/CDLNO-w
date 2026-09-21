import unittest

import torch
from torch import nn

from cdlno.linearno_loop.v2.core import LinearNOFFNLoopCore
from cdlno.linearno_loop.v2.factories import industrial_factories, standard_factories


class TinyAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.dim = hidden; self.heads = 1; self.dim_head = hidden; self.rank = 2; self.variant = "plain"
        self.linear = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.linear(x)

    def forward_with_context(self, x, processor):
        context=x.mean(dim=1)[:,None,None,:].expand(-1,1,self.rank,-1)
        delta=processor(context)-context
        return self.linear(x+delta.mean(dim=2).squeeze(1)[:,None,:])


class TinyMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__(); self.linear = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.linear(x)


class TinyBlock(nn.Module):
    def __init__(self, hidden, last_layer):
        super().__init__(); self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden); self.Attn = TinyAttention(hidden)
        self.ln_2 = nn.LayerNorm(hidden); self.mlp = TinyMLP(hidden)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden); self.mlp2 = nn.Linear(hidden, hidden)


def tiny_core(mode="sr_1_over_r", P=1, C=2, R=2, S=1, hidden=4):
    return LinearNOFFNLoopCore(prefix_blocks=P, recurrent_core_blocks=C, loop_repeats=R,
        suffix_blocks=S, residual_mode=mode, core_ffn_mode="round_specific",
        block_factory=lambda *, last_layer: TinyBlock(hidden, last_layer),
        operator_factory=lambda: (nn.LayerNorm(hidden), TinyAttention(hidden)),
        point_ffn_factory=lambda: (nn.LayerNorm(hidden), TinyMLP(hidden)))


def independent_core_oracle(core, x):
    for physical in core.prefix:
        block = physical.block
        x = x + block.Attn(block.ln_1(x)); x = x + block.mlp(block.ln_2(x))
    anchor = x
    if core.residual_mode == "sr_1_over_r":
        for r in range(core.loop_repeats):
            for p in range(core.recurrent_core_blocks):
                x = x + core.core_operators[p].Attn(core.core_operators[p].ln_1(x)) / core.loop_repeats
                x = x + core.core_ffns[p][r].mlp(core.core_ffns[p][r].ln_2(x)) / core.loop_repeats
    elif core.residual_mode == "rb_attnres":
        complete = [anchor]
        for r, row in enumerate(core.rb_receivers):
            partial = None
            for p in range(core.recurrent_core_blocks):
                h = row[2*p](tuple(complete) if partial is None else (*complete, partial))
                u = core.core_operators[p].Attn(core.core_operators[p].ln_1(h))
                partial = u if partial is None else partial + u
                h = row[2*p+1]((*complete, partial))
                partial = partial + core.core_ffns[p][r].mlp(core.core_ffns[p][r].ln_2(h))
            complete.append(partial)
        x = core.rb_output(tuple(complete))
    else:
        deltas = []
        for r in range(core.loop_repeats):
            entry = x
            for p in range(core.recurrent_core_blocks):
                x = x + core.core_operators[p].Attn(core.core_operators[p].ln_1(x)) / core.loop_repeats
                x = x + core.core_ffns[p][r].mlp(core.core_ffns[p][r].ln_2(x)) / core.loop_repeats
            deltas.append(x-entry)
            sources = (anchor, *deltas)
            x = core.lb_boundaries[r](sources) if r < core.loop_repeats-1 else core.lb_output(sources)
    for index, physical in enumerate(core.suffix):
        block = physical.block
        x = x + block.Attn(block.ln_1(x)); x = x + block.mlp(block.ln_2(x))
        if index == core.suffix_blocks-1:
            x = block.mlp2(block.ln_3(x))
    return x


class LF2CoreTests(unittest.TestCase):
    def test_oracle_three_modes_and_topologies(self):
        for topology in ((1,3,2,1), (2,2,2,2), (0,2,3,1)):
            for mode in ("sr_1_over_r", "rb_attnres", "lb_attnres_1_over_r"):
                torch.manual_seed(8); core = tiny_core(mode, *topology).double()
                if mode != "sr_1_over_r":
                    for module in core.modules():
                        if hasattr(module, "query"): module.query.data.fill_(0.13)
                x = torch.randn(2, 5, 4, dtype=torch.double)
                self.assertTrue(torch.allclose(core(x), independent_core_oracle(core, x), atol=1e-12, rtol=1e-12))

    def test_ownership_keys_calls_and_gradients(self):
        core = tiny_core(C=3, R=2)
        operator_ids = [[id(parameter) for parameter in module.parameters()] for module in core.core_operators]
        ffn_ids = [[[id(parameter) for parameter in core.core_ffns[p][r].parameters()] for r in range(2)] for p in range(3)]
        self.assertTrue(all(set(ffn_ids[p][0]).isdisjoint(ffn_ids[p][1]) for p in range(3)))
        self.assertEqual(len({item for row in operator_ids for item in row}), sum(map(len, operator_ids)))
        keys = tuple(core.state_dict())
        self.assertFalse(any("round" in key and "operator" in key for key in keys))
        self.assertEqual(core.point_ffn_instance_count, 1 + 3*2 + 1)
        calls = {"operators": [0]*3, "ffns": [[0]*2 for _ in range(3)]}
        handles=[]
        for p,module in enumerate(core.core_operators):
            handles.append(module.register_forward_hook(lambda m,a,o,p=p: calls["operators"].__setitem__(p,calls["operators"][p]+1)))
        for p,row in enumerate(core.core_ffns):
            for r,module in enumerate(row):
                handles.append(module.register_forward_hook(lambda m,a,o,p=p,r=r: calls["ffns"][p].__setitem__(r,calls["ffns"][p][r]+1)))
        x=torch.randn(2,5,4,requires_grad=True); core(x).square().mean().backward()
        for handle in handles: handle.remove()
        self.assertEqual(calls["operators"],[2,2,2]); self.assertEqual(calls["ffns"],[[1,1]]*3)
        self.assertTrue(all(parameter.grad is not None for parameter in core.core_operators[0].parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in core.core_ffns[0][1].parameters()))

    def test_six_variant_factories_forward_and_strict_roundtrip(self):
        cases=[]
        for variant in ("plain","temp","conv","conv_temp"):
            cases.append((variant, standard_factories(hidden=8,heads=2,rank=4,variant=variant,
                dropout=0.,mlp_ratio=1,H=2,W=3,out_dim=2), 6 if "conv" in variant else 5))
        cases.append(("airfrans",industrial_factories("airfrans",hidden=8,heads=2,rank=4,dropout=0.,mlp_ratio=1,out_dim=2),5))
        cases.append(("shapenet",industrial_factories("car",hidden=8,heads=2,rank=4,dropout=0.,mlp_ratio=1,out_dim=2),5))
        for variant,factories,N in cases:
            with self.subTest(variant=variant):
                torch.manual_seed(3)
                core=LinearNOFFNLoopCore(prefix_blocks=1,recurrent_core_blocks=2,loop_repeats=2,suffix_blocks=1,
                    residual_mode="sr_1_over_r",core_ffn_mode="round_specific",block_factory=factories[0],
                    operator_factory=factories[1],point_ffn_factory=factories[2]).double()
                x=torch.randn(2,N,8,dtype=torch.double,requires_grad=True); y=core(x); y.square().mean().backward()
                clone=LinearNOFFNLoopCore(prefix_blocks=1,recurrent_core_blocks=2,loop_repeats=2,suffix_blocks=1,
                    residual_mode="sr_1_over_r",core_ffn_mode="round_specific",block_factory=factories[0],
                    operator_factory=factories[1],point_ffn_factory=factories[2]).double()
                clone.load_state_dict(core.state_dict(),strict=True)
                self.assertTrue(torch.equal(y.detach(),clone(x.detach())))

    def test_invalid_and_inactive_router_keys(self):
        for kwargs in (dict(C=0),dict(R=0),dict(S=0)):
            with self.assertRaises(ValueError): tiny_core(**kwargs)
        sr=tiny_core("sr_1_over_r"); keys=tuple(sr.state_dict())
        self.assertFalse(any("receiver" in key or "boundar" in key for key in keys))


if __name__ == "__main__": unittest.main()
