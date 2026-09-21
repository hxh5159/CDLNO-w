"""Loop core with shared operators and execution-position-specific point FFNs."""

import torch
from torch import nn

from linearno_loop.contracts import RESIDUAL_MODES
from linearno_loop.v2.contracts import CORE_FFN_MODES
from cdlno.linearno_loop.attnres import PointDepthAttnRes
from cdlno.linearno_loop.body import LinearNOBlockBody
from cdlno.linearno_loop.core import PhysicalBlock, topology


class SharedCoreOperator(nn.Module):
    """Own one native ln_1 and LinearNO operator for one physical core slot."""

    def __init__(self, ln_1, attention):
        super().__init__()
        if not isinstance(ln_1, nn.LayerNorm) or not isinstance(attention, nn.Module):
            raise TypeError("shared operator requires LayerNorm and attention modules")
        self.ln_1 = ln_1
        self.Attn = attention

    def forward(self, x, context_processor=None):
        normalized = self.ln_1(x)
        if context_processor is None:
            return self.Attn(normalized)
        method = getattr(self.Attn, "forward_with_context", None)
        if method is None:
            raise TypeError("attention does not support a latent context processor")
        return method(normalized, context_processor)


class RoundPointFFN(nn.Module):
    """Own one native ln_2 and point MLP for exactly one (core, round)."""

    def __init__(self, ln_2, mlp):
        super().__init__()
        if not isinstance(ln_2, nn.LayerNorm) or not isinstance(mlp, nn.Module):
            raise TypeError("round point FFN requires LayerNorm and MLP modules")
        self.ln_2 = ln_2
        self.mlp = mlp

    def forward(self, x):
        return self.mlp(self.ln_2(x))


class LinearNOFFNLoopCore(nn.Module):
    """P/C/R/S loop with shared O[p] and independent F[p,r].

    Factories create only the modules they return. In particular, the operator
    and point-FFN factories must not construct and discard complete blocks.
    """

    def __init__(self, *, prefix_blocks, recurrent_core_blocks, loop_repeats,
                 suffix_blocks, residual_mode, core_ffn_mode, block_factory,
                 operator_factory, point_ffn_factory):
        super().__init__()
        if residual_mode not in RESIDUAL_MODES:
            raise ValueError(f"residual_mode must be one of {RESIDUAL_MODES}")
        if core_ffn_mode not in CORE_FFN_MODES:
            raise ValueError(f"core_ffn_mode must be one of {CORE_FFN_MODES}")
        P, C, R, S = prefix_blocks, recurrent_core_blocks, loop_repeats, suffix_blocks
        topology(P, C, R, S)
        self.prefix_blocks, self.recurrent_core_blocks = P, C
        self.loop_repeats, self.suffix_blocks = R, S
        self.residual_mode, self.core_ffn_mode = residual_mode, core_ffn_mode
        self.prefix = nn.ModuleList([PhysicalBlock(block_factory(last_layer=False)) for _ in range(P)])
        self.core_operators = nn.ModuleList([
            SharedCoreOperator(*operator_factory()) for _ in range(C)
        ])
        self.core_ffns = nn.ModuleList([
            nn.ModuleList([RoundPointFFN(*point_ffn_factory()) for _ in range(R)]) for _ in range(C)
        ])
        self.suffix = nn.ModuleList([
            PhysicalBlock(block_factory(last_layer=index == S - 1)) for index in range(S)
        ])
        hidden = self.core_operators[0].Attn.dim
        if residual_mode == "rb_attnres":
            self.rb_receivers = nn.ModuleList([
                nn.ModuleList([PointDepthAttnRes(hidden) for _ in range(2 * C)]) for _ in range(R)
            ])
            self.rb_output = PointDepthAttnRes(hidden)
        elif residual_mode == "lb_attnres_1_over_r":
            self.lb_boundaries = nn.ModuleList([PointDepthAttnRes(hidden) for _ in range(R - 1)])
            self.lb_output = PointDepthAttnRes(hidden)
        self.validate_structure()

    def install_latent_ffns(self, *, inner_width, feature_seed):
        """Install C context FFNs after common-tree init and placeholder draw."""
        if self.core_ffn_mode != "round_specific_latent":
            raise ValueError("latent FFNs are valid only in round_specific_latent mode")
        if hasattr(self, "latent_ffns"):
            raise RuntimeError("latent FFNs are already installed")
        if type(feature_seed) is not int or isinstance(feature_seed, bool) or not 0 <= feature_seed < 2**63:
            raise ValueError("feature_seed must be an integer in [0,2^63)")
        from .latent import LatentContextFFN
        hidden = self.core_operators[0].Attn.dim
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(feature_seed)
            modules = nn.ModuleList([LatentContextFFN(hidden, inner_width) for _ in range(self.recurrent_core_blocks)])
            for module in modules:
                module.initialize_release_identity()
        self.latent_ffns = modules
        self.feature_seed = feature_seed
        self.validate_structure(require_latent=True)

    @property
    def unique_depth(self):
        return self.prefix_blocks + self.recurrent_core_blocks + self.suffix_blocks

    @property
    def executed_depth(self):
        return self.prefix_blocks + self.recurrent_core_blocks * self.loop_repeats + self.suffix_blocks

    @property
    def point_ffn_instance_count(self):
        return self.prefix_blocks + self.recurrent_core_blocks * self.loop_repeats + self.suffix_blocks

    def validate_structure(self, *, require_latent=False):
        topology(self.prefix_blocks, self.recurrent_core_blocks, self.loop_repeats, self.suffix_blocks)
        if len(self.prefix) != self.prefix_blocks or len(self.core_operators) != self.recurrent_core_blocks:
            raise ValueError("v2 physical module count disagrees with topology")
        if len(self.core_ffns) != self.recurrent_core_blocks or any(len(row) != self.loop_repeats for row in self.core_ffns):
            raise ValueError("v2 requires one point FFN per core position and round")
        if len(self.suffix) != self.suffix_blocks:
            raise ValueError("v2 suffix count disagrees with topology")
        seen = set()
        for module in (*self.prefix, *self.core_operators,
                       *(ffn for row in self.core_ffns for ffn in row), *self.suffix):
            for parameter in module.parameters():
                if id(parameter) in seen:
                    raise ValueError("v2 modules must not alias parameters across registered paths")
                seen.add(id(parameter))
        for index, physical in enumerate((*self.prefix, *self.suffix)):
            expected_last = index == self.prefix_blocks + self.suffix_blocks - 1
            if physical.block.last_layer is not expected_last:
                raise ValueError("only final suffix may own the output head")
        hidden = self.core_operators[0].Attn.dim
        signature = None
        for operator in self.core_operators:
            current = tuple(getattr(operator.Attn, name, None) for name in
                            ("dim", "heads", "dim_head", "rank", "variant"))
            if None in current or signature is not None and current != signature:
                raise ValueError("all shared core operators require one LinearNO signature")
            signature = current
        for row in self.core_ffns:
            for ffn in row:
                if tuple(ffn.ln_2.normalized_shape) != (hidden,):
                    raise ValueError("round point FFN LayerNorm must match hidden width")
        if self.residual_mode == "rb_attnres":
            if len(self.rb_receivers) != self.loop_repeats or any(len(row) != 2 * self.recurrent_core_blocks for row in self.rb_receivers):
                raise ValueError("RB requires 2*C receivers per round")
        elif hasattr(self, "rb_receivers") or hasattr(self, "rb_output"):
            raise ValueError("non-RB mode must not register RB receivers")
        if self.residual_mode == "lb_attnres_1_over_r":
            if len(self.lb_boundaries) != self.loop_repeats - 1:
                raise ValueError("LB requires R-1 boundary receivers")
        elif hasattr(self, "lb_boundaries") or hasattr(self, "lb_output"):
            raise ValueError("non-LB mode must not register LB receivers")
        if self.core_ffn_mode == "round_specific":
            if hasattr(self, "latent_ffns"):
                raise ValueError("round_specific must not register latent modules")
        elif require_latent:
            if not hasattr(self, "latent_ffns") or len(self.latent_ffns) != self.recurrent_core_blocks:
                raise ValueError("latent mode requires one context FFN per core position")
            for position, module in enumerate(self.latent_ffns):
                if module.hidden != hidden:
                    raise ValueError(f"latent FFN {position} hidden mismatch")

    def _operator(self, position, x):
        processor = self.latent_ffns[position] if self.core_ffn_mode == "round_specific_latent" else None
        return self.core_operators[position](x, processor)

    @staticmethod
    def _rb_receive(receiver, sources, anchor):
        canonical = anchor.dtype
        local = tuple(source if source.dtype == canonical else source.to(dtype=canonical)
                      for source in sources)
        return receiver(local)

    def _branch_visit(self, x, position, round_index, scale):
        operator = self._operator(position, x)
        x = x + scale * operator
        point = self.core_ffns[position][round_index](x)
        return x + scale * point

    def _sr_forward(self, x):
        scale = 1.0 / self.loop_repeats
        for round_index in range(self.loop_repeats):
            for position in range(self.recurrent_core_blocks):
                x = self._branch_visit(x, position, round_index, scale)
        return x

    def _rb_forward(self, anchor):
        completed = [anchor]
        for round_index, receivers in enumerate(self.rb_receivers):
            partial = None
            for position in range(self.recurrent_core_blocks):
                sources = tuple(completed) if partial is None else (*completed, partial)
                hidden = self._rb_receive(receivers[2 * position], sources, anchor)
                raw = self._operator(position, hidden)
                partial = raw if partial is None else partial + raw
                hidden = self._rb_receive(receivers[2 * position + 1], (*completed, partial), anchor)
                raw = self.core_ffns[position][round_index](hidden)
                partial = partial + raw
            completed.append(partial)
        return self._rb_receive(self.rb_output, tuple(completed), anchor)

    def _lb_forward(self, anchor):
        x, deltas = anchor, []
        scale = 1.0 / self.loop_repeats
        for round_index in range(self.loop_repeats):
            entry = x
            for position in range(self.recurrent_core_blocks):
                x = self._branch_visit(x, position, round_index, scale)
            deltas.append(x - entry)
            sources = (anchor, *deltas)
            x = (self.lb_boundaries[round_index](sources)
                 if round_index < self.loop_repeats - 1 else self.lb_output(sources))
        return x

    def forward(self, x):
        self.validate_structure(require_latent=self.core_ffn_mode == "round_specific_latent")
        for block in self.prefix:
            x = block(x)
        if self.residual_mode == "sr_1_over_r":
            x = self._sr_forward(x)
        elif self.residual_mode == "rb_attnres":
            x = self._rb_forward(x)
        else:
            x = self._lb_forward(x)
        for index, block in enumerate(self.suffix):
            x = block(x, finalize=index == self.suffix_blocks - 1)
        return x
