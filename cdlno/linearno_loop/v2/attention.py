"""LinearNO context boundary used only by the v2 recurrent operators."""

import torch

from cdlno.linearno.attention import LinearNOAttention


class ContextLinearNOAttention(LinearNOAttention):
    """Native LinearNO with an optional processor between K^T V and Q readout.

    Plain ``forward`` delegates to the inherited implementation. This makes the
    round-specific mode use the exact existing formula instead of a second
    nominally equivalent implementation.
    """

    def forward(self, x):
        return super().forward(x)

    def forward_with_context(self, x, context_processor):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("LinearNO attention requires a tensor [B,N,dim]")
        B, N, channels = x.shape
        if B < 1 or N < 1 or channels != self.dim or not x.is_floating_point():
            raise ValueError("invalid LinearNO context-adapter input")
        if self.variant in ("conv", "conv_temp"):
            if N != self.H * self.W:
                raise ValueError(f"LinearNO conv requires N={self.H * self.W}; got {N}")
            grid = x.transpose(1, 2).reshape(B, channels, self.H, self.W)
            projected = self.in_project_x(grid)
            features = projected.reshape(B, self.heads, self.dim_head, N).transpose(-1, -2)
        else:
            projected = self.in_project_x(x)
            features = projected.reshape(B, N, self.heads, self.dim_head).transpose(1, 2)
            if self.variant == "airfrans":
                features = features.contiguous()
        q_logits, k_logits, values = self.to_q(features), self.to_k(features), self.to_v(features)
        if self.variant in ("temp", "conv_temp"):
            q_logits = q_logits / self.temperature_q.clamp(0.01, 1.0)
            k_logits = k_logits / self.temperature_k.clamp(0.01, 1.0)
        elif self.variant == "shapenet":
            q_logits = q_logits / self.tempreature_q.clamp(0.1, 2.0)
            k_logits = k_logits / self.tempreature_k.clamp(0.1, 2.0)
        queries = q_logits.softmax(dim=-1)
        keys = k_logits.softmax(dim=-2)
        context = torch.einsum("bhnm,bhnd->bhmd", keys, values)
        processed = context_processor(context)
        if not isinstance(processed, torch.Tensor) or processed.shape != context.shape:
            raise ValueError("context processor must preserve [B,h,M,d_h]")
        if processed.device != context.device or processed.dtype != context.dtype:
            raise ValueError("context processor must preserve context device/dtype")
        readout = torch.einsum("bhnm,bhmd->bhnd", queries, processed)
        merged = readout.transpose(1, 2).reshape(B, N, self.heads * self.dim_head)
        return self.to_out(merged)

