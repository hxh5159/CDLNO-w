"""LinearNO attention with optional v4 dual-end adaptive temperatures."""
import torch
from torch import nn
from cdlno.linearno.attention import LinearNOAttention
from .temperature import PointTemperaturePredictor, LatentTemperaturePredictor, stable_seed, multiplier

class V4LinearNOAttention(LinearNOAttention):
    def install_temperature_predictors(self, *, mode, public_seed, logical_depth):
        if getattr(self, "_predictors_installed", False):
            raise RuntimeError("temperature predictors already installed")
        if mode not in ("base", "latent_k_point_q", "point_k_point_q"):
            raise ValueError("unknown temperature mode")
        self.temperature_mode = mode
        if mode != "base":
            qseed = stable_seed(public_seed, logical_depth, "q", "shared")
            self.q_temperature = PointTemperaturePredictor(self.dim_head, 1, self.rank, qseed)
            if mode == "latent_k_point_q":
                kseed = stable_seed(public_seed, logical_depth, "k", mode)
                self.k_temperature = LatentTemperaturePredictor(self.dim_head, self.rank, self.rank, kseed)
            else:
                kseed = stable_seed(public_seed, logical_depth, "k", mode)
                self.k_temperature = PointTemperaturePredictor(self.dim_head, 1, self.rank, kseed)
        self._predictors_installed = True

    def _features(self, x):
        B,N,channels=x.shape
        if self.variant in ("conv", "conv_temp"):
            if N != self.H*self.W:
                raise ValueError(f"LinearNO conv requires N=H*W={self.H}*{self.W}; got {N}")
            projected=self.in_project_x(x.transpose(1,2).reshape(B,channels,self.H,self.W))
            return projected.reshape(B,self.heads,self.dim_head,N).transpose(-1,-2)
        projected=self.in_project_x(x)
        features=projected.reshape(B,N,self.heads,self.dim_head).transpose(1,2)
        return features.contiguous() if self.variant=="airfrans" else features

    def forward(self, x, *, logical_depth=None):
        planned=getattr(self,"_planned_temperature_mode",'base')
        installed=getattr(self,"_predictors_installed",False)
        if planned!='base' and not installed:
            raise RuntimeError('dynamic V4 attention requires installed temperature predictors')
        if not installed:
            return super().forward(x)
        if self.temperature_mode == "base":
            return super().forward(x)
        if logical_depth is not None and (type(logical_depth) is not int or logical_depth < 0):
            raise ValueError("logical_depth must be a nonnegative integer")
        if not isinstance(x,torch.Tensor) or x.ndim!=3 or not x.is_floating_point():
            raise ValueError("attention input must be floating [B,N,H]")
        B,N,channels=x.shape
        if channels!=self.dim or B<1 or N<1: raise ValueError("attention input shape mismatch")
        features=self._features(x)
        q_logits,k_logits,values=self.to_q(features),self.to_k(features),self.to_v(features)
        q_delta = self.q_temperature(features) if self.temperature_mode != "base" else None
        k_delta = self.k_temperature(features) if self.temperature_mode != "base" else None
        if self.variant in ("temp","conv_temp"):
            tau_q=self.temperature_q.clamp(.01,1.)
            tau_k=self.temperature_k.clamp(.01,1.)
        elif self.variant=="shapenet":
            tau_q=self.tempreature_q.clamp(.1,2.)
            tau_k=self.tempreature_k.clamp(.1,2.)
        else:
            tau_q=tau_k=1.0
        if self.temperature_mode != "base":
            q_logits=q_logits / (tau_q * multiplier(q_delta))
            k_logits=k_logits / (tau_k * multiplier(k_delta))
        else:
            q_logits=q_logits/tau_q; k_logits=k_logits/tau_k
        queries=q_logits.softmax(dim=-1); keys=k_logits.softmax(dim=-2)
        context=torch.einsum("bhnm,bhnd->bhmd",keys,values)
        readout=torch.einsum("bhnm,bhmd->bhnd",queries,context)
        merged=readout.transpose(1,2).reshape(B,N,self.heads*self.dim_head)
        return self.to_out(merged)
