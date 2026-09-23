"""Seed-isolated K/Q temperature predictors."""
import math
import torch
from torch import nn

class _PredictorBase(nn.Module):
    def __init__(self, in_width, out_width, hidden_width, seed):
        super().__init__()
        if type(seed) is not int or seed < 0:
            raise ValueError("predictor seed must be a nonnegative integer")
        with torch.random.fork_rng(devices=[]):
            # CPU allocation must not seed or initialize any CUDA generator.
            torch.random.default_generator.manual_seed(seed)
            self.fc1 = nn.Linear(in_width, hidden_width)
            self.act = nn.GELU(approximate="tanh")
            self.fc2 = nn.Linear(hidden_width, out_width)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

class PointTemperaturePredictor(_PredictorBase):
    def forward(self, s):
        return self.fc2(self.act(self.fc1(s)))

class LatentTemperaturePredictor(_PredictorBase):
    def forward(self, s):
        # [B,h,N,d] -> [B,h,d] -> [B,h,1,M]
        return self.fc2(self.act(self.fc1(s.mean(dim=2)))).unsqueeze(2)

def stable_seed(public_seed, depth, branch, strategy):
    import hashlib
    value = f"{int(public_seed)}:{int(depth)}:{branch}:{strategy}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63)

def multiplier(delta):
    return torch.exp(math.log(2.0) * torch.tanh(delta))
