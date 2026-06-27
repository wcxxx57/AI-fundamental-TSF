import math

import torch
import torch.nn as nn


class ResidualShrinkageFusion(nn.Module):
    """Constrain a complex fusion head to be a small residual over a stable base.

    The output is base + s * (candidate - base), where s is a learnable scalar in
    (0, 1). Initializing s near zero keeps the model close to the original
    fixed-weight MM-TSFlib fusion and lets validation decide whether a more
    complex correction is useful.
    """

    def __init__(self, init_shrink=0.1, max_shrink=0.5, signed=False):
        super().__init__()
        self.max_shrink = float(max_shrink)
        if self.max_shrink <= 0.0 or self.max_shrink > 1.0:
            raise ValueError("max_shrink must be in (0, 1].")
        self.signed = bool(signed)
        if self.signed:
            init_shrink = min(max(float(init_shrink), -self.max_shrink + 1e-4), self.max_shrink - 1e-4)
            init_logit = math.atanh(init_shrink / self.max_shrink)
        else:
            init_shrink = min(max(float(init_shrink), 1e-4), self.max_shrink - 1e-4)
            init_ratio = init_shrink / self.max_shrink
            init_logit = math.log(init_ratio / (1.0 - init_ratio))
        self.logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self.last_shrinkage = None

    def forward(self, base, candidate):
        if self.signed:
            shrinkage = self.max_shrink * torch.tanh(self.logit)
        else:
            shrinkage = self.max_shrink * torch.sigmoid(self.logit)
        self.last_shrinkage = float(shrinkage.detach().cpu())
        return base + shrinkage * (candidate - base)

    def value(self):
        with torch.no_grad():
            if self.signed:
                return float((self.max_shrink * torch.tanh(self.logit)).detach().cpu())
            return float((self.max_shrink * torch.sigmoid(self.logit)).detach().cpu())
