import torch
import torch.nn as nn
from .ste import round_clamp_ste

GROUP_SIZE = 128  # paper default; 64 also supported everywhere


def ternarize_weight(w, group_size=GROUP_SIZE):
    """Ternarize an FP weight to {-1,0,1} with group-wise alpha (Bonsai g128).

    One alpha per group of `group_size` consecutive weights along the last
    dim — matches the Ternary-Bonsai on-disk format (verified: within each
    g128 group, w/alpha rounds exactly to {-1,0,1}). Last dim must divide
    evenly by group_size (2048, 1024, 6144 all do; embeddings 2048 too).
    alpha is detached. STE applies to the round_clamp. Returns w_q, same
    shape as w.
    """
    K = group_size
    shape = w.shape
    wg = w.reshape(*shape[:-1], shape[-1] // K, K)
    with torch.no_grad():
        a = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    out = round_clamp_ste(wg / a) * a
    return out.reshape(shape)


class TernaryLinear(nn.Linear):
    """Drop-in nn.Linear with ternary fake-quant on weight.

    Subclasses nn.Linear so unsloth's target discovery
    (`isinstance(module, torch.nn.Linear)`) sees it as a trainable Linear.
    Owns an FP master weight; forward applies ternary STE on the weight
    before matmul. Optimizer sees the FP weight via autograd.
    bias=False by default (Qwen3 has no_bias).
    """

    def __init__(self, weight, bias=None, group_size=GROUP_SIZE):
        super().__init__(weight.shape[1], weight.shape[0], bias is not None)
        self.weight = nn.Parameter(weight.detach())  # reuse storage; no clone
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        self.group_size = group_size

    def forward(self, x):
        w_q = ternarize_weight(self.weight, self.group_size).to(x.dtype)
        return torch.nn.functional.linear(x, w_q, self.bias)

    def extra_repr(self):
        return f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, ternary=True"
