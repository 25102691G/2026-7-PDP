"""Joint FP64 bounded contributions. Empirical bounds are not a rounding proof."""
import math
import torch


def clip(vector):
    x=vector.detach().to(dtype=torch.float64).reshape(-1).clone()
    if not bool(torch.isfinite(x).all()):return torch.zeros_like(x)
    length=float(torch.linalg.vector_norm(x))
    if not math.isfinite(length):return torch.zeros_like(x)
    x.mul_(min(1.,3./(length+1e-6)))
    post=float(torch.linalg.vector_norm(x))
    safe=3.*(1.-1e-12)
    if post>safe:x.mul_(safe/post)
    return x


def read_gradient(parameters):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad.detach()).reshape(-1).double()
                      for p in parameters])


def double_clip(clipped_positions):
    if tuple(clipped_positions.shape)!=(4,4,2340):raise ValueError('fixed4x4x2340 layout required')
    # Caller has already clipped each position; second clip bounds each slot.
    b=torch.stack([clip(clipped_positions[j].double().sum(0)) for j in range(4)])
    return b.sum(0),b
