"""Muon, the optimizer Stable Audio 3 is actually trained with.

Its shipped config asks for `MuonAdamW`: Muon on the attention and feed-forward
matrices, AdamW on everything else. That pairing is not a detail. Muon
orthogonalises the momentum before applying it, so every matrix takes a step of
the same spectral size no matter how its gradients are scaled. Adam does not,
and substituting it means picking a single learning rate that is simultaneously
too large for some matrices and too small for others -- which is how a finetune
quietly destroys a pretrained model while its loss curve keeps falling.

The orthogonalisation is the Newton-Schulz iteration from Jordan et al., run in
bfloat16 because only the direction of the result is used.
"""

from __future__ import annotations

import torch


def orthogonalise(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of `g` by quintic Newton-Schulz."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.T
        x = a * x + (b * aa + c * aa @ aa) @ x
    return (x.T if transposed else x).to(g.dtype)


def fused_chunks(name: str, shape, embed_dim: int) -> int:
    """How many independent matrices a fused projection holds.

    A fused qkv weight is several projections stacked along the output dimension.
    Orthogonalising the stack treats them as one matrix, which is wrong: the
    reference splits them, which is what its `fused_layer_patterns` setting is for.
    """
    out = shape[0]
    if name.endswith("proj.weight") and ".ff." in name:
        return 2 if out % 2 == 0 else 1     # GEGLU: gate and value
    if any(k in name for k in ("to_qkv", "to_kv", "to_q")):
        return max(1, out // embed_dim)
    return 1


class Muon(torch.optim.Optimizer):
    """Momentum with the update orthogonalised before it is applied.

    Args:
        params: 2-D parameters only; give everything else to AdamW.
        lr: step size. The orthogonal update has unit spectral norm, so this is
            directly the size of the step, unlike Adam where it interacts with
            the gradient scale.
        momentum: heavy-ball coefficient, applied Nesterov-style.
        chunks: per-parameter split count for fused projections, keyed by id().
    """

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.95,
                 weight_decay: float = 0.0, chunks=None, momentum_dtype=torch.bfloat16):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
        self.chunks = chunks or {}
        self.momentum_dtype = momentum_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    # bfloat16 buffer: 2.9 GB instead of 5.8 GB for this model, and
                    # only the direction of the orthogonalised result survives anyway.
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=self.momentum_dtype)
                buf = state["momentum_buffer"]
                g = p.grad.reshape(p.shape[0], -1)
                buf.mul_(mu).add_(g.to(buf.dtype))
                g = g.add(buf.to(g.dtype), alpha=mu)       # Nesterov

                n = self.chunks.get(id(p), 1)
                u = (torch.cat([orthogonalise(s) for s in g.chunk(n, dim=0)], 0)
                     if n > 1 else orthogonalise(g))
                # Wider-than-tall matrices carry more directions; the usual
                # correction keeps the per-element step size comparable across shapes.
                scale = max(1.0, p.shape[0] / u.shape[1]) ** 0.5
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(u.reshape(p.shape), alpha=-lr * scale)
        return loss
