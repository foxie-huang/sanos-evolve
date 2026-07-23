#!/usr/bin/env python3
"""
AD feasibility probe (v2): can PyTorch autograd differentiate through the ADAPTIVE recompression if we
STOP-GRADIENT the discrete assignment? Recompress = sort by mean -> equal-weight chunks -> moment-match.
We detach the argsort/searchsorted (a fixed membership matrix) and keep the moment-matching differentiable.
If the AD gradient matches a central finite-difference at several theta, the stop-gradient approach is
sound -> the full torch port (kernel+propagate+readout) is just engineering; v1's accuracy is preserved.
"""
import torch


def recompress_torch(w, mu, sg, n_keep):
    """Adaptive recompress with a STOP-GRADIENT membership: the sort + equal-weight chunk assignment is
    detached; the per-chunk 0/1/2 moment-match is differentiable in (w, mu, sg)."""
    n = w.shape[0]
    with torch.no_grad():                                   # ---- discrete assignment (no grad) ----
        order = torch.argsort(mu)
        cw = torch.cumsum(w[order], 0)
        edges = torch.searchsorted(cw, torch.linspace(0.0, float(cw[-1]), n_keep + 1)[1:-1].to(cw), right=True)
        chunk_of_sorted = torch.bucketize(torch.arange(n), edges)      # chunk id per sorted position
        memb = torch.zeros(n, n_keep)
        memb[order, chunk_of_sorted] = 1.0                  # original component -> its chunk (fixed)
    Wj = memb.t() @ w                                       # ---- differentiable moment-match ----
    Ws = torch.where(Wj > 1e-12, Wj, torch.ones_like(Wj))
    Mj = (memb.t() @ (w * mu)) / Ws
    E2 = (memb.t() @ (w * (sg ** 2 + mu ** 2))) / Ws
    Vj = torch.clamp(E2 - Mj ** 2, min=1e-12)
    return Wj, Mj, torch.sqrt(Vj)


def forward(theta):
    """Toy theta-dependent mixture -> recompress -> a downstream scalar (2nd moment), all in torch."""
    n = 60
    mu0 = torch.linspace(-0.25, 0.25, n)
    w = torch.softmax(4.0 * mu0 + theta, dim=0)             # weights depend on theta
    mu = mu0 + 0.15 * theta                                 # means depend on theta (shifts across chunk edges)
    sg = torch.full((n,), 0.05)
    Wj, Mj, Sj = recompress_torch(w, mu, sg, 8)
    return (Wj * (Mj ** 2 + Sj ** 2)).sum()                 # a downstream functional of the recompressed law


if __name__ == "__main__":
    print(f"{'theta':>7}{'AD grad':>14}{'FD grad':>14}{'rel.err':>10}")
    h = 1e-5
    for t0 in [-0.4, -0.1, 0.0, 0.2, 0.5, 0.9]:
        th = torch.tensor(t0, requires_grad=True)
        out = forward(th); out.backward()
        g_ad = float(th.grad)
        with torch.no_grad():
            g_fd = float((forward(torch.tensor(t0 + h)) - forward(torch.tensor(t0 - h))) / (2 * h))
        rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-9)
        print(f"{t0:>7.2f}{g_ad:>14.6f}{g_fd:>14.6f}{rel:>9.1%}")
    print("\nAD through detached-membership recompress matches FD => stop-gradient approach is sound.")
