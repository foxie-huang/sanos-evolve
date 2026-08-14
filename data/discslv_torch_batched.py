#!/usr/bin/env python3
"""
BATCHED grid propagate (v2 perf): propagate all B = n_f*n_s*nz sigma_ATM(z,f,s) chains together as a
batch dimension, instead of the Python loop. The recompress becomes a BATCHED detached-membership merge
(sort/chunk over the component axis, batched over B*regimes). Reuses build_kernel/lev_torch/stationary_pi/
lev_increment/interp_batch from discslv_torch. Same result as the loop version; far fewer ops -> fast +
still differentiable.  Validate+bench:  python3 discslv_torch_batched.py
"""
import os

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss
import discslv_torch as T

_JENSEN = os.environ.get("JENSEN", "1") != "0"   # intra-component leverage correction, on by default

torch.set_default_dtype(torch.float64)


def _merge_batch(w, mu, sg, nk):
    """(B,M)->(B,nk) batched recompress: v1's sort->equal-weight chunk (STOP-GRADIENT membership) ->
    moment-match (differentiable). Matches _fast_merge per row. Aggregation via scatter_add instead of a
    dense (Bn,M,nk) one-hot einsum -- identical result + identical gradient (membership is stop-grad, the
    sums are diff'able in w/mu/sg), ~nk-fold less memory (the one-hot was the eval's dominant tensor)."""
    Bn, M = w.shape
    with torch.no_grad():
        order = torch.argsort(mu, dim=1)
        cw = torch.cumsum(torch.gather(w, 1, order), dim=1)
        qt = cw[:, -1:] * (torch.arange(1, nk, dtype=w.dtype) / nk)[None, :]   # (B, nk-1)
        edges = torch.searchsorted(cw, qt, right=True)                         # (B, nk-1)
        pos = torch.arange(M)[None, :].expand(Bn, M).contiguous()
        chunk = torch.searchsorted(edges, pos, right=True)                    # (B,M) chunk id per SORTED pos:
        #   count edges<=pos == the old (pos[:,:,None]>=edges[:,None,:]).sum(-1), but WITHOUT the (Bn,M,nk)
        #   int64 broadcast that was the eval's true ~12GB transient peak (searchsorted is exact, verified 0 mismatch)
        chunk_of = torch.empty(Bn, M, dtype=torch.long).scatter_(1, order, chunk)  # chunk id per ORIGINAL pos

    def seg(x):                                                                # segment-sum into nk bins (diff'able in x)
        return torch.zeros(Bn, nk, dtype=x.dtype, device=x.device).scatter_add(1, chunk_of, x)
    Wj = seg(w)
    Ws = torch.where(Wj > 1e-15, Wj, torch.ones_like(Wj))
    Mj = torch.where(Wj > 1e-15, seg(w * mu) / Ws, torch.zeros_like(Wj))
    E2 = seg(w * (sg ** 2 + mu ** 2)) / Ws
    return Wj, Mj, torch.sqrt(torch.clamp(E2 - Mj ** 2, min=1e-12))


def _clamped_ratio(a, mu, sg, m):
    """E[exp(a*clamp(z,-m,m))] / exp(a*clamp(mu,-m,m)) for z ~ N(mu, sg^2). Exact.

    The naive lognormal identity exp(a^2 sg^2/2) is WRONG here and unstable: lev_torch clamps its
    argument to +-zmax, so lambda is flat outside that band, and a component wider than the band (sd
    0.245 against zmax 0.066 by step 2) got its variance inflated 64x, which widened the next
    component, which inflated more -- divergence to NaN by step 4. Splitting the expectation at the
    clamp gives the exact value and bounds it by exp(2|a|m), since clamp(z) never leaves [-m, m], so
    the feedback cannot run away.
    """
    Phi = lambda x: 0.5 * (1.0 + torch.erf(x * 0.70710678118654752))
    sg = torch.clamp(sg, min=1e-12)
    cmu = torch.clamp(mu, -m, m)
    lo, hi = (-m - mu) / sg, (m - mu) / sg
    t_lo = torch.exp(a * (-m - cmu)) * Phi(lo)                                  # mass below -m
    t_hi = torch.exp(a * (m - cmu)) * (1.0 - Phi(hi))                           # mass above +m
    ex = torch.clamp(a * (mu - cmu) + 0.5 * a * a * sg * sg, max=40.0)          # guard the interior
    t_mid = torch.exp(ex) * (Phi(hi - a * sg) - Phi(lo - a * sg))
    b = float(np.exp(2.0 * abs(a) * m))
    return torch.clamp(t_lo + t_mid + t_hi, 1.0 / b, b)


def propagate_batch(state, ker, lam, nk):
    """One fused step for a BATCH of chains. state = (W,MU,SG,F,S) each (B,N). Returns (B,N')."""
    W, MU, SG, F, S = state
    B, N = W.shape; nl, nf, ns = ker["n_l"], ker["n_f"], ker["n_s"]; wl = ker["wl"]
    Vlr = ker["Vl"][F, S]                                    # (B,N,nl)
    lm = torch.clamp(lam(MU), min=1e-6)                      # (B,N)
    # Jensen correction for freezing lambda at the component MEAN (JENSEN=0 disables). lambda is
    # log-linear in z, so against a Gaussian component of width SG the exact intra-component averages
    # are lambda(MU)*exp(c1^2 SG^2/2) and lambda(MU)^2*exp(2 c1^2 SG^2). Without it the injected
    # variance is understated by ~23% by step 13 and the error grows monotonically with step count.
    # Exact only where lev_torch's clamps do not bind (|MU| <= zmax, lambda inside `safety`).
    c1 = getattr(lam, "c1", 0.0) if _JENSEN else 0.0
    if c1:
        m = getattr(lam, "zmax", 0.0)
        lv = lm ** 2 * _clamped_ratio(2.0 * c1, MU, SG, m)   # E[lambda^2] / lambda(MU)^2
        lt = lm * _clamped_ratio(c1, MU, SG, m)              # E[lambda]   / lambda(MU)
    else:
        lv, lt = lm ** 2, lm
    Vl = lv[..., None] * Vlr
    mtil = -0.5 * Vl + lt[..., None] * ker["tilt"][F, S]
    A = torch.log((wl * torch.exp(mtil + 0.5 * Vl)).sum(-1)) # (B,N)
    Dl = mtil - A[..., None]                                 # (B,N,nl)
    mu_c = MU[..., None] + Dl                                # (B,N,nl)
    sg_c = torch.sqrt(SG[..., None] ** 2 + Vl)
    Tf_g = ker["Tf"][:, F, :].permute(1, 2, 0, 3)            # (B,N,nl,nf)
    Ts_g = ker["Ts"][:, S, :].permute(1, 2, 0, 3)            # (B,N,nl,ns)
    w_c = W[:, :, None, None, None] * wl[None, None, :, None, None] * Tf_g[..., :, None] * Ts_g[..., None, :]
    w_c = w_c / w_c.sum(dim=(1, 2, 3, 4), keepdim=True)      # normalize per chain
    M = N * nl
    w_r = w_c.permute(0, 3, 4, 1, 2).reshape(B * nf * ns, M)                     # (B*15, M) per dest regime
    mu_r = mu_c[:, None, None].expand(B, nf, ns, N, nl).reshape(B * nf * ns, M)
    sg_r = sg_c[:, None, None].expand(B, nf, ns, N, nl).reshape(B * nf * ns, M)
    if M > nk:
        Wj, Mj, SGj = _merge_batch(w_r, mu_r, sg_r, nk); K = nk
    else:
        Wj, Mj, SGj = w_r, mu_r, sg_r; K = M
    Wo = Wj.reshape(B, nf, ns, K); MUo = Mj.reshape(B, nf, ns, K); SGo = SGj.reshape(B, nf, ns, K)
    Fo = torch.arange(nf)[None, :, None, None].expand(B, nf, ns, K)
    So = torch.arange(ns)[None, None, :, None].expand(B, nf, ns, K)
    Wo = Wo.reshape(B, -1); MUo = MUo.reshape(B, -1); SGo = SGo.reshape(B, -1)
    Fo = Fo.reshape(B, -1).long(); So = So.reshape(B, -1).long()
    Arel = torch.log((Wo * torch.exp(MUo + 0.5 * SGo ** 2)).sum(-1, keepdim=True))   # per-chain re-lock
    return Wo, MUo - Arel, SGo, Fo, So


def _Phi(x):
    return 0.5 * (1.0 + torch.erf(x / (2.0 ** 0.5)))


def gm_call_batch(Kstrike, W, MU, SG):
    """Batched GM call price, per chain (B,). Kstrike scalar."""
    Fw = torch.exp(MU + 0.5 * SG ** 2)
    d1 = (torch.log(Fw / Kstrike) + 0.5 * SG ** 2) / SG
    return (W * (Fw * _Phi(d1) - Kstrike * _Phi(d1 - SG))).sum(-1)


def atm_skew_batch(W, MU, SG, Tmat, dm=6e-3, iters=8):
    """Batched ATM vol + skew (B,) via probit guess + unrolled Newton (matches v1 atm_skew_of)."""
    sqrtT = float(Tmat) ** 0.5
    Klo = float(np.exp(-dm)); Khi = float(np.exp(dm))
    Ca = gm_call_batch(1.0, W, MU, SG); Cl = gm_call_batch(Klo, W, MU, SG); Ch = gm_call_batch(Khi, W, MU, SG)
    u0 = 2.0 * (2.0 ** 0.5) * torch.erfinv(torch.clamp(Ca, -0.999, 0.999))       # (B,)

    def inv(Kk, C):
        u = u0
        for _ in range(iters):
            d1 = (-np.log(Kk) + 0.5 * u ** 2) / u
            vega = torch.clamp(torch.exp(-0.5 * d1 ** 2) / (2 * np.pi) ** 0.5, min=1e-10)
            u = torch.clamp(u - (_Phi(d1) - Kk * _Phi(d1 - u) - C) / vega, min=1e-6)
        return u
    ua = inv(1.0, Ca); ul = inv(Klo, Cl); uh = inv(Khi, Ch)
    return ua / sqrtT, (uh - ul) / sqrtT / (2 * dm)


def fused_ssr_ts_batched(theta, lam_fns, n, dt, nz=9, zmax=0.12, Q=5, nk=16, n_f=5, n_s=3, n_l=5):
    """Full term-structure SSR with the BATCHED grid. Same result as discslv_torch.fused_ssr_ts."""
    ker = T.build_kernel(theta, dt, n_f, n_s, n_l)
    zg = torch.linspace(-zmax, zmax, nz); iz0 = nz // 2
    f0 = torch.arange(n_f).repeat_interleave(n_s * nz)      # B = nf*ns*nz chains: (f0,s0,iz) grid
    s0 = torch.arange(n_s).repeat_interleave(nz).repeat(n_f)
    z0 = zg.repeat(n_f * n_s)
    B = n_f * n_s * nz
    st = (torch.ones(B, 1), z0[:, None].clone(), torch.full((B, 1), 1e-4), f0[:, None].long(), s0[:, None].long())
    for k in range(n):
        st = propagate_batch(st, ker, lam_fns[k], nk)
    sig, sk = atm_skew_batch(st[0], st[1], st[2], n * dt)   # (B,), (B,)
    sigg = sig.reshape(n_f, n_s, nz)
    skw0 = sk.reshape(n_f, n_s, nz)[:, :, iz0]
    pi = T.stationary_pi(ker); lam0 = lam_fns[0](torch.zeros(1, dtype=ker["wl"].dtype, device=ker["wl"].device))[0]
    zq_np, wq_np = hermegauss(Q); wq_np = wq_np / wq_np.sum()
    dd = dict(dtype=ker["wl"].dtype, device=ker["wl"].device)
    zq = torch.as_tensor(zq_np, **dd); wq = torch.as_tensor(wq_np, **dd); wl = ker["wl"]
    siggflat = sigg.reshape(n_f * n_s, nz)
    Vl = lam0 ** 2 * ker["Vl"]                              # (nf,ns,nl) all-regime one-step moments
    mtil = -0.5 * Vl + lam0 * ker["tilt"]
    Dl = mtil - torch.log((wl * torch.exp(mtil + 0.5 * Vl)).sum(-1, keepdim=True))
    mean_r = (wl * Dl).sum(-1)                              # (nf,ns)
    var = (pi * ((wl * (Dl ** 2 + Vl)).sum(-1) - mean_r ** 2)).sum()
    sd = torch.sqrt(torch.clamp(Vl, min=1e-16))
    r = Dl[..., None] + sd[..., None] * zq                  # (nf,ns,nl,Q)
    SIG = T.interp_batch(r, zg, siggflat).reshape(n_f, n_s, n_f, n_s, n_l, Q)  # (a,b,f,s,l,q) dest at r
    P = torch.einsum("lfa,lsb->fslab", ker["Tf"], ker["Ts"])                   # (f,s,l,a,b)
    sig_dest = torch.einsum("fslab,abfslq->fslq", P, SIG)                      # (f,s,l,q)
    wlq = wl[:, None] * wq[None, :]
    cov = (wlq * (sig_dest - sigg[:, :, iz0][..., None, None]) * (r - mean_r[..., None, None])).sum((-1, -2))
    beta = (pi * cov).sum() / var; skew = (pi * skw0).sum()
    return beta / skew, beta, skew


def fused_ssr_ts_multi(theta, lam_fns, snaps, dt, nz=9, zmax=0.12, Q=5, nk=16, n_f=5, n_s=3, n_l=5):
    """SSR at MULTIPLE maturities `snaps` from ONE propagation of length max(snaps). Bit-exact to
    [fused_ssr_ts_batched(theta, lam_fns[:n], n, ...) for n in snaps]: the per-maturity chains are PREFIXES
    (same init, same term-structure leverage lam_fns[0..n-1]), so snapshotting the longest chain at each
    maturity reproduces every shorter chain exactly. Turns sum(snaps) propagation-steps into max(snaps).
    lam_fns must have length >= max(snaps); the readout's return distribution (from lam_fns[0]) is shared."""
    nmax = max(snaps)
    ker = T.build_kernel(theta, dt, n_f, n_s, n_l)
    zg = torch.linspace(-zmax, zmax, nz); iz0 = nz // 2
    f0 = torch.arange(n_f).repeat_interleave(n_s * nz)
    s0 = torch.arange(n_s).repeat_interleave(nz).repeat(n_f)
    z0 = zg.repeat(n_f * n_s); B = n_f * n_s * nz
    st = (torch.ones(B, 1), z0[:, None].clone(), torch.full((B, 1), 1e-4), f0[:, None].long(), s0[:, None].long())
    # n-INDEPENDENT return distribution (uses lam_fns[0] = step-0 leverage, identical for every maturity)
    pi = T.stationary_pi(ker)
    lam0 = lam_fns[0](torch.zeros(1, dtype=ker["wl"].dtype, device=ker["wl"].device))[0]
    zq_np, wq_np = hermegauss(Q); wq_np = wq_np / wq_np.sum()
    dd = dict(dtype=ker["wl"].dtype, device=ker["wl"].device)
    zq = torch.as_tensor(zq_np, **dd); wq = torch.as_tensor(wq_np, **dd); wl = ker["wl"]
    Vl = lam0 ** 2 * ker["Vl"]; mtil = -0.5 * Vl + lam0 * ker["tilt"]
    Dl = mtil - torch.log((wl * torch.exp(mtil + 0.5 * Vl)).sum(-1, keepdim=True))
    mean_r = (wl * Dl).sum(-1)
    var = (pi * ((wl * (Dl ** 2 + Vl)).sum(-1) - mean_r ** 2)).sum()
    sd = torch.sqrt(torch.clamp(Vl, min=1e-16))
    r = Dl[..., None] + sd[..., None] * zq
    P = torch.einsum("lfa,lsb->fslab", ker["Tf"], ker["Ts"])
    wlq = wl[:, None] * wq[None, :]
    snapset = set(snaps); out = {}
    for k in range(nmax):
        st = propagate_batch(st, ker, lam_fns[k], nk)
        if (k + 1) in snapset:                                  # read SSR off the intermediate state
            sig, sk = atm_skew_batch(st[0], st[1], st[2], (k + 1) * dt)
            sigg = sig.reshape(n_f, n_s, nz); skw0 = sk.reshape(n_f, n_s, nz)[:, :, iz0]
            SIG = T.interp_batch(r, zg, sigg.reshape(n_f * n_s, nz)).reshape(n_f, n_s, n_f, n_s, n_l, Q)
            sig_dest = torch.einsum("fslab,abfslq->fslq", P, SIG)
            cov = (wlq * (sig_dest - sigg[:, :, iz0][..., None, None]) * (r - mean_r[..., None, None])).sum((-1, -2))
            out[k + 1] = (pi * cov).sum() / var / (pi * skw0).sum()
    return torch.stack([out[n] for n in snaps])


if __name__ == "__main__":
    import sys, os, time
    HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
    sys.path.insert(0, POC)
    import discslv_slv
    from slv_fast import propagate_vec, fused_ssr_exact_ts
    discslv_slv.propagate = propagate_vec
    from discslv_2f import TwoFactorSV
    from discslv_slv import Epi_V, nu_bar, raw_increment
    from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at

    DT = 1.0 / 52.0; NS = [1, 4, 13]
    kw = dict(nu_f=0.208, nu_s=0.411, nu_l=1.070, lam_skew=-0.303, lam_f=0.633, lam_s=2.092, kap_f=0.937, kap_s=2.706)
    OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
    chain = sanos_chain(OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"); sig = ref_vol(chain)
    gbar = solve_gbar(kw, sig, dt=DT)
    K = TwoFactorSV(gbar=gbar, dt=DT, n_f=5, n_s=3, n_l=5, **kw); EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    th0 = np.array([gbar, kw["nu_f"], kw["nu_s"], kw["nu_l"], kw["lam_skew"], kw["lam_f"], kw["lam_s"], kw["kap_f"], kw["kap_s"]])
    LT = {n: [T.lev_torch(l.coef, l.zmax, l.safety) for l in [leverage_at(chain, (k + 1) * DT, EV, dt=DT) for k in range(n)]] for n in NS}
    LV = {n: [leverage_at(chain, (k + 1) * DT, EV, dt=DT) for k in range(n)] for n in NS}

    print("=== batched SSR vs v1 ===")
    for n, lab in [(1, "1wk"), (4, "1m"), (13, "3m")]:
        v1 = fused_ssr_exact_ts(K, LV[n], n, EV, nub, Vlr, tiltr, 16, DT, nz=9)[0]
        tb = float(fused_ssr_ts_batched(torch.tensor(th0), LT[n], n, DT, nz=9)[0])
        print(f"  {lab}: v1={v1:.6f}  batched={tb:.6f}  |diff|={abs(v1 - tb):.2e}")

    def ssr_vec(th):
        return torch.stack([fused_ssr_ts_batched(th, LT[n], n, DT, nz=9)[0] for n in NS])
    t0 = time.time(); [ssr_vec(torch.tensor(th0)) for _ in range(1)]; tf = time.time() - t0
    t0 = time.time(); J = torch.func.jacrev(ssr_vec)(torch.tensor(th0)); tj = time.time() - t0
    print(f"\n  batched forward (3-mat): {tf:.2f}s   batched jacrev VECTOR Jac: {tj:.2f}s")
    print(f"  (loop was: forward 6.47s, jacrev 34.40s; v1 FD serial 107s / parallel-8core ~15s)")
