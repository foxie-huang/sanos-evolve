#!/usr/bin/env python3
"""How exactly does the leverage overlay carry the conditional variance? Measured, not assumed.

THE CLAIM UNDER TEST. The overlay is meant to realise the Gyongy relation sigma_LV^2 = sigma_Dupire^2
/ E[nu|z]. The paper used to state the finite-step version as an identity: scale the branch variances
by L^2 and the whole conditional variance scales by L^2. That is false at finite dt, and the reason is
visible in `propagate.py`: the step scales VARIANCES by L^2 but the skew tilt by L,

    V^LV_l = L^2 V_l,      mtil^LV_l = -1/2 L^2 V_l + L t_l,     t_l = lam_skew zeta_l sqrt(V_l),

so the branch dispersion of the drift does not scale as L^2. Writing Vbar(u) = E_l[V_l] +
Var_l(mtil_l) for the full one-step increment variance,

    Var_{K^L}(dZ | z,u) = L^2 Vbar(u) + R,
    R = (L^4 - L^2)/4 Var_l(V_l) - (L^3 - L^2) Cov_l(V_l, t_l).

Both terms vanish at L = 1. Since V_l = O(dt) and t_l = O(sqrt(dt)), R = O(dt^{3/2}) against a target
of O(dt), so the RELATIVE error is O(sqrt(dt)) -- which at dt = 1/52 is not automatically negligible
and has to be measured. The common re-lock that follows restores the forward exactly and shifts every
branch by the same constant, so it does not touch this remainder.

WHAT IS MEASURED. delta = Var_{K^L}(dZ | component, p) / (L^2 Vbar) - 1, with BOTH variances taken
over the joint (factor-node, branch) law the production lock normalises over -- weights w_x (x) w_l --
because that is the information set the implemented kernel conditions on. Evaluated at the nodes the
production step visits (the propagated component mixture and its price abscissas p = MU + SG z_p,
where the ladder's lambda is read) and weighted by the production weights W (x) w_p, so the summary
is probability-weighted rather than a maximum over tail nodes carrying no mass.

LADDER. Stage 2 dates were fitted on the mollified ladder, stage 1 dates were not, so the tag selects
both the record and the ladder. One process per tag, as in `jacobian_7free.py` and `skew_margin.py`.

    DEV=mps LAMH=1.0 python3 leverage_remainder.py --tag _n9
    DEV=mps LAMH=4.0 PILLARAWARE=0 python3 leverage_remainder.py --tag _dw9
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True, choices=["_n9", "_dw9"])
ap.add_argument("--out", default=None, help="json to write (default: vix_joint_refit/lev_remainder<tag>.json)")
A = ap.parse_args()

os.environ.setdefault("LADDER", "42")
os.environ.setdefault("VOVLAMTEN", "avg")
NEED = {"_n9": {"LAMH": "1.0"}, "_dw9": {"LAMH": "4.0", "PILLARAWARE": "0"}}[A.tag]
for k, v in NEED.items():
    os.environ[k] = v
DEV = os.environ.get("DEV", "mps")
sys.argv = [sys.argv[0], "cpu"]

import torch                                                          # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kernel_fast"))
sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                                   # noqa: E402
import consts, fkernel as kernel                                      # noqa: E402
import propagate as PR                                                # noqa: E402
import calibrate_slv_exact_ts as C                                    # noqa: E402
import end_to_end as E                                                # noqa: E402

SHIPPED = {"2012-06-01": "_dw9", "2016-06-01": "_dw9", "2017-06-01": "_dw9", "2018-06-01": "_dw9",
           "2019-06-03": "_n9", "2020-06-01": "_n9", "2021-06-01": "_n9", "2022-06-01": "_n9",
           "2024-06-03": "_n9"}


def step_delta(state, ker, lam):
    """delta and its production weight at every (component, price node, factor node) the step visits.

    Mirrors the front half of `propagate.step` exactly -- same gg clamp, same lambda floor, same
    quadrature -- and then does the branch algebra in closed form instead of propagating.
    """
    W, MU, SG, mf, ms, Vff, Vss, Vfs = state
    K = ker["K"]
    nu_f, nu_s, nu_l = ker["nu_f"], ker["nu_s"], ker["nu_l"]
    mX = nu_f * mf + nu_s * ms
    vX = torch.clamp(nu_f.pow(2) * Vff + nu_f * nu_s * Vfs * K.two + nu_s.pow(2) * Vss, min=1e-16)
    x = mX[..., None] + torch.sqrt(vX)[..., None] * K.zx                  # (B,N,nx)
    gg = torch.clamp(ker["gbar"] + x[..., None] + nu_l * K.zl, min=K.lo_g, max=K.hi_g)
    V = torch.exp(gg) * K.dt_t                                            # (B,N,nx,nl)  = V_l(u)
    sig = torch.sqrt(V)
    t = ker["lam_skew"] * K.zl * sig                                      # skew tilt
    p = MU[..., None] + SG[..., None] * K.zp                              # (B,N,np)
    L = torch.clamp(lam(p), min=1e-6)[..., None, None]                    # (B,N,np,1,1)

    # JOINT (k, l) moments. The production lock (`propagate.step`) normalises over the factor
    # quadrature AND the branches together, so the state the kernel is conditioned on is the
    # component, not the (component, factor node) pair. Taking Var/Cov branchwise at each factor
    # node -- as the first version of this script did -- measures a finer conditioning than the
    # implementation uses and understates the dispersion the overlay has to carry.
    wkl = (K.wx[:, None] * K.wl[None, :])                                 # (nx,nl), sums to 1
    E_ = lambda a: (wkl * a).sum((-2, -1))                                # noqa: E731
    EV, Et = E_(V), E_(t)
    VarV = E_(V.pow(2)) - EV.pow(2)
    Vart = E_(t.pow(2)) - Et.pow(2)
    Cvt = E_(V * t) - EV * Et
    Vbar = EV + (0.25 * VarV - Cvt + Vart)                                # (B,N)

    Lp = L[..., 0, 0]                                                     # (B,N,np)
    L2b = Lp.pow(2)
    L3b, L4b = L2b * Lp, L2b.pow(2)
    VarVb, Cvtb, Vbb = VarV[..., None], Cvt[..., None], Vbar[..., None]   # (B,N,1)
    R = 0.25 * (L4b - L2b) * VarVb - (L3b - L2b) * Cvtb                   # (B,N,np)
    delta = R / torch.clamp(L2b * Vbb, min=1e-30)

    w = W[..., None] * K.wp[None, None, :]
    return delta.reshape(-1), w.reshape(-1)


def wsummary(d, w):
    w = w / w.sum()
    o = np.argsort(np.abs(d))
    d, w = np.abs(d[o]), w[o]
    c = np.cumsum(w)
    q = lambda a: float(d[np.searchsorted(c, a)])                         # noqa: E731
    return dict(rms=float(np.sqrt((w * d ** 2).sum())), median=q(0.50),
                p99=q(0.99), p999=q(0.999), max=float(d[-1]))


def measure(date, th8):
    ctx, _c, _n = E.ctx_rebuilt(date, "SPX")
    LAM, SIG = ctx["LT"][max(ctx["LT"])], ctx["sig_ref"]
    K = consts.Consts(DEV, torch.float32)
    th = torch.tensor(th8, dtype=torch.float32, device=DEV)
    D_, Wt = [], []
    with torch.no_grad():
        g = kernel.solve_gbar(th, SIG, K)
        kk = kernel.build_kernel(kernel.th9(th, g, K), K)
        # one chain seeded on the STATIONARY factor law at spot 0, so every weight below is a
        # genuine probability: the mixture weights W and the two quadrature rules w_p, w_x.
        one = torch.ones(1, 1, dtype=K.dtype, device=K.device)
        z = torch.zeros(1, 1, dtype=K.dtype, device=K.device)
        st = (one.clone(), z.clone(), torch.full((1, 1), 1e-6, dtype=K.dtype, device=K.device),
              z.clone(), z.clone(), one.clone(), one.clone(),
              kernel.stationary_corr(kk).reshape(1, 1))
        for k in range(max(K.NS)):
            d, w = step_delta(st, kk, LAM[k])
            D_.append(d.cpu().numpy()); Wt.append(w.cpu().numpy())
            st = PR.step(st, kk, LAM[k])
    if DEV == "mps":
        torch.mps.empty_cache()
    return np.concatenate(D_), np.concatenate(Wt)


if __name__ == "__main__":
    rows, allD, allW = {}, [], []
    for date, tag in SHIPPED.items():
        if tag != A.tag:
            continue
        f = json.load(open(os.path.join(_P.DATA, f"fit_kf{tag}_{date}.json")))
        th8 = [f["theta"][n] for n in C.NAMES_N] + [f["kap_s"]]
        d, w = measure(date, th8)
        s = wsummary(d, w)
        rows[date] = s
        allD.append(d); allW.append(w)
        print(f"  {date} {tag:5s}  weighted RMS {100*s['rms']:6.3f}%   median {100*s['median']:6.3f}%"
              f"   p99 {100*s['p99']:6.3f}%   p99.9 {100*s['p999']:6.3f}%   max {100*s['max']:6.3f}%",
              flush=True)
    pool = wsummary(np.concatenate(allD), np.concatenate(allW))
    print(f"\n  pooled ({len(rows)} dates)  weighted RMS {100*pool['rms']:.3f}%"
          f"   median {100*pool['median']:.3f}%   p99 {100*pool['p99']:.3f}%"
          f"   p99.9 {100*pool['p999']:.3f}%   max {100*pool['max']:.3f}%")
    out = A.out or os.path.join(_P.DATA, f"lev_remainder{A.tag}.json")
    json.dump(dict(tag=A.tag, ladder=NEED, per_date=rows, pooled=pool,
                   definition="delta = Var_KL(dZ|z,u)/(L^2 Vbar(u)) - 1 at production nodes, "
                              "weights W (x) w_p (x) w_x"), open(out, "w"), indent=1)
    print(f"  wrote {out}")
