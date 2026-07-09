#!/usr/bin/env python3
"""
Honest e2e benchmark: does AD actually speed up the CALIBRATION? The fit is least-squares (residual
vector m vs params n=8). Reverse-mode AD (jacrev) wins big only for SCALAR outputs (1 backward); for a
vector Jacobian it costs ~m backward passes. So we time all the shapes that matter:
  - torch forward (the 3-maturity SSR vector)
  - jacrev of the VECTOR SSR (the least-squares Jacobian: m=3 outputs -> ~3 backward passes)
  - jacrev of a SCALAR loss (sum-of-squares proxy -> 1 backward pass -- the case AD is built for)
  - v1 numpy forward + FD Jacobian (n=8 evals) -- the incumbent
"""
import sys, os, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import discslv_torch as T
import discslv_slv
from slv_fast import propagate_vec, fused_ssr_exact_ts
discslv_slv.propagate = propagate_vec
from discslv_2f import TwoFactorSV
from discslv_slv import Epi_V, nu_bar, raw_increment
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at

DT = 1.0 / 52.0; NS = [1, 4, 13]
kw = dict(nu_f=0.208, nu_s=0.411, nu_l=1.070, lam_skew=-0.303, lam_f=0.633, lam_s=2.092, kap_f=0.937, kap_s=2.706)
OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"
chain = sanos_chain(OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"); sig = ref_vol(chain)
gbar = solve_gbar(kw, sig, dt=DT)
K = TwoFactorSV(gbar=gbar, dt=DT, n_f=5, n_s=3, n_l=5, **kw); EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
th0 = np.array([gbar, kw["nu_f"], kw["nu_s"], kw["nu_l"], kw["lam_skew"], kw["lam_f"], kw["lam_s"], kw["kap_f"], kw["kap_s"]])
LAM_T = {n: [T.lev_torch(l.coef, l.zmax, l.safety) for l in [leverage_at(chain, (k + 1) * DT, EV, dt=DT) for k in range(n)]] for n in NS}
LAM_V1 = {n: [leverage_at(chain, (k + 1) * DT, EV, dt=DT) for k in range(n)] for n in NS}


def ssr_vec_t(th):
    return torch.stack([T.fused_ssr_ts(th, LAM_T[n], n, DT, nz=9)[0] for n in NS])


def timeit(fn, k=1):
    t0 = time.time()
    for _ in range(k):
        r = fn()
    return (time.time() - t0) / k, r


tha = torch.tensor(th0)
tf, _ = timeit(lambda: ssr_vec_t(tha))
tjv, _ = timeit(lambda: torch.func.jacrev(ssr_vec_t)(torch.tensor(th0)))          # vector Jacobian (m=3 backward)
tjs, _ = timeit(lambda: torch.func.jacrev(lambda th: (ssr_vec_t(th) ** 2).sum())(torch.tensor(th0)))  # scalar grad (1 backward)


def ssr_vec_np():
    return np.array([fused_ssr_exact_ts(K, LAM_V1[n], n, EV, nub, Vlr, tiltr, 16, DT, nz=9)[0] for n in NS])


tfn, _ = timeit(ssr_vec_np)
def fd_np():
    J = np.zeros((len(NS), 9)); h = 1e-6
    f0 = ssr_vec_np()  # (reused base)
    for i in range(9):
        tp = th0.copy(); tp[i] += h
        kwp = dict(zip(["gbar"] + list(kw.keys()), tp)) if False else None
        # rebuild v1 kernel at tp
        kp = dict(zip(["gbar", "nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"], tp))
        Kp = TwoFactorSV(gbar=kp["gbar"], dt=DT, n_f=5, n_s=3, n_l=5, **{q: kp[q] for q in kw})
        EVp = Epi_V(Kp); nubp = nu_bar(Kp, EVp); Vlp, tip = raw_increment(Kp)
        fp = np.array([fused_ssr_exact_ts(Kp, LAM_V1[n], n, EVp, nubp, Vlp, tip, 16, DT, nz=9)[0] for n in NS])
        J[:, i] = (fp - f0) / h
    return J


tfd, _ = timeit(fd_np)

print(f"                                      seconds")
print(f"  torch forward (3-maturity SSR)     {tf:8.2f}")
print(f"  torch jacrev VECTOR Jac (3x8)      {tjv:8.2f}   (least-squares Jacobian -- the fit's actual need)")
print(f"  torch jacrev SCALAR grad (8)       {tjs:8.2f}   (1 backward -- what AD is built for)")
print(f"  v1 numpy forward                   {tfn:8.2f}")
print(f"  v1 numpy FD Jacobian (8 evals)     {tfd:8.2f}   (the incumbent)")
print(f"\n  AD vector-Jac vs v1 FD:  {tfd/tjv:.2f}x   |   AD scalar-grad vs v1 FD:  {tfd/tjs:.2f}x")
