#!/usr/bin/env python3
"""STAGE 3+4 validation: full SSR in torch vs v1 (forward), then the Jacobian via torch.func.jacrev vs
finite-difference (values + timing) -- the payoff: 8 FD evals -> 1 reverse pass."""
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

DT = 1.0 / 52.0
kw = dict(nu_f=0.208, nu_s=0.411, nu_l=1.070, lam_skew=-0.303, lam_f=0.633, lam_s=2.092, kap_f=0.937, kap_s=2.706)
OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"
chain = sanos_chain(OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"); sig = ref_vol(chain)
gbar = solve_gbar(kw, sig, dt=DT)
K = TwoFactorSV(gbar=gbar, dt=DT, n_f=5, n_s=3, n_l=5, **kw); EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
th0 = np.array([gbar, kw["nu_f"], kw["nu_s"], kw["nu_l"], kw["lam_skew"], kw["lam_f"], kw["lam_s"], kw["kap_f"], kw["kap_s"]])

print("=== Stage 3: full SSR, torch vs v1 (nz=9) ===")
for n, lab in [(1, "1wk"), (4, "1m"), (13, "3m")]:
    lam_v1 = [leverage_at(chain, (k + 1) * DT, EV, dt=DT) for k in range(n)]
    v1 = fused_ssr_exact_ts(K, lam_v1, n, EV, nub, Vlr, tiltr, 16, DT, nz=9)[0]
    lam_t = [T.lev_torch(l.coef, l.zmax, l.safety) for l in lam_v1]
    tt = float(T.fused_ssr_ts(torch.tensor(th0), lam_t, n, DT, nz=9)[0])
    print(f"  {lab}: v1={v1:.6f}  torch={tt:.6f}  |diff|={abs(v1-tt):.2e}")

# Stage 4: jacobian at n=4 (1m)
n = 4; lam_v1 = [leverage_at(chain, (k + 1) * DT, EV, dt=DT) for k in range(n)]
lam_t = [T.lev_torch(l.coef, l.zmax, l.safety) for l in lam_v1]
def ssr(th):
    return T.fused_ssr_ts(th, lam_t, n, DT, nz=9)[0]

print("\n=== Stage 4: Jacobian d(SSR)/d(theta) at 1m ===")
tha = torch.tensor(th0, requires_grad=True)
t0 = time.time(); J_ad = torch.func.jacrev(ssr)(tha).detach().numpy(); t_ad = time.time() - t0
t0 = time.time(); J_fd = np.zeros(9); h = 1e-6
for i in range(9):
    tp = th0.copy(); tp[i] += h; tm = th0.copy(); tm[i] -= h
    with torch.no_grad():
        J_fd[i] = float((ssr(torch.tensor(tp)) - ssr(torch.tensor(tm))) / (2 * h))
t_fd = time.time() - t0
print(f"  {'param':>9}{'jacrev':>13}{'FD':>13}{'rel.err':>9}")
for i, nm in enumerate(T.PNAMES):
    print(f"  {nm:>9}{J_ad[i]:>13.5f}{J_fd[i]:>13.5f}{abs(J_ad[i]-J_fd[i])/(abs(J_fd[i])+1e-9):>8.1%}")
print(f"\n  jacrev (1 reverse pass): {t_ad:.2f}s   |   FD (8 evals): {t_fd:.2f}s   |   speedup {t_fd/t_ad:.1f}x")
