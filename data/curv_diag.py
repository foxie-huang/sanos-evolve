#!/usr/bin/env python3
"""
DIAGNOSTIC (not a fit target): the FITTED model's ATM smile curvature + log-return excess kurtosis vs the
REALISED (data) ones. The GM marginals are closed-form, so both are analytic: kurtosis from the GM central
moments; ATM curvature d2(sigma)/dk^2 from a quadratic fit of the closed-form GM implied smile (same window
for model & data). Model marginal = the fusion propagated from spot at the fitted theta; data = SANOS marginal.
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import discslv_slv                                                   # noqa: E402
from slv_fast import propagate_vec                                   # noqa: E402
discslv_slv.propagate = propagate_vec
from discslv_slv import (propagate, marginal, initial_state, iv_at, E_nu_given_z_vec,   # noqa: E402
                         Epi_V, nu_bar, raw_increment)
from discslv_2f import TwoFactorSV                                   # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at   # noqa: E402
from slv_interp import interp_marginal                              # noqa: E402
import calibrate_slv_exact_ts as C                                  # noqa: E402

DT = 1.0 / 52.0; NAMES = C.NAMES
OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
# (date, ticker, fitted theta)
CASES = [
    ("2019-06-03", "SPX", [0.832, 0.362, 0.874, -0.302, 0.490, 2.426, 0.738, 2.834]),  # SPX reference
    ("2018-06-01", "NDX", [0.842, 0.737, 0.842, -2.350, 0.714, 3.263, 0.879, 2.077]),
    ("2020-06-01", "NDX", [0.687, 0.503, 0.186, -0.288, 0.424, 2.411, 0.845, 2.818]),
    ("2021-06-01", "NDX", [1.114, 0.518, 0.573, -0.861, 0.848, 3.478, 0.764, 2.214]),
    ("2022-06-01", "NDX", [1.199, 0.804, 0.844, -0.863, 0.599, 2.581, 0.668, 2.573]),
]


def exkurt(mu):                                                      # closed-form log-return excess kurtosis of a GM
    W, MU, SG = mu; m = np.sum(W * MU); d = MU - m
    mu2 = np.sum(W * (d ** 2 + SG ** 2))
    mu4 = np.sum(W * (d ** 4 + 6 * d ** 2 * SG ** 2 + 3 * SG ** 4))
    return float(mu4 / mu2 ** 2 - 3)


def nearest(chain, wk):                                            # raw SANOS marginal at the expiry nearest wk*DT
    T = wk * DT; j = min(range(len(chain)), key=lambda i: abs(chain[i][0] - T))
    return chain[j][0], chain[j][1]


def model_marg(chain, sig, theta, n):
    kw0 = dict(zip(NAMES, C.X0_MAP["dense"]))
    EV0 = Epi_V(TwoFactorSV(gbar=solve_gbar(kw0, sig, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw0))
    kw = dict(zip(NAMES, theta))
    K = TwoFactorSV(gbar=solve_gbar(kw, sig, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    st = initial_state(K)
    for k in range(1, n + 1):
        lf = leverage_at(chain, k * DT, EV0)
        st, _ = propagate(K, st, lambda mc, l=lf, cur=st: l(mc) ** 2 / np.clip(E_nu_given_z_vec(mc, cur, nub), 0.3, 3.0),
                          EV, nub, Vlr, tiltr, 16)
    return marginal(st)


print("Log-return EXCESS KURTOSIS, closed-form, RAW SANOS marginals (model from-spot @ fitted theta):")
print(f"{'date':>12}{'tk':>5}  |  {'1m (mdl/data)':>16}  {'3m (mdl/data)':>16}  {'6m (mdl/data)':>16}")
for date, tk, th in CASES:
    chain = sanos_chain(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz", ticker=tk); sig = ref_vol(chain)
    cells = []
    for wk in [4, 13, 26]:
        Te, md = nearest(chain, wk); n = max(1, round(Te / DT))
        mm = model_marg(chain, sig, th, n)
        cells.append(f"{exkurt(mm):>6.1f}/{exkurt(md):<6.1f}")
    print(f"{date:>12}{tk:>5}  |  {cells[0]:>16}  {cells[1]:>16}  {cells[2]:>16}")
