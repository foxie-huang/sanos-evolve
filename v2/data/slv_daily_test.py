#!/usr/bin/env python3
"""
Confirm daily dt (1/252) fixes the 1wk overshoot: at the SAME 1wk maturity, weekly is n=1 (LV
undiluted) but daily is n=5 (LV should dilute like 1m does). Also time a daily 3m readout (n=63)
to gauge the full-fit cost before committing the whole grid. Uses the fitted 1wk-3m theta.
"""
import sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment, fused_ssr_readout  # noqa: E402
from slv_fast import propagate_vec                                    # noqa: E402
discslv_slv.propagate = propagate_vec                                 # vectorized drop-in
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402

date = os.path.join(os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod")), "SPX-NDX-RUT-VIX_2015-06-01.json.gz")
chain = sanos_chain(date); sig_ref = ref_vol(chain)
kw = dict(nu_f=0.452, nu_s=0.463, nu_l=0.568, lam_skew=-1.521, lam_f=0.702, lam_s=2.991, kap_f=0.980, kap_s=2.533)


def readout(dt, n):
    K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=dt), dt=dt, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    lam = leverage_at(chain, n * dt, EV, dt=dt)
    return K, EV, nub, Vlr, tiltr, fused_ssr_readout(K, lam, n, EV, nub, Vlr, tiltr, 16, dt)


print("1wk (T~1/52): weekly n=1 vs daily n=5  --  does the undiluted LV dilute?")
for dt, n, lab in [(1.0 / 52.0, 1, "weekly"), (1.0 / 252.0, 5, "daily ")]:
    *_, (tot, lv, sv) = readout(dt, n)
    print(f"  {lab} (n={n}): fused={tot:.3f}  LV={lv:.3f}  SV={sv:.3f}   (emp 1wk ~2.03)")

dt = 1.0 / 252.0
K, EV, nub, Vlr, tiltr, _ = readout(dt, 63)
lam = leverage_at(chain, 63 * dt, EV, dt=dt)
t0 = time.time(); fused_ssr_readout(K, lam, 63, EV, nub, Vlr, tiltr, 16, dt)
print(f"\ndaily 3m readout (n=63): {time.time()-t0:.1f}s  -> x5 maturities x ~15 evals gauges the full fit")
