#!/usr/bin/env python3
"""
slv_termstruct.py -- test the leverage TERM-STRUCTURE hypothesis for the 2m-3m LV collapse.

Compare the fused SSR computed with (a) a SINGLE frozen T-maturity leverage at every step (what
fused_ssr_readout does) vs (b) a PER-STEP leverage lam_fns[k] = leverage at maturity (k+1)*dt
(steep short horizons, flat long). If (b)'s LV stops collapsing at 2m/3m, the term structure is the
cause. Evaluated at the fitted 1wk-3m theta.
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment, fused_ssr_readout  # noqa: E402
from slv_fast import propagate_vec, fused_ssr_readout_ts              # noqa: E402
discslv_slv.propagate = propagate_vec                                 # vectorized drop-in
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402

DT = 1.0 / 52.0
date = os.path.join(os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod")), "SPX-NDX-RUT-VIX_2015-06-01.json.gz")
chain = sanos_chain(date); sig_ref = ref_vol(chain)
kw = dict(nu_f=0.452, nu_s=0.463, nu_l=0.568, lam_skew=-1.521, lam_f=0.702, lam_s=2.991, kap_f=0.980, kap_s=2.533)
K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K); nk = 16
emp = {8: 1.538, 13: 1.457}                                          # de-inflated 2m / 3m targets

print(f"{'T':>4}{'mode':>9}{'fused':>8}{'LV':>8}{'SV':>8}{'emp':>7}")
for n, lab in [(8, "2m"), (13, "3m")]:
    t1, l1, s1 = fused_ssr_readout(K, leverage_at(chain, n * DT, EV), n, EV, nub, Vlr, tiltr, nk, DT)
    lam_fns = [leverage_at(chain, (k + 1) * DT, EV) for k in range(n)]
    t2, l2, s2 = fused_ssr_readout_ts(K, lam_fns, n, EV, nub, Vlr, tiltr, nk, DT)
    print(f"{lab:>4}{'single':>9}{t1:>8.3f}{l1:>8.3f}{s1:>8.3f}{emp[n]:>7.2f}")
    print(f"{lab:>4}{'per-step':>9}{t2:>8.3f}{l2:>8.3f}{s2:>8.3f}{emp[n]:>7.2f}", flush=True)
