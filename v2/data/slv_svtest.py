#!/usr/bin/env python3
"""Is the Gyongy-decoupled fit valid? Compare the fused SV part (leverage applied, lambda~1) to the
bare kernel ssr_2f at the default theta. If SV ~ ssr_2f, the fit can use fast ssr_2f for the SV and
add the (precomputed, weakly theta-dependent) LV -- instead of the ~5h full fused least_squares."""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV, ssr_2f                          # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment, fused_ssr_readout   # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at  # noqa: E402

DT = 1.0 / 52.0; NS = [4, 13, 26, 52]; LABELS = ["1m", "3m", "6m", "1y"]
date = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod/SPX-NDX-RUT-VIX_2015-06-01.json.gz"
chain = sanos_chain(date); sig_ref = ref_vol(chain)
kw = dict(nu_f=0.43, nu_s=0.50, lam_skew=-1.48, lam_f=0.98, lam_s=1.65, kap_f=1.00, kap_s=2.34, nu_l=0.14)
K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
for nk in [16, 32, 48]:
    print(f"\n--- nk={nk} (finer grid -> gap should shrink toward benign ~-13% if collocation) ---")
    print(f"{'T':>4}{'fusedSV':>10}{'ssr_2f':>10}{'gap':>8}{'LV':>8}")
    for n, lab in zip(NS, LABELS):
        lam = leverage_at(chain, n * DT, EV)
        _, lv, sv = fused_ssr_readout(K, lam, n, EV, nub, Vlr, tiltr, nk, DT)
        bare = ssr_2f(K, n, nk=nk)[0]
        print(f"{lab:>4}{sv:>10.3f}{bare:>10.3f}{100*(sv-bare)/bare:>7.0f}%{lv:>8.3f}", flush=True)
