#!/usr/bin/env python3
"""Job 1 validation: exact closed-form beta (fused_ssr_exact) vs MC (fused_ssr_mc, exact-by-sampling)
vs the leading-order readout (fused_ssr_readout). Expect: exact ~ MC (both exact, exact has no
sampling noise), and both differ from the readout by the 9-27% we saw. Weekly, 1wk + 1m."""
import sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment, fused_ssr_readout, fused_ssr_mc  # noqa: E402
from slv_fast import propagate_vec, fused_ssr_exact                   # noqa: E402
discslv_slv.propagate = propagate_vec
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402

WK = 1.0 / 52.0
date = os.path.join(os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod")), "SPX-NDX-RUT-VIX_2015-06-01.json.gz")
chain = sanos_chain(date); sig_ref = ref_vol(chain)
kw = dict(nu_f=0.452, nu_s=0.463, nu_l=0.568, lam_skew=-1.521, lam_f=0.702, lam_s=2.991, kap_f=0.980, kap_s=2.533)
K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=WK), dt=WK, n_f=5, n_s=3, n_l=5, **kw)
EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)

print(f"{'mat':>4}{'readout':>9}{'MC':>8}{'exact':>8}{'exact-MC':>10}{'exact-readout':>15}")
for n, lab in [(1, "1wk"), (4, "1m")]:
    lam = leverage_at(chain, n * WK, EV, dt=WK)
    rd = fused_ssr_readout(K, lam, n, EV, nub, Vlr, tiltr, 16, WK)[0]
    mc = fused_ssr_mc(K, lam, n, EV, nub, Vlr, tiltr, 16, WK, N=400_000)
    ex = fused_ssr_exact(K, lam, n, EV, nub, Vlr, tiltr, 16, WK)[0]
    print(f"{lab:>4}{rd:>9.3f}{mc:>8.3f}{ex:>8.3f}{100*(ex-mc)/mc:>9.0f}%{100*(ex-rd)/rd:>14.0f}%", flush=True)
