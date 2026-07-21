#!/usr/bin/env python3
"""
Ground-truth check: the closed-form fused_ssr_readout (lv+sv, leading-order in the step) vs
fused_ssr_mc (direct simulation, exact -- no decomposition, no truncation, no collocation).
Run at 1wk and 1m, weekly and daily-rescaled (kap rescaled so the continuous model is preserved).

Tells us: (i) how far the closed-form is from exact at these step sizes, and (ii) whether the EXACT
SSR is dt-invariant (with kap rescaled) -- if MC is dt-invariant while the readout isn't, the
readout's dt-dependence (the LV dilution) is confirmed as a closed-form artifact, not the model.
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment, fused_ssr_readout, fused_ssr_mc  # noqa: E402
from slv_fast import propagate_vec                                    # noqa: E402
discslv_slv.propagate = propagate_vec
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402
from slv_dtscale import rescale_kap, WK, DY                           # noqa: E402

date = os.path.join(os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod")), "SPX-NDX-RUT-VIX_2015-06-01.json.gz")
chain = sanos_chain(date); sig_ref = ref_vol(chain)
kw = dict(nu_f=0.452, nu_s=0.463, nu_l=0.568, lam_skew=-1.521, lam_f=0.702, lam_s=2.991, kap_f=0.980, kap_s=2.533)

print(f"{'':>16}{'readout':>9}{'MC (exact)':>12}{'gap':>7}")
for dt, kwx, mats, lab in [(WK, kw, [(1, "1wk"), (4, "1m")], "weekly"),
                           (DY, rescale_kap(kw, WK, DY), [(5, "1wk"), (21, "1m")], "daily-rescaled")]:
    K = TwoFactorSV(gbar=solve_gbar(kwx, sig_ref, dt=dt), dt=dt, n_f=5, n_s=3, n_l=5, **kwx)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    for n, mlab in mats:
        lam = leverage_at(chain, n * dt, EV, dt=dt)
        rd = fused_ssr_readout(K, lam, n, EV, nub, Vlr, tiltr, 16, dt)[0]
        mc = fused_ssr_mc(K, lam, n, EV, nub, Vlr, tiltr, 16, dt, N=300_000)
        print(f"{lab+' '+mlab:>16}{rd:>9.3f}{mc:>12.3f}{100*(rd-mc)/mc:>6.0f}%", flush=True)
