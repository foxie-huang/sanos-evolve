#!/usr/bin/env python3
"""
Is the 1wk overshoot GENUINE or a dt artifact? Now that kap is rescaled (so the kernel dynamics are
preserved across dt, per slv_dtscale), compare the 1wk FUSED SSR weekly (n=1) vs daily-rescaled
(n=5). With kap correct, any remaining difference is the pure LV-dilution effect (undiluted single
step vs 5 diluted steps). If they agree, the 1wk overshoot is a genuine model feature; if daily
drops, the LV genuinely dilutes and weekly n=1 was overstating it.
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment, fused_ssr_readout  # noqa: E402
from slv_fast import propagate_vec                                    # noqa: E402
discslv_slv.propagate = propagate_vec
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402
from slv_dtscale import rescale_kap, WK, DY                           # noqa: E402

date = os.path.join(os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod")), "SPX-NDX-RUT-VIX_2015-06-01.json.gz")
chain = sanos_chain(date); sig_ref = ref_vol(chain)
kw = dict(nu_f=0.452, nu_s=0.463, nu_l=0.568, lam_skew=-1.521, lam_f=0.702, lam_s=2.991, kap_f=0.980, kap_s=2.533)

print("1wk FUSED SSR (kap rescaled so dynamics are preserved -> isolates LV dilution):")
for dt, n, kwx, lab in [(WK, 1, kw, "weekly       n=1"), (DY, 5, rescale_kap(kw, WK, DY), "daily-rescaled n=5")]:
    K = TwoFactorSV(gbar=solve_gbar(kwx, sig_ref, dt=dt), dt=dt, n_f=5, n_s=3, n_l=5, **kwx)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    tot, lv, sv = fused_ssr_readout(K, leverage_at(chain, n * dt, EV, dt=dt), n, EV, nub, Vlr, tiltr, 16, dt)
    print(f"  {lab}: fused={tot:.3f}  LV={lv:.3f}  SV={sv:.3f}   (emp 1wk ~2.0)", flush=True)
