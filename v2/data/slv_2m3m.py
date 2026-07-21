#!/usr/bin/env python3
"""
Analyze the 2m-3m collapse. Decompose the exact fused SSR at the fitted theta into:
  SV  = fused_ssr_exact with a FLAT leverage (lam=1, no z-skew) -> pure kernel spot-vol response,
  Tot = fused_ssr_exact with the real SANOS leverage           -> LV + SV,
  LV  = Tot - SV                                                -> the leverage's z-skew contribution.
Also the leverage's ATM slope dlam/dz (the local-vol skew driving the LV) per maturity. Shows whether
the collapse is the LV washing out, the SV fading, or the leverage suppressing the SV.
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment                  # noqa: E402
from slv_fast import propagate_vec, fused_ssr_exact                   # noqa: E402
discslv_slv.propagate = propagate_vec
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402

DT = 1.0 / 52.0; NS = [1, 2, 4, 8, 13]; LABELS = ["1wk", "2wk", "1m", "2m", "3m"]
date = os.path.join(os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod")), "SPX-NDX-RUT-VIX_2015-06-01.json.gz")
chain = sanos_chain(date); sig_ref = ref_vol(chain)
kw = dict(nu_f=0.280, nu_s=0.293, nu_l=0.465, lam_skew=-2.113, lam_f=1.164, lam_s=2.460, kap_f=0.992, kap_s=2.518)
K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
flat = lambda z: np.ones_like(np.asarray(z, float))                   # flat leverage -> SV only

print(f"{'mat':>4}{'Tot(LV+SV)':>11}{'SV(flat)':>9}{'LV':>7}{'dlam/dz':>9}{'emp':>7}")
for n, lab in zip(NS, LABELS):
    lam = leverage_at(chain, n * DT, EV, dt=DT)
    tot = fused_ssr_exact(K, lam, n, EV, nub, Vlr, tiltr, 16, DT, nz=13)[0]
    sv = fused_ssr_exact(K, flat, n, EV, nub, Vlr, tiltr, 16, DT, nz=13)[0]
    dslope = (lam(0.02) - lam(-0.02)) / 0.04
    print(f"{lab:>4}{tot:>11.3f}{sv:>9.3f}{tot-sv:>7.3f}{float(dslope):>9.1f}", flush=True)
