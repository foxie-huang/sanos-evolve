#!/usr/bin/env python3
"""
Does the smile CURVATURE carry vol-of-vol info in the FUSION model? Crux test before building the backbone.
With the leverage held FIXED (as in the fit, EV0 from the reference theta), sweep nu (nu_f,nu_s) in the
kernel and read the model's ATM smile curvature at 1m/3m -- does it move with nu, and does the NDX data
curvature fall inside the swept range (=> nu identifiable from curvature)? If curvature is nu-INVARIANT it is
statics-only (circular, backbone useless); if nu-SENSITIVE it firms nu.
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
DATE = "2021-06-01"; THETA = [1.114, 0.518, 0.573, -0.861, 0.848, 3.478, 0.764, 2.214]   # NDX 2021 fit


def curv_of(mu, T, dm=8e-3):
    a = iv_at(mu, T, [0.0])[0]; lo, hi = iv_at(mu, T, [-dm, dm])
    return float((hi - 2 * a + lo) / dm ** 2)                        # d2 sigma / d logm^2 at ATM


chain = sanos_chain(f"{OUT}/SPX-NDX-RUT-VIX_{DATE}.json.gz", ticker="NDX"); sig = ref_vol(chain)
kw0 = dict(zip(NAMES, C.X0_MAP["dense"]))
EV0 = Epi_V(TwoFactorSV(gbar=solve_gbar(kw0, sig, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw0))   # FIXED leverage norm
LEV = {k: leverage_at(chain, k * DT, EV0) for k in range(1, 14)}

print(f"NDX {DATE}  sig_ref={sig:.3f}")
print("DATA smile curvature (NDX SANOS marginal):")
for n, lab in [(4, "1m"), (13, "3m")]:
    print(f"   {lab}: {curv_of(interp_marginal(chain, n * DT), n * DT):+.2f}")

print("\nMODEL curvature vs nu (leverage FIXED, kernel nu scaled):")
print(f"   {'nu_scale':>9}{'1m':>9}{'3m':>9}")
for sc in [0.5, 1.0, 1.6]:
    kw = dict(zip(NAMES, THETA)); kw["nu_f"] *= sc; kw["nu_s"] *= sc
    K = TwoFactorSV(gbar=solve_gbar(kw, sig, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    row = []
    for n in [4, 13]:
        st = initial_state(K)
        for k in range(1, n + 1):
            lf = LEV[k]
            st, _ = propagate(K, st, lambda mc, l=lf, cur=st: l(mc) ** 2 / np.clip(E_nu_given_z_vec(mc, cur, nub), 0.3, 3.0),
                              EV, nub, Vlr, tiltr, 16)
        row.append(curv_of(marginal(st), n * DT))
    print(f"   {sc:>9.1f}{row[0]:>+9.2f}{row[1]:>+9.2f}")
