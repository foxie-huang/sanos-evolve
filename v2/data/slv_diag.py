#!/usr/bin/env python3
"""Diagnose the leverage skew: is the frozen-shape interpolation's lambda(z) monotone (carries the
SANOS skew -> non-zero LV SSR) or symmetric (skew washed out -> LV~0)? Compare dlambda/dz@0 to the
SANOS marginal's own implied-vol skew (the statics skew the leverage should track)."""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                   # noqa: E402
from discslv_slv import Epi_V                                       # noqa: E402
from discslv import GMM                                             # noqa: E402
from slv_wire import sanos_chain, ref_vol, leverage_at, solve_gbar  # noqa: E402
from slv_interp import interp_marginal                              # noqa: E402

DT = 1.0 / 52.0
date = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod/SPX-NDX-RUT-VIX_2015-06-01.json.gz"
chain = sanos_chain(date); sig_ref = ref_vol(chain)
kw = dict(nu_f=0.43, nu_s=0.50, lam_skew=-1.48, lam_f=0.98, lam_s=1.65, kap_f=1.00, kap_s=2.34, nu_l=0.14)
gbar = solve_gbar(kw, sig_ref)
K = TwoFactorSV(gbar=gbar, dt=DT, n_f=5, n_s=3, n_l=5, **kw)
EV = Epi_V(K)
print(f"sig_ref={sig_ref:.3f}  gbar={gbar:.2f}  EV={EV:.2e}  (target sig_ref^2*dt={sig_ref**2*DT:.2e})\n")


def sanos_skew(T, dm=6e-3):
    mu = interp_marginal(chain, T); g = GMM(mu[0], mu[1], mu[2], F=1.0); F = g.forward()
    iv = lambda k: float(g.implied_vol(F * np.exp(k), T)[0])
    return iv(0.0), (iv(dm) - iv(-dm)) / (2 * dm)


zg = np.linspace(-0.10, 0.10, 11)
for lab, n in [("1m", 4), ("3m", 13), ("6m", 26), ("1y", 52)]:
    T = n * DT; lam = leverage_at(chain, T, EV)
    lams = np.array([lam(z) for z in zg])
    dslope = (lam(0.02) - lam(-0.02)) / 0.04                        # dlambda/dz at 0
    vol, skew = sanos_skew(T)
    print(f"{lab}: lam(z=-.1..+.1) = " + " ".join(f"{v:.2f}" for v in lams))
    print(f"     dlam/dz@0 = {dslope:+.2f}   | SANOS ATM vol {vol:.3f}, impl skew {skew:+.2f}")
