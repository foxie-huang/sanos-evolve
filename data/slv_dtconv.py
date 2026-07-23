#!/usr/bin/env python3
"""
Job 2 -- structured dt-scaling check via a convergence study.

The exact SSR at a FIXED maturity must be dt-invariant if the model has a proper continuous limit
and the params are scaled correctly. Compute the exact 1m SSR (fused_ssr_exact) at dt = 1/52, 1/104,
1/252 (weekly, bi-weekly, daily), each with kap rescaled from the weekly baseline (nu unscaled, gbar
re-solved). Also report the invariants that SHOULD be dt-independent (stationary log-var spread,
per-year autocorrelation, per-step spot-vol covariance) to localize any drift:
  - converges + invariants flat  -> kap-rescaling suffices;
  - drifts / an invariant moves   -> that parameter (lam_f/lam_s/lam_skew) needs its own dt-scaling.
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment, stationary_pi   # noqa: E402
from slv_fast import propagate_vec, fused_ssr_exact                   # noqa: E402
discslv_slv.propagate = propagate_vec
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402
from slv_dtscale import rescale_kap, factor_autocorr, WK              # noqa: E402

date = os.path.join(os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod")), "SPX-NDX-RUT-VIX_2015-06-01.json.gz")
chain = sanos_chain(date); sig_ref = ref_vol(chain)
kw = dict(nu_f=0.452, nu_s=0.463, nu_l=0.568, lam_skew=-1.521, lam_f=0.702, lam_s=2.991, kap_f=0.980, kap_s=2.533)


def spot_vol_cov(K):
    """Per-step Cov(return, log-variance change) -- the leverage's spot-vol coupling."""
    pi = stationary_pi(K); zf, zs = K.zf, K.zs; cov = 0.0
    for f in range(K.n_f):
        for s in range(K.n_s):
            d = K.D[f, s]
            for l in range(K.n_l):
                Pf, Ps = K.trans_f(l, f), K.trans_s(l, s)
                dx = sum(Pf[fp] * Ps[sp] * (K.nu_f * (zf[fp] - zf[f]) + K.nu_s * (zs[sp] - zs[s]))
                         for fp in range(K.n_f) for sp in range(K.n_s))
                cov += pi[f, s] * K.wl[l] * d[l] * dx
    return cov


def stationary_spread(K):
    pi = stationary_pi(K); zf, zs = K.zf, K.zs
    x = np.array([[K.nu_f * zf[f] + K.nu_s * zs[s] for s in range(K.n_s)] for f in range(K.n_f)])
    return float(np.sum(pi * x ** 2) - np.sum(pi * x) ** 2)


print(f"{'dt':>8}{'n(1m)':>6}{'exactSSR':>10}{'spread':>9}{'acf_f/yr':>10}{'acf_s/yr':>10}{'spotvolCov':>12}")
for dt in [WK, 1.0 / 104.0, 1.0 / 252.0]:
    kwx = kw if abs(dt - WK) < 1e-12 else rescale_kap(kw, WK, dt)
    n1m = int(round((1.0 / 12.0) / dt))
    K = TwoFactorSV(gbar=solve_gbar(kwx, sig_ref, dt=dt), dt=dt, n_f=5, n_s=3, n_l=5, **kwx)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    ssr = fused_ssr_exact(K, leverage_at(chain, n1m * dt, EV, dt=dt), n1m, EV, nub, Vlr, tiltr, 16, dt)[0]
    rf, rs = factor_autocorr(K)
    print(f"{dt:>8.5f}{n1m:>6}{ssr:>10.3f}{stationary_spread(K):>9.4f}{rf**(1/dt):>10.4f}"
          f"{rs**(1/dt):>10.4f}{spot_vol_cov(K):>12.2e}", flush=True)
