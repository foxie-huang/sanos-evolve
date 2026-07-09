#!/usr/bin/env python3
"""
(b) dt-rescaling of the kernel's per-step transition rates.

The transition matrices Tf/Ts use kap_f/kap_s (and lam_f/lam_s) with NO dt factor, so they are
PER-STEP; changing dt with fixed kap changes the per-YEAR vol dynamics (the bug in the naive daily
switch). Fix: rescale kap so the per-year factor autocorrelation rho^(1/dt) is dt-invariant (i.e.
preserve the continuous OU reversion e^{-kappa} over a year). The amplitudes nu_f/nu_s do NOT enter
the transition (only the variance), so they're dt-independent; gbar is re-solved for the level.

Verify: the BARE-kernel SSR term structure (ssr_2f, tests the kernel dynamics) at matching calendar
maturities should agree weekly vs rescaled-daily. NOTE lam_f/lam_s also enter the transition (the
leverage coupling) -- held fixed here, so this preserves the vol MEMORY; whether the leverage also
needs a dt-scaling is what the SSR-match check reveals (a residual mismatch => lam needs scaling too).
"""
import sys, os
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV, ssr_2f                          # noqa: E402

WK = 1.0 / 52.0; DY = 1.0 / 252.0
kw = dict(nu_f=0.452, nu_s=0.463, nu_l=0.568, lam_skew=-1.521, lam_f=0.702, lam_s=2.991, kap_f=0.980, kap_s=2.533)


def _autocorr(P, z):
    pi = np.ones(len(z)) / len(z)
    for _ in range(2000):
        pi = pi @ P
    pi /= pi.sum()
    zc = z - float(np.sum(pi * z))
    return float(np.sum(pi * zc * (P @ zc)) / np.sum(pi * zc * zc))


def factor_autocorr(K):
    """Per-step lag-1 autocorrelation of the fast + slow log-vol factors (l-averaged transitions)."""
    Pf = np.einsum('l,lfg->fg', K.wl, K.Tf); Ps = np.einsum('l,lsg->sg', K.wl, K.Ts)
    return _autocorr(Pf, K.zf), _autocorr(Ps, K.zs)


def _rho_at_kap(kap, which, base, dt):
    d = dict(base); d['kap_' + which] = kap
    rf, rs = factor_autocorr(TwoFactorSV(gbar=-5.0, dt=dt, n_f=5, n_s=3, n_l=5, **d))
    return rf if which == 'f' else rs


def rescale_kap(base, dt_from, dt_to):
    """kap_f/kap_s at dt_to preserving the per-year autocorrelation rho^(1/dt)."""
    out = dict(base)
    for which, lo, hi in [('f', 0.02, 4.0), ('s', 0.05, 8.0)]:
        rho_from = _rho_at_kap(base['kap_' + which], which, base, dt_from)
        rho_to = rho_from ** (dt_to / dt_from)                        # rho_from^(1/dt_from) = rho_to^(1/dt_to)
        out['kap_' + which] = brentq(lambda k: _rho_at_kap(k, which, base, dt_to) - rho_to, lo, hi)
    return out


if __name__ == "__main__":
    Kw = TwoFactorSV(gbar=-5.0, dt=WK, n_f=5, n_s=3, n_l=5, **kw)
    rf_w, rs_w = factor_autocorr(Kw)
    print(f"weekly  per-step autocorr: fast {rf_w:.4f}  slow {rs_w:.4f}   | per-YEAR fast {rf_w**52:.3f} slow {rs_w**52:.3f}")
    kw_dy = rescale_kap(kw, WK, DY)
    Kd = TwoFactorSV(gbar=-5.0, dt=DY, n_f=5, n_s=3, n_l=5, **kw_dy)
    rf_d, rs_d = factor_autocorr(Kd)
    print(f"daily   per-step autocorr: fast {rf_d:.4f}  slow {rs_d:.4f}   | per-YEAR fast {rf_d**252:.3f} slow {rs_d**252:.3f}")
    print(f"\nrescaled kap:  kap_f {kw['kap_f']:.3f} -> {kw_dy['kap_f']:.3f}   kap_s {kw['kap_s']:.3f} -> {kw_dy['kap_s']:.3f}\n")
    print(f"{'mat':>4}{'weekly SSR':>12}{'daily-rescaled':>16}{'gap':>7}")
    for lab, nw, nd in [("1m", 4, 21), ("3m", 13, 63), ("6m", 26, 126)]:
        sw = ssr_2f(Kw, nw, nk=16)[0]; sd = ssr_2f(Kd, nd, nk=16)[0]
        print(f"{lab:>4}{sw:>12.3f}{sd:>16.3f}{100*(sd-sw)/sw:>6.0f}%", flush=True)
