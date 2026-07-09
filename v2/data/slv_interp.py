#!/usr/bin/env python3
"""
slv_interp.py -- OU-semigroup marginal interpolation (paper Sec. interp, §1197).

The latent log-variance splits as x_t = g_0(t) + X^F_t + X^S_t: a time-inhomogeneous
forward-variance LEVEL g_0 plus a time-homogeneous OU generator. So one interpolates only the
1-D level curve (here the total variance, monotone -> calendar no-arb automatic) and carries the
time-homogeneous SHAPE. The SANOS marginals are GM (not OU-decomposed), so the tractable
realization is: monotone-interpolate the total variance onto the target maturity and rescale the
bracketing-expiry standardized GM shape to it. Exact enough for a stable LOCAL (one-dt) discrete
Dupire; the one approximation vs the exact OU semigroup is the frozen within-bracket shape (the
skew does not flatten inside a bracket -- negligible over a single dt).
"""
import numpy as np


def total_var(mu):
    """Total variance of log-spot for a GM marginal (W, MU, SG)."""
    W, MU, SG = mu
    m = float(np.sum(W * MU))
    return float(np.sum(W * (MU ** 2 + SG ** 2)) - m ** 2)


def interp_marginal(chain, T):
    """Resample the SANOS marginal chain [(T_j,(W,MU,SG))] to maturity T: monotone-interpolated
    total variance + bracketing-expiry standardized shape, martingale re-locked."""
    Ts = np.array([c[0] for c in chain]); Vs = np.array([total_var(c[1]) for c in chain])
    V = float(np.interp(T, Ts, Vs))                                # monotone level g_0(T)
    j = int(np.argmin(np.abs(Ts - T)))                             # nearest expiry supplies the shape
    W, MU, SG = chain[j][1]
    m = float(np.sum(W * MU)); s = np.sqrt(max(total_var(chain[j][1]), 1e-12)); snew = np.sqrt(max(V, 1e-12))
    MUn = m + (MU - m) * snew / s; SGn = SG * snew / s             # rescale standardized shape to V
    MUn = MUn - np.log(np.sum(W * np.exp(MUn + 0.5 * SGn ** 2)))   # re-lock the martingale
    return (W, MUn, SGn)
