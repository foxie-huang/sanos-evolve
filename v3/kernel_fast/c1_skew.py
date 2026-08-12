#!/usr/bin/env python3
"""Is lambda's z-slope c1(T) consistent with the MARKET's skew term structure -- and what does
smoothing it cost in observable units? No fitting; arithmetic on the LP marginals already on disk.

WHY THIS IS THE RIGHT TEST. `log lambda_k(z) = c1 z + c0` with DEG=1, so c1 IS the entire
strike-dependence the overlay supplies: c1(T) is a SKEW TERM STRUCTURE, an observable, not a
numerical coefficient. Every comparison in 6e.15-6e.21 ranked c1-smoothers on a joint SSR/vov cost,
which cannot see strike calibration at all. This measures the thing that objective is blind to.

Three questions, in order:

  (1) IS THE ROUGHNESS IN c1(T) PRESENT IN THE MARKET? Implied-vol skew is computed from the LP
      marginal AT EACH PILLAR -- no weekly interpolation involved. If the market skew term structure
      is smooth where c1(T) jumps, the jump is numerical (6e.13's claim) and there is something to
      remove. If the market is rough there too, smoothing is destroying signal.

  (2) HOW BIG IS THE SMOOTHING PERTURBATION, in vol terms? A change dc1 multiplies the local variance
      at log-moneyness z by exp(dc1*z), i.e. local VOL by exp(dc1*z/2). Reported at z = +-1 SD of the
      maturity, as a percentage of local vol -- directly comparable to the ~26 ATM vol bp bid-ask bar
      that verify_blend.py uses.

  (3) DOES SMOOTHING MOVE c1 TOWARD OR AWAY FROM THE MARKET SKEW? If c1(T) is a skew, then the
      smoothed version should track the market's own skew term structure BETTER, not worse. If it
      moves away, the smoothing is buying SSR/vov fit by mis-calibrating strikes -- which is exactly
      the 6e.22 objection, made quantitative.

    python3 c1_skew.py
    python3 c1_skew.py 2016-06-01
"""
import os
import sys

DATES = [a for a in sys.argv[1:] if a[:2] == "20"] or \
        ["2012-06-01", "2016-06-01", "2017-06-01", "2018-06-01", "2019-06-03",
         "2020-06-01", "2021-06-01", "2022-06-01", "2024-06-03"]
os.environ.setdefault("LADDER", "42")
for _k in ("LAMSMOOTH", "LAMKEEP", "LAMSLOPE", "LAMSG", "LAMDSG", "LAMH", "LAMDATES"):
    os.environ.pop(_k, None)                      # the RAW ladder is the reference here
sys.argv = [sys.argv[0], "cpu"]

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
torch.set_num_threads(1)
from scipy.stats import norm                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                           # noqa: E402
import end_to_end as E                                        # noqa: E402
import sanos_leverage as SL                                   # noqa: E402

DEG, KEEP = 3, 1                                              # the withdrawn `_c9` configuration


def bs_call(F, k, sig):
    if sig <= 1e-12:
        return max(F - k, 0.0)
    d1 = (np.log(F / k) + 0.5 * sig * sig) / sig
    return F * norm.cdf(d1) - k * norm.cdf(d1 - sig)


def implied(F, k, price):
    """Total implied vol (sigma*sqrt(T)) by bisection. Returns nan outside the no-arb band."""
    lo, hi = 1e-8, 5.0
    if price <= max(F - k, 0.0) + 1e-14 or price >= F - 1e-14:
        return np.nan
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_call(F, k, mid) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def market_skew(K, V, qs, Ts, T, eta):
    """d(implied vol)/d(log-moneyness) at the money, from the LP marginal AT T. Total-vol units."""
    q, var = SL.blend(K, V, qs, Ts, T, eta)
    F = float(np.sum(q * K))
    m2 = float(np.sum(q * K ** 2) * np.exp(var))
    sd = float(np.sqrt(max(np.log(max(m2 / F ** 2, 1 + 1e-12)), 1e-12)))
    d = 0.5 * sd                                              # +-half a standard deviation
    ks = np.array([F * np.exp(-d), F * np.exp(d)])
    px = SL.call(K, q, var, ks)
    iv = np.array([implied(F, k, p) for k, p in zip(ks, px)])
    if not np.all(np.isfinite(iv)):
        return np.nan, sd
    return float((iv[1] - iv[0]) / (2 * d)), sd


def rough(x):
    """Mean |relative jump| between neighbours -- the 6e.13 statistic."""
    x = np.asarray(x, float)
    ok = np.isfinite(x[:-1]) & np.isfinite(x[1:]) & (np.abs(x[:-1]) > 1e-12)
    if ok.sum() < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x))[ok] / np.abs(x[:-1])[ok]))


print(f"  {'yr':5s} | {'c1(T) rough':>11s} {'mkt skew rough':>14s} {'ratio':>6s} | "
      f"{'smoothing pert. at 1SD':>22s} | {'corr(c1,skew)':>13s} {'raw':>6s} {'smoothed':>8s}")
print(f"  {'':5s} | {'weekly':>11s} {'AT PILLARS':>14s} {'':>6s} | "
      f"{'mean %':>10s} {'max %':>10s} | {'':13s} {'':6s} {'':8s}")
for date in DATES:
    K, V, qs, Ts = E.joint_cached(date, "SPX", SL.ETA)
    _s, lev, _c, _n = E.rebuilt_static(date, "SPX")
    wk = sorted(lev)
    c1 = np.array([lev[k].coef[0] for k in wk], float)
    # the withdrawn smoother, reproduced here so the perturbation can be priced
    x = np.log(np.asarray(wk, float))
    c1s = np.polyval(np.polyfit(x, c1, DEG), x)
    c1s[:KEEP] = c1[:KEEP]; c1s[len(wk) - KEEP:] = c1[len(wk) - KEEP:]

    # (1) market skew at the PILLARS -- no weekly interpolation anywhere in this
    Tp = np.asarray(Ts, float)
    use = Tp[(Tp > 0) & (Tp <= max(wk) * SL.DT)]
    sk, sds = [], []
    for T in use:
        s, sd = market_skew(K, V, qs, Ts, float(T), SL.ETA)
        sk.append(s); sds.append(sd)
    sk = np.array(sk, float)

    # (2) perturbation in LOCAL VOL terms at +-1 SD of each maturity
    sd_wk = np.interp(np.array(wk, float) * SL.DT, use, np.array(sds, float))
    pert = np.abs(np.exp(np.abs(c1s - c1) * sd_wk / 2.0) - 1.0) * 100.0

    # (3) does smoothing move c1 toward the market skew? Compare both on the pillar grid.
    c1_at = np.interp(use, np.array(wk, float) * SL.DT, c1)
    c1s_at = np.interp(use, np.array(wk, float) * SL.DT, c1s)
    ok = np.isfinite(sk)
    cr = np.corrcoef(c1_at[ok], sk[ok])[0, 1] if ok.sum() > 2 else np.nan
    cs = np.corrcoef(c1s_at[ok], sk[ok])[0, 1] if ok.sum() > 2 else np.nan

    print(f"  {date[:4]:5s} | {rough(c1):11.4f} {rough(sk):14.4f} "
          f"{rough(c1)/max(rough(sk),1e-9):6.1f}x | {pert.mean():9.2f}% {pert.max():9.2f}% | "
          f"{'':13s} {cr:6.3f} {cs:8.3f}")
print("\n  ratio >> 1 : the weekly ladder is rougher than the market's own skew term structure,")
print("               i.e. the roughness is NUMERICAL and there is something to remove.")
print("  corr raw vs smoothed: if smoothing MOVES c1 AWAY from the market skew, it is buying")
print("               SSR/vov fit by mis-calibrating strikes -- the 6e.22 objection, quantified.")
print("  perturbation: local VOL change at 1SD, in %. Compare with the ~26 ATM vol bp bid-ask bar")
print("               (0.26%) that verify_blend.py holds the blend to.")
