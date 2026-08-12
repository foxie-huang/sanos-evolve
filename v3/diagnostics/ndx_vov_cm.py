#!/usr/bin/env python3
"""NDX realised vol-of-vol on a CONSTANT-MATURITY series -- removes the roll artefact.

THE DEFECT. `ndx_vov_readout.expiry()` snaps to the NEAREST listed expiry, so the daily series used
by `rvov` sits at a dte that JUMPS discretely (30d target: 32d, 31d, ... 17d, then 45d). `rvov` is
std(diff(log(x))), so every one of those jumps enters as a "vol move" when it is really a step ALONG
the variance term structure. Measured on the cached series: roll severity (mean |relative day-to-day
dte change|) falls monotonically with tenor -- 0.168 at 7d to 0.009 at 365d -- and WITHIN each year
rvov correlates with it across tenors at +0.550, positive in 10 of 10 years. At the two fitted
anchors the 30d series rolls 1.40x as hard as the 90d one (0.047 vs 0.034), so the SHORT anchor is
inflated more than the long one and the target decay ratio vov(90)/vov(30) is biased STEEP -- which
is exactly the "steep-decay limitation" the 2-anchor fit cannot reach.

THE FIX. Build the series at a FIXED maturity by interpolating between the two bracketing listed
expiries, LINEARLY IN TOTAL VARIANCE:

    w(T) = sigma(T)^2 * T                                  (total variance; var_swap returns sigma)
    w(tau) = w_lo + (tau - T_lo)/(T_hi - T_lo) * (w_hi - w_lo)
    sigma(tau) = sqrt(w(tau) / tau)

Linear-in-total-variance is the calendar-consistent interpolation (it is what a constant-maturity
index does, and it cannot create calendar arbitrage between the bracketing expiries). The resulting
series sits at EXACTLY tau every day, so there is no roll and no term-structure step to mistake for
a vol move.

Extrapolation is REFUSED, not fudged: a day whose listed expiries do not bracket tau is dropped. That
keeps the estimator honest at 7d and 365d, where bracketing often fails, at the cost of fewer obs.

WRITES ITS OWN CACHE (`.ndx_cm_cache/`). It must NEVER touch `.ndx_cache/vov_<year>.npz` (the FIT's
targets, keyed on TENORS=[30,90]) nor `.ndx_oos_cache/` (the snapped OOS series). Those are inputs to
published numbers; this is a candidate replacement and must be comparable to them, not overwrite them.

    python3 ndx_vov_cm.py              # all years
    python3 ndx_vov_cm.py 2021         # one year
"""
import glob
import gzip
import json
import os
import sys
import time

import numpy as np

YEARS_ARG = [a for a in sys.argv[1:] if a[:2] == "20"]
sys.argv = [sys.argv[0], "cpu"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths as _P                                                  # noqa: E402
DATA = _P.DATA
sys.path.insert(0, os.path.normpath(os.path.join(DATA, "..", "v2", "data")))
from ndx_vov_readout import var_swap, rvov, rvov_spread              # noqa: E402
import calibrate_joint_torch as J                                    # noqa: E402

OUT = J.OUT
CACHE = os.path.join(DATA, ".ndx_cm_cache")
os.makedirs(CACHE, exist_ok=True)
TEN = [7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365]
MIN_DTE = 7


def cm_level(recs, ticker, tau):
    """Constant-maturity sqrt(variance) at exactly `tau` days, or None if tau is not bracketed."""
    tk = [r for r in recs if r["ticker"] == ticker and r["dte"] >= MIN_DTE]
    dtes = sorted(set(r["dte"] for r in tk))
    lo = [d for d in dtes if d <= tau]
    hi = [d for d in dtes if d >= tau]
    if not lo or not hi:
        return None                                    # refuse to extrapolate
    Tl, Th = max(lo), min(hi)
    try:
        sl = var_swap([r for r in tk if r["dte"] == Tl], Tl / 365.0)
        if Th == Tl:
            return float(sl)
        sh = var_swap([r for r in tk if r["dte"] == Th], Th / 365.0)
    except Exception:
        return None
    wl, wh = sl ** 2 * Tl, sh ** 2 * Th                # total variance at each bracketing expiry
    w = wl + (tau - Tl) / (Th - Tl) * (wh - wl)        # LINEAR IN TOTAL VARIANCE
    return float(np.sqrt(max(w / tau, 1e-12)))


def year_cm(year, ticker="NDX"):
    k = os.path.join(CACHE, f"vov_cm_{ticker}_{year}.npz")
    if os.path.exists(k):
        return dict(np.load(k))
    t0 = time.time()
    ser = {t: [] for t in TEN}
    files = sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz"))
    for f in files:
        try:
            recs = json.load(gzip.open(f))["strikes"]
        except Exception:
            continue
        for t in TEN:
            v = cm_level(recs, ticker, t)
            if v is not None and np.isfinite(v) and v > 0:
                ser[t].append(v)
    out = dict(tenors=np.array(TEN, float),
               rvov=np.array([rvov(np.array(ser[t])) if len(ser[t]) > 5 else np.nan for t in TEN]),
               spread=np.array([rvov_spread(np.array(ser[t])) if len(ser[t]) > 130 else np.nan
                                for t in TEN]),
               n_obs=np.array([len(ser[t]) for t in TEN], float),
               wall=np.array([time.time() - t0]), n_files=np.array([len(files)]))
    np.savez(k, **out)
    return out


if __name__ == "__main__":
    years = YEARS_ARG or ["2012", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
    print(f"  CONSTANT-MATURITY NDX vol-of-vol (no roll) vs the SNAPPED series\n")
    print(f"  {'yr':6s} {'tenor':>6s} {'snapped':>9s} {'const-mat':>10s} {'change':>8s}   n_obs")
    S, C = {}, {}
    for y in years:
        cm = year_cm(y)
        sn = dict(np.load(os.path.join(DATA, ".ndx_oos_cache", f"vov_oos_NDX_{y}.npz")))
        S[y], C[y] = sn, cm
        for i, t in enumerate(TEN):
            if t in (30, 90):
                a, b = sn["rvov"][i], cm["rvov"][i]
                print(f"  {y:6s} {t:6.0f}d {a:9.3f} {b:10.3f} {100*(b/a-1):+7.1f}% {cm['n_obs'][i]:7.0f}")
    i30, i90 = TEN.index(30), TEN.index(90)
    rs = np.array([S[y]["rvov"][i90] / S[y]["rvov"][i30] for y in years])
    rc = np.array([C[y]["rvov"][i90] / C[y]["rvov"][i30] for y in years])
    print(f"\n  DECAY RATIO vov(90)/vov(30) -- the quantity the 2-anchor fit is asked to reproduce")
    print(f"    snapped        mean {rs.mean():.3f}   range {rs.min():.3f}-{rs.max():.3f}")
    print(f"    constant-mat   mean {rc.mean():.3f}   range {rc.min():.3f}-{rc.max():.3f}")
    print(f"    -> ratio moves {100*(rc.mean()/rs.mean()-1):+.1f}%; the model's reachable floor is ~0.62")
    print(f"    dates below 0.62 (unreachable): snapped {int((rs < 0.62).sum())}/{len(years)}, "
          f"const-mat {int((rc < 0.62).sum())}/{len(years)}")
    p_s = -np.log(rs) / np.log(3.0)
    p_c = -np.log(rc) / np.log(3.0)
    print(f"\n  implied 2-point decay exponent p:  snapped {p_s.mean():.3f}   const-mat {p_c.mean():.3f}"
          f"   (robust 11-tenor reference 0.327)")
