#!/usr/bin/env python3
"""Does the NDX vov roughness come from CONSTANT-MATURITY BRACKET CHURN? A time-domain test.

THE HYPOTHESIS. `cm_level` interpolates each fixed tenor between the two BRACKETING listed expiries.
Those two expiries are a fixed pair on most days, but occasionally the pair CHANGES -- the near one
rolls past the tenor, or a newly listed expiry slots in between. At that moment the interpolation
switches to a different pair of smiles, and any inconsistency between them appears as a one-day jump
in the level. `rvov` counts that jump as a vol move.

WHY THIS TEST AND NOT THE LAST ONE. 6e.42 tried to isolate the artefact by CROSS-SECTIONAL structure
(factor decomposition) and failed: adjacent tenors share bracketing expiries, so their artefacts are
correlated and land in the common factors rather than the idiosyncratic residual. Cross-section cannot
separate them. But the artefact has a sharp TIME signature -- it can only occur on a churn day -- and
that is directly checkable.

IDENTITY IS THE EXPIRY DATE, NOT THE DTE. A fixed expiry's dte counts down by one every day, so
comparing dte would flag every single day. The bracketing pair is identified as
(file_date + dte_lo, file_date + dte_hi) and churn is a change in THAT pair.

CHEAP BY CONSTRUCTION: only the set of listed dtes is needed, never `var_swap`. The expensive smile
integration is skipped entirely.

PREDICTIONS, fixed before running:
  * If the hypothesis holds, |increment| on churn days should exceed |increment| on quiet days by a
    wide margin, and churn days should carry a share of total squared variation far above their share
    of days.
  * The effect should be STRONGER at tenors with the worst roughness (180d, 45-60d) than at the
    cleanest (14d, 21d).
  * If churn days look like every other day, the mechanism is wrong and the roughness is something
    else. That would be a clean falsification.

MEASUREMENT ONLY. Writes a cache and prints; touches no target.

    python3 ndx_bracket_churn.py
"""
import glob
import gzip
import json
import os
import sys
import time
from datetime import date, timedelta

import numpy as np

sys.argv = [sys.argv[0], "cpu"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                                   # noqa: E402
DATA = _P.DATA
sys.path.insert(0, os.path.normpath(os.path.join(DATA, "..", "v2", "data")))
import calibrate_joint_torch as J                                     # noqa: E402

OUT = J.OUT
CACHE = os.path.join(DATA, ".ndx_scr_cache")
FIT = [14, 21, 30, 45, 60, 90, 120, 180]
YEARS = ["2012", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2024"]
MIN_DTE = 7
VLO, VHI = 0.02, 2.0


def brackets(year, ticker="NDX"):
    """Per day, per tenor: the bracketing expiry pair as ABSOLUTE dates. Cached."""
    k = os.path.join(CACHE, f"brackets_{ticker}_{year}.npz")
    if os.path.exists(k):
        return dict(np.load(k, allow_pickle=True))
    t0 = time.time()
    files = sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz"))
    lo = np.zeros((len(files), len(FIT))); hi = np.zeros((len(files), len(FIT)))
    lo[:] = np.nan; hi[:] = np.nan
    for i, f in enumerate(files):
        try:
            recs = json.load(gzip.open(f))["strikes"]
        except Exception:
            continue
        d0 = date.fromisoformat(os.path.basename(f).split("_")[-1][:-8])
        dtes = sorted({int(r["dte"]) for r in recs
                       if r["ticker"] == ticker and int(r["dte"]) >= MIN_DTE})
        if not dtes:
            continue
        for j, t in enumerate(FIT):
            below = [d for d in dtes if d <= t]
            above = [d for d in dtes if d >= t]
            if not below or not above:
                continue
            # absolute expiry dates -> a fixed expiry keeps the same identity as its dte counts down
            lo[i, j] = (d0 + timedelta(days=max(below))).toordinal()
            hi[i, j] = (d0 + timedelta(days=min(above))).toordinal()
    out = dict(lo=lo, hi=hi, n=np.array([len(files)]), wall=np.array([time.time() - t0]))
    np.savez(k, **out)
    return out


def series(year):
    z = np.load(os.path.join(CACHE, f"series_NDX_{year}.npz"))
    n = int(z["n_files"][0])
    M = np.full((n, len(FIT)), np.nan)
    for j, t in enumerate(FIT):
        idx = z[f"idx_{t}"].astype(int); val = z[f"val_{t}"].astype(float)
        ok = (val >= VLO) & (val <= VHI)
        M[idx[ok], j] = val[ok]
    return M


if __name__ == "__main__":
    print("  BRACKET CHURN vs QUIET days: |daily log-move| in the const-maturity level\n")
    print(f"  {'yr':6s} {'tenor':>6s} {'churn d':>8s} {'%days':>6s} {'|r| churn':>10s} "
          f"{'|r| quiet':>10s} {'ratio':>6s} {'%var on churn':>14s}")
    ROW = []
    for y in YEARS:
        B = brackets(y); M = series(y)
        n = min(len(M), B["lo"].shape[0])
        for j, t in enumerate(FIT):
            lo, hi = B["lo"][:n, j], B["hi"][:n, j]
            x = M[:n, j]
            ok = np.isfinite(x)
            r, ch = [], []
            for i in range(n - 1):
                if not (ok[i] and ok[i + 1]):
                    continue
                if not (np.isfinite(lo[i]) and np.isfinite(lo[i + 1])
                        and np.isfinite(hi[i]) and np.isfinite(hi[i + 1])):
                    continue
                r.append(np.log(x[i + 1]) - np.log(x[i]))
                ch.append((lo[i + 1] != lo[i]) or (hi[i + 1] != hi[i]))
            r = np.array(r); ch = np.array(ch, bool)
            if len(r) < 30 or ch.sum() < 3 or (~ch).sum() < 3:
                continue
            ac, aq = float(np.mean(np.abs(r[ch]))), float(np.mean(np.abs(r[~ch])))
            share = float(np.sum(r[ch] ** 2) / np.sum(r ** 2))
            ROW.append((y, t, int(ch.sum()), 100 * ch.mean(), ac, aq, ac / aq, 100 * share))
    for y in YEARS:
        for rr in [r for r in ROW if r[0] == y]:
            print(f"  {rr[0]:6s} {rr[1]:5d}d {rr[2]:8d} {rr[3]:5.1f}% {rr[4]:10.4f} {rr[5]:10.4f} "
                  f"{rr[6]:6.2f} {rr[7]:13.1f}%")
    A = np.array([[r[3], r[6], r[7]] for r in ROW])
    print(f"\n  OVERALL: churn days are {A[:,0].mean():.1f}% of days, carry {A[:,2].mean():.1f}% of the")
    print(f"  squared variation, and their |move| is {A[:,1].mean():.2f}x a quiet day's.")
    print(f"\n  BY TENOR (mean over years)")
    print(f"  {'tenor':>6s} {'%days churn':>12s} {'|r| ratio':>10s} {'%var on churn':>14s}")
    for t in FIT:
        s = np.array([[r[3], r[6], r[7]] for r in ROW if r[1] == t])
        if len(s):
            print(f"  {t:5d}d {s[:,0].mean():11.1f}% {s[:,1].mean():10.2f} {s[:,2].mean():13.1f}%")
    print(f"\n  A ratio near 1.0 falsifies the mechanism: churn days would be ordinary days.")
