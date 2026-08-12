#!/usr/bin/env python3
"""Screened NDX realised vol-of-vol: fixes a splice bug AND removes single-day data errors.

MOTIVATION. The 2018 panel carries a realised vov of 9.7 at 45d against neighbouring tenors near 1.5
-- a 6x spike at ONE tenor with full daily coverage. That is a broken `var_swap` on some day, not a
market feature. It is harmless while the fit uses 30d/90d only, but a multi-tenor objective is a sum
of RELATIVE residuals, so one 6x point would dominate the whole fit.

TWO DEFECTS, FIXED SEPARATELY AND REPORTED SEPARATELY -- changing both at once and reporting one
number would make it impossible to say which did the work.

  1. SPLICE. `cm_level` refuses to extrapolate, so a day whose listed expiries do not bracket the
     tenor is DROPPED. `rvov` then takes diff(log(.)) straight down the surviving list, so an
     increment can silently span a multi-day gap and be counted as one day's move. Fix: keep the
     trading-day INDEX alongside the level and form increments only between ADJACENT days.

  2. SINGLE-DAY OUTLIERS. Fix: rolling-median filter on the log level (window w), drop points whose
     deviation exceeds K robust MADs. This targets a spike that REVERTS -- the signature of a bad
     print -- while a genuine regime shift moves the rolling median with it within a few days and
     survives. K=6 is deliberately conservative: the aim is egregious errors, not tail trimming.

WHY NOT WINSORISE THE INCREMENTS. Because vol-of-vol legitimately has fat tails, and trimming large
daily moves would bias vov DOWN exactly in the crisis years the model most needs to fit. Screening the
LEVEL against its own local median removes bad prints without touching real moves. The per-year drop
counts are printed so a screen that starts eating 2020 is visible immediately.

WRITES ITS OWN CACHE (`.ndx_scr_cache/`), never `.ndx_cache/` or `.ndx_cm_cache/`, so `_cm9` and
`_pq9` stay exactly reproducible. It also caches the DAILY SERIES, not just the aggregate, which is
what makes re-screening with different (w, K) instant instead of a 30-minute rebuild.

    python3 ndx_vov_screen.py                 # all years, w=11 K=6
    python3 ndx_vov_screen.py --k 4 --w 21    # re-screen from the cached series, seconds
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=float, default=6.0, help="drop points beyond K robust MADs of the local median")
ap.add_argument("--w", type=int, default=11, help="rolling median window (trading days)")
# DEFAULT none: the PHYSICAL bound and the splice fix are always applied; a STATISTICAL screen on top
# is opt-in and, on the evidence below, not warranted.
#   * median   REJECTED. Screened 2020 HARDEST (2.41%, cutting 180d by 61.5%) -- its scale is the MAD
#     of deviations from a lagging rolling median, so a fast year sets a tight band exactly when the
#     market moves. It also MISSED the motivating case.
#   * reversal REJECTED on the evidence, not on principle. It also missed the motivating case (the bad
#     point is at the series start, so there is no increment INTO it), it moves 14/72 fit cells at 5.3%
#     mean against the bound's 5/72 at 1.6%, it still screens 2020 hardest, and on the INDEPENDENT test
#     -- vov must decay with tenor -- it buys almost nothing: violations 15 -> 14 of 63, and 2020 gets
#     WORSE (2 -> 3). The bound alone cuts the worst violation from 589% to 110%.
ap.add_argument("--mode", default="none", choices=["none", "reversal", "median"],
                help="none (default: physical bound + splice only); the statistical screens are rejected")
ap.add_argument("--alpha", type=float, default=0.3, help="reversal tightness: |r_t+r_t+1| < alpha*|r_t|")
# PHYSICAL validity bound, applied BEFORE any statistical screen. 2% to 200% annualised vol. This is
# not a percentile: it is the range in which an index variance-swap level can exist at all. Measured
# over all 25,576 NDX points, exactly ONE falls outside -- a 0.0001 (1bp vol) at 2018 45d, whose
# single increment is 92.5% of that cell's entire variance and drives rvov to 9.7 against neighbours
# near 1.5. Distribution for scale: p0.01 = 0.084, median 0.209, p99.9 = 0.660, max 0.835.
# The statistical screens both MISSED it -- the median filter because it re-centres on a corrupt
# neighbourhood, the reversal test because the bad point sits at the series start so there is no
# increment INTO it to judge. A physical bound cannot eat signal: no index trades at 1bp vol.
ap.add_argument("--vlo", type=float, default=0.02, help="lowest physically possible level")
ap.add_argument("--vhi", type=float, default=2.0, help="highest physically possible level")
# CHURNFROM: drop increments spanning a constant-maturity BRACKET CHANGE, at this tenor and above.
# VALIDATED OUT OF SAMPLE (6e.44): on a fit that saw 14-45d only, held-out 60-180d prediction improves
# 21.7%, while dropping the SAME NUMBER of increments AT RANDOM improves it 0.3%. So it is the churn
# days specifically, not thinning. 60d is the threshold because below it churn days are ORDINARY
# (|r| ratio 1.04-1.27, variance concentration ~1.05) -- expiries are dense there and a bracket switch
# is seamless -- while at 180d they are rare (7.5% of days) and violent (2.11x, 4.63x concentration).
# Dropping churn everywhere costs 29.6% of all increments and makes the two clean years WORSE.
# 0 disables. Only applies to tenors present in the brackets cache.
ap.add_argument("--churn-from", dest="churnfrom", type=float, default=60.0,
                help="drop bracket-change increments at this tenor and above; 0 = off")
ap.add_argument("--ticker", default="NDX")
ap.add_argument("years", nargs="*")
A = ap.parse_args()

sys.argv = [sys.argv[0], "cpu"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                                  # noqa: E402
DATA = _P.DATA
sys.path.insert(0, os.path.normpath(os.path.join(DATA, "..", "v2", "data")))
import calibrate_joint_torch as J                                    # noqa: E402
sys.path.insert(0, HERE)
from ndx_vov_cm import cm_level, TEN                                 # noqa: E402

OUT = J.OUT
CACHE = os.path.join(DATA, ".ndx_scr_cache")
os.makedirs(CACHE, exist_ok=True)
YEARS = A.years or ["2012", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]


def year_series(year, ticker):
    """Daily CM var-swap level per tenor WITH the trading-day index. Cached: screening is then free."""
    k = os.path.join(CACHE, f"series_{ticker}_{year}.npz")
    if os.path.exists(k):
        return dict(np.load(k))
    t0 = time.time()
    idx = {t: [] for t in TEN}
    val = {t: [] for t in TEN}
    files = sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz"))
    for i, f in enumerate(files):
        try:
            recs = json.load(gzip.open(f))["strikes"]
        except Exception:
            continue
        for t in TEN:
            v = cm_level(recs, ticker, t)
            if v is not None and np.isfinite(v) and v > 0:
                idx[t].append(i); val[t].append(v)
    out = {"n_files": np.array([len(files)]), "wall": np.array([time.time() - t0])}
    for t in TEN:
        out[f"idx_{t}"] = np.asarray(idx[t], float)
        out[f"val_{t}"] = np.asarray(val[t], float)
    np.savez(k, **out)
    return out


def keep_mask(val, w, K):
    """True where the log level is within K robust MADs of its own rolling median."""
    lg = np.log(np.asarray(val, float))
    n = len(lg)
    if n < 5:
        return np.ones(n, bool), 0.0
    h = max(1, w // 2)
    med = np.array([np.median(lg[max(0, i - h):min(n, i + h + 1)]) for i in range(n)])
    dev = lg - med
    mad = float(np.median(np.abs(dev - np.median(dev))) * 1.4826)
    if not np.isfinite(mad) or mad <= 0:
        return np.ones(n, bool), 0.0
    return np.abs(dev) <= K * mad, mad


def keep_mask_reversal(idx, val, K, alpha):
    """Flag a level point whose move IN is large and is immediately UNDONE -- a one-day round trip.

    WHY NOT THE MEDIAN FILTER (mode=median, kept so the failure is reproducible). Its scale is the MAD
    of deviations from a ROLLING MEDIAN, which lags a trending series; in a fast year every real move
    reads as a large deviation, so it screened 2020 HARDEST (2.41% of points, cutting 180d by 61.5%)
    -- the exact signature of eating signal that this script's own diagnostic was written to catch.

    The round-trip test cannot do that. A genuine move in the variance level PERSISTS; only a bad
    print goes out and comes straight back. Scale here is the MAD of the INCREMENTS themselves, so a
    violent year raises its own bar instead of lowering it."""
    lg = np.log(np.asarray(val, float))
    i = np.asarray(idx, float)
    n = len(lg)
    keep = np.ones(n, bool)
    if n < 8:
        return keep, 0.0
    r = np.diff(lg)
    adj = np.diff(i) == 1
    rr = r[adj]
    if len(rr) < 5:
        return keep, 0.0
    s = float(np.median(np.abs(rr - np.median(rr))) * 1.4826)
    if not np.isfinite(s) or s <= 0:
        return keep, 0.0
    for j in range(1, n - 1):
        if i[j] - i[j - 1] != 1 or i[j + 1] - i[j] != 1:
            continue                                   # need both neighbours adjacent to judge
        a, b = r[j - 1], r[j]
        if abs(a) > K * s and abs(a + b) < alpha * abs(a):
            keep[j] = False
    return keep, s


def rv_churn(idx, val, lo, hi, mask):
    """Annualised vol using only ADJACENT-day increments that do NOT span a bracket change.

    Masks at the INCREMENT level, not the point level: a bracket change contaminates the one increment
    that straddles it, and dropping the endpoint would also discard the clean increment on its far
    side. `lo`/`hi` are indexed by TRADING DAY; `idx` maps series position -> day."""
    i = np.asarray(idx, float)
    v = np.asarray(val, float)
    if mask is not None:
        i, v = i[mask], v[mask]
    d = []
    for k in range(len(v) - 1):
        a, b = int(i[k]), int(i[k + 1])
        if b - a != 1:
            continue                                   # gap: same discipline as gap_aware
        if not (np.isfinite(lo[a]) and np.isfinite(lo[b])
                and np.isfinite(hi[a]) and np.isfinite(hi[b])):
            continue
        if lo[b] != lo[a] or hi[b] != hi[a]:
            continue                                   # bracket pair changed -> drop this increment
        d.append(np.log(v[k + 1]) - np.log(v[k]))
    return float(np.std(d) * np.sqrt(252.0)) if len(d) > 4 else np.nan


def rv(idx, val, gap_aware, mask=None):
    """Annualised vol of the daily log level. gap_aware forms increments only between ADJACENT days."""
    i, v = np.asarray(idx, float), np.asarray(val, float)
    if mask is not None:
        i, v = i[mask], v[mask]
    if len(v) < 6:
        return np.nan
    d = np.diff(np.log(v))
    if gap_aware:
        d = d[np.diff(i) == 1]
    return float(np.std(d) * np.sqrt(252.0)) if len(d) > 4 else np.nan


if __name__ == "__main__":
    print(f"  SCREENED NDX realised vov  (mode={A.mode}, K={A.k}, alpha={A.alpha}, w={A.w})\n")
    print(f"  {'yr':6s} {'tnr':>5s} {'spliced':>8s} {'gap+valid':>10s} {'+screened':>10s} "
          f"{'d_valid':>7s} {'d_scr':>7s} {'drop':>5s}")
    ALL = {}
    from ndx_bracket_churn import FIT as BRK_TEN, brackets as _brk       # noqa: E402
    for y in YEARS:
        s = year_series(y, A.ticker)
        nchurn = {}
        BRK = None
        if A.churnfrom:
            try:
                BRK = _brk(y, A.ticker)
            except Exception as e:
                print(f"  WARNING {y}: no brackets ({e}) -- churn drop NOT applied this year")
        out = dict(tenors=np.array(TEN, float))
        rows = []
        for t in TEN:
            i, v = s[f"idx_{t}"], s[f"val_{t}"]
            a = rv(i, v, False)                                  # what .ndx_cm_cache holds today
            valid = (v >= A.vlo) & (v <= A.vhi)          # physical bound first
            nv = int((~valid).sum())
            iv, vv = i[valid], v[valid]
            b = rv(iv, vv, True)                          # splice fixed + physically valid
            if A.mode == "none":
                m2, _sc = np.ones(len(vv), bool), 0.0
            elif A.mode == "reversal":
                m2, _sc = keep_mask_reversal(iv, vv, A.k, A.alpha)
            else:
                m2, _sc = keep_mask(vv, A.w, A.k)
            m = np.zeros(len(v), bool); m[np.where(valid)[0][m2]] = True
            c = rv(i, v, True, m)                                # splice fixed + screened
            if A.churnfrom and t >= A.churnfrom and BRK is not None and t in BRK_TEN:
                jb = BRK_TEN.index(t)
                cc = rv_churn(i, v, BRK["lo"][:, jb], BRK["hi"][:, jb], m)
                if np.isfinite(cc):
                    c = cc; nchurn[t] = True
            nd = int((~m).sum())
            rows.append((t, a, b, c, nd, len(v), nv))
        out["rvov"] = np.array([r[3] for r in rows])
        out["rvov_spliced"] = np.array([r[1] for r in rows])
        out["rvov_gap"] = np.array([r[2] for r in rows])
        out["n_dropped"] = np.array([r[4] for r in rows], float)
        out["n_obs"] = np.array([r[5] for r in rows], float)
        out["n_invalid"] = np.array([r[6] for r in rows], float)
        out["churn_applied"] = np.array(sorted(nchurn), dtype=float)
        np.savez(os.path.join(CACHE, f"vov_scr_{A.ticker}_{y}.npz"), **out)
        ALL[y] = out
        for t, a, b, c, nd, n, nv in rows:
            flag = "  <<" if (np.isfinite(a) and np.isfinite(c) and abs(c / a - 1) > 0.15) else ""
            print(f"  {y:6s} {t:4.0f}d {a:8.3f} {b:10.3f} {c:10.3f} "
                  f"{100*(b/a-1):+6.1f}% {100*(c/b-1):+6.1f}% {nd:5d}"
                  f"{(' INVALID x%d' % nv) if nv else ''}{flag}")
    D = np.array([[ALL[y]["n_dropped"][i] for i in range(len(TEN))] for y in YEARS])
    N = np.array([[ALL[y]["n_obs"][i] for i in range(len(TEN))] for y in YEARS])
    print(f"\n  DROPPED points by year (of ~252 x {len(TEN)} tenors)")
    for j, y in enumerate(YEARS):
        print(f"    {y}  {int(D[j].sum()):4d}  ({100*D[j].sum()/max(1,N[j].sum()):.2f}%)")
    print(f"    TOTAL {int(D.sum())} of {int(N.sum())} = {100*D.sum()/N.sum():.2f}%")
    print(f"\n  If 2020 is not among the least-screened years, the filter is eating real moves.")
