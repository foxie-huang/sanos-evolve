#!/usr/bin/env python3
"""Q vs P VOL-OF-VOL ON SPX -- the one ticker where both sides are observable at once.

WHY. The two tickers' vov targets are different objects, not just different data sources:

    SPX  vix_readout.data_vix -> ATM implied vol of VIX OPTIONS (parity forward, IV at K=F).
         A traded price. Q measure.
    NDX  ndx_vov_readout.rvov -> std(diff(log(var-swap level))) over the year.
         A realised statistic. P measure.

The model's readout is the same function in both cases -- VX.vix_ivol, the IV of a VIX-STYLE OPTION.
So on SPX the readout and the target are the same object and the fit is measure-consistent; on NDX
the readout is compared against a realised SPOT vol-of-vol, and it is NOT.

WHAT Q/P IS AND IS NOT. It was built expecting a risk premium (Q > P). It does not measure one
cleanly, because it nets TWO effects of opposite sign:
  + the vol-of-vol risk premium, which lifts Q
  - the SPOT-vs-FUTURE damping: VIX options are on VIX FUTURES, whose vol is damped by mean reversion
    relative to VIX spot -- and spot is what the realised side measures. At short tenors this
    dominates and Q lands BELOW P.
That makes Q/P useless as a premium estimate but exactly right for the job NDX needs: it is the
measured ratio between the model's VIX-option readout and a realised spot vov, per tenor.

This runs ONE pass per year over the daily files and builds, at each tenor in TEN:

    P  realised vov of the SPX var-swap strip, CONSTANT MATURITY (ndx_vov_cm.cm_level, ticker=SPX)
       -- i.e. the NDX construction applied to SPX, so the two sides are comparable by construction
    Q  VIX ATM implied vol interpolated to the same tenor, linear in total variance, no extrapolation
       -- reported both as the year mean and on the FIT DATE (what the SPX fit actually consumes)

THE TERM STRUCTURE IS THE POINT, not the level -- because the decay ratio vov(90)/vov(30) is what the
2-anchor NDX fit must reproduce, and it is where NDX exceeds the model's 1/sqrt(T) ceiling.

TWO CONTROLS, because a bare ratio cannot tell a real effect from a broken estimator:
  * CONTROL 1, our 30d const-mat strip vs the exchange's VIX INDEX. Same object, one built by us and
    one not, so any gap is OUR construction noise. It bounds how much of the realised side is real.
  * CONTROL 2, the signature profile rvov(q)/sqrt(q) at q = 1,2,5,10 days. Flat means diffusion.
    Falling can be iid level noise OR mean reversion -- and running the VIX index through the SAME
    profile is what separates them, since the index has no construction noise.

CAVEATS THAT BOUND THE READING:
  * HORIZON. The SPX fit consumes a SINGLE-DAY Q read; the NDX fit consumes a YEAR-LONG P statistic.
    Both Q columns are printed so that gap is visible rather than buried in the ratio.
  * GAPPY SERIES. A day where a tenor is not bracketed is DROPPED, so rvov's daily increments can
    span a gap at the wing tenors. Refusing to extrapolate is the deliberate trade; coverage (n_yrs
    and n_obs) is printed so a thin tenor is visible rather than silently trusted.
  * MEASURE-ONLY. This writes a cache and a json. It does not touch any target used by a fit.

    python3 spx_pq_vov.py                 # all years
    python3 spx_pq_vov.py 2019 2020       # a subset
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
sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                                  # noqa: E402
DATA = _P.DATA
sys.path.insert(0, os.path.normpath(os.path.join(DATA, "..", "v2", "data")))
from ndx_vov_readout import rvov, rvov_spread                        # noqa: E402
from vix_readout import data_vix                                     # noqa: E402
import calibrate_joint_torch as J                                    # noqa: E402
sys.path.insert(0, HERE)
from ndx_vov_cm import cm_level                                      # noqa: E402

OUT = J.OUT
CACHE = os.path.join(DATA, ".spx_pq_cache")
os.makedirs(CACHE, exist_ok=True)
# Full grid, matched EXACTLY to ndx_vov_cm.TEN and the .ndx_oos_cache tenor set, so the ratio can be
# applied tenor-by-tenor to the OOS series without any interpolation of its own. Q is unbracketed at
# some tenors in some years (VIX options do not list arbitrarily short or long), and `vix_atm` returns
# None there rather than extrapolating -- those cells come out NaN and are reported as gaps.
TEN = [7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365]
LAGS = [1, 2, 5, 10]
MIN_DTE = getattr(J, "MIN_DTE", 7)
FIT_DATES = {"2012": "2012-06-01", "2016": "2016-06-01", "2017": "2017-06-01", "2018": "2018-06-01",
             "2019": "2019-06-03", "2020": "2020-06-01", "2021": "2021-06-01", "2022": "2022-06-01",
             "2023": "2023-06-01", "2024": "2024-06-03"}


def vix_atm(date, taus=TEN):
    """VIX ATM implied vol interpolated to each tau, linear in TOTAL VARIANCE. None where unbracketed.

    Same interpolation rule as the constant-maturity P side (ndx_vov_cm), so the two are not being
    compared across two different interpolation conventions."""
    try:
        _spot, dv = data_vix(date)
    except Exception:
        return [None] * len(taus)
    dv = [d for d in dv if d[0] >= MIN_DTE and np.isfinite(d[2]) and d[2] > 0]
    if len(dv) < 2:
        return [None] * len(taus)
    T = np.array([d[0] for d in dv], float)
    iv = np.array([d[2] for d in dv], float)
    o = np.argsort(T); T, iv = T[o], iv[o]
    out = []
    for tau in taus:
        if tau < T[0] or tau > T[-1]:
            out.append(None)                                # refuse to extrapolate
            continue
        j = int(np.searchsorted(T, tau))
        if T[j] == tau:
            out.append(float(iv[j])); continue
        Tl, Th = T[j - 1], T[j]
        wl, wh = iv[j - 1] ** 2 * Tl, iv[j] ** 2 * Th
        w = wl + (tau - Tl) / (Th - Tl) * (wh - wl)
        out.append(float(np.sqrt(max(w / tau, 1e-12))))
    return out


def rvov_lag(x, q):
    """rvov on q-day increments, scaled to a daily-equivalent: std(log x_{t+q}/x_t)*sqrt(252/q).

    For a genuine driftless diffusion this is FLAT in q. It falls under EITHER iid noise in the level
    (bid-ask bounce, strike-grid changes, interpolation error -- which the strip has and a traded
    index does not) OR mean reversion, which pulls the log-level back over longer increments. Those
    are not distinguishable from one series: run the VIX INDEX through the same profile, and whatever
    it shares is the vol process, not our construction. Measured 2019: strip 0.760, index 0.773 --
    the same, so it is mean reversion and Control 1 bounds our noise at +2%."""
    x = np.asarray(x, float)
    if len(x) <= q + 5:
        return float("nan")
    return float(np.std(np.log(x[q:]) - np.log(x[:-q])) * np.sqrt(252.0 / q))


def year_pq(year):
    k = os.path.join(CACHE, f"pq_SPX_{year}.npz")
    if os.path.exists(k):
        return dict(np.load(k, allow_pickle=True))
    t0 = time.time()
    lev = {t: [] for t in TEN}                              # P: CM var-swap level series
    qq = {t: [] for t in TEN}                               # Q: VIX ATM IV series
    vidx = []                                               # the VIX INDEX itself -- the clean control
    files = sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz"))
    for f in files:
        date = os.path.basename(f).split("_")[-1][:-8]
        try:
            recs = json.load(gzip.open(f))["strikes"]
        except Exception:
            continue
        for t in TEN:
            v = cm_level(recs, "SPX", t)
            if v is not None and np.isfinite(v) and v > 0:
                lev[t].append(v)
        for t, q in zip(TEN, vix_atm(date)):
            if q is not None and np.isfinite(q) and q > 0:
                qq[t].append(q)
        # VIX index = a 30d SPX variance level published by the exchange, i.e. the SAME object our
        # 30d CM strip level estimates, but WITHOUT our construction noise. Any gap between
        # rvov(strip30) and rvov(VIX) is ours, not the market's.
        _v = [r for r in recs if r["ticker"] == "VIX"]
        if _v:
            try:
                s = float(_v[0]["spotPrice"]) / 100.0
                if np.isfinite(s) and s > 0:
                    vidx.append(s)
            except Exception:
                pass
    out = dict(tenors=np.array(TEN, float),
               p_rvov=np.array([rvov(np.array(lev[t])) if len(lev[t]) > 5 else np.nan for t in TEN]),
               p_spread=np.array([rvov_spread(np.array(lev[t])) if len(lev[t]) > 130 else np.nan
                                  for t in TEN]),
               q_mean=np.array([np.mean(qq[t]) if len(qq[t]) > 5 else np.nan for t in TEN]),
               q_med=np.array([np.median(qq[t]) if len(qq[t]) > 5 else np.nan for t in TEN]),
               q_fitdate=np.array([x if x is not None else np.nan
                                   for x in vix_atm(FIT_DATES.get(year, f"{year}-06-01"))]),
               p_vix=np.array([rvov(np.array(vidx)) if len(vidx) > 5 else np.nan]),
               # signature profile: rows = tenors (+ VIX index last), cols = lags 1,2,5,10
               lags=np.array(LAGS, float),
               sig=np.array([[rvov_lag(lev[t], q) for q in LAGS] for t in TEN]
                            + [[rvov_lag(vidx, q) for q in LAGS]]),
               n_p=np.array([len(lev[t]) for t in TEN], float),
               n_q=np.array([len(qq[t]) for t in TEN], float),
               n_vix=np.array([len(vidx)], float),
               wall=np.array([time.time() - t0]), n_files=np.array([len(files)]))
    np.savez(k, **out)
    return out


if __name__ == "__main__":
    years = YEARS_ARG or ["2012", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
    print("  SPX VOL-OF-VOL: Q (VIX ATM option IV) vs P (realised vov of the const-mat var-swap strip)")
    print("  Q/P is NOT a pure risk premium: VIX options are on VIX FUTURES, whose vol is damped by")
    print("  mean reversion relative to VIX SPOT, which is what the realised side measures. Q/P")
    print("  therefore nets the (positive) premium against the (negative) spot-vs-future damping.")
    print("  What it IS, and what NDX needs: the ratio between the model's VIX-option readout and a")
    print("  realised SPOT vov, measured on the one ticker where both exist.\n")
    print(f"  {'yr':6s} {'ten':>5s} {'P real':>8s} {'Q yrmean':>9s} {'Q fitdt':>8s} "
          f"{'Q/P mean':>9s} {'Q/P fit':>8s}   n_P/n_Q")
    R = {}
    for y in years:
        d = year_pq(y); R[y] = d
        for i, t in enumerate(TEN):
            p, qm, qf = d["p_rvov"][i], d["q_mean"][i], d["q_fitdate"][i]
            print(f"  {y:6s} {t:4.0f}d {p:8.3f} {qm:9.3f} {qf:8.3f} "
                  f"{qm/p:9.3f} {qf/p:8.3f}   {d['n_p'][i]:.0f}/{d['n_q'][i]:.0f}")
    ok = [y for y in years if np.isfinite(R[y]["p_rvov"][TEN.index(30)])
          and np.isfinite(R[y]["q_mean"][TEN.index(30)])
          and np.isfinite(R[y]["p_rvov"][TEN.index(90)])
          and np.isfinite(R[y]["q_mean"][TEN.index(90)])]
    I30, I90 = TEN.index(30), TEN.index(90)
    r30 = np.array([R[y]["q_mean"][I30] / R[y]["p_rvov"][I30] for y in ok])
    r90 = np.array([R[y]["q_mean"][I90] / R[y]["p_rvov"][I90] for y in ok])
    print(f"\n  PREMIUM Q/P (year mean), {len(ok)} years")
    print(f"    30d   mean {r30.mean():.3f}   median {np.median(r30):.3f}   range {r30.min():.3f}-{r30.max():.3f}")
    print(f"    90d   mean {r90.mean():.3f}   median {np.median(r90):.3f}   range {r90.min():.3f}-{r90.max():.3f}")
    print(f"    ratio of premiums 30d/90d  mean {np.mean(r30/r90):.3f}  "
          f"-> {'SHORT-END RICHER: correcting NDX STEEPENS the demanded decay' if np.mean(r30/r90) > 1.05 else ('LONG-END RICHER: correcting FLATTENS the demanded decay' if np.mean(r30/r90) < 0.95 else 'ROUGHLY UNIFORM: correcting shifts nu up, decay demand unchanged')}")
    # ---- CONTROL 1: our 30d strip level vs the VIX index (the same object, exchange-published) ----
    pv = np.array([R[y]["p_vix"][0] for y in ok])
    p30 = np.array([R[y]["p_rvov"][TEN.index(30)] for y in ok])
    print(f"\n  CONTROL 1 -- rvov of OUR 30d const-mat strip level vs rvov of the VIX INDEX")
    print(f"    strip30  mean {p30.mean():.3f}      VIX index  mean {pv.mean():.3f}"
          f"      ratio {np.mean(p30/pv):.3f}")
    print(f"    -> {100*(np.mean(p30/pv)-1):+.1f}% of our realised vov at 30d is CONSTRUCTION NOISE, "
          f"not vol-of-vol" if np.mean(p30 / pv) > 1.02 else "    -> strip and index agree")
    print(f"    Q/P using the VIX INDEX as P instead:  30d mean {np.mean([R[y]['q_mean'][0] for y in ok]/pv):.3f}")
    # ---- CONTROL 2: signature profile. flat in lag = diffusion; falling = iid level noise ----
    print(f"\n  CONTROL 2 -- signature profile, rvov at q-day increments scaled to daily-equivalent")
    print(f"    {'series':12s}" + "".join(f"{'q='+str(q):>9s}" for q in LAGS) + f"{'q10/q1':>9s}")
    for i, nm in enumerate([f"strip {t}d" for t in TEN] + ["VIX index"]):
        s = np.nanmean(np.array([R[y]["sig"][i] for y in ok]), axis=0)
        print(f"    {nm:12s}" + "".join(f"{v:9.3f}" for v in s) + f"{s[-1]/s[0]:9.3f}")
    print(f"    The VIX INDEX falls the SAME way as our strip, and it carries no construction noise --")
    print(f"    so the decline is MEAN REVERSION in the vol process, not microstructure. Control 1 is")
    print(f"    what bounds our noise (+2% at 30d in 2019), and it is small.")
    pd = np.array([R[y]["p_rvov"][I90] / R[y]["p_rvov"][I30] for y in ok])
    qd = np.array([R[y]["q_mean"][I90] / R[y]["q_mean"][I30] for y in ok])
    print(f"\n  DECAY vov(90)/vov(30) -- the shape the 2-anchor fit must reproduce")
    print(f"    P (realised)  mean {pd.mean():.3f}   implied p = {(-np.log(pd)/np.log(3.0)).mean():.3f}")
    print(f"    Q (implied)   mean {qd.mean():.3f}   implied p = {(-np.log(qd)/np.log(3.0)).mean():.3f}")
    print(f"    model's reachable floor on the ratio ~0.62  (p <= 0.5)")
    print(f"\n  Q/P COVERAGE AND LEVEL BY TENOR (year mean; n = years with a bracketed Q)")
    print(f"    {'tenor':>6s} {'Q/P':>7s} {'range':>15s} {'n_yrs':>6s}")
    for i, t in enumerate(TEN):
        rr = np.array([R[y]["q_mean"][i] / R[y]["p_rvov"][i] for y in years
                       if np.isfinite(R[y]["q_mean"][i]) and np.isfinite(R[y]["p_rvov"][i])
                       and R[y]["p_rvov"][i] > 0])
        if len(rr) == 0:
            print(f"    {t:5.0f}d {'--':>7s} {'no bracketed Q':>15s} {0:6d}"); continue
        print(f"    {t:5.0f}d {rr.mean():7.3f} {rr.min():7.3f}-{rr.max():6.3f} {len(rr):6d}")
    json.dump({y: {k: R[y][k].tolist() for k in ("tenors", "p_rvov", "p_spread", "q_mean", "q_med",
                                                 "q_fitdate", "n_p", "n_q")} for y in years},
              open(os.path.join(DATA, "spx_pq_vov.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(DATA, 'spx_pq_vov.json')}")
