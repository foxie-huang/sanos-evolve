"""
Cross-strike Hull-White (2017) minimum-variance delta -- the faithful industry baseline.
========================================================================================

The simple baseline in wire_orats.py regresses only the ATM vol on the return. Hull-White (2017,
"Optimal Delta Hedging for Options") instead pool the WHOLE smile: they model the expected change
in implied vol per unit spot move as a quadratic in the option's delta,

    E[Δσ(δ)] = (a + b·δ + c·δ²) · ΔS/(S·√τ),

and estimate (a,b,c) by a trailing pooled regression across strikes. Evaluating at the ATM delta
(δ≈0.5) gives a MORE STABLE ATM spot-vol sensitivity than the single-point ATM regression, because
it borrows strength across the cross-section. That is the fair, strong industry baseline.

This script builds the constant-delta 1-month vol surface from ORATS, forms the cross-strike
Hull-White MV delta, and re-runs the SSR-forecast test with it as the baseline -- so we see (i)
whether the option-implied SSR (F^Q) still beats the STRONGER baseline, and (ii) whether the
cross-strike Hull-White beats the simple ATM MV delta.

Usage:  python3 hullwhite.py [Y0] [Y1] [WINDOW]   (default 2015 2019 63)
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "v2", "data")))
import ssr_forecast_eval as E
from orats_loader import load_day
from empirical_ssr import atm_vol_skew, DT

ORATS = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))  # set ORATS_EOD_DIR to your ORATS EOD chain directory
TENORS = np.array([4, 13, 26, 52]) * DT                     # 1m/3m/6m/1y (years)
DELTAS = np.arange(0.1, 0.91, 0.1)                          # call-equivalent delta buckets
CACHE = os.path.join(HERE, ".orats_cache")


# ----------------------------------------------------------------- per-day features
def _pick_1m(day, lo=21, hi=45, target=30):
    best, bd = None, 1e9
    for s in day.values():
        dte = s.get("dte")
        if s["F"] and s["T"] and dte and lo <= dte <= hi and abs(dte - target) < bd:
            bd, best = abs(dte - target), s
    return best


def _delta_smile(smile):
    """(δ_c ascending, σ, τ) for one expiry: call-equivalent forward delta δ_c=N(d1)."""
    k = np.asarray(smile["logm"], float)
    sig = np.array([v if v else np.nan for v in smile["iv"]], float)
    tau = smile["T"]
    ok = np.isfinite(k) & np.isfinite(sig) & (sig > 0)
    k, sig = k[ok], sig[ok]
    if len(k) < 6:
        return None
    d1 = (-k + 0.5 * sig ** 2 * tau) / (sig * np.sqrt(tau))
    dc = norm.cdf(d1)                                       # in (0,1), ATM≈0.5
    o = np.argsort(dc)
    return dc[o], sig[o], tau


def _day_features(day):
    """(spot, vol[4], skew[4], sigma_on_delta_grid[ndelta]) or None.
    spot = MEDIAN stockPrice across rows (ORATS stockPrice is corrupt in 2022-23; see wire_orats)."""
    T, vol, sk, allsp = [], [], [], []
    for s in day.values():
        if s.get("spot"):
            allsp.append(s["spot"])
        if not s["F"] or not s["T"]:
            continue
        vs = atm_vol_skew(s)
        if vs:
            T.append(s["T"]); vol.append(vs[0]); sk.append(vs[1])
    if len(T) < 3 or not allsp:
        return None
    spot = float(np.median(allsp))
    o = np.argsort(T); T = np.array(T)[o]; vol = np.array(vol)[o]; sk = np.array(sk)[o]
    s1 = _pick_1m(day)
    if s1 is None:
        return None
    ds = _delta_smile(s1)
    if ds is None:
        return None
    dc, sig, _ = ds
    sig_grid = np.interp(DELTAS, dc, sig)                   # constant-delta 1m smile
    return spot, np.interp(TENORS, T, vol), np.interp(TENORS, T, sk), sig_grid


def load_all(y0, y1, ticker="SPX"):
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, f"HW_{ticker}_{y0}_{y1}_clean.npz")
    if os.path.exists(key):
        z = np.load(key)
        return z["dates"], z["spot"], z["vol"], z["skew"], z["sgrid"]
    paths = []
    for y in range(y0, y1 + 1):
        paths += sorted(glob.glob(f"{ORATS}/{ticker}-*_{y}-*.json.gz"))
    dates, spot, vol, skew, sgrid = [], [], [], [], []
    for p in paths:
        f = _day_features(load_day(p, [ticker]).get(ticker, {}))
        if f is None:
            continue
        ymd = int(os.path.basename(p).split("_")[-1].split(".")[0].replace("-", ""))
        dates.append(ymd); spot.append(f[0]); vol.append(f[1]); skew.append(f[2]); sgrid.append(f[3])
    dates = np.array(dates); spot = np.array(spot, float)
    vol = np.vstack(vol); skew = np.vstack(skew); sgrid = np.vstack(sgrid)
    np.savez(key, dates=dates, spot=spot, vol=vol, skew=skew, sgrid=sgrid)
    return dates, spot, vol, skew, sgrid


# ----------------------------------------------------------------- the Hull-White MV delta
def hullwhite_R_series(sgrid, skew1m, spot, window, deltas=DELTAS, atm_delta=0.5):
    """R_HW[t] (SSR units) = (∂σ_ATM/∂logS)/skew, from a trailing pooled quadratic-in-δ regression
    Δσ(δ) = α + (a + b·δ + c·δ²)·r over [t-window+1, t] across all delta buckets."""
    n = len(spot)
    r = np.full(n, np.nan); r[1:] = np.log(spot[1:] / spot[:-1])
    dsig = np.full_like(sgrid, np.nan); dsig[1:] = sgrid[1:] - sgrid[:-1]
    nd = len(deltas)
    R = np.full(n, np.nan)
    for t in range(window, n):
        S = np.arange(t - window + 1, t + 1)
        rr = np.repeat(r[S], nd)                            # (W*nd,)
        dd = np.tile(deltas, len(S))
        yy = dsig[S].reshape(-1)
        m = np.isfinite(yy) & np.isfinite(rr)
        if m.sum() < 4 * nd:
            continue
        X = np.column_stack([np.ones(m.sum()), rr[m], dd[m] * rr[m], dd[m] ** 2 * rr[m]])
        beta, *_ = np.linalg.lstsq(X, yy[m], rcond=None)
        _, a, b, c = beta
        sens = a + b * atm_delta + c * atm_delta ** 2       # ∂σ_ATM/∂logS at δ=0.5
        sk = skew1m[t]
        R[t] = sens / sk if abs(sk) > 1e-9 else np.nan
    return R


def _arr_fc(arr):
    def f(panel, t, tenor):
        return float(arr[t]) if 0 <= t < arr.size and np.isfinite(arr[t]) else np.nan
    return f


# ----------------------------------------------------------------- Doeff-Kamal F^Q (as in wire_orats)
def _dk_forecaster(skew_mat, tenors):
    A = np.column_stack([np.ones(len(tenors)), np.log(tenors)])
    def f(panel, t, tenor):
        sk = np.abs(skew_mat[t])
        if not np.all(np.isfinite(sk)) or np.any(sk <= 0):
            return np.nan
        (_, slope), *_ = np.linalg.lstsq(A, np.log(sk), rcond=None)
        return float(slope + 2.0)
    return f


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 63
    dates, spot, vol, skew, sgrid = load_all(y0, y1)
    print(f"ORATS SPX {y0}-{y1}: {len(dates)} days, delta grid {np.round(DELTAS,2)}, HW window {W}")

    panel = E.Panel(dates=dates, spot=spot,
                    atmiv={TENORS[j]: vol[:, j] for j in range(4)},
                    skew={TENORS[j]: skew[:, j] for j in range(4)})
    TENOR, HORIZON = TENORS[0], 21

    R_hw = hullwhite_R_series(sgrid, skew[:, 0], spot, W)          # cross-strike MV delta
    model = _dk_forecaster(skew, TENORS)                          # F^Q (option-implied)
    bench = _arr_fc(R_hw)                                         # BASELINE = cross-strike Hull-White
    extras = {"atm_MV": E.RealisedSSRForecaster(window=W, as_ssr=True),   # simple ATM MV (old F^P)
              "const": (lambda p, t, ten: 1.5),
              "persistence": E.persistence_forecaster(HORIZON, as_ssr=True)}

    frame = E.build_frame(panel, TENOR, HORIZON, model=model, bench=bench,
                          train_frac=0.5, min_trail=W, as_ssr=True, extra_forecasters=extras)
    ev = E.evaluate(frame, bias_correct=True)
    fc = {"f_model": frame.f_model, "f_bench": frame.f_bench,
          "atm_MV": frame.extra["atm_MV"], "const": frame.extra["const"],
          "persistence": frame.extra["persistence"]}
    hr = E.hedging_replay(panel, frame, TENOR, fc, baseline="f_bench")

    print("\nF^Q = Doeff-Kamal skew-decay SSR  |  BASELINE (f_bench) = CROSS-STRIKE HULL-WHITE MV delta")
    print("  (extra 'atm_MV' = the simple single-point ATM MV delta from wire_orats, for contrast)")
    print(E.format_report(ev, hr))
    q, p, y = frame.f_model, frame.f_bench, frame.target
    a = frame.extra["atm_MV"]
    print("-" * 78)
    print(f"[desc] F^Q sd={np.nanstd(q):.2f} | Hull-White(cross) sd={np.nanstd(p):.2f} | "
          f"ATM-MV(simple) sd={np.nanstd(a):.2f} | target sd={np.nanstd(y):.2f}")
    print(f"[desc] corr(HW-cross, ATM-MV-simple)={np.corrcoef(p[np.isfinite(p)&np.isfinite(a)], a[np.isfinite(p)&np.isfinite(a)])[0,1]:+.3f} "
          f"(how different is the cross-strike estimate from the single-point one)")

    import json
    en = ev["encompassing"]
    rec = dict(sample=f"{y0}-{y1}", window=W, n_test=ev["n_test"], baseline="cross_strike_hull_white",
               rmse=ev["rmse"], encompassing_verdict=en["verdict"],
               hedging_var_vs_HWbaseline=hr["mean_resid_var"],
               hedging_dm_model_vs_HW_p=float(hr["dm_model_vs_baseline"]["p"]) if hr["dm_model_vs_baseline"] else None)
    out = os.path.join(HERE, f"results_hw_{y0}_{y1}.json")
    json.dump(rec, open(out, "w"), indent=2, default=float)
    print(f"[saved] {out}")
    return ev, hr


if __name__ == "__main__":
    main()
