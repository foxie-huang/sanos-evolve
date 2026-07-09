"""
REAL-DATA run of the SSR-forecast test on ORATS SPX EOD.
========================================================

Builds the daily Panel from the existing pipeline (v2/data/empirical_ssr.date_row) and runs the
harness (ssr_forecast_eval) for real.

The option-implied forecast F^Q here is the DOEFF-KAMAL skew-decay SSR (= H + 3/2, with H read
from the ATM-skew term structure  |skew(tau)| ~ tau^{H-1/2}). This is the genuinely option-implied,
forward-looking SSR the paper's own §6.4 mechanism uses to pin the SSR from the smile -- and it is
the RIGHT F^Q here because on SPX EOD there is NO traded Q SSR observable (forward-start/cliquet
skew is OTC), so a full-model SSR would be calibrated TO the realised target and the test would be
circular. The skew-decay SSR is independent of the realised estimate, so the head-to-head is honest.

F^P = trailing realised SSR (the cheap backward-looking benchmark).  Target = forward realised SSR.
All in SSR units. Cached to .orats_cache/ so re-runs are instant.

Usage:  python3 wire_orats.py [YEAR_START] [YEAR_END]   (default 2015 2019)
"""
from __future__ import annotations
import os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_V2DATA = os.path.abspath(os.path.join(HERE, "..", "..", "v2", "data"))
sys.path.insert(0, _V2DATA)

import ssr_forecast_eval as E
from empirical_ssr import DT, atm_vol_skew                  # reuse the validated smile ATM/skew
from orats_loader import load_day

ORATS = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"
TENORS_STEPS = [4, 13, 26, 52]                              # 1m/3m/6m/1y in DT=1/52 steps
TENORS = np.array(TENORS_STEPS) * DT                        # years
CACHE = os.path.join(HERE, ".orats_cache")


def clean_row(path, ticker="SPX", tt=TENORS):
    """(spot, vol[4], skew[4]) for one date, with a CLEAN spot.
    NOTE: ORATS `stockPrice` is corrupt in 2022-2023 (varies across strikes, bad prints up to +20%);
    date_row/empirical_ssr pick one row's value and get spurious ±20% return jumps that inflate
    realized vol (2023 -> 55% vs the true 13%) and destroy the realized comovement. Fix: spot =
    MEDIAN stockPrice across all rows (robust). vol/skew use the parity-forward F + mid-IVs, which
    are clean, so they are unaffected."""
    day = load_day(path, [ticker]).get(ticker, {})
    allsp, T, V, K = [], [], [], []
    for s in day.values():
        if s.get("spot"):
            allsp.append(s["spot"])
        if s["F"] and s["T"]:
            vs = atm_vol_skew(s)
            if vs:
                T.append(s["T"]); V.append(vs[0]); K.append(vs[1])
    if len(T) < 3 or not allsp:
        return None
    o = np.argsort(T); T = np.array(T)[o]; V = np.array(V)[o]; K = np.array(K)[o]
    return float(np.median(allsp)), np.interp(tt, T, V), np.interp(tt, T, K)


# ---------------------------------------------------------------------------- load & cache
def load_series(y0: int, y1: int, ticker: str = "SPX"):
    """Return (dates[int YYYYMMDD], spot[n], vol[n,4], skew[n,4]) across [y0, y1], CLEAN spot."""
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, f"{ticker}_{y0}_{y1}_clean.npz")
    if os.path.exists(key):
        z = np.load(key)
        return z["dates"], z["spot"], z["vol"], z["skew"]
    paths = []
    for y in range(y0, y1 + 1):
        paths += sorted(glob.glob(f"{ORATS}/{ticker}-*_{y}-*.json.gz"))
    dates, spot, vol, skew = [], [], [], []
    for p in paths:
        r = clean_row(p, ticker)
        if r is None:
            continue
        base = os.path.basename(p)
        ymd = int(base.split("_")[-1].split(".")[0].replace("-", ""))
        dates.append(ymd); spot.append(r[0]); vol.append(r[1]); skew.append(r[2])
    dates = np.array(dates); spot = np.array(spot, float)
    vol = np.vstack(vol); skew = np.vstack(skew)
    np.savez(key, dates=dates, spot=spot, vol=vol, skew=skew)
    return dates, spot, vol, skew


def build_panel(dates, spot, vol, skew) -> E.Panel:
    atmiv = {TENORS[j]: vol[:, j] for j in range(len(TENORS))}
    skewd = {TENORS[j]: skew[:, j] for j in range(len(TENORS))}
    return E.Panel(dates=dates, spot=spot, atmiv=atmiv, skew=skewd)


# ------------------------------------------------------- Doeff-Kamal skew-decay F^Q (option-implied)
def doeff_kamal_forecaster(skew_matrix, tenors):
    """SSR^Q_t = H + 3/2, with (H-1/2) the log-log slope of |ATM skew| vs tenor at date t.
    Purely option-implied (from the day's skew term structure); independent of realised comovement."""
    logtau = np.log(tenors)
    A = np.column_stack([np.ones_like(logtau), logtau])
    def f(panel: E.Panel, t: int, tenor: float) -> float:
        sk = np.abs(skew_matrix[t])
        if not np.all(np.isfinite(sk)) or np.any(sk <= 0):
            return np.nan
        (_, slope), *_ = np.linalg.lstsq(A, np.log(sk), rcond=None)   # log|skew| = c + (H-1/2) log tau
        return float(slope + 2.0)                                     # H + 3/2 = (slope + 1/2) + 3/2
    return f


# ---------------------------------------------------------------------------- run
def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    dates, spot, vol, skew = load_series(y0, y1)
    print(f"ORATS SPX {y0}-{y1}: {len(dates)} trading days "
          f"({dates[0]}..{dates[-1]}), tenors={np.round(TENORS,3)} yrs")
    panel = build_panel(dates, spot, vol, skew)

    TENOR = TENORS[0]                      # evaluate the 1m realised comovement (hedging tenor)
    HORIZON = 21                           # ~1-month forward window
    TRAIL_W = int(sys.argv[3]) if len(sys.argv) > 3 else 63   # trailing window for the MV baseline

    model = doeff_kamal_forecaster(skew, TENORS)          # F^Q: option-implied skew-decay SSR
    bench = E.RealisedSSRForecaster(window=TRAIL_W, as_ssr=True)   # F^P: trailing realised SSR
    extras = {"persistence": E.persistence_forecaster(HORIZON, as_ssr=True),
              "const": (lambda p, t, ten: 1.5)}            # fixed R (~paper's structural value)

    frame = E.build_frame(panel, TENOR, HORIZON, model=model, bench=bench,
                          train_frac=0.5, as_ssr=True, extra_forecasters=extras)
    ev = E.evaluate(frame, bias_correct=True)
    fc = {"f_model": frame.f_model, "f_bench": frame.f_bench,
          "persistence": frame.extra["persistence"], "const": frame.extra["const"]}
    hr = E.hedging_replay(panel, frame, TENOR, fc)

    print("\nF^Q = Doeff-Kamal skew-decay SSR (option-implied)  |  "
          "F^P = trailing realised SSR = the industry MIN-VARIANCE / Hull-White (2017) delta")
    print(E.format_report(ev, hr))

    # descriptive: how much does each forecaster actually vary, and the raw Q-vs-P level gap
    print("-" * 78)
    q, p, y = frame.f_model, frame.f_bench, frame.target
    print(f"[desc] F^Q mean={np.nanmean(q):.2f} sd={np.nanstd(q):.2f} | "
          f"F^P mean={np.nanmean(p):.2f} sd={np.nanstd(p):.2f} | "
          f"target mean={np.nanmean(y):.2f} sd={np.nanstd(y):.2f}")
    print(f"[desc] raw corr(F^Q, target)={np.corrcoef(q, y)[0,1]:+.3f}  "
          f"corr(F^P, target)={np.corrcoef(p, y)[0,1]:+.3f}  "
          f"(the SSR risk premium is the F^P-target level gap: {np.nanmean(p-y):+.2f})")

    # persist a compact result record (numbers, not just stdout)
    import json
    en = ev["encompassing"]
    rec = dict(sample=f"{y0}-{y1}", n_days=int(len(dates)), n_test=ev["n_test"],
               tenor=float(TENOR), horizon=HORIZON, trail_w=TRAIL_W,
               rmse=ev["rmse"], dm_accuracy_p=float(ev["dm"]["p"]),
               encompassing=dict(bQ=float(en["bQ"]), p_bQ=float(en["p"][1]),
                                 bP=float(en["bP"]), p_bP=float(en["p"][2]),
                                 incr_r2_model=float(en["delta_r2_model"]), verdict=en["verdict"]),
               hedging_var_vs_MVbaseline=hr["mean_resid_var"],
               hedging_var_vs_black=hr.get("mean_resid_var_vs_black"),
               hedging_dm_model_vs_MVbaseline_p=float(hr["dm_model_vs_baseline"]["p"]) if hr["dm_model_vs_baseline"] else None,
               desc=dict(fQ_mean=float(np.nanmean(q)), fQ_sd=float(np.nanstd(q)),
                         fP_mean=float(np.nanmean(p)), fP_sd=float(np.nanstd(p)),
                         target_mean=float(np.nanmean(y)), target_sd=float(np.nanstd(y)),
                         corr_fQ_target=float(np.corrcoef(q, y)[0, 1]),
                         corr_fP_target=float(np.corrcoef(p, y)[0, 1])))
    out = os.path.join(HERE, f"results_{y0}_{y1}.json")
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, default=float)
    print(f"[saved] {out}")
    return ev, hr


if __name__ == "__main__":
    main()
