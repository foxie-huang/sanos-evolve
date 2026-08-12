#!/usr/bin/env python3
"""OUT-OF-SAMPLE test of the Q/P object correction: _cm9 vs _pq9 on tenors neither fit ever saw.

WHY THIS AND NOT THE FIT RESIDUAL. `NDXVOVPQ` changes the vov TARGETS, so in-sample vov RMS compares
each run against its own target and cannot say which is better -- lowering a target you then hit is
not evidence. The fit sees 30d and 90d ONLY. Every other tenor is a genuine prediction, identical for
both runs, and cannot be gamed by making the fitted targets easier.

SCORED BOTH WAYS, because neither scoring is neutral on its own:
  * vs RAW realised        -- the object _cm9 was fit to. Favours _cm9 by construction.
  * vs Q/P-CORRECTED realised -- the object the model's readout actually computes (an option on a
    FUTURE, not on spot). Favours _pq9 by construction.
Reporting only one would be picking the answer. If the correction is real, _pq9 should win the
corrected scoring by MORE than _cm9 wins the raw one, and _pq9's advantage should show up in the
SHAPE across tenors rather than only in the level.

REALISED SIDE is `.ndx_cm_cache` (constant maturity), matching what both fits were trained on -- not
`.ndx_oos_cache` (nearest-expiry snapped), which carries the roll artefact of 6e.29.

TENOR COVERAGE. The leveraged readout propagates n_opt = round(tau/dt) steps and then reads a 30d
window (n_var), so it needs n_opt+n_var <= LADDER rungs; at LADDER=42 that caps it near 267d. Tenors
past the cap are dropped rather than extrapolated, and the drop is printed.

    python3 ndx_oos_score.py                       # _cm9 vs _pq9
    python3 ndx_oos_score.py --tags _cm9,_pq9,_n9
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="_cm9,_pq9")
ap.add_argument("--fitted", default="30,90", help="tenors the fit saw; excluded from the OOS score")
A = ap.parse_args()
TAGS = [t for t in A.tags.split(",") if t]
FITTED = [int(x) for x in A.fitted.split(",")]

os.environ.setdefault("LADDER", "42")
os.environ.setdefault("VOVLAMTEN", "avg")
sys.argv = [sys.argv[0], "cpu"]

import torch                                                          # noqa: E402
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kernel_fast"))
sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                                   # noqa: E402
import consts, fkernel as kernel, vix as VX                           # noqa: E402
import calibrate_slv_exact_ts as C                                    # noqa: E402
import end_to_end as E                                                # noqa: E402

DATES = ["2012-06-01", "2016-06-01", "2017-06-01", "2018-06-01", "2019-06-03",
         "2020-06-01", "2021-06-01", "2022-06-01", "2024-06-03"]
PQ = json.load(open(os.path.join(_P.DATA, "spx_pq_vov.json")))


def realised(year):
    """Constant-maturity realised vov + the per-tenor Q/P ratio for that year. NaN where Q is absent."""
    z = np.load(os.path.join(_P.DATA, ".ndx_cm_cache", f"vov_cm_NDX_{year}.npz"))
    T = np.array([int(x) for x in z["tenors"]])
    rv = np.array(z["rvov"], float)
    p = PQ[year]
    Tp = [int(x) for x in p["tenors"]]
    q = np.array(p["q_mean"], float); pp = np.array(p["p_rvov"], float)
    ratio = np.full(len(T), np.nan)
    for i, t in enumerate(T):
        if t in Tp:
            j = Tp.index(t)
            if np.isfinite(q[j]) and np.isfinite(pp[j]) and pp[j] > 0:
                ratio[i] = q[j] / pp[j]
    return T, rv, ratio


def model_curve(tag, date, T):
    """Model vov at each tenor in T, NaN past the ladder cap. Returns (values, cap_days)."""
    j = json.load(open(os.path.join(_P.DATA, f"fit_kf{tag}_{date}_ndx.json")))
    ctx, _c, _n = E.ctx_rebuilt(date, "NDX")
    K = consts.Consts("cpu", torch.float32)
    LAM, SIG, SPOT = ctx["LT"][max(ctx["LT"])], ctx["sig_ref"], ctx["spot"]
    n_var = max(1, int(round((30.0 / 365.0) / K.dt)))
    cap = (len(LAM) - n_var) * K.dt * 365.0
    th = torch.tensor([j["theta"][k] for k in C.NAMES_N] + [j["kap_s"]], dtype=torch.float32)
    g = kernel.solve_gbar(th, SIG, K)
    kk = kernel.build_kernel(kernel.th9(th, g, K), K)
    out = np.full(len(T), np.nan)
    with torch.no_grad():
        u0 = VX.solve_us0(kk, SIG, SPOT, n_var)
        for i, t in enumerate(T):
            if t <= cap:
                out[i] = float(VX.vix_ivol(kk, SIG, float(t) / 365.0, SPOT, lam_fns=LAM, us0=u0)[1])
    return out, cap


def rms(m, t):
    ok = np.isfinite(m) & np.isfinite(t) & (t > 0)
    return (100 * np.sqrt(np.mean(((m[ok] - t[ok]) / t[ok]) ** 2)), int(ok.sum())) if ok.any() else (np.nan, 0)


if __name__ == "__main__":
    print(f"  OUT-OF-SAMPLE vov, tenors EXCLUDING the fitted {FITTED}. Realised = const-maturity.\n")
    acc = {t: {"raw": [], "cor": []} for t in TAGS}
    cap_seen = None
    for d in DATES:
        yr = d[:4]
        T, rv, ratio = realised(yr)
        oos = ~np.isin(T, FITTED)
        line = f"  {yr}  "
        for tag in TAGS:
            m, cap = model_curve(tag, d, T)
            cap_seen = cap
            sel = oos & np.isfinite(m)
            r_raw, n1 = rms(m[sel], rv[sel])
            selc = sel & np.isfinite(ratio)
            r_cor, n2 = rms(m[selc], (rv * ratio)[selc])
            acc[tag]["raw"].append(r_raw); acc[tag]["cor"].append(r_cor)
            line += f"| {tag} raw {r_raw:6.1f}% (n{n1})  corrected {r_cor:6.1f}% (n{n2}) "
        print(line)
    print(f"\n  model curve capped at {cap_seen:.0f}d; tenors beyond it dropped, not extrapolated")
    print(f"\n  {'tag':6s} {'OOS RMS vs RAW':>16s} {'OOS RMS vs CORRECTED':>22s}")
    for tag in TAGS:
        a = np.array(acc[tag]["raw"], float); b = np.array(acc[tag]["cor"], float)
        print(f"  {tag:6s} {np.nanmean(a):15.1f}% {np.nanmean(b):21.1f}%")
    if len(TAGS) == 2:
        x, y = TAGS
        ax, bx = np.array(acc[x]["raw"]), np.array(acc[x]["cor"])
        ay, by = np.array(acc[y]["raw"]), np.array(acc[y]["cor"])
        print(f"\n  vs RAW        {y} beats {x} at {int((ay < ax).sum())}/{len(DATES)} dates "
              f"({100*(np.nanmean(ay)/np.nanmean(ax)-1):+.1f}% mean)")
        print(f"  vs CORRECTED  {y} beats {x} at {int((by < bx).sum())}/{len(DATES)} dates "
              f"({100*(np.nanmean(by)/np.nanmean(bx)-1):+.1f}% mean)")
        print(f"\n  READ IT AS: each scoring favours the run trained on that object. The correction is")
        print(f"  supported only if {y}'s win on CORRECTED is bigger than {x}'s win on RAW.")
    json.dump({t: {k: list(map(float, v)) for k, v in acc[t].items()} for t in TAGS},
              open(os.path.join(_P.DATA, "ndx_oos_score.json"), "w"), indent=1)
