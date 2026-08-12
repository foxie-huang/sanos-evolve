#!/usr/bin/env python3
"""HELD-OUT check on the bracket-churn correction, with a placebo control.

THE DESIGN. `_h9` is fitted on 14/21/30/45d ONLY -- the tenors where churn days are ordinary (|r|
ratio 1.04-1.27, 6e.43), so the `drop >=60d` rule leaves the fitted targets bit-identical. 60/90/120/
180d are never seen by the optimiser. The model's prediction there is then scored against three
versions of the held-out target:

    RAW       the shipped screened series
    CHURN     churn-day increments dropped at 60d+
    PLACEBO   the SAME NUMBER of increments dropped at 60d+, chosen AT RANDOM

**THE PLACEBO IS THE POINT.** Dropping increments reduces roughness, and the model curve is smooth, so
ANY thinning of the long end moves the target toward the model and would score better. That would
support "smoothing helps", not "churn is the mechanism". The correction is only supported if it beats
a random drop of equal size. Several draws are averaged and the spread is reported, so a win inside
the placebo's own scatter is visible as a non-result.

Q/P is applied to all three identically (6e.30), since the fitted targets carry it.

MEASUREMENT ONLY.

    python3 ndx_churn_heldout.py [--draws 40]
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_h9")
ap.add_argument("--draws", type=int, default=40)
ap.add_argument("--seed", type=int, default=12345)
A = ap.parse_args()

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

CACHE = os.path.join(_P.DATA, ".ndx_scr_cache")
ALL = [14, 21, 30, 45, 60, 90, 120, 180]
FITTED = [14, 21, 30, 45]
HELD = [60, 90, 120, 180]
YEARS = {"2012-06-01": "2012", "2016-06-01": "2016", "2017-06-01": "2017", "2018-06-01": "2018",
         "2019-06-03": "2019", "2020-06-01": "2020", "2021-06-01": "2021", "2022-06-01": "2022",
         "2024-06-03": "2024"}
VLO, VHI = 0.02, 2.0
PQ = json.load(open(os.path.join(_P.DATA, "spx_pq_vov.json")))


def qp(year, tenors):
    z = PQ[year]; T = [int(x) for x in z["tenors"]]
    q = np.asarray(z["q_mean"], float); p = np.asarray(z["p_rvov"], float)
    return np.array([q[T.index(t)] / p[T.index(t)] for t in tenors])


def increments(year):
    """Per held-out tenor: (log increments, churn flags), adjacent days, physically valid levels."""
    z = np.load(os.path.join(CACHE, f"series_NDX_{year}.npz"))
    b = np.load(os.path.join(CACHE, f"brackets_NDX_{year}.npz"))
    n = int(z["n_files"][0]); lo, hi = b["lo"], b["hi"]
    out = {}
    for t in HELD:
        j = ALL.index(t)
        x = np.full(n, np.nan)
        idx = z[f"idx_{t}"].astype(int); val = z[f"val_{t}"].astype(float)
        ok = (val >= VLO) & (val <= VHI)
        x[idx[ok]] = val[ok]
        r, ch = [], []
        for i in range(min(n, lo.shape[0]) - 1):
            if not (np.isfinite(x[i]) and np.isfinite(x[i + 1])):
                continue
            if not (np.isfinite(lo[i, j]) and np.isfinite(lo[i + 1, j])):
                continue
            r.append(np.log(x[i + 1]) - np.log(x[i]))
            ch.append((lo[i + 1, j] != lo[i, j]) or (hi[i + 1, j] != hi[i, j]))
        out[t] = (np.array(r), np.array(ch, bool))
    return out


def model_vov(tag, date, tenors):
    j = json.load(open(os.path.join(_P.DATA, f"fit_kf{tag}_{date}_ndx.json")))
    ctx, _c, _n = E.ctx_rebuilt(date, "NDX")
    K = consts.Consts("cpu", torch.float32)
    LAM, SIG, SPOT = ctx["LT"][max(ctx["LT"])], ctx["sig_ref"], ctx["spot"]
    n_var = max(1, int(round((30.0 / 365.0) / K.dt)))
    th = torch.tensor([j["theta"][k] for k in C.NAMES_N] + [j["kap_s"]], dtype=torch.float32)
    g = kernel.solve_gbar(th, SIG, K)
    kk = kernel.build_kernel(kernel.th9(th, g, K), K)
    with torch.no_grad():
        u0 = VX.solve_us0(kk, SIG, SPOT, n_var)
        return np.array([float(VX.vix_ivol(kk, SIG, float(t) / 365.0, SPOT,
                                           lam_fns=LAM, us0=u0)[1]) for t in tenors])


if __name__ == "__main__":
    rng = np.random.default_rng(A.seed)
    print(f"  HELD-OUT: {A.tag} fitted on {FITTED} only; scored at {HELD}.")
    print(f"  RAW vs CHURN-dropped vs PLACEBO (same count dropped at random, {A.draws} draws)\n")
    print(f"  {'yr':6s} {'RAW':>8s} {'CHURN':>8s} {'PLACEBO':>16s} {'churn beats placebo by':>23s}")
    R, Cc, Pm = [], [], []
    for d, y in YEARS.items():
        inc = increments(y)
        m = model_vov(A.tag, d, HELD)
        ratio = qp(y, HELD)
        raw = np.array([np.std(inc[t][0]) * np.sqrt(252) for t in HELD]) * ratio
        chn = np.array([np.std(inc[t][0][~inc[t][1]]) * np.sqrt(252) for t in HELD]) * ratio
        pls = []
        for _ in range(A.draws):
            v = []
            for t in HELD:
                r, ch = inc[t]
                keep = np.ones(len(r), bool)
                keep[rng.choice(len(r), size=int(ch.sum()), replace=False)] = False
                v.append(np.std(r[keep]) * np.sqrt(252))
            pls.append(np.array(v) * ratio)
        pls = np.array(pls)
        rms = lambda tg: 100 * np.sqrt(np.mean(((m - tg) / tg) ** 2))
        pr = np.array([rms(p) for p in pls])
        R.append(rms(raw)); Cc.append(rms(chn)); Pm.append(pr.mean())
        print(f"  {y:6s} {rms(raw):7.1f}% {rms(chn):7.1f}% {pr.mean():9.1f}+-{pr.std():4.1f}% "
              f"{pr.mean()-rms(chn):+22.1f}pp")
    R, Cc, Pm = map(np.array, (R, Cc, Pm))
    print(f"\n  {'MEAN':6s} {R.mean():7.1f}% {Cc.mean():7.1f}% {Pm.mean():9.1f}%")
    print(f"\n  churn vs RAW      {100*(Cc.mean()/R.mean()-1):+6.1f}%   better at {int((Cc<R).sum())}/9 dates")
    print(f"  churn vs PLACEBO  {100*(Cc.mean()/Pm.mean()-1):+6.1f}%   better at {int((Cc<Pm).sum())}/9 dates")
    print(f"  placebo vs RAW    {100*(Pm.mean()/R.mean()-1):+6.1f}%   <- how much is GENERIC thinning")
    print(f"\n  If churn only matches the placebo, the mechanism is not what helps -- thinning is.")
