#!/usr/bin/env python3
"""Does the tenor OSCILLATION appear on SPX, where no Q/P transport is needed?

THE QUESTION (6e.48). After removing the model's own log-log curvature, NDX's vov residual still
carries a significant tenor oscillation: model OVER at 14-21d (21d t=-4.36), UNDER at 30-60d
(45d +19.6%), OVER again at 90d (t=-3.03). Every estimator fix has left it standing. The prime suspect
is the ONE unverifiable step in the NDX pipeline -- the Q/P ratio measured on SPX and TRANSPORTED to
NDX, which cannot be checked on NDX because there are no liquid NDX vol options.

WHY SPX SETTLES IT. On SPX the model's readout and the target are the SAME OBJECT (6e.30): both are
the ATM implied vol of a VIX-style option. No transport, no correction, nothing assumed. So:

    FLAT on SPX      -> the oscillation is NDX-specific -> the transport is implicated.
    SAME SHAPE on SPX -> it is a property of the model's readout -> the transport is exonerated
                         and the target is not at fault.

LIKE FOR LIKE. The target is `q_fitdate` -- the VIX ATM IV on the FIT DATE interpolated to each tenor
-- not the year mean, so a single-date model is compared with a single-date read. Tenors where VIX
options do not bracket are NaN and are skipped, not extrapolated.

Both series are measured against their OWN power law and then differenced, exactly as in 6e.48, so the
two numbers are directly comparable.

    python3 spx_profile_check.py [--tag _n9]
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_n9")
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

FIT = [14, 21, 30, 45, 60, 90, 120, 180]
DATES = {"2012-06-01": "2012", "2016-06-01": "2016", "2017-06-01": "2017", "2018-06-01": "2018",
         "2019-06-03": "2019", "2020-06-01": "2020", "2021-06-01": "2021", "2022-06-01": "2022",
         "2024-06-03": "2024"}
PQ = json.load(open(os.path.join(_P.DATA, "spx_pq_vov.json")))
# 6e.48, NDX: TARGET-minus-MODEL profile after removing the model's own curvature
NDX_PROFILE = np.array([-7.4, -12.0, +8.5, +19.6, +8.5, -8.3, -3.3, -5.7])


def model_vov(date, tenors):
    j = json.load(open(os.path.join(_P.DATA, f"fit_kf{A.tag}_{date}.json")))
    ctx, _c, _n = E.ctx_rebuilt(date, "SPX")
    K = consts.Consts("cpu", torch.float32)
    LAM, SIG, SPOT = ctx["LT"][max(ctx["LT"])], ctx["sig_ref"], ctx["spot"]
    nv = max(1, int(round((30.0 / 365.0) / K.dt)))
    th = torch.tensor([j["theta"][k] for k in C.NAMES_N] + [j["kap_s"]], dtype=torch.float32)
    g = kernel.solve_gbar(th, SIG, K)
    kk = kernel.build_kernel(kernel.th9(th, g, K), K)
    with torch.no_grad():
        u0 = VX.solve_us0(kk, SIG, SPOT, nv)
        return np.array([float(VX.vix_ivol(kk, SIG, float(t) / 365.0, SPOT,
                                           lam_fns=LAM, us0=u0)[1]) for t in tenors])


def prof(v, tenors):
    ok = np.isfinite(v) & (v > 0)
    f = np.log(np.asarray(tenors, float))
    b, a = np.polyfit(f[ok], np.log(v[ok]), 1)
    r = np.full(len(v), np.nan)
    r[ok] = 100 * (np.log(v[ok]) - (a + b * f[ok]))
    return r


if __name__ == "__main__":
    print(f"  SPX: model ({A.tag}) vs the TRUE Q (VIX ATM IV on the fit date). No transport.\n")
    RT, RM = [], []
    for d, y in DATES.items():
        z = PQ[y]; T = [int(x) for x in z["tenors"]]
        q = np.array([z["q_fitdate"][T.index(t)] for t in FIT], float)
        m = model_vov(d, FIT)
        if np.sum(np.isfinite(q)) < 5:
            print(f"  {y}: only {int(np.sum(np.isfinite(q)))} bracketed Q tenors -- skipped"); continue
        RT.append(prof(q, FIT)); RM.append(prof(m, FIT))
    RT, RM = np.array(RT), np.array(RM)
    mt = np.nanmean(RT, axis=0); mm = np.nanmean(RM, axis=0)
    res = RT - RM
    mr = np.nanmean(res, axis=0)
    n = np.sum(np.isfinite(res), axis=0)
    ts = mr / (np.nanstd(res, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)))
    print(f"  {'':12s} " + " ".join(f"{t:>7d}d" for t in FIT))
    print(f"  {'Q profile':12s} " + " ".join(f"{v:+8.1f}" for v in mt))
    print(f"  {'model prof':12s} " + " ".join(f"{v:+8.1f}" for v in mm))
    print(f"  {'Q - MODEL':12s} " + " ".join(f"{v:+8.1f}" for v in mr))
    print(f"  {'t-stat':12s} " + " ".join(f"{v:+8.2f}" for v in ts))
    print(f"  {'n years':12s} " + " ".join(f"{v:8d}" for v in n))
    print(f"\n  {'NDX (6e.48)':12s} " + " ".join(f"{v:+8.1f}" for v in NDX_PROFILE))
    ok = np.isfinite(mr)
    print(f"\n  SPX oscillation RMS      {np.sqrt(np.nanmean(mr[ok]**2)):.1f}%")
    print(f"  NDX oscillation RMS      {np.sqrt(np.mean(NDX_PROFILE[ok]**2)):.1f}%")
    print(f"  corr(SPX profile, NDX profile) = {np.corrcoef(mr[ok], NDX_PROFILE[ok])[0,1]:+.3f}")
    print(f"  cells with |t| > 2 on SPX: {int(np.sum(np.abs(ts[ok]) > 2))}/{int(ok.sum())}")
    print(f"\n  FLAT on SPX -> NDX-specific -> transport implicated.")
    print(f"  SAME SHAPE  -> a readout property -> transport exonerated.")
