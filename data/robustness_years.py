#!/usr/bin/env python3
"""
Cross-regime robustness of the E[nu|z] second-order leverage correction. For each date (spread across
years/vol-regimes): fit theta (joint SSR+VIX ridge), then compare LEADING vs CORRECTED leverage on
  (a) the from-spot ATM term structure vs the real SANOS ATM -- the DIRECT test: does the correction lift
      the under-accumulated from-spot level toward real?
  (b) step-3 liquid marginal residual -- should stay a ~noop (re-seeded);
  (c) step-4 raw forward ATM + the inversion (corrected forward ATM should rise above the near spot vol).
Writes robustness_years.json.  Run detached.
"""
import sys, os, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import real_pipeline_34 as RP                                        # FIRST: binds RP.propagate = original
import calibrate_joint_torch as J                                   # noqa: E402 (patches discslv_slv.propagate; RP.* unaffected)

DATES = ["2017-06-01", "2018-06-01", "2019-06-03", "2020-06-01", "2023-06-01"]
WK = 1.0 / 52.0
WINDOWS = [(4 * WK, 8 * WK), (4 * WK, 13 * WK), (13 * WK, 26 * WK), (26 * WK, 52 * WK)]
MATS = [4, 8, 13, 26]


def fit_theta(date):
    ctx = J.build_date_ctx(date)
    anchor = np.asarray(J.WARM.get(date[:4], J.C.X0_MAP["ts"]), float)
    res = J.fit_date(ctx, anchor, anchor=anchor, w_reg=J.W_REG, max_nfev=30)
    m = J.model_torch(torch.tensor(res.x, dtype=torch.float32), ctx["LT"], ctx["sig_ref"],
                      ctx["spot"], ctx["vdtes"]).detach().numpy()
    ns = len(ctx["emp"]); s, v = m[:ns], m[ns:]
    ssr = 100 * np.sqrt(np.mean(((s - ctx["emp"]) / ctx["emp"]) ** 2))
    vix = 100 * np.sqrt(np.mean(((v - ctx["vov_d"]) / ctx["vov_d"]) ** 2))
    return res.x, float(ssr), float(vix)


def from_spot_atm(date, theta, correct):
    RP.THETA[date] = list(theta); RP.CORRECT = correct
    chain, sig, K, EV, nub, Vlr, tiltr = RP.build_kernel(date)
    out = {}
    for n in MATS:
        st = RP.initial_state(K)
        for k in range(1, n + 1):
            lf = RP.leverage_at(chain, k * RP.DT, EV)
            st, _ = RP.propagate(K, st, RP._lev2(lf, st, nub), EV, nub, Vlr, tiltr, RP.NK)
        out[n] = float(RP.atm_skew_of(RP.marginal(st), n * RP.DT)[0])
    return out


def real_atm(date):
    chain = RP.sanos_chain(RP.OUT + f"/SPX-NDX-RUT-VIX_{date}.json.gz")
    return {n: float(RP.atm_skew_of(RP.interp_marginal(chain, n * RP.DT), n * RP.DT)[0]) for n in MATS}


if __name__ == "__main__":
    res = {}
    for date in DATES:
        print(f"\n{'='*64}\n{date}\n{'='*64}", flush=True)
        theta, ssr, vix = fit_theta(date); RP.THETA[date] = list(theta)
        print(f"  fit: SSR {ssr:.1f}%  VIX {vix:.1f}%   theta={np.round(theta,3)}", flush=True)
        ra = real_atm(date); fl = from_spot_atm(date, theta, False); fc = from_spot_atm(date, theta, True)
        print("  from-spot ATM (wk):  real  / leading / corrected   (closer?)", flush=True)
        wins = 0
        for n in MATS:
            better = abs(fc[n] - ra[n]) < abs(fl[n] - ra[n]); wins += better
            print(f"    {n:2d}wk: {ra[n]:.4f} / {fl[n]:.4f} / {fc[n]:.4f}   {'YES' if better else 'no'}", flush=True)
        RP.CORRECT = False; s3l = RP.step3(date); s4l = RP.step4(date, WINDOWS)
        RP.CORRECT = True;  s3c = RP.step3(date); s4c = RP.step4(date, WINDOWS)
        liq = lambda s: [r["call_bp"] for r in s if r["dte"] <= 190]
        m3l, m3c = float(np.median(liq(s3l))), float(np.median(liq(s3c)))
        print(f"  step3 liquid call-bp median: leading {m3l:.1f} -> corrected {m3c:.1f}  (noop check)", flush=True)
        print("  step4 raw fwd ATM  lead -> corr  (real fwd-var):  + inversion(corr fwd>spot1m)?", flush=True)
        spot1m_c = fc[4]                                              # corrected from-spot ~1m vol
        for i, w in enumerate(["1m2m", "1m3m", "3m6m", "6m1y"]):
            inv = "OK" if s4c[i]["fwd_atm_vol"] > spot1m_c or i > 0 else "INVERTED"
            print(f"    {w}: {s4l[i]['fwd_atm_vol']:.4f} -> {s4c[i]['fwd_atm_vol']:.4f}  ({s4c[i]['real_fwd_vol']:.4f})  {inv if i==0 else ''}", flush=True)
        res[date] = dict(theta=[round(float(x), 4) for x in theta], fit_ssr=round(ssr, 1), fit_vix=round(vix, 1),
                         real_atm=ra, atm_lead=fl, atm_corr=fc, atm_corr_wins=f"{wins}/{len(MATS)}",
                         step3_liq_lead=round(m3l, 1), step3_liq_corr=round(m3c, 1),
                         fwd_atm=[dict(w=w, lead=round(s4l[i]["fwd_atm_vol"], 4), corr=round(s4c[i]["fwd_atm_vol"], 4),
                                       real=round(s4c[i]["real_fwd_vol"], 4)) for i, w in enumerate(["1m2m", "1m3m", "3m6m", "6m1y"])])
    json.dump(res, open(os.path.join(HERE, "robustness_years.json"), "w"), indent=1)
    print(f"\nwrote robustness_years.json ({len(res)} dates)")
