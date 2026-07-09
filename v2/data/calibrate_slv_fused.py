#!/usr/bin/env python3
"""
calibrate_slv_fused.py -- the FAITHFUL alg:calib on the leverage-locked fused model.

Fit the 8 dynamic knobs so the TOTAL fused SSR (LV from the SANOS Dupire leverage + SV from the
kernel, Gyongy-decoupled) matches the empirical SSR. Statics locked to SANOS by the leverage;
gbar solved per-theta so EV = sig_ref^2*dt (lambda ~ 1, paper §748). No new model code -- reuses
discslv_slv.fused_ssr_readout + slv_wire.{sanos_chain, ref_vol, solve_gbar, leverage_at}.
Expensive (fused readout); run detached.
"""
import sys, os, glob, time
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                   # noqa: E402
import discslv_slv                                                   # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment, fused_ssr_readout   # noqa: E402
from slv_fast import propagate_vec                                   # noqa: E402
discslv_slv.propagate = propagate_vec                                # vectorized drop-in (verified bit-for-bit; ~6x)
from empirical_ssr import empirical_ssr                             # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at  # noqa: E402

DT = 1.0 / 252.0; NS = [5, 10, 21, 42, 63]; LABELS = ["1wk", "2wk", "1m", "2m", "3m"]; NK = 16   # DAILY dt, 1wk-3m (trading days)
WSSR = np.array([1.0, 1.5, 2.0, 1.5, 1.0])                              # emphasize the 1m/belly (canonical), taper the noisy edges
NAMES = ["nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
LO = np.array([0.10, 0.10, 0.10, -3.0, 0.0, 0.0, 0.05, 0.5])
HI = np.array([1.20, 1.20, 1.00, 0.0, 8.0, 8.0, 1.0, 4.0])
X0 = np.array([0.43, 0.50, 0.14, -1.48, 0.98, 1.65, 1.00, 2.34])


def fused_ssr_vec(x, chain, sig_ref, nk=NK, split=False):
    kw = dict(zip(NAMES, x)); gbar = solve_gbar(kw, sig_ref, dt=DT)
    K = TwoFactorSV(gbar=gbar, dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    tot, lv, sv = [], [], []
    for n in NS:
        lam = leverage_at(chain, n * DT, EV, dt=DT)
        t, l, s = fused_ssr_readout(K, lam, n, EV, nub, Vlr, tiltr, nk, DT)
        tot.append(t); lv.append(l); sv.append(s)
    return (np.array(tot), np.array(lv), np.array(sv)) if split else np.array(tot)


if __name__ == "__main__":
    OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"
    date = sys.argv[1] if len(sys.argv) > 1 else OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"
    yr = os.path.basename(date).split("_")[-1][:4]
    chain = sanos_chain(date); sig_ref = ref_vol(chain)
    emp, nd = empirical_ssr(sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz")), ns=NS, dt=DT)
    t0 = time.time()
    print(f"empirical SSR ({nd} {yr} dates): {np.round(emp, 3)}  weights {WSSR}   (SD-scaled skew, deinflated)", flush=True)
    res = least_squares(lambda x: WSSR * (fused_ssr_vec(x, chain, sig_ref) - emp), X0, bounds=(LO, HI),
                        diff_step=4e-2, max_nfev=100, xtol=1e-6, ftol=1e-6, verbose=2)
    tot, lv, sv = fused_ssr_vec(res.x, chain, sig_ref, nk=16, split=True)
    print(f"FUSED leverage-locked calibration -- {os.path.basename(date)}  ({res.nfev} evals, {time.time()-t0:.0f}s)")
    print(f"statics locked to SANOS ({len(chain)} expiries); gbar solved per-theta (lambda~1)\n")
    print("theta: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, res.x)) + "\n")
    print(f"{'':6}" + "".join(f"{l:>9}" for l in LABELS))
    print(f"{'fused':6}" + "".join(f"{v:9.3f}" for v in tot))
    print(f"{'  LV':6}" + "".join(f"{v:9.3f}" for v in lv))
    print(f"{'  SV':6}" + "".join(f"{v:9.3f}" for v in sv))
    print(f"{'emp':6}" + "".join(f"{v:9.3f}" for v in emp))
    print(f"{'err':6}" + "".join(f"{100*(t-e)/e:8.0f}%" for t, e in zip(tot, emp)))
