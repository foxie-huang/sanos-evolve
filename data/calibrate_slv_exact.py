#!/usr/bin/env python3
"""
Re-run: calibrate 1wk-3m at the native weekly dt, but with the EXACT closed-form SSR (fused_ssr_exact,
Job 1) instead of the leading-order readout we fit to before (which was +36% off at 1m). Fixed weekly
dt (dt-invariance is a separate structural task, Job 2). Coarse sigma-grid (nz=9) for fit speed.
"""
import sys, os, time
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                    # noqa: E402
import discslv_slv                                                    # noqa: E402
from discslv_slv import Epi_V, nu_bar, raw_increment                  # noqa: E402
from slv_fast import propagate_vec, fused_ssr_exact                   # noqa: E402
discslv_slv.propagate = propagate_vec
from empirical_ssr import empirical_ssr                               # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402

DT = 1.0 / 52.0; NS = [1, 2, 4, 8, 13]; LABELS = ["1wk", "2wk", "1m", "2m", "3m"]; NZ = 9
WSSR = np.array([1.0, 1.5, 2.0, 1.5, 1.0])
NAMES = ["nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
LO = np.array([0.10, 0.10, 0.10, -3.0, 0.0, 0.0, 0.05, 0.5])
HI = np.array([1.20, 1.20, 1.00, 0.0, 8.0, 8.0, 1.0, 4.0])
X0 = np.array([0.45, 0.46, 0.57, -1.52, 0.70, 2.99, 0.98, 2.53])      # start at the readout-fit theta


def exact_vec(x, chain, sig_ref, nz=NZ):
    kw = dict(zip(NAMES, x)); K = TwoFactorSV(gbar=solve_gbar(kw, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    out = []
    for n in NS:
        lam = leverage_at(chain, n * DT, EV, dt=DT)
        out.append(fused_ssr_exact(K, lam, n, EV, nub, Vlr, tiltr, 16, DT, nz=nz)[0])
    return np.array(out)


if __name__ == "__main__":
    OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
    date = OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"; yr = "2015"
    chain = sanos_chain(date); sig_ref = ref_vol(chain)
    emp, nd = empirical_ssr(sorted(__import__("glob").glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz")), ns=NS, dt=DT)
    print(f"empirical SSR ({nd} {yr}): {np.round(emp, 3)}  weights {WSSR}  (EXACT beta, weekly)", flush=True)
    t0 = time.time()
    res = least_squares(lambda x: WSSR * (exact_vec(x, chain, sig_ref) - emp), X0, bounds=(LO, HI),
                        diff_step=5e-2, max_nfev=60, xtol=1e-6, ftol=1e-6, verbose=2)
    mod = exact_vec(res.x, chain, sig_ref, nz=15)
    print(f"\nEXACT-beta calibration ({res.nfev} evals, {time.time()-t0:.0f}s)")
    print("theta: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, res.x)) + "\n")
    print(f"{'':6}" + "".join(f"{l:>8}" for l in LABELS))
    print(f"{'exact':6}" + "".join(f"{v:8.3f}" for v in mod))
    print(f"{'emp':6}" + "".join(f"{v:8.3f}" for v in emp))
    print(f"{'err':6}" + "".join(f"{100*(m-e)/e:7.0f}%" for m, e in zip(mod, emp)))
