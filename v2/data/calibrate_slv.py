#!/usr/bin/env python3
"""
calibrate_slv.py -- faithful alg:calib in its SSR-target realization (the dynamics).

Statics: the real SANOS marginals are locked in by the discrete-Dupire leverage (slv_wire +
discslv_slv, validated end-to-end) -- they are NOT fit here. Dynamics: fit the 8 SSR-carrying
knobs so the kernel's realized SSR matches the empirical SSR term structure. gbar is fixed at the
real vol level (nearly inert per the paper, Sec. fusion). While the leverage's LV-SSR contribution
is damped by the frozen-shape interpolation (a documented refinement), the fused SSR ~ the kernel's
own ssr_2f, so the dynamics fit uses the fast ssr_2f. Reuses discslv_2f.ssr_2f + empirical_ssr +
slv_wire.{sanos_chain, ref_vol}. No new model code.
"""
import sys, os, glob
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV, ssr_2f                          # noqa: E402
from empirical_ssr import empirical_ssr                             # noqa: E402
from slv_wire import sanos_chain, ref_vol                           # noqa: E402

DT = 1.0 / 52.0; NS = [4, 13, 26, 52]; LABELS = ["1m", "3m", "6m", "1y"]
NAMES = ["nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]   # gbar fixed
LO = np.array([0.10, 0.10, 0.10, -3.0, 0.0, 0.0, 0.05, 0.5])
HI = np.array([1.20, 1.20, 1.00, 0.0, 8.0, 8.0, 1.0, 4.0])
X0 = np.array([0.43, 0.50, 0.14, -1.48, 0.98, 1.65, 1.00, 2.34])


def kernel(x, gbar):
    return TwoFactorSV(gbar=gbar, nu_f=x[0], nu_s=x[1], nu_l=x[2], lam_skew=x[3],
                       lam_f=x[4], lam_s=x[5], kap_f=x[6], kap_s=x[7], dt=DT, n_f=5, n_s=3, n_l=5)


def model_ssr(x, gbar, nk=16):
    K = kernel(x, gbar)
    return np.array([ssr_2f(K, n, nk=nk)[0] for n in NS])


if __name__ == "__main__":
    OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"
    date = sys.argv[1] if len(sys.argv) > 1 else OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"
    yr = os.path.basename(date).split("_")[-1][:4]
    chain = sanos_chain(date); gbar = float(np.log(ref_vol(chain) ** 2))
    emp, nd = empirical_ssr(sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz")))
    res = least_squares(lambda x: model_ssr(x, gbar) - emp, X0, bounds=(LO, HI),
                        diff_step=3e-2, max_nfev=160, xtol=1e-8, ftol=1e-8)
    mod = model_ssr(res.x, gbar, nk=24)
    print(f"leverage-locked SSR calibration -- {os.path.basename(date)}, gbar={gbar:.2f} (real vol {np.exp(gbar/2):.3f})")
    print(f"statics: locked to SANOS chain ({len(chain)} expiries) by the Dupire leverage\n")
    print("theta: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, res.x)) + "\n")
    print(f"{'':6}" + "".join(f"{l:>9}" for l in LABELS))
    print(f"{'SSR mod':6}" + "".join(f"{v:9.3f}" for v in mod))
    print(f"{'SSR emp':6}" + "".join(f"{v:9.3f}" for v in emp))
    print(f"{'err':6}" + "".join(f"{100*(m-e)/e:8.0f}%" for m, e in zip(mod, emp)))
