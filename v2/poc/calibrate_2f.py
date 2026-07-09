"""
calibrate_2f.py -- calibrate the 9 two-factor knobs to SPX-like statics + the SSR term structure.

Knobs x = [gbar, nu_f, nu_s, nu_l, lam_skew, lam_f, lam_s, kap_f, kap_s].
Targets (verified-canonical SPX, 1m-1y): ATM vol ~flat, ATM skew flattening, SSR decaying ~1.45->1.05
(short-T -> H+3/2 ~1.5, long-T -> ~1.0 sticky-strike).  Closed-form observables (no MC), least-squares.

Run from poc/ :  python3 calibrate_2f.py
"""
import time
import numpy as np
import warnings
from scipy.optimize import least_squares
from discslv_2f import TwoFactorSV, ssr_2f

warnings.filterwarnings("ignore")
DT = 1.0 / 52.0
NK_FIT = 16            # convergence ~2%, fast
NS = [4, 13, 26, 52]; LABELS = ["1m", "3m", "6m", "1y"]
TGT_SSR = np.array([1.45, 1.30, 1.15, 1.05])
TGT_VOL = np.array([0.17, 0.17, 0.17, 0.17])
TGT_SKEW = np.array([-0.50, -0.40, -0.34, -0.28])
W = (20.0, 120.0, 25.0)            # weights: SSR, vol, skew (balance residual scales)
NAMES = ["gbar", "nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
LO = np.array([np.log(0.005), 0.10, 0.10, 0.10, -3.0, 0.0, 0.0, 0.05, 0.5])
HI = np.array([np.log(0.15), 1.20, 1.20, 1.00, 0.0, 8.0, 8.0, 1.0, 4.0])
X0 = np.array([np.log(0.04), 0.55, 0.55, 0.5, -0.8, 3.0, 3.0, 0.2, 2.0])


def kernel(x):
    return TwoFactorSV(gbar=x[0], nu_f=x[1], nu_s=x[2], lam_skew=x[4], lam_f=x[5], lam_s=x[6],
                       kap_f=x[7], kap_s=x[8], dt=DT, nu_l=x[3], n_f=5, n_s=3, n_l=5)


def observables(x, nk=NK_FIT):
    K = kernel(x); ssr, vol, skew = [], [], []
    for n in NS:
        r, a, k = ssr_2f(K, n, nk=nk); ssr.append(r); vol.append(a); skew.append(k)
    return np.array(ssr), np.array(vol), np.array(skew)


def residuals(x):
    ssr, vol, skew = observables(x)
    return np.concatenate([W[0] * (ssr - TGT_SSR), W[1] * (vol - TGT_VOL), W[2] * (skew - TGT_SKEW)])


def main():
    t0 = time.time()
    res = least_squares(residuals, X0, bounds=(LO, HI), diff_step=3e-2, max_nfev=160, xtol=1e-8, ftol=1e-8)
    x = res.x
    ssr, vol, skew = observables(x, nk=24)        # verify at the converged nk
    print(f"calibration: {res.nfev} evals, {time.time()-t0:.0f}s, cost {res.cost:.1f}\n")
    print("theta:  " + "  ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, x)) + "\n")
    print(f"{'':<6}" + "".join(f"{l:>8}" for l in LABELS))
    print(f"{'SSR':<6}" + "".join(f"{v:8.2f}" for v in ssr) + "   | tgt " + " ".join(f"{t:.2f}" for t in TGT_SSR))
    print(f"{'vol':<6}" + "".join(f"{v:8.3f}" for v in vol) + "   | tgt " + " ".join(f"{t:.2f}" for t in TGT_VOL))
    print(f"{'skew':<6}" + "".join(f"{v:8.3f}" for v in skew) + "   | tgt " + " ".join(f"{t:.2f}" for t in TGT_SKEW))


if __name__ == "__main__":
    main()
