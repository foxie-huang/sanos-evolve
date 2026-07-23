#!/usr/bin/env python3
"""
Diagnostic: is the 2015-06-01 under-fit a WEIGHT imbalance or a kernel CAPACITY ceiling?

Re-run calibrate_2f's own least_squares, but with RELATIVE-ERROR residuals -- each observable
normalized by its own target magnitude, so skew is no longer out-weighted 6:1 by vol. If the
skew closes, it was the (canonical-tuned) weights; if skew stays ~half and lam_skew rails to -3,
it's the 9-knob kernel's capacity for this steep real surface. Touches no poc code.
"""
import sys, os
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC)
import calibrate_2f as C                                  # noqa: E402

# real targets from calibrate_real.py's completed 2015-06-01 / 2015-window run (deterministic)
TGT_SSR  = np.array([2.163, 1.460, 1.364, 1.313])
TGT_VOL  = np.array([0.121, 0.130, 0.143, 0.156])
TGT_SKEW = np.array([-1.115, -0.703, -0.500, -0.368])


def resid_rel(x):
    ssr, vol, skew = C.observables(x)
    return np.concatenate([(ssr - TGT_SSR) / np.abs(TGT_SSR),
                           (vol - TGT_VOL) / np.abs(TGT_VOL),
                           (skew - TGT_SKEW) / np.abs(TGT_SKEW)])


res = least_squares(resid_rel, C.X0, bounds=(C.LO, C.HI), diff_step=3e-2, max_nfev=200, xtol=1e-10, ftol=1e-10)
x = res.x; ssr, vol, skew = C.observables(x, nk=24)
rail = ["{}={:.3f}{}".format(n, v, "  <-LO" if abs(v - lo) < 1e-3 else ("  <-HI" if abs(v - hi) < 1e-3 else ""))
        for n, v, lo, hi in zip(C.NAMES, x, C.LO, C.HI)]
print(f"relative-error fit: {res.nfev} evals, cost {res.cost:.3f}\n")
print("theta:  " + "  ".join(rail) + "\n")
print(f"{'':6}" + "".join(f"{l:>9}" for l in C.LABELS))
for nm, mod, tgt in [("SSR", ssr, TGT_SSR), ("vol", vol, TGT_VOL), ("skew", skew, TGT_SKEW)]:
    print(f"{nm+' fit':6}" + "".join(f"{v:9.3f}" for v in mod))
    print(f"{nm+' tgt':6}" + "".join(f"{v:9.3f}" for v in tgt))
    print(f"{nm+' err':6}" + "".join(f"{100*(m-t)/abs(t):8.0f}%" for m, t in zip(mod, tgt)))
