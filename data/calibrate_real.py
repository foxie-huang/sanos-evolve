#!/usr/bin/env python3
"""
calibrate_real.py -- run the EXISTING calibrate_2f against REAL targets.

No new model code: extract the real statics (ATM vol/skew, one date) and the real dynamics
(empirical realized SSR, a window), swap them in for calibrate_2f's hardcoded canonical
TGT_VOL/TGT_SKEW/TGT_SSR, and call calibrate_2f's own least_squares. This is `alg:calib`
in its ATM-observable form, ported to real SPX.
"""
import sys, os, glob
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import calibrate_2f                                     # noqa: E402
from real_targets import statics_targets                # noqa: E402
from empirical_ssr import empirical_ssr                 # noqa: E402


def calibrate_real(statics_date, ssr_window):
    st = statics_targets(statics_date)                  # real ATM vol/skew at 1m/3m/6m/1y
    ssr, ndates = empirical_ssr(ssr_window)             # real realized SSR at 1m/3m/6m/1y
    calibrate_2f.TGT_VOL = st["iv"]                     # swap in the real targets (module globals)
    calibrate_2f.TGT_SKEW = st["sk"]
    calibrate_2f.TGT_SSR = ssr
    res = least_squares(calibrate_2f.residuals, calibrate_2f.X0,
                        bounds=(calibrate_2f.LO, calibrate_2f.HI),
                        diff_step=3e-2, max_nfev=160, xtol=1e-8, ftol=1e-8)
    m_ssr, m_vol, m_skew = calibrate_2f.observables(res.x, nk=24)
    return res.x, dict(ssr=(m_ssr, ssr), vol=(m_vol, st["iv"]), skew=(m_skew, st["sk"]), ndates=ndates)


if __name__ == "__main__":
    OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
    date = sys.argv[1] if len(sys.argv) > 1 else OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"
    yr = os.path.basename(date).split("_")[-1][:4]
    window = sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz"))
    x, fit = calibrate_real(date, window)
    L = calibrate_2f.LABELS
    print(f"REAL calibration -- statics {os.path.basename(date)}, SSR from {fit['ndates']} {yr} dates\n")
    print("theta:  " + "  ".join(f"{n}={v:.3f}" for n, v in zip(calibrate_2f.NAMES, x)) + "\n")
    print(f"{'':6}" + "".join(f"{l:>9}" for l in L))
    for nm, (mod, tgt) in [("SSR", fit["ssr"]), ("vol", fit["vol"]), ("skew", fit["skew"])]:
        print(f"{nm+' fit':6}" + "".join(f"{v:9.3f}" for v in mod))
        print(f"{nm+' tgt':6}" + "".join(f"{v:9.3f}" for v in tgt))
