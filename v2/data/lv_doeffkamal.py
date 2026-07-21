#!/usr/bin/env python3
"""
The LV (local-vol backbone SSR) the RIGHT way: the Doeff-Kamal relation, forward from the SANOS static
skew term structure -- no kernel, no residual, no nu->0 degeneracy.

    SSR_LV(T) = H + 3/2,   S_T ~ tau^{H-1/2}   =>   SSR_LV = 2 + d log|S_T| / d log T

S_T = the SANOS static ATM implied-vol skew at maturity T (read off the interpolated SANOS marginal).
The local-vol SSR is then just 2 + the log-log slope of that skew term structure. Compare 2015 vs 2019:
if the 2019 skew term structure has a DIFFERENT decay slope, that is the real, statics-side reason the
LV differs between regimes (the honest version of my earlier hand-wave).
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import discslv_slv                                                    # noqa: E402
from slv_fast import propagate_vec                                    # noqa: E402
discslv_slv.propagate = propagate_vec
from discslv_slv import atm_skew_of                                   # noqa: E402
from slv_wire import sanos_chain                                      # noqa: E402
from slv_interp import interp_marginal                                # noqa: E402

OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
DAYS = np.array([7, 14, 21, 30, 45, 60, 90, 120, 180])               # maturity grid (calendar days)
FIT = slice(0, 6)                                                     # 1wk..2m for the slope (the calibrated range)


def skew_ts(date):
    chain = sanos_chain(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz")
    S = []
    for d in DAYS:
        T = d / 365.0
        mu = interp_marginal(chain, T)
        _, sk = atm_skew_of(mu, T)
        S.append(sk)
    return np.array(S)


for date in ["2015-06-01", "2019-06-03"]:
    S = skew_ts(date)
    lT = np.log(DAYS / 365.0); lS = np.log(np.abs(S))
    p_glob = np.polyfit(lT[FIT], lS[FIT], 1)[0]                       # global power-law slope over the fit range
    ssr_lv_glob = 2 + p_glob; H_glob = p_glob + 0.5
    print(f"\n=== {date} ===  SANOS static skew term structure S_T:")
    print("  " + "  ".join(f"{d}d:{s:+.3f}" for d, s in zip(DAYS, S)))
    print(f"  global log-log slope over 1wk-2m: p={p_glob:+.3f}  =>  H={H_glob:+.2f}  SSR_LV={ssr_lv_glob:.3f}")
    print(f"  {'mat(d)':>7}{'S_T':>9}{'local p':>9}{'SSR_LV':>9}")
    for i in range(1, len(DAYS) - 1):                                # local slope by central difference in log-log
        p_loc = (lS[i + 1] - lS[i - 1]) / (lT[i + 1] - lT[i - 1])
        print(f"  {DAYS[i]:>7}{S[i]:>9.3f}{p_loc:>9.3f}{2 + p_loc:>9.3f}")
