#!/usr/bin/env python3
"""
ssr_audit.py -- audit the empirical SSR estimator over the 1wk-3m informative range.

Is the belly target (2.16/1.46/1.36) inflated by an under-read skew denominator? SSR = beta/skew,
so break it into pieces per maturity: beta = Cov(dSigma_ATM, r)/Var(r) (the regression slope) and
skew = mean ATM skew (the denominator). Then cross-check the raw-quadratic ATM skew (what the
estimator uses) against the SANOS-marginal skew (the whole-surface fit). If the raw skew is
too small (less negative), that alone inflates the SSR.
"""
import sys, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from empirical_ssr import atm_vol_skew                                # noqa: E402
from orats_loader import load_day                                     # noqa: E402
from real_targets import statics_targets                             # noqa: E402

DT = 1.0 / 52.0
NS = [1, 2, 4, 8, 13, 26]; TT = np.array(NS) * DT                     # 1wk 2wk 1m 2m 3m 6m
LABELS = ["1wk", "2wk", "1m", "2m", "3m", "6m"]
OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))


def date_row(path):
    day = load_day(path, ["SPX"]).get("SPX", {})
    T, vol, sk, spot, tmin = [], [], [], None, None
    for exp, s in day.items():
        if not s["F"] or not s["T"]:
            continue
        vs = atm_vol_skew(s)
        if vs:
            spot = s["spot"]; T.append(s["T"]); vol.append(vs[0]); sk.append(vs[1])
    if len(T) < 3 or spot is None:
        return None
    o = np.argsort(T); T = np.array(T)[o]; vol = np.array(vol)[o]; sk = np.array(sk)[o]
    return float(spot), np.interp(TT, T, vol), np.interp(TT, T, sk), float(T[0])


if __name__ == "__main__":
    yr = sys.argv[1] if len(sys.argv) > 1 else "2015"
    rows = [r for r in (date_row(p) for p in sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz"))) if r]
    spots = np.array([r[0] for r in rows]); vols = np.vstack([r[1] for r in rows]); sks = np.vstack([r[2] for r in rows])
    front = np.median([r[3] for r in rows])
    ret = np.diff(np.log(spots)); dvol = np.diff(vols, axis=0)
    print(f"{yr}: {len(rows)} dates | ann ret vol {ret.std()*np.sqrt(252):.2f} | median front expiry {front*252:.1f} trading days\n")
    print(f"{'T':>5}{'ATMvol':>8}{'beta':>9}{'skew':>9}{'SSR=b/s':>9}   note")
    for j, lab in enumerate(LABELS):
        beta = np.cov(dvol[:, j], ret)[0, 1] / np.var(ret); skew = np.mean(sks[:, j])
        note = "EXTRAPOLATED (< front)" if TT[j] < front else ""
        print(f"{lab:>5}{np.mean(vols[:,j]):>8.3f}{beta:>9.3f}{skew:>9.3f}{beta/skew:>9.2f}   {note}")
    print("\nskew cross-check @ 2015-06-01  (denominator sanity):")
    st = statics_targets(OUT + f"/SPX-NDX-RUT-VIX_{yr}-06-01.json.gz")
    print("  SANOS-marginal skew  1m/3m/6m/1y :", np.round(st["sk"], 2))
    print("  raw-quadratic skew   1m/3m/6m    :", np.round([np.mean(sks[:, 2]), np.mean(sks[:, 4]), np.mean(sks[:, 5])], 2))
