#!/usr/bin/env python3
"""Does the 30-60d vov bulge come from WHERE each tenor sits between its bracketing expiries?

THE BULGE. 6e.37 measured each year's vov term structure against its own power law and found a
SYSTEMATIC oscillation, not noise: mean residual -10.2% at 14d (t=-3.35), -10.3% at 21d (t=-3.43),
then +7.9 / +16.5 / +23.2% at 30/45/60d, then -11.0 / -17.2% at 90/120d. It has survived all four
validated corrections (6e.45) and none of the confirmed mechanisms reaches it: bracket churn is
ORDINARY at 30-45d (concentration ~1.05, 6e.43), and the idiosyncratic variance share does not track
it either (45d is LOW at 10.1% yet its roughness is high, 6e.42).

THE HYPOTHESIS. `cm_level` interpolates LINEARLY IN TOTAL VARIANCE between the bracketing listed
expiries. Interpolation error vanishes when the target tenor sits ON a listed expiry and is maximal
when it sits MIDWAY between two. NDX lists weeklies near-dated and monthlies further out, so the
tenors that happen to land on listed expiries should be clean and those that land between them should
carry extra variance -- which inflates `rvov`, a POSITIVE bias, matching the bulge's sign.

MEASURED PER TENOR:
  * gap      = Th - Tl, the bracket width in days
  * w        = (tau - Tl) / (Th - Tl), the interpolation position
  * w(1-w)   proportional to linear-interpolation error variance; 0 on an expiry, 0.25 midway
  * exact    fraction of days where the tenor lands ON a listed expiry (no interpolation at all)

FALSIFIED IF w(1-w) is flat across tenors, or fails to correlate with the 6e.37 roughness profile.
Note the profile has BOTH signs, so the test is against |residual| and against the signed profile
separately -- a mechanism that adds variance can only explain the POSITIVE lobe.

Reconstructs each file's date from its NAME (no parsing) and reuses the cached bracket ordinals.

    python3 ndx_interp_position.py
"""
import glob
import os
import sys
from datetime import date

import numpy as np

sys.argv = [sys.argv[0], "cpu"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                                   # noqa: E402
DATA = _P.DATA
sys.path.insert(0, os.path.normpath(os.path.join(DATA, "..", "v2", "data")))
import calibrate_joint_torch as J                                     # noqa: E402

OUT = J.OUT
CACHE = os.path.join(DATA, ".ndx_scr_cache")
FIT = [14, 21, 30, 45, 60, 90, 120, 180]
YEARS = ["2012", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2024"]
# 6e.37 mean signed residual from each year's own power law, same tenor order
ROUGH_SIGNED = np.array([-10.2, -10.3, +7.9, +16.5, +23.2, -11.0, -17.2, +0.9])


def day_ordinals(year):
    return np.array([date.fromisoformat(os.path.basename(f).split("_")[-1][:-8]).toordinal()
                     for f in sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz"))], float)


if __name__ == "__main__":
    print("  WHERE does each tenor sit between its bracketing listed expiries?\n")
    G, W, E = [], [], []
    for y in YEARS:
        b = np.load(os.path.join(CACHE, f"brackets_NDX_{y}.npz"))
        d0 = day_ordinals(y)
        lo, hi = b["lo"], b["hi"]
        n = min(len(d0), lo.shape[0])
        g, w, e = [], [], []
        for j, t in enumerate(FIT):
            Tl = lo[:n, j] - d0[:n]
            Th = hi[:n, j] - d0[:n]
            ok = np.isfinite(Tl) & np.isfinite(Th)
            gap = Th[ok] - Tl[ok]
            pos = np.where(gap > 0, (t - Tl[ok]) / np.where(gap > 0, gap, 1), 0.0)
            g.append(np.mean(gap))
            w.append(np.mean(pos * (1 - pos)))
            e.append(np.mean(gap == 0))
        G.append(g); W.append(w); E.append(e)
    G, W, E = np.array(G), np.array(W), np.array(E)
    print(f"  {'tenor':>6s} {'bracket gap d':>14s} {'w(1-w)':>9s} {'% exact hit':>12s} "
          f"{'6e.37 residual':>15s}")
    for j, t in enumerate(FIT):
        print(f"  {t:5d}d {G[:,j].mean():13.1f} {W[:,j].mean():9.4f} {100*E[:,j].mean():11.1f}% "
              f"{ROUGH_SIGNED[j]:+14.1f}%")
    w = W.mean(axis=0)
    print(f"\n  corr(w(1-w), SIGNED residual)   = {np.corrcoef(w, ROUGH_SIGNED)[0,1]:+.3f}")
    print(f"  corr(w(1-w), |residual|)        = {np.corrcoef(w, np.abs(ROUGH_SIGNED))[0,1]:+.3f}")
    print(f"  corr(bracket gap, |residual|)   = {np.corrcoef(G.mean(axis=0), np.abs(ROUGH_SIGNED))[0,1]:+.3f}")
    print(f"  corr(%exact, SIGNED residual)   = {np.corrcoef(E.mean(axis=0), ROUGH_SIGNED)[0,1]:+.3f}")
    pos = ROUGH_SIGNED > 0
    print(f"\n  POSITIVE-lobe tenors {[FIT[i] for i in range(len(FIT)) if pos[i]]}: "
          f"mean w(1-w) {w[pos].mean():.4f}, exact {100*E.mean(axis=0)[pos].mean():.1f}%")
    print(f"  NEGATIVE-lobe tenors {[FIT[i] for i in range(len(FIT)) if not pos[i]]}: "
          f"mean w(1-w) {w[~pos].mean():.4f}, exact {100*E.mean(axis=0)[~pos].mean():.1f}%")
    print(f"\n  A variance-adding mechanism can only explain the POSITIVE lobe; if w(1-w) is higher")
    print(f"  there and lower on the negative lobe, the hypothesis survives.")
