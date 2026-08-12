#!/usr/bin/env python3
"""Is the NDX vov term-structure roughness IDIOSYNCRATIC? Factor decomposition of the daily increments.

THE PREMISE TO TEST. `rvov` is computed INDEPENDENTLY per tenor, so it throws away the largest thing
in the data: on any given day the increments at 14d...180d are near-collinear, because one vol shock
moves the whole curve. A genuine shock lands in the COMMON factors. A tenor-specific artefact -- a
constant-maturity bracket change at 45d on day t with no matching move at 30d or 60d -- is
IDIOSYNCRATIC by construction. If the term-structure roughness of 6e.41 is really estimator artefact,
the idiosyncratic variance fraction should track it, per tenor and per year.

A SANITY NUMBER FIRST. A realised vol from n~250 increments has relative standard error ~1/sqrt(2n) =
4.5%. The measured roughness is 24%. So pure sampling error cannot be the explanation and something
tenor-specific must be; this asks whether the panel can see it.

METHOD.
  * Build the aligned day x tenor increment panel. A row is kept only where EVERY tenor has a finite
    level on both t and t+1 AND the two days are ADJACENT -- same gap discipline as the shipped
    estimator (6e.34), so no increment spans a hole.
  * Physical bound [vlo, vhi] applied first, exactly as shipped.
  * Eigendecompose the increment covariance. With k factors the common part is V_k L_k V_k'; the
    per-tenor COMMUNALITY is its diagonal over the total diagonal. 1 - communality is idiosyncratic.
  * vov_common(T) = sqrt(252 * common_TT) -- what the estimator would report if it kept only what
    moves with the curve.

WHAT WOULD FALSIFY THE PREMISE: idiosyncratic fractions that are flat across tenors, or uncorrelated
with the roughness. Then the roughness is not tenor-specific noise and this whole direction is wrong.

DOES NOT WRITE ANY TARGET. Measurement only.

    python3 ndx_vov_factor.py
    python3 ndx_vov_factor.py --k 3
"""
import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=2, help="number of common factors")
ap.add_argument("--vlo", type=float, default=0.02)
ap.add_argument("--vhi", type=float, default=2.0)
A = ap.parse_args()

sys.argv = [sys.argv[0], "cpu"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                                   # noqa: E402

CACHE = os.path.join(_P.DATA, ".ndx_scr_cache")
FIT = [14, 21, 30, 45, 60, 90, 120, 180]
YEARS = ["2012", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2024"]


def panel(year):
    """Aligned daily log-increment panel (rows = days, cols = FIT tenors). Adjacent days only."""
    z = np.load(os.path.join(CACHE, f"series_NDX_{year}.npz"))
    n = int(z["n_files"][0])
    M = np.full((n, len(FIT)), np.nan)
    for j, t in enumerate(FIT):
        idx = z[f"idx_{t}"].astype(int); val = z[f"val_{t}"].astype(float)
        ok = (val >= A.vlo) & (val <= A.vhi)
        M[idx[ok], j] = val[ok]
    good = np.all(np.isfinite(M), axis=1)
    rows = [np.log(M[t + 1]) - np.log(M[t])
            for t in range(n - 1) if good[t] and good[t + 1]]
    return np.array(rows) if rows else np.zeros((0, len(FIT)))


def decomp(R, k):
    S = np.cov(R, rowvar=False)
    w, V = np.linalg.eigh(S)
    o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    Ck = (V[:, :k] * w[:k]) @ V[:, :k].T
    comm = np.diag(Ck) / np.diag(S)
    return S, w, comm, np.sqrt(252.0 * np.clip(np.diag(Ck), 0, None)), np.sqrt(252.0 * np.diag(S))


def rough(v):
    """RMS % deviation of a term structure from its own power law."""
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 3:
        return np.nan
    b, a = np.polyfit(np.log(np.array(FIT, float)[ok]), np.log(v[ok]), 1)
    return float(np.sqrt(np.mean((100 * (np.log(v[ok]) - (a + b * np.log(np.array(FIT, float)[ok])))) ** 2)))


if __name__ == "__main__":
    print(f"  NDX vov: is the term-structure roughness IDIOSYNCRATIC?   k = {A.k} common factors\n")
    print(f"  {'yr':6s} {'ndays':>6s} {'var expl by PC1':>15s} {'PC1+2':>7s} {'PC1-3':>7s} "
          f"{'rough(indep)':>13s} {'rough(common)':>14s}")
    IDIO, ROUGH, ALL = [], [], []
    for y in YEARS:
        R = panel(y)
        if len(R) < 30:
            print(f"  {y:6s} {len(R):6d}   too few aligned days"); continue
        S, w, comm, vc, vi = decomp(R, A.k)
        ev = w / w.sum()
        ri, rc = rough(vi), rough(vc)
        IDIO.append(1 - comm); ROUGH.append((y, ri, rc)); ALL.append((y, comm, vi, vc))
        print(f"  {y:6s} {len(R):6d} {100*ev[0]:14.1f}% {100*ev[:2].sum():6.1f}% {100*ev[:3].sum():6.1f}% "
              f"{ri:12.1f}% {rc:13.1f}%")
    I = np.array(IDIO)
    print(f"\n  IDIOSYNCRATIC variance fraction by tenor (1 - communality), % -- the artefact channel")
    print(f"  {'yr':6s} " + " ".join(f"{t:>6d}d" for t in FIT))
    for (y, comm, vi, vc), row in zip(ALL, I):
        print(f"  {y:6s} " + " ".join(f"{100*v:6.1f}" for v in row))
    print(f"  {'MEAN':6s} " + " ".join(f"{100*v:6.1f}" for v in I.mean(axis=0)))
    ri = np.array([r[1] for r in ROUGH]); rc = np.array([r[2] for r in ROUGH])
    mi = I.mean(axis=1)
    print(f"\n  ROUGHNESS: independent estimator {ri.mean():.1f}%  ->  common-factor {rc.mean():.1f}%"
          f"   ({100*(rc.mean()/ri.mean()-1):+.1f}%)")
    print(f"  corr(mean idiosyncratic fraction, roughness of the independent estimator) = "
          f"{np.corrcoef(mi, ri)[0,1]:+.3f}   over {len(ri)} years")
    # per-CELL test: does the tenor with the most idiosyncratic variance carry the roughness?
    cells_i, cells_r = [], []
    for (y, comm, vi, vc), row in zip(ALL, I):
        f = np.array(FIT, float); ok = np.isfinite(vi) & (vi > 0)
        b, a = np.polyfit(np.log(f[ok]), np.log(vi[ok]), 1)
        dev = np.abs(100 * (np.log(vi) - (a + b * np.log(f))))
        cells_i.extend(row.tolist()); cells_r.extend(dev.tolist())
    print(f"  corr over all {len(cells_i)} (year,tenor) CELLS = "
          f"{np.corrcoef(cells_i, cells_r)[0,1]:+.3f}")
