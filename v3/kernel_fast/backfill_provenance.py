#!/usr/bin/env python3
"""Backfill NDX fit provenance by VERIFYING it against the caches, not by asserting it.

WHY. `NDXVOVSCR` and `NDXTENORS` were only added to the fit record on 2026-08-12, after `_t9`, `_t9s`,
`_h9` and the shipped `_c9` had already run. Their records therefore do not state which realised-vov
series they were fitted to -- and since `NDXVOVSCR=1` later came to MEAN the churn drop as well
(6e.45), the flag is ambiguous even where present. For a published panel that is not good enough.

THE METHOD. A fit's stored `vov_target` is a FINGERPRINT of the configuration that produced it.
Reconstruct the target from every candidate series and see which one reproduces it to float precision:

    snapped        .ndx_cache/vov_<yr>.npz                     (the original nearest-expiry series)
    const-mat      .ndx_cm_cache/vov_cm_NDX_<yr>.npz           (NDXVOVCM=1)
    screened       .ndx_scr_cache -> `rvov_gap`                (bound+splice, NO churn)
    screened+churn .ndx_scr_cache -> `rvov`                    (bound+splice+churn at 60d+)

each optionally times the per-year per-tenor SPX Q/P ratio (NDXVOVPQ=1). That is 8 candidates; the one
that matches is the configuration, established from the data rather than from memory.

`rvov_gap` is what makes the PRE-churn generation recoverable even though the cache was overwritten:
the screen stores the splice-fixed value alongside the churn-dropped one, so both generations are
still present in the current npz.

Fields are written as `ndxvovscr` / `ndxtenors` plus a `_provenance` block recording HOW each was
established, so a reconstructed value is never mistaken for one recorded at fit time. Anything that
cannot be verified is left absent and reported, not guessed.

    python3 backfill_provenance.py --dry-run
    python3 backfill_provenance.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
A = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                                   # noqa: E402

D = _P.DATA
SCR = os.path.join(D, ".ndx_scr_cache")
CM = os.path.join(D, ".ndx_cm_cache")
SNAP = os.path.normpath(os.path.join(D, "..", "v2", "data", ".ndx_cache"))
PQ = json.load(open(os.path.join(D, "spx_pq_vov.json")))


def qp(year, tenors):
    z = PQ[year]; T = [int(x) for x in z["tenors"]]
    q = np.asarray(z["q_mean"], float); p = np.asarray(z["p_rvov"], float)
    out = []
    for t in tenors:
        if int(t) not in T:
            return None
        i = T.index(int(t))
        if not (np.isfinite(q[i]) and np.isfinite(p[i]) and p[i] > 0):
            return None
        out.append(q[i] / p[i])
    return np.array(out)


def series(kind, year, tenors):
    try:
        if kind == "snapped":
            z = np.load(os.path.join(SNAP, f"vov_{year}.npz"))
            v = np.asarray(z["rvov"], float)
            return v if len(v) == len(tenors) else None      # snapped cache is keyed on [30,90] only
        if kind == "const-mat":
            z = np.load(os.path.join(CM, f"vov_cm_NDX_{year}.npz")); key = "rvov"
        else:
            z = np.load(os.path.join(SCR, f"vov_scr_NDX_{year}.npz"))
            key = "rvov" if kind == "screened+churn" else "rvov_gap"
        T = [int(x) for x in z["tenors"]]
        return np.array([float(z[key][T.index(int(t))]) for t in tenors])
    except Exception:
        return None


if __name__ == "__main__":
    fits = sorted(glob.glob(os.path.join(D, "fit_kf*_ndx.json")))
    KINDS = ["snapped", "const-mat", "screened", "screened+churn"]
    n_ok = n_amb = n_fail = n_skip = 0
    summary = {}
    for f in fits:
        j = json.load(open(f))
        tgt = np.asarray(j.get("vov_target", []), float)
        ten = j.get("vov_tenor_d")
        yr = j["date"][:4]
        # `vov_tenor_d` predates most of these records. The axis is RECOVERABLE: try the candidate
        # sets and keep the one that reproduces the target -- a wrong axis indexes the wrong cache
        # cells and cannot match, so a hit confirms the axis as well as the series.
        axis_inferred = ten is None
        cand_axes = ([ten] if ten is not None else
                     [a for a in ([30, 90], [14, 21, 30, 45], [14, 21, 30, 45, 60, 90, 120, 180])
                      if len(a) == len(tgt)])
        if not len(tgt) or not cand_axes:
            n_skip += 1
            summary.setdefault("NO TARGET / NO CANDIDATE AXIS", []).append(os.path.basename(f))
            continue
        hits = []
        for ax in cand_axes:
            for k in KINDS:
                sv = series(k, yr, ax)
                if sv is None or len(sv) != len(tgt):
                    continue
                if np.allclose(sv, tgt, rtol=1e-6):
                    hits.append((k, False, tuple(ax)))
                r = qp(yr, ax)
                if r is not None and np.allclose(sv * r, tgt, rtol=1e-6):
                    hits.append((k, True, tuple(ax)))
        tag = os.path.basename(f).split("_20")[0].replace("fit_kf", "")
        # Several series can match when they produce IDENTICAL targets -- e.g. at 14-45d the churn
        # rule (60d+) is inert and the physical bound touched only one cell, so const-mat, screened
        # and screened+churn coincide. That is not a failure: the fit is bit-identical either way, so
        # the flag is genuinely unidentifiable and nothing is lost. Record the SET, do not pick one.
        axes = {h[2] for h in hits}
        if len(hits) >= 1 and len(axes) == 1:
            ax = list(axes)[0]
            pqs = {h[1] for h in hits}
            kinds = sorted({h[0] for h in hits})
            if len(pqs) == 1 and len(kinds) >= 1:
                kind, pq_on = kinds[0] if len(kinds) == 1 else "|".join(kinds), list(pqs)[0]
                j["ndxtenors"] = ",".join(str(int(t)) for t in ax)
                j["ndxvovpq"] = bool(pq_on)
                if len(kinds) == 1:
                    j["ndxvovscr"] = kind.startswith("screened")
                    j["ndxvovcm"] = (kind == "const-mat")
                j["_provenance"] = {
                    "method": "reconstructed 2026-08-12 by matching vov_target against the caches",
                    "series": kind if len(kinds) == 1 else kinds,
                    "series_identifiable": len(kinds) == 1,
                    "qp_applied": bool(pq_on),
                    "tenor_axis_inferred": bool(axis_inferred),
                    "note": ("NDXVOVSCR/NDXTENORS were not recorded at fit time; VERIFIED by "
                             "reproducing vov_target to rtol=1e-6, not asserted from memory. Where "
                             "series_identifiable is false the candidates produce IDENTICAL targets, "
                             "so the fit is unaffected by which flag was set."),
                }
                if not A.dry_run:
                    json.dump(j, open(f, "w"), indent=1)
                if len(kinds) == 1:
                    n_ok += 1
                    summary.setdefault(f"{tag}: {kind}" + (" x Q/P" if pq_on else "")
                                       + (" [axis inferred]" if axis_inferred else ""), []).append(yr)
                else:
                    n_amb += 1
                    summary.setdefault(f"{tag}: {'/'.join(kinds)} INDISTINGUISHABLE (identical targets)"
                                       + (" x Q/P" if pq_on else ""), []).append(yr)
                continue
        if False:
            kind, pq_on = None, None
            j["ndxvovscr"] = kind.startswith("screened")
            j["ndxvovcm"] = (kind == "const-mat")
            j["ndxvovpq"] = pq_on
            j["ndxtenors"] = ",".join(str(int(t)) for t in ten)
            j["_provenance"] = {
                "method": "reconstructed 2026-08-12 by matching vov_target against the caches",
                "series": kind, "qp_applied": bool(pq_on),
                "note": ("NDXVOVSCR/NDXTENORS were not recorded at fit time; these were VERIFIED by "
                         "reproducing vov_target to rtol=1e-6, not asserted from memory."),
            }
            if not A.dry_run:
                json.dump(j, open(f, "w"), indent=1)
            n_ok += 1
            summary.setdefault(f"{tag}: {kind}" + (" x Q/P" if pq_on else ""), []).append(yr)
        elif False:
            pass
        else:
            n_fail += 1
            summary.setdefault(f"{tag}: NO MATCH (targets not reproducible from any cache)", []).append(yr)
    print(f"  {'DRY RUN -- ' if A.dry_run else ''}{len(fits)} NDX fit records\n")
    for k in sorted(summary):
        print(f"  {k}   [{len(summary[k])} dates]")
    print(f"\n  verified {n_ok}   ambiguous {n_amb}   unreproducible {n_fail}   no tenor axis {n_skip}")
