#!/usr/bin/env python3
"""Build the per-date lambda-smoothing decision table that `end_to_end._lam_on` consults.

WHY A MEASURED GATE AND NOT A ROUGHNESS THRESHOLD. Slope smoothing (LAMSLOPE=1) helps at six of the
nine SPX dates and hurts at three, and NOTHING computable before the fit predicts which. Thirteen
smoothness statistics were tried -- ladder-side (RMS of what the mollifier removes; mean and max
relative weekly slope jump over weeks 2-12, over the full ladder, over the tail; the same weighted by
how many fitted tenors read each week) and residual-side (oscillatory amplitude of the pre-smoothing
vov residual, its share of total residual, sign flips, mean |second difference|) -- and every one
OVERLAPS between the helped and hurt sets. The strongest reaches corr = -0.73, carried entirely by
2012/2016 being both roughest and most improved; among the other seven there is no signal at all
(2020/2021/2022 have the smoothest ladders and gain 10-34%; 2017 is third-roughest and loses).

Nor is the loss an optimiser artifact. 2017/2019/2024 were refit COLD under the same smoothing
(`_sl9c`) with 10-12 Jacobian evaluations against the warm run's 2-4, and 2019/2024 landed within
0.2pp of the warm result. The smoothed objective is genuinely worse there.

So the gate measures instead of predicting: fit both ways, keep the lower DATA cost.

WHAT THIS COSTS, STATED PLAINLY. The decision is made on the SAME data the model is fitted to, so the
resulting panel total is in-sample optimistic by up to one bit per date. It is a discrete choice
between two variants with IDENTICAL parameter counts -- not a fitted degree of freedom -- but it is
still selection, and the honest comparison for any downstream claim is the blanket-smoothed panel.

RESOLUTION. The objective is a staircase (recompression banding, stop-gradient) whose measured noise
floor is 10-17% of cost (handoff 6e.3). A date whose two variants differ by less than that is a COIN
FLIP and the table says so: the gate genuinely resolves only the dates outside the floor.

    python3 lam_gate.py                    # all recorded runs -> lam_smooth_dates.json
    python3 lam_gate.py --dry              # print the table, write nothing
    python3 lam_gate.py --raw _ref2 --smooth _sl9,_sl9c,_sl9d --auto _g9 --floor 0.17

--raw/--smooth are for LEGACY tags that predate the `lam` provenance block; --auto is for tags that
carry it and are classified per date. Never declare a GATED panel as one variant -- see main().
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                          # noqa: E402

PANEL = ["2012-06-01", "2016-06-01", "2017-06-01", "2018-06-01", "2019-06-03",
         "2020-06-01", "2021-06-01", "2022-06-01", "2024-06-03"]


def data_cost(j):
    """The DATA half of the objective -- no ridge. Same weighting refit.py builds the residual with."""
    s = np.asarray(j["ssr"], float); t = np.asarray(j["ssr_target"], float)
    v = np.asarray(j["vov"], float); vt = np.asarray(j["vov_target"], float)
    w = 0.8 * np.sqrt(len(s) / len(v))
    return 0.5 * np.sum(((s - t) / t) ** 2) + 0.5 * np.sum((((v - vt) / vt) * w) ** 2)


def load(tag, date):
    p = os.path.join(_P.DATA, f"fit_kf{tag}_{date}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def split(s):
    return [t.strip() for t in (s or "").split(",") if t.strip()]


def smoothed(j, tag, date):
    """Was THIS fit's ladder smoothed? Read from the fit's own `lam` provenance, never from its tag.

    `applied` is the per-date gate's decision, so a gated panel classifies correctly date by date.
    Refuses to guess when the block is absent -- a tag with no provenance must be declared via
    --raw/--smooth, or it would silently join the wrong pool and the gate would compare a variant
    against itself.
    """
    L = j.get("lam")
    if not L:
        raise SystemExit(f"{tag} {date}: no `lam` provenance -- list it under --raw or --smooth "
                         f"instead of --auto (it predates refit.py recording the lambda config)")
    if L.get("applied") is False:                 # the gate turned smoothing off at this date
        return False
    return bool(L.get("smooth") or L.get("sg"))


def main():
    ap = argparse.ArgumentParser()
    # BOTH SIDES MUST GET THE SAME START PROTOCOL, or the gate measures the optimiser instead of the
    # model. The first table compared `_ref2` (raw, DISPLACED start) against `_sl9` (smoothed,
    # undisplaced) -- and displacement alone is worth -7.2% panel-wide, -11% at 2017. Re-checked at
    # all three raw dates with `_sl9d` (smoothed + displaced): margins narrowed 7.2->5.1, 14.3->13.5,
    # 24.8->22.2 and NOTHING FLIPPED, but only because they exceeded the protocol effect. Pass every
    # run of both variants and let the best of each side compete.
    #
    # A TAG IS NOT A VARIANT. `_g9` is the GATED panel: smoothed at six dates, raw at three. Passing
    # it wholesale as "raw" scored its SMOOTHED 2012 fit against the smoothed pool and reported
    # "raw wins by +56.5%". So tags in --auto are classified PER DATE from the `lam` block refit.py
    # records; only legacy tags predating that block are declared by hand.
    ap.add_argument("--raw", default="_ref2",
                    help="comma-separated tags known raw (legacy runs with no `lam` provenance)")
    ap.add_argument("--smooth", default="_sl9,_sl9c,_sl9d",
                    help="comma-separated tags known smoothed (legacy runs with no `lam` provenance)")
    ap.add_argument("--auto", default="_g9",
                    help="comma-separated tags carrying a `lam` block; each is classified per date")
    ap.add_argument("--floor", type=float, default=0.17,
                    help="staircase noise floor as a fraction of cost; |delta| below it = coin flip")
    ap.add_argument("--out", default=os.path.join(_P.DATA, "lam_smooth_dates.json"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    tab, rec, miss = {}, {}, []
    tot_raw = tot_sm = tot_gate = 0.0
    print(f"  raw={a.raw}  smooth={a.smooth}  auto={a.auto} (classified per date from `lam`)  "
          f"floor={a.floor:.0%} of cost\n")
    print(f"  {'date':12s} {'raw':>9s} {'smoothed':>9s} {'delta':>8s}  {'decision':>8s}  resolution")
    for d in PANEL:
        raw_p, sm_p = {}, {}                    # {tag: cost}, best of each pool competes
        for t in split(a.raw):
            j = load(t, d)
            if j is not None:
                raw_p[t] = data_cost(j)
        for t in split(a.smooth):
            j = load(t, d)
            if j is not None:
                sm_p[t] = data_cost(j)
        for t in split(a.auto):
            j = load(t, d)
            if j is None:
                continue
            (sm_p if smoothed(j, t, d) else raw_p)[t] = data_cost(j)
        if not raw_p or not sm_p:
            miss.append(d); continue
        cr, cs = min(raw_p.values()), min(sm_p.values())
        rel = cs / cr - 1.0
        on = cs < cr
        # Resolution: is the gap bigger than the objective's own noise? Below the floor the two
        # variants are indistinguishable and the decision is arbitrary -- report it as such rather
        # than laundering a coin flip into a model specification.
        decided = abs(rel) > a.floor
        tab[d] = bool(on)
        rec[d] = dict(raw=cr, smoothed=cs, rel=rel, smooth=bool(on), decided=bool(decided),
                      raw_pool=raw_p, smooth_pool=sm_p)
        tot_raw += cr; tot_sm += cs; tot_gate += min(cr, cs)
        print(f"  {d:12s} {cr:9.5f} {cs:9.5f} {rel:+7.1%}  {'SMOOTH' if on else 'raw':>8s}  "
              f"{'decided' if decided else 'COIN FLIP (inside the noise floor)'}")
    if miss:
        print(f"\n  MISSING both tags at: {', '.join(miss)} -- not in the table, `_lam_on` will raise")

    nd = sum(1 for v in rec.values() if not v["decided"])
    print(f"\n  TOTAL DATA   raw {tot_raw:.5f}   blanket-smoothed {tot_sm:.5f} "
          f"({tot_sm/tot_raw-1:+.1%})   gated {tot_gate:.5f} ({tot_gate/tot_raw-1:+.1%})")
    print(f"  the gate buys {1-tot_gate/tot_sm:.1%} over blanket smoothing, and {nd} of {len(rec)} "
          f"decisions are inside the staircase noise floor")
    print(f"  IN-SAMPLE: the decision uses the same data as the fit. Quote the blanket number as the "
          f"honest baseline.")

    if a.dry:
        print("\n  --dry: nothing written")
        return
    json.dump(dict(smooth=tab, detail=rec,
                   note="per-date lambda slope-smoothing decision, MEASURED (fit both ways, keep the "
                        "lower DATA cost). Consumed by end_to_end._lam_on via LAMDATES=<this file>. "
                        "In-sample: selection uses the same data as the fit.",
                   floor=a.floor, raw_tags=a.raw, smooth_tags=a.smooth, auto_tags=a.auto),
              open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")
    print(f"  use it with:  LAMDATES={a.out} LAMSMOOTH=3 LAMKEEP=1 LAMSLOPE=1 ...")


if __name__ == "__main__":
    main()
