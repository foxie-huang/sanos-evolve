#!/usr/bin/env python3
"""Collect the nine per-date `fit_kf<TAG>_*.json` into one panel record: JSON + a flat CSV.

The JSON carries the configuration ONCE (it must be identical across the nine, and this asserts it)
plus per-date theta, series and diagnostics. The CSV is one row per (date, block, tenor) for anything
that wants a dataframe. `panel_ref_SPX.json` / `panel_ref2_SPX.json` / `panel_sl9_SPX.json` were built
inline in a session and had no script; this is that script, so the next panel is one command.

    python3 panel_record.py _g9
    python3 panel_record.py _g9 --title "SPX gated panel"
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                          # noqa: E402

PANEL = ["2012-06-01", "2016-06-01", "2017-06-01", "2018-06-01", "2019-06-03",
         "2020-06-01", "2021-06-01", "2022-06-01", "2024-06-03"]
SSR_WK = [1, 2, 4, 8, 13]
# Configuration that must agree across the nine dates or the panel is not one panel.
SHARED = ["vovlev", "vixfix", "vovlamten", "ladder", "blend", "backend", "device",
          "kap_s_fitted", "box", "seed0", "noridge", "monopen"]


def data_cost(j):
    """The DATA half of the objective -- no ridge. Matches refit.py's residual weighting."""
    s = np.asarray(j["ssr"], float); t = np.asarray(j["ssr_target"], float)
    v = np.asarray(j["vov"], float); vt = np.asarray(j["vov_target"], float)
    w = 0.8 * np.sqrt(len(s) / len(v))
    return 0.5 * np.sum(((s - t) / t) ** 2) + 0.5 * np.sum((((v - vt) / vt) * w) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="fit tag, e.g. _g9")
    ap.add_argument("--title", default=None)
    ap.add_argument("--date", default="2026-08-10", help="record date (scripts cannot read the clock)")
    ap.add_argument("--ticker", default="SPX", help="SPX (default) or NDX; NDX fits carry an _ndx suffix")
    a = ap.parse_args()

    fits, cfg = {}, None
    for d in PANEL:
        sfx = "" if a.ticker.upper() == "SPX" else f"_{a.ticker.lower()}"
        p = os.path.join(_P.DATA, f"fit_kf{a.tag}_{d}{sfx}.json")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run the panel first")
        fits[d] = json.load(open(p))
        c = {k: fits[d].get(k) for k in SHARED}
        if cfg is None:
            cfg, cfg_date = c, d
        elif c != cfg:
            diff = {k: (cfg[k], c[k]) for k in c if c[k] != cfg[k]}
            raise SystemExit(f"CONFIG MISMATCH {cfg_date} vs {d}: {diff} -- these are not one panel")

    # The lambda block varies BY DESIGN under a per-date gate, so it is recorded per date and
    # summarised, not asserted equal.
    lam = {d: fits[d].get("lam") for d in PANEL}
    out = dict(panel=a.title or f"{a.ticker.upper()} {a.tag}", ticker=a.ticker.upper(),
               created=a.date, config=cfg,
               lam_gate={d: (lam[d] or {}).get("applied") for d in PANEL},
               lam_config={k: (lam[PANEL[0]] or {}).get(k)
                           for k in ("smooth", "keep", "slope_only", "sg", "nofreeze", "dates")},
               dates={})
    rows = []
    tot = 0.0
    for d in PANEL:
        j = fits[d]
        vd = j.get("vov_tenor_d")
        if vd is None:
            # Fits do not store the vov tenors, so take them from the CONTEXT. This replaced a
            # fallback that read `panel_ref2_SPX.json` -- an SPX file, which could not serve NDX at
            # all, and which the 6e.23a purge had already broken once.
            import end_to_end as _E                                   # noqa: E402  (lazy: pulls torch)
            vd = [int(x) for x in _E.ctx_rebuilt(d, a.ticker.upper())[0]["vdtes"]]
        # The context is rebuilt from the CURRENT environment, so a fit made with NDXTENORS set and
        # replayed without it gets the default 2-tenor axis against 8 values. Refuse rather than
        # write a record whose x and y disagree -- matplotlib catching it downstream is luck.
        if len(vd) != len(j["vov"]):
            raise SystemExit(f"{d}: fit has {len(j['vov'])} vov values but the tenor axis has "
                             f"{len(vd)} ({vd}). If this fit used NDXTENORS, set it for this run too.")
        c = data_cost(j); tot += c
        out["dates"][d] = dict(theta=j["theta"], kap_s=j["kap_s"], data_cost=c,
                               ssr_rms=j["ssr_rms"], vov_rms=j["vov_rms"], njev=j["njev"],
                               wall=j.get("wall"), seed=j.get("seed"),
                               lam_applied=(lam[d] or {}).get("applied"),
                               ssr_tenor_wk=SSR_WK, ssr=j["ssr"], ssr_target=j["ssr_target"],
                               vov_tenor_d=vd, vov=j["vov"], vov_target=j["vov_target"])
        for lab, blk, xs in (("ssr", "ssr", SSR_WK), ("vov", "vov", vd)):
            for i, x in enumerate(xs):
                m, t = j[blk][i], j[f"{blk}_target"][i]
                rows.append([d, lab, f"{x}{'wk' if lab == 'ssr' else 'd'}",
                             f"{m:.6f}", f"{t:.6f}", f"{100*(m/t-1):.3f}"])
    out["total_data_cost"] = tot

    pj = os.path.join(_P.DATA, f"panel{a.tag}_{a.ticker.upper()}.json")
    pc = os.path.join(_P.DATA, f"panel{a.tag}_{a.ticker.upper()}.csv")
    json.dump(out, open(pj, "w"), indent=1)
    with open(pc, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["date", "block", "tenor", "model", "target", "rel_err_pct"])
        w.writerows(rows)
    print(f"  wrote {pj}\n  wrote {pc}")
    print(f"  TOTAL DATA cost {tot:.5f} over {len(PANEL)} dates")
    on = [d[:4] for d in PANEL if out["lam_gate"][d]]
    off = [d[:4] for d in PANEL if out["lam_gate"][d] is False]
    if on or off:
        print(f"  gate: smoothed {on}   raw {off}")


if __name__ == "__main__":
    main()
