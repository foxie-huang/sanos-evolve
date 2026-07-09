#!/usr/bin/env python3
"""
CONTROL for the faithful bridge: run the identical procedure on NON-EVENT dates. If the pooled event skew
(39/44 negative) is real, controls should give J3 ~ 0, sign-balanced. If controls are ALSO systematically
negative, the negative "event skew" is a Level-1-kernel artifact (the constant kernel under-produces the
diffusive put-skew, leaving a spurious negative residual). Parallel across dates. Records wall-time.
    python3 deevent_bridge_control.py [TICKER=SPX]
"""
import sys, os, time, glob
import numpy as np
from datetime import date as D
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from deevent_bridge_slv import build_flank as build                   # noqa: E402  one-step-from-flank (skew-faithful)
from deevent_termstruct import FOMC, CPI, OUT                              # noqa: E402
from deevent_ssr import nfp_dates                                          # noqa: E402

TICKER = sys.argv[1] if len(sys.argv) > 1 else "SPX"


def _run(d):
    try:
        return build(d, TICKER, 9)
    except Exception as e:
        return {"event": d, "err": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    t0 = time.time()
    evd = [D.fromisoformat(e) for e in set(FOMC) | set(CPI) | set(nfp_dates([2022, 2023, 2024]))]
    cands = []                                                             # trading days >=5d from ANY event
    for p in sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_202[234]-*.json.gz")):
        d = os.path.basename(p).split("_")[1][:10]; dd = D.fromisoformat(d)
        if all(abs((dd - e).days) >= 5 for e in evd):
            cands.append(d)
    ctrl = cands[::max(1, len(cands) // 24)][:24]                          # ~24 controls spread across 2022-2024
    with ProcessPoolExecutor(max_workers=min(12, (os.cpu_count() or 4) - 2)) as ex:
        res = list(ex.map(_run, ctrl))
    ok = [r for r in res if r and "J2" in r and np.isfinite(r.get("J2", np.nan)) and r["J2"] > 0]
    J2 = np.array([r["J2"] for r in ok]); J3 = np.array([r["J3"] for r in ok]); n = len(ok)
    seJ3 = np.std(J3) / np.sqrt(n)
    print(f"CONTROL (non-event) faithful bridge -- {TICKER}, {n}/{len(ctrl)} clean dates (2022-2024)\n")
    print(f"  pseudo-move   sqrt(mean J2) = {np.sqrt(J2.mean())*100:.2f}%   (should be << 1.5%, the event move)")
    print(f"  pseudo-skew   mean(J3)/mean(J2)^1.5 = {J3.mean()/J2.mean()**1.5:+.2f}   (SE {seJ3/J2.mean()**1.5:.2f})")
    print(f"  J3 sign: {int((J3<0).sum())}/{n} negative    t(mean J3) = {J3.mean()/seJ3:+.1f}")
    print(f"\n  READ: controls ~0 & balanced -> event skew REAL; controls also ~90% neg -> Level-1 artifact.")
    print(f"wall {time.time()-t0:.0f}s")
