#!/usr/bin/env python3
"""
Pool the FAITHFUL clean-flank bridge over 2022-2024 events (daily-grid era) -- the clean test of whether the
index-macro event skew is a real systematic tilt or averages to zero (the Zhong-vs-us frontier).

The bridge runs on GLIDE's numpy/scipy code (no GPU); the right speedup for a POOL is event-level parallelism
(each event independent). Aggregates the event variance (implied move) and skew, pooling J3 and J2 separately
(mean(J3)/mean(J2)^1.5) so per-event skew noise averages down. Records wall-time.
    python3 deevent_bridge_pool.py [TICKER=SPX]
"""
import sys, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from deevent_bridge_slv import build_flank as build                   # noqa: E402  one-step-from-flank (skew-faithful)
from deevent_termstruct import FOMC, CPI                                   # noqa: E402

TICKER = sys.argv[1] if len(sys.argv) > 1 else "SPX"


def _run(ev):
    try:
        return build(ev[0], TICKER, 9) | {"type": ev[1]}
    except Exception as e:
        return {"event": ev[0], "type": ev[1], "err": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    events = [(e, "FOMC") for e in FOMC if e >= "2022"] + [(e, "CPI") for e in CPI if e >= "2022"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(12, (os.cpu_count() or 4) - 2)) as ex:
        res = list(ex.map(_run, events))
    ok = [r for r in res if r and "J2" in r and np.isfinite(r.get("J2", np.nan)) and r["J2"] > 0]
    J2 = np.array([r["J2"] for r in ok]); J3 = np.array([r["J3"] for r in ok])
    mv = np.array([r["move"] for r in ok]); n = len(ok)
    skagg = J3.mean() / J2.mean() ** 1.5
    seJ3 = np.std(J3) / np.sqrt(n)
    print(f"FAITHFUL bridge pooled -- {TICKER}, {n}/{len(events)} events (2022-2024 FOMC+CPI, daily grid)\n")
    print(f"  implied event move   sqrt(mean J2) = {np.sqrt(J2.mean())*100:.2f}%   "
          f"(per-event {mv.mean()*100:.2f}% +/- {mv.std()*100:.2f})")
    print(f"  aggregate event skew  mean(J3)/mean(J2)^1.5 = {skagg:+.2f}   (SE {seJ3/J2.mean()**1.5:.2f})")
    print(f"  J3 sign: {int((J3<0).sum())}/{n} negative    t(mean J3) = {J3.mean()/seJ3:+.1f}")
    for typ in ("FOMC", "CPI"):
        s = [r for r in ok if r["type"] == typ]
        if s:
            j2 = np.array([r["J2"] for r in s]); j3 = np.array([r["J3"] for r in s])
            print(f"    {typ}: n={len(s)}  move {np.sqrt(j2.mean())*100:.2f}%  "
                  f"skew {j3.mean()/j2.mean()**1.5:+.2f}  ({int((j3<0).sum())}/{len(s)} neg)")
    bad = [r for r in res if r not in ok]
    if bad:
        print(f"\n  dropped {len(bad)}: " + ", ".join(f"{r.get('event')}({r.get('err','?')[:18]})" for r in bad[:6]))
    print(f"\nwall {time.time()-t0:.0f}s  ({(time.time()-t0)/max(n,1):.1f}s/event effective)")
