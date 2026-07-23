#!/usr/bin/env python3
"""
Target caching: the fit's objective is (model(theta)-target)/target, and the TARGET is theta-independent
-- the realized-SSR regression over the year (~30-60s), the VIX quotes, the SANOS chain/leverage. In a
per-date backtest re-fitting each day, computing these once and reusing them removes the ~1-min setup from
every fit. This caches the expensive empirical-SSR (per year x NS x dt) to disk; the per-date statics are
cheap and computed live.  Demo:  python3 target_cache.py
"""
import os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from empirical_ssr import empirical_ssr                              # noqa: E402

CACHE = os.path.join(HERE, ".target_cache"); os.makedirs(CACHE, exist_ok=True)
OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))


def emp_ssr_cached(year, ns, dt):
    """Empirical (realized) SSR target for a year, cached to disk by (year, NS, dt). First call runs the
    ~30-60s regression; later calls load instantly. This is the target reused across a year's per-date fits."""
    key = os.path.join(CACHE, f"empssr_{year}_{'-'.join(map(str, ns))}_{round(dt, 6)}.npz")
    if os.path.exists(key):
        d = np.load(key); return d["emp"], int(d["nd"]), True
    paths = sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz"))
    emp, nd = empirical_ssr(paths, ns=ns, dt=dt)
    np.savez(key, emp=emp, nd=nd)
    return emp, nd, False


if __name__ == "__main__":
    import time
    DT = 1.0 / 52.0; NS = [1, 2, 4, 8, 13]
    key = os.path.join(CACHE, f"empssr_2015_{'-'.join(map(str, NS))}_{round(DT, 6)}.npz")
    if os.path.exists(key):
        os.remove(key)                                               # force the cold path for the demo
    t0 = time.time(); emp, nd, hit = emp_ssr_cached("2015", NS, DT); t_cold = time.time() - t0
    t0 = time.time(); emp2, nd2, hit2 = emp_ssr_cached("2015", NS, DT); t_warm = time.time() - t0
    print(f"  cold (compute + cache): {t_cold:6.1f}s   hit={hit}")
    print(f"  warm (load from cache): {t_warm:6.3f}s   hit={hit2}   speedup {t_cold/max(t_warm,1e-6):.0f}x")
    print(f"  emp SSR 2015 ({nd} days): {np.round(emp, 3)}   (identical: {np.allclose(emp, emp2)})")
    print(f"\n  In a 252-date backtest: the empirical target is computed ONCE, not 252x")
    print(f"  -> removes ~{t_cold:.0f}s setup from each subsequent fit; per-fit cost becomes just the iterations.")
