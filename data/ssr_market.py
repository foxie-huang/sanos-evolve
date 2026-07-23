#!/usr/bin/env python3
"""Market OBSERVED (realized, P-measure) SSR term structure for SPX and NDX, side by side, per year.
SSR(T) = Cov(dSigma_ATM(T), r)/Var(r)/mean(skew(T)) over the year (empirical_ssr). Records wall-time.
    python3 ssr_market.py [YEAR ...]
"""
import sys, os, glob, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from empirical_ssr import empirical_ssr                                      # noqa: E402

OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
NS = [1, 2, 4, 8, 13, 26, 52]; LAB = ["1wk", "2wk", "1m", "2m", "3m", "6m", "1y"]
YEARS = sys.argv[1:] or ["2017", "2019", "2021"]

if __name__ == "__main__":
    t0 = time.time()
    print("Market observed (realized) SSR term structure -- Cov(dSigma_ATM, r)/Var(r)/mean(skew)\n")
    print(f"{'year':>6} {'tkr':>4} {'n':>4} | " + "  ".join(f"{l:>5}" for l in LAB))
    for yr in YEARS:
        paths = sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{yr}-*.json.gz"))
        for tkr in ("SPX", "NDX"):
            try:
                ssr, n = empirical_ssr(paths, ns=NS, ticker=tkr)
                print(f"{yr:>6} {tkr:>4} {n:>4} | " + "  ".join(f"{v:>5.2f}" for v in ssr))
            except Exception as e:
                print(f"{yr:>6} {tkr:>4}    - | ({type(e).__name__}: {e})")
        print()
    print(f"wall {time.time()-t0:.0f}s")
