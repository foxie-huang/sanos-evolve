#!/usr/bin/env python3
"""Freeze the SANOS static layer's contribution to the kernel fit -- 40 floats per date-ticker.

Everything the joint LP produces (K anchors, V variances, qs densities, Ts) is consumed to build
exactly two objects that reach `model_torch`:

    sig_ref   one scalar; pins gamma-bar via E_pi[V] = sig_ref^2 dt, and enters the VIX leg
    LT        13 degree-1 log-lambda polynomials (coef[2], zmax), one per weekly step

That is 1 + 13*3 = 40 floats. The LP output for SPX 2022 alone is 9,750 floats, so the fit sees
0.41% of it. Freezing those 40 removes `rebuilt_static` -- and with it the Epi_V/TwoFactorSV
construction and the npz load -- from every worker startup.

The npz is written at FULL precision. panel_leverage_record.json is the human-readable record and is
rounded (5-6 dp); do not fit from it.

    python3 static_payload.py            # freeze all 18
    python3 static_payload.py --verify   # freeze, then check against live rebuilt_static
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths as _P                        # noqa: E402
HERE = _P.DATA                             # code moved; fits/caches/records did not

# (was sys.path.insert(0, HERE) -- HERE is now the DATA dir; sibling modules come
#  from _paths, which puts the v3_scripts code dirs ahead of it.)
import sanos_true as ST   # noqa: E402  -- owns ETA; see its docstring
PAYLOAD = os.path.join(HERE, "static_payload.npz")
PANEL = {"2012": "2012-06-01", "2016": "2016-06-01", "2017": "2017-06-01", "2018": "2018-06-01",
         "2019": "2019-06-03", "2020": "2020-06-01", "2021": "2021-06-01", "2022": "2022-06-01",
         "2024": "2024-06-03"}
# PAYLOAD_NWK sets how many weekly lambda rungs the payload stores. 13 was max(J.NS) -- correct when
# only SSR read LT, but VOVLEV=1's leveraged VIX propagates to max_vov_tenor + the 30d VIX window:
# 42 weeks at 2021/2022/2024 (262d tenor + 30d = 292d), 29-30 elsewhere. Frozen at 13 the tail is
# lambda-frozen and LADDER>13 needs NOFREEZE=1, i.e. a full rebuild on every fit.
# `load` already reads the count back from _meta[2], so a payload frozen deeper is read correctly
# with no change there. Freeze with: LADDER=42 PAYLOAD_NWK=42 python3 static_payload.py --verify
NWK = int(os.environ.get("PAYLOAD_NWK", "13"))   # was max(J.NS)=13; see above
ETA, DEG = ST.ETA, 1          # re-export, NOT a second declaration


def key(ticker, date):
    return f"{ticker}_{date}"


def freeze(eta=ETA, deg=DEG, path=PAYLOAD):
    """Solve/replay the static layer for the panel and write the 40-float payload per date-ticker."""
    # (was sys.path.insert(0, HERE) -- HERE is now the DATA dir; sibling modules come
#  from _paths, which puts the v3_scripts code dirs ahead of it.)
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "v2", "poc")))
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "v2", "data")))
    sys.argv = [sys.argv[0], "cpu"]
    import end_to_end as E

    out = {}
    was_off = E.FREEZE_OFF
    E.FREEZE_OFF = True          # read the LP, NOT the payload this call is about to overwrite
    try:
        for date in PANEL.values():
            for tk in ("SPX", "NDX"):
                sig_ref, lev, clipped, npil = E.rebuilt_static(date, tk, eta=eta, deg=deg)
                k = key(tk, date)
                out[f"{k}/sig_ref"] = np.float64(sig_ref)
                out[f"{k}/coef"] = np.array([lev[i + 1].coef for i in range(NWK)], dtype=np.float64)
                out[f"{k}/zmax"] = np.array([lev[i + 1].zmax for i in range(NWK)], dtype=np.float64)
                out[f"{k}/diag"] = np.array([clipped, npil], dtype=np.int64)
                print(f"  {k}  sig_ref {sig_ref:.6f}  clip {clipped}  pillars {npil}", flush=True)
    finally:
        E.FREEZE_OFF = was_off
    out["_meta"] = np.array([eta, deg, NWK] + list(E.SAFETY), dtype=np.float64)
    np.savez(path, **out)
    _CACHE.pop(path, None)       # drop the in-process copy of the file we just replaced
    n = len(PANEL) * 2
    print(f"\nfroze {n} date-tickers -> {path}  ({os.path.getsize(path)/1024:.0f} KB, "
          f"{n * (1 + 3 * NWK)} floats)")
    return path


class Lev:
    """Same interface as end_to_end.Lev / slv_wire._Lev, so lev_torch consumes it unchanged."""

    __slots__ = ("coef", "zmax", "safety")

    def __init__(self, coef, zmax, safety):
        self.coef, self.zmax, self.safety = np.asarray(coef, float), float(zmax), safety


_CACHE = {}


def load(date, ticker, eta=ETA, deg=DEG, path=PAYLOAD):
    """(sig_ref, {k: Lev}, clipped, npillars) from the frozen payload, or None if unavailable.

    Returns None -- rather than raising -- when the file is absent or was frozen under different
    (eta, deg), so callers fall back to recomputation instead of silently fitting the wrong layer.
    """
    if not os.path.exists(path):
        return None
    z = _CACHE.get(path)
    if z is None:
        z = _CACHE[path] = dict(np.load(path))
    meta = z.get("_meta")
    if meta is None or abs(float(meta[0]) - eta) > 1e-12 or int(meta[1]) != int(deg):
        return None
    k = key(ticker, date)
    if f"{k}/sig_ref" not in z:
        return None
    nwk, safety = int(meta[2]), (float(meta[3]), float(meta[4]))
    coef, zmax, diag = z[f"{k}/coef"], z[f"{k}/zmax"], z[f"{k}/diag"]
    lev = {i + 1: Lev(coef[i], zmax[i], safety) for i in range(nwk)}
    return float(z[f"{k}/sig_ref"]), lev, int(diag[0]), int(diag[1])


if __name__ == "__main__":
    want_verify = "--verify" in sys.argv        # freeze() overwrites sys.argv for the torch device arg
    freeze()
    if want_verify:
        import end_to_end as E
        print("\nverify frozen vs live rebuilt_static:")
        ws = wc = 0.0
        for date in PANEL.values():
            for tk in ("SPX", "NDX"):
                s0, l0, c0, p0 = E.rebuilt_static(date, tk, eta=ETA, deg=DEG)
                s1, l1, c1, p1 = load(date, tk)
                ws = max(ws, abs(s0 - s1))
                wc = max(wc, max(float(np.abs(l0[k].coef - l1[k].coef).max()) for k in l0))
                assert (c0, p0) == (c1, p1), f"{tk} {date}: diag {(c0,p0)} != {(c1,p1)}"
                assert all(l0[k].zmax == l1[k].zmax and l0[k].safety == l1[k].safety for k in l0)
        print(f"  worst sig_ref drift {ws:.3e} | worst coef drift {wc:.3e}  (expect 0.0 -- full precision)")
