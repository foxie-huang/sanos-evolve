#!/usr/bin/env python3
"""Is the model bit-reproducible on this device? Residual AND Jacobian, repeated evaluations.

WHY IT CAN FAIL. Float `scatter_add` / `index_add` lower to atomic adds on a GPU. The order in
which the ~75 contributions to one output slot arrive depends on warp scheduling, float addition is
not associative, so the sum differs run to run. Measured raw on MPS at (B, M, G)=(225, 18000, 240):

    op                            CPU         MPS
    scatter_add (18000 -> 240)   0.00e+00    6.08e-07    <-- non-deterministic
    index_add   (18000 -> 240)   0.00e+00    5.06e-07    <-- non-deterministic
    sum / cumsum / argsort       0.00e+00    0.00e+00
    searchsorted / elementwise   0.00e+00    0.00e+00

6e-07 would be harmless on its own. It is not harmless here, because recompression bands by
EQUAL WEIGHT: a rounding-level change in a cumulative weight tips a component across a quantile
boundary, it lands in a different cell, and a band flip costs O(1e-3) on the readout. That is a
four-order amplification, and it made the same theta score differently on every evaluation --
which silently corrupts any finite-difference check, any tolerance probe, and least_squares itself,
since trf compares a predicted reduction against a measured one.

THE FIX (see bands.seg_bounds, propagate.recompress). Both segment sums are built from primitives
that ARE bit-deterministic on MPS -- argsort, cumsum, gather, searchsorted -- by sorting on the
segment key and differencing a prefix sum, so the summation order is fixed by the data rather than
by the scheduler. `DETSEG=0` restores the old float scatter_add for A/B.

Two traps met while fixing it, both recorded in bands.py:
  * counting segment members with an INTEGER scatter_add (exact, so order cannot matter) is correct
    but 9x SLOWER than the float scatter it replaces -- int64 scatter_add is 83.6 ms on MPS vs
    9.0 ms for fp32. `searchsorted` on the already-sorted keys costs 9.7 ms and needs no counts.
  * prepending a zero column to make an exclusive prefix (`torch.cat`) copies the whole (B, 18000)
    tensor per call: 82.57 s vs 0.45 s on the SSR block, MPS, matched otherwise. `_pref` indexes
    `idx - 1` under a guard instead.

    python3 det_check.py                 # both devices, 5 repeats
    DEV=mps REP=10 python3 det_check.py  # one device, harder
"""
import os
import sys
import time

DATE = os.environ.get("DATE", "2016-06-01")
REP = int(os.environ.get("REP", "5"))
DEVS = [os.environ["DEV"]] if os.environ.get("DEV") else ["cpu", "mps"]
sys.argv = [sys.argv[0], "cpu"]

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                         # noqa: E402,F401
import consts, fkernel as kernel, readouts, vix as VX       # noqa: E402
import discslv_torch as D                                   # noqa: E402
import calibrate_joint_torch as J                           # noqa: E402
import calibrate_slv_exact_ts as C                          # noqa: E402
import end_to_end as E                                      # noqa: E402

ctx, _c, _p = E.ctx_rebuilt(DATE, "SPX")
LAM, SIG, SPOT, VD = ctx["LT"][max(J.NS)], ctx["sig_ref"], ctx["spot"], list(ctx["vdtes"])


def spread(vs):
    """Max elementwise range across repeats, relative to the mean magnitude."""
    m = np.stack([np.asarray(v, np.float64).ravel() for v in vs])
    return float((m.max(0) - m.min(0)).max() / max(np.abs(m).mean(), 1e-30))


print(f"SPX {DATE}   DETSEG={os.environ.get('DETSEG', '1')}   BANDMODE="
      f"{os.environ.get('BANDMODE', 'stable')}   {REP} repeats at one fixed theta\n")
print(f"  {'dev':4s} {'quantity':12s} {'repeats':>7s} {'spread':>11s} {'s/eval':>8s}   verdict")

for dev in DEVS:
    K = consts.Consts(dev, torch.float32)
    n_var = max(1, int(round((30.0 / 365.0) / K.dt)))
    th = torch.tensor(list(C.X0_MAP_N["ts"]) + [C.KAP_S_FIXED], dtype=torch.float32, device=dev)

    def model(t):
        """SSR term structure and the LEVERAGED (VOVLEV=1) vov, i.e. what the fit actually sees."""
        g = kernel.solve_gbar(t, SIG, K, C.KAP_S_FIXED)
        kk = kernel.build_kernel(kernel.th9(t, g, K, C.KAP_S_FIXED), K)
        ssr = readouts.ssr_ts(kk, LAM, D._interp_lin, readouts.atm_skew)
        u0 = VX.solve_us0(kk, SIG, SPOT, n_var)
        vov = torch.stack([VX.vix_ivol(kk, SIG, float(d) / 365.0, SPOT, lam_fns=LAM, us0=u0)[1]
                           for d in VD])
        return torch.cat([ssr, vov])

    for name, fn in (("residual", lambda: model(th)),
                     ("jacobian", lambda: torch.func.jacfwd(model)(th))):
        try:
            fn()                                            # warm; excluded from the timing
            t0 = time.time()
            vs = [fn().detach().cpu().numpy() for _ in range(REP)]
            per = (time.time() - t0) / REP
        except Exception as e:
            print(f"  {dev:4s} {name:12s} {'-':>7s} {'-':>11s} {'-':>8s}   FAILED: {str(e)[:44]}")
            continue
        s = spread(vs)
        print(f"  {dev:4s} {name:12s} {REP:7d} {s:11.3e} {per:8.2f}   "
              f"{'DETERMINISTIC' if s == 0.0 else 'NON-DETERMINISTIC'}")
