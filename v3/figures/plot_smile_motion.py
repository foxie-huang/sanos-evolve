#!/usr/bin/env python3
"""Where the calibrated model sits on the sticky map -- `figs/real_dynamics.png`.

REPLACES the v2 route (`v2/data/sticky_check.py` -> `generate_figs.py` FIGURE 4), which built its
lambda with `leverage_at` -- the legacy per-expiry static layer, NOT the SANOS-faithful
`rebuilt_static` the pipeline actually uses -- and propagated on the superseded engine. Both panels
are recomputed here at the shipped theta through `end_to_end.ctx_rebuilt`.

The five coefficients come from `readouts.smile_response`, which is `ssr_ts`'s own closed-form
realised-covariance machinery applied to the skew and curvature fields as well as to the level.
Nothing is finite-differenced by hand and no convention is re-derived. Two internal checks are
printed rather than assumed: beta must equal SSR x skew0 exactly (it is the same quantity before the
last division), and the SSR column must reproduce the fitted record.

    python3 plot_smile_motion.py                 # SPX 2019, the paper's date
    python3 plot_smile_motion.py --date 2020-06-01 --tag _n9
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--date", default="2019-06-03")
ap.add_argument("--tag", default="_n9")
ap.add_argument("--panel", default="1m")      # tenor drawn in panel (b)
ap.add_argument("--out", default=None)
A = ap.parse_args()

os.environ.setdefault("LADDER", "42")
os.environ.setdefault("VOVLAMTEN", "avg")
sys.argv = [sys.argv[0], "cpu"]

import torch                                                          # noqa: E402
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kernel_fast"))
sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                                   # noqa: E402
import consts, fkernel as kernel, readouts as R                       # noqa: E402
import calibrate_slv_exact_ts as C                                    # noqa: E402
import end_to_end as E                                                # noqa: E402
import discslv_torch as DT                                            # noqa: E402

OUT = A.out or os.path.join(os.path.dirname(os.path.dirname(HERE)), "manuscript_v3", "figs")
LAB = ["1wk", "2wk", "1m", "2m", "3m"]
WKS = [1, 2, 4, 8, 13]
C0, C1, C2 = "#1f6f8b", "#b02a37", "#9a6b00"

j = json.load(open(os.path.join(_P.DATA, f"fit_kf{A.tag}_{A.date}.json")))
ctx, _c, _n = E.ctx_rebuilt(A.date, "SPX")                # the SANOS-faithful layer
K = consts.Consts("cpu", torch.float32)
LAM, SIG = ctx["LT"][max(ctx["LT"])], ctx["sig_ref"]
th = torch.tensor([j["theta"][k] for k in C.NAMES_N] + [j["kap_s"]], dtype=torch.float32)
g = kernel.solve_gbar(th, SIG, K)
kk = kernel.build_kernel(kernel.th9(th, g, K), K)
with torch.no_grad():
    M = R.smile_response(kk, LAM, DT._interp_lin).numpy()
D = {LAB[i]: dict(zip(["beta", "d_skew", "d_curv", "skew0", "curv0", "SSR"], M[i])) for i in range(5)}

# --- checks, printed not assumed -------------------------------------------
bad = max(abs(D[l]["beta"] - D[l]["SSR"] * D[l]["skew0"]) for l in LAB)
print(f"  identity  beta == SSR * skew0 : max |diff| {bad:.2e}")
rec = j["ssr"]
print(f"  vs fitted record (ssr_ts)     : max |diff| "
      f"{max(abs(D[l]['SSR'] - r) for l, r in zip(LAB, rec)):.2e}   "
      f"(CPU here, {j.get('device')} in the fit)")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
plt.rcParams.update({"font.size": 9})

# (a) LEVEL -- the SSR term structure against the sticky-map reference levels
ax[0].plot(WKS, [D[l]["SSR"] for l in LAB], "o-", c=C0, zorder=3, label="model")
for y, lab, c in [(0, "sticky-moneyness", C2), (1, "sticky-strike", "gray"), (2, "local vol", C1)]:
    ax[0].axhline(y, ls="--", c=c, lw=1)
    ax[0].text(0.35, y + 0.075, lab, va="bottom", ha="left", fontsize=7.5, color=c)
ax[0].set(title="(a) Level: SSR term structure", xlabel="maturity (weeks)", ylabel="SSR",
          xlim=(0, 14.5), ylim=(-0.25, 2.45))
# no legend in (a): it would carry a single entry, and the three reference levels are already
# labelled in place by the text annotations above.
ax[0].grid(alpha=.3)

# (b) SHAPE -- the smile's own response to a -1% spot move, by moment
d = D[A.panel]
k = np.linspace(-0.05, 0.05, 101)
r = -0.01
ax[1].plot(k * 100, r * (d["beta"] + d["d_skew"] * k + 0.5 * d["d_curv"] * k ** 2) * 100,
           c=C0, lw=1.8, label="model")
ax[1].axhline(0, ls="--", c=C2, lw=1.2, label="sticky-moneyness")
ax[1].plot(k * 100, r * (d["skew0"] + d["curv0"] * k) * 100, ls="--", c="gray", lw=1.2,
           label="sticky-strike")
ax[1].set(title=rf"(b) Shape: {A.panel} smile shift on $-1\%$ spot",
          xlabel=r"log-moneyness $k$ (%)", ylabel=r"$\Delta\sigma(k)$ (vol pts)")
ax[1].legend(frameon=False, fontsize=8)
ax[1].grid(alpha=.3)

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "real_dynamics.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
json.dump({l: {kk_: float(vv) for kk_, vv in D[l].items()} for l in LAB},
          open(os.path.join(_P.DATA, "smile_response_spx.json"), "w"), indent=1)
print(f"  wrote {p}")
print(f"  panel (b) at {A.panel}: beta {d['beta']:+.3f}  d_skew {d['d_skew']:+.3f}  "
      f"skew0 {d['skew0']:+.3f}  -> d_skew/(2*curv0) = {d['d_skew']/(2*d['curv0']):+.4f}")
