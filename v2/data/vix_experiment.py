#!/usr/bin/env python3
"""No-extra-factor VIX-vov experiments on 2022-06-01:
   (A) is the vov TERM-STRUCTURE SHAPE node-sensitive?  (finer GH grid at FIXED theta -> power-law h)
   (B) up-weight the vov and re-fit -> SSR<->vov tradeoff (does a slower kappa_S flatten it?)
Baseline (5/3 nodes, w_vov=0.8): SSR 6.2% / VIX 28.1%; data vov power-law h=0.112 vs model h=0.36.
    python3 vix_experiment.py [mps|cpu]"""
import os, sys, json, time
import numpy as np, torch
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "mps"
torch.set_default_dtype(torch.float32); torch.set_default_device(DEVICE)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import calibrate_joint_torch as J
DTt, TB, DT, NS = J.DTt, J.TB, J.DT, J.NS

DATE = "2022-06-01"
ctx = J.build_date_ctx(DATE)
vdtes, vov_d, sig_ref, spot = ctx["vdtes"], ctx["vov_d"], ctx["sig_ref"], ctx["spot"]
th8 = np.asarray(json.load(open(os.path.join(HERE, "fit_summary.json")))[DATE]["theta"], float)
lx = np.log(vdtes)
def hfit(y): return -np.polyfit(lx, np.log(y), 1)[0]        # power-law exponent (steepness)
def rms(v):  return 100 * np.sqrt(np.mean(((v - vov_d) / vov_d) ** 2))
print(f"data vov: power-law h={hfit(vov_d):.3f}  ({vov_d[0]:.2f} @ {int(vdtes[0])}d -> {vov_d[-1]:.2f} @ {int(vdtes[-1])}d)")

# ---- (A) node sensitivity of the vov SHAPE at the fitted theta (gbar solved at 5/3; compare shapes) ----
print("\n(A) finer GH grid at FIXED theta (does the decay flatten toward data h=0.11?):")
th9 = torch.cat([J.solve_gbar_torch(torch.tensor(th8, dtype=torch.float32), sig_ref, DT).reshape(1),
                 torch.tensor(th8, dtype=torch.float32)])
for nf, ns in [(5, 3), (7, 4), (9, 5)]:
    ker = DTt.build_kernel(th9, DT, nf, ns, 5)
    vov = np.array([float(DTt.vix_ivol(ker, sig_ref, float(d) / 365.0, spot)[1]) for d in vdtes])
    print(f"   n_f={nf} n_s={ns}:  model h={hfit(vov):.3f}   vov {vov[0]:.2f}->{vov[-1]:.2f}   "
          f"(shape-RMS at this theta {rms(vov):.1f}%)")

# ---- (B) up-weight the vov and re-fit (SSR<->vov tradeoff) --------------------------------------------
print("\n(B) up-weight vov and re-fit (baseline w_vov=0.8 -> SSR 6.2%/VIX 28.1%):")
anchor = np.asarray(J.WARM.get(DATE[:4], J.C.X0_MAP["ts"]), float)
for wv in [2.0, 5.0]:
    t0 = time.time()
    res = J.fit_date(ctx, anchor, w_vov=wv, anchor=anchor, w_reg=J.W_REG, max_nfev=30)
    m = J.model_torch(torch.tensor(res.x, dtype=torch.float32), ctx["LT"], sig_ref, spot, vdtes).detach().cpu().numpy()
    nsr = len(ctx["emp"]); s, v = m[:nsr], m[nsr:]
    ssr = 100 * np.sqrt(np.mean(((s - ctx["emp"]) / ctx["emp"]) ** 2))
    kap_s = res.x[-1]                                       # slow rate (last param)
    print(f"   w_vov={wv}:  SSR {ssr:.1f}%  VIX {rms(v):.1f}%  model-h {hfit(v):.3f}  kap_s={kap_s:.2f}  "
          f"({res.nfev} evals, {time.time()-t0:.0f}s)", flush=True)
print("\n[done]")
