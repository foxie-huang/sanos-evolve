#!/usr/bin/env python3
"""Corrected VIX experiment: MULTI-START re-fits (single-start fell into a bad basin).
   Does a properly-fit two-factor theta flatten the vov, or is h~0.34 a real floor?
   - w_vov=2.0  : proper SSR<->vov tradeoff (multi-start)
   - w_vov=20   : vov-FOCUSED (near-ignore SSR) -> the best two-factor vov achievable
Baseline (w_vov=0.8): SSR 6.2%/VIX 28.1%, model h=0.36; data h=0.112.
    python3 vix_experiment2.py [mps|cpu]"""
import os, sys, json, time
import numpy as np, torch
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "mps"
torch.set_default_dtype(torch.float32); torch.set_default_device(DEVICE)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import calibrate_joint_torch as J

DATE = "2022-06-01"
ctx = J.build_date_ctx(DATE)
vdtes, vov_d, sig_ref, spot = ctx["vdtes"], ctx["vov_d"], ctx["sig_ref"], ctx["spot"]
lx = np.log(vdtes)
def hfit(y): return -np.polyfit(lx, np.log(y), 1)[0]
def rms(a, b): return 100 * np.sqrt(np.mean(((a - b) / b) ** 2))
anchor = np.asarray(J.WARM.get(DATE[:4], J.C.X0_MAP["ts"]), float)
print(f"data vov h={hfit(vov_d):.3f}  ({vov_d[0]:.2f}->{vov_d[-1]:.2f});  baseline w_vov=0.8 -> SSR 6.2% VIX 28.1%")

for wv in [2.0, 20.0]:
    t0 = time.time()
    res = J.fit_date_multistart(ctx, anchor, J.W_REG, max_nfev=30, w_vov=wv)
    m = J.model_torch(torch.tensor(res.x, dtype=torch.float32), ctx["LT"], sig_ref, spot, vdtes).detach().cpu().numpy()
    nsr = len(ctx["emp"]); s, v = m[:nsr], m[nsr:]
    ssr, vix = rms(s, ctx["emp"]), rms(v, vov_d)
    costs = getattr(res, "costs", None)
    print(f"\nw_vov={wv} (multi-start {getattr(res,'n_starts','?')} seeds, best cost, {time.time()-t0:.0f}s):")
    print(f"   SSR {ssr:.1f}%   VIX {vix:.1f}%   model-h {hfit(v):.3f}   kap_f={res.x[-2]:.2f} kap_s={res.x[-1]:.2f}")
    print(f"   seed costs: {[round(float(c),4) for c in costs] if costs is not None else 'n/a'}")
    print(f"   model vov: " + " ".join(f"{int(d)}d:{a:.2f}(data {b:.2f})" for d, a, b in zip(vdtes[::2], v[::2], vov_d[::2])))
print("\n[done]")
