#!/usr/bin/env python3
"""Where does the grind VIX-vov RMS come from? Per-maturity decomposition of the MULTI-START fit
   (the theta now in the paper, fit_summary_ms.json). Is the ~22% RMS pulled up by long-dated jumps?
       python3 vix_maturity_breakdown.py [mps|cpu]"""
import os, sys, json
import numpy as np, torch
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "mps"
torch.set_default_dtype(torch.float32); torch.set_default_device(DEVICE)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import calibrate_joint_torch as J

MS = json.load(open(os.path.join(HERE, "fit_summary_ms.json")))
DATES = ["2020-06-01", "2022-06-01", "2023-06-01"]   # COVID (9%) + the two grinds (~22%)

def hfit(x, y):                                       # power-law decay exponent vov~tau^-h
    return -np.polyfit(np.log(x), np.log(y), 1)[0]

for date in DATES:
    if date not in MS: continue
    ctx = J.build_date_ctx(date)
    vdtes, vov_d = np.asarray(ctx["vdtes"], float), np.asarray(ctx["vov_d"], float)
    # stored theta is rounded to 3dp; polish back into the basin so numbers match the paper's exact fit
    anchor = np.asarray(J.WARM.get(date[:4], J.C.X0_MAP["ts"]), float)
    res = J.fit_date(ctx, np.asarray(MS[date]["theta"], float), w_vov=0.8,
                     anchor=anchor, w_reg=J.W_REG, max_nfev=25)
    th = np.asarray(res.x, float)
    m = J.model_torch(torch.tensor(th, dtype=torch.float32), ctx["LT"], ctx["sig_ref"],
                      ctx["spot"], ctx["vdtes"]).detach().cpu().numpy()
    v = m[len(ctx["emp"]):]                            # model vov
    rel = (v - vov_d) / vov_d                          # signed relative error per maturity
    sq  = rel ** 2
    rms_all = 100 * np.sqrt(sq.mean())
    print(f"\n=== {date} ({MS[date]['regime']}): paper VIX RMS {MS[date]['vix']}%  "
          f"(recomputed {rms_all:.1f}%) ===")
    print(f"  data vov h={hfit(vdtes, vov_d):.3f}   model vov h={hfit(vdtes, v):.3f}   "
          f"(steeper model => underhang at the long end)")
    print(f"  {'dte':>5} {'model':>7} {'data':>7} {'relerr%':>8} {'RMS-share%':>10}")
    share = sq / sq.sum() * 100
    for d, mo, da, r, sh in zip(vdtes, v, vov_d, rel, share):
        print(f"  {int(d):>5} {mo:>7.3f} {da:>7.3f} {100*r:>+8.1f} {sh:>10.1f}")
    # RMS if we drop the longest k maturities
    order = np.argsort(vdtes)
    print("  cumulative RMS keeping only the shortest N maturities:")
    for n in range(2, len(vdtes) + 1):
        idx = order[:n]
        rms_n = 100 * np.sqrt(sq[idx].mean())
        print(f"     shortest {n} (<= {int(vdtes[idx].max())}d): RMS {rms_n:>5.1f}%")
print("\n[done]")
