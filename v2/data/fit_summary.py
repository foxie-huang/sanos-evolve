#!/usr/bin/env python3
"""Fit quality (SSR/VIX RMS floors, theta) + WALL-TIME for the 5 cross-regime dates, on GPU (MPS) by default
-- the full joint jacrev is 3.7x faster on MPS than CPU (6s vs 22s). Empirical-SSR targets are cached.
    python3 fit_summary.py [mps|cpu]"""
import sys, os, json, time
import numpy as np
import torch

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "mps"
torch.set_default_dtype(torch.float32); torch.set_default_device(DEVICE)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import calibrate_joint_torch as J                                   # noqa: E402

DATES = ["2017-06-01", "2018-06-01", "2019-06-03", "2020-06-01", "2023-06-01"]
out = {}
t_all = time.time()
for date in DATES:
    t0 = time.time(); ctx = J.build_date_ctx(date); t_ctx = time.time() - t0
    anchor = np.asarray(J.WARM.get(date[:4], J.C.X0_MAP["ts"]), float)
    res = J.fit_date(ctx, anchor, anchor=anchor, w_reg=J.W_REG, max_nfev=30)
    m = J.model_torch(torch.tensor(res.x, dtype=torch.float32), ctx["LT"], ctx["sig_ref"],
                      ctx["spot"], ctx["vdtes"]).detach().cpu().numpy()
    ns = len(ctx["emp"]); s, v = m[:ns], m[ns:]
    ssr = float(100 * np.sqrt(np.mean(((s - ctx["emp"]) / ctx["emp"]) ** 2)))
    vix = float(100 * np.sqrt(np.mean(((v - ctx["vov_d"]) / ctx["vov_d"]) ** 2)))
    out[date] = dict(ssr=round(ssr, 1), vix=round(vix, 1), theta=[round(float(x), 3) for x in res.x],
                     nfev=int(res.nfev), fit_s=round(res.wall, 1), ctx_s=round(t_ctx, 1), device=DEVICE)
    print(f"{date}: SSR {ssr:.1f}% VIX {vix:.1f}%  {res.nfev} evals  fit {res.wall:.0f}s  ctx {t_ctx:.1f}s  [{DEVICE}]", flush=True)
tot = sum(v["fit_s"] for v in out.values())
out["_meta"] = dict(device=DEVICE, total_fit_s=round(tot, 1), avg_fit_s=round(tot / len(DATES), 1),
                    wall_s=round(time.time() - t_all, 1))
json.dump(out, open(os.path.join(HERE, "fit_summary.json"), "w"), indent=1)
print(f"\n[{DEVICE}] total fit {tot:.0f}s, avg {tot/len(DATES):.0f}s/date; wall {time.time()-t_all:.0f}s -> fit_summary.json")
