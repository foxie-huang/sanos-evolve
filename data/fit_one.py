#!/usr/bin/env python3
"""Fit ONE date's joint SSR+VIX with the (now clean-spot) pipeline and append to fit_summary.json.
Mirrors fit_summary.py exactly (same anchor / ridge W_REG / max_nfev=30 / device), so the result is
directly comparable to the cross-regime dates already there.
    python3 fit_one.py 2022-06-01 [mps|cpu]"""
import sys, os, json, time
import numpy as np, torch

DATE = sys.argv[1]
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "mps"
torch.set_default_dtype(torch.float32); torch.set_default_device(DEVICE)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import calibrate_joint_torch as J                                     # noqa: E402

t0 = time.time(); ctx = J.build_date_ctx(DATE); t_ctx = time.time() - t0
anchor = np.asarray(J.WARM.get(DATE[:4], J.C.X0_MAP["ts"]), float)
res = J.fit_date(ctx, anchor, anchor=anchor, w_reg=J.W_REG, max_nfev=30)
m = J.model_torch(torch.tensor(res.x, dtype=torch.float32), ctx["LT"], ctx["sig_ref"],
                  ctx["spot"], ctx["vdtes"]).detach().cpu().numpy()
ns = len(ctx["emp"]); s, v = m[:ns], m[ns:]
ssr = float(100 * np.sqrt(np.mean(((s - ctx["emp"]) / ctx["emp"]) ** 2)))
vix = float(100 * np.sqrt(np.mean(((v - ctx["vov_d"]) / ctx["vov_d"]) ** 2)))
print(f"{DATE}: SSR {ssr:.1f}% VIX {vix:.1f}%  emp_target={np.round(ctx['emp'],3).tolist()}  "
      f"{res.nfev} evals  fit {res.wall:.0f}s  ctx {t_ctx:.1f}s  [{DEVICE}]", flush=True)

fp = os.path.join(HERE, "fit_summary.json")
d = json.load(open(fp)) if os.path.exists(fp) else {}
d[DATE] = dict(ssr=round(ssr, 1), vix=round(vix, 1), theta=[round(float(x), 3) for x in res.x],
               nfev=int(res.nfev), fit_s=round(res.wall, 1), ctx_s=round(t_ctx, 1), device=DEVICE)
json.dump(d, open(fp, "w"), indent=1)
print(f"appended {DATE} to fit_summary.json")
