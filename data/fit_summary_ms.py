#!/usr/bin/env python3
"""MULTI-START cross-regime fits at the paper's standard w_vov=0.8 (fixes the single-start bad-basin
issue found 2026-07-08: single-start 2022 gave VIX 28% but multi-start w_vov=2 gave 17%). Same dates as
tab:crossregime. Writes fit_summary_ms.json; compare to fit_summary.json (single-start).
    python3 fit_summary_ms.py [mps|cpu]"""
import sys, os, json, time
import numpy as np, torch
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "mps"
torch.set_default_dtype(torch.float32); torch.set_default_device(DEVICE)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import calibrate_joint_torch as J

DATES = ["2015-06-01", "2019-06-03", "2020-06-01", "2022-06-01", "2023-06-01"]   # tab:crossregime regimes
LABEL = {"2015": "calm", "2019": "moderate", "2020": "COVID", "2022": "grind", "2023": "grind"}
out = {}; t_all = time.time()
for date in DATES:
    t0 = time.time(); ctx = J.build_date_ctx(date); t_ctx = time.time() - t0
    anchor = np.asarray(J.WARM.get(date[:4], J.C.X0_MAP["ts"]), float)
    res = J.fit_date_multistart(ctx, anchor, J.W_REG, max_nfev=30, w_vov=0.8)
    m = J.model_torch(torch.tensor(res.x, dtype=torch.float32), ctx["LT"], ctx["sig_ref"],
                      ctx["spot"], ctx["vdtes"]).detach().cpu().numpy()
    ns = len(ctx["emp"]); s, v = m[:ns], m[ns:]
    ssr = float(100 * np.sqrt(np.mean(((s - ctx["emp"]) / ctx["emp"]) ** 2)))
    vix = float(100 * np.sqrt(np.mean(((v - ctx["vov_d"]) / ctx["vov_d"]) ** 2)))
    costs = [round(float(c), 4) for c in getattr(res, "costs", [])]
    out[date] = dict(regime=LABEL[date[:4]], ssr=round(ssr, 1), vix=round(vix, 1),
                     theta=[round(float(x), 3) for x in res.x], n_starts=int(getattr(res, "n_starts", 0)),
                     seed_costs=costs, fit_s=round(getattr(res, "wall", time.time() - t0), 1), device=DEVICE)
    print(f"{date} ({LABEL[date[:4]]}): SSR {ssr:.1f}%  VIX {vix:.1f}%  "
          f"[multistart {getattr(res,'n_starts','?')} seeds, costs {costs}]  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(HERE, "fit_summary_ms.json"), "w"), indent=1)   # incremental save
print(f"\n[done] wall {time.time()-t_all:.0f}s -> fit_summary_ms.json")
