#!/usr/bin/env python3
"""
OFF-SPX NDX calibration -- fit theta to NDX with OWN-STRIP + REALISED targets only (no VIX, no cliquets),
the paper's off-SPX recipe (Sec. 'what stands in for VIX'). Statics = NDX SANOS marginals; leverage/SSR
target = NDX REALISED SSR (own strip); vov target = NDX REALISED variance-of-variance (realised vol of the
own-strip variance-swap level at [30,90]d). Ridge-stabilised joint fit (reuses calibrate_joint_torch.fit_date).
Records wall-time.  Caveats: the vov is P (realised), matched to the model's Q readout (~risk-premium bias);
the model vov term structure (option-expiry) vs realised (variance-tenor) is an approximate alignment.
    python3 calibrate_ndx.py [cpu|mps] [DATE ...]
"""
import sys, os, glob, gzip, json, time
import numpy as np
import torch

DEVICE = "mps"
if len(sys.argv) > 1 and sys.argv[1] in ("cpu", "mps"):
    DEVICE = sys.argv.pop(1)
torch.set_default_dtype(torch.float32); torch.set_default_device(DEVICE)

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import calibrate_joint_torch as J                                   # noqa: E402
import calibrate_slv_exact_ts as C                                  # noqa: E402
from empirical_ssr import empirical_ssr                             # noqa: E402
from ndx_vov_readout import expiry, var_swap, rvov                  # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at  # noqa: E402
from discslv_2f import TwoFactorSV                                  # noqa: E402
from discslv_slv import Epi_V                                       # noqa: E402
import discslv_torch as DTt                                         # noqa: E402

OUT = J.OUT; NAMES = J.NAMES; DT = J.DT; NS = C.NS
CACHE = os.path.join(HERE, ".ndx_cache"); os.makedirs(CACHE, exist_ok=True)
TENORS = [30, 90]


def ndx_ssr_cached(year):
    k = os.path.join(CACHE, f"ssr_{year}.npz")
    if os.path.exists(k):
        d = np.load(k); return d["ssr"], int(d["nd"])
    paths = sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz"))
    ssr, nd = empirical_ssr(paths, ns=NS, dt=DT, ticker="NDX")
    np.savez(k, ssr=ssr, nd=nd); return ssr, nd


def ndx_vov_cached(year):
    k = os.path.join(CACHE, f"vov_{year}.npz")
    if os.path.exists(k):
        return np.load(k)["rvov"]
    series = {t: [] for t in TENORS}
    for f in sorted(glob.glob(f"{OUT}/SPX-NDX-RUT-VIX_{year}-*.json.gz")):
        try:
            recs = json.load(gzip.open(f))["strikes"]
            for t in TENORS:
                ch = expiry(recs, "NDX", t)
                if ch:
                    series[t].append(var_swap(ch, ch[0]["dte"] / 365))
        except Exception:
            continue
    rv = np.array([rvov(np.array(series[t])) for t in TENORS])
    np.savez(k, rvov=rv); return rv


def ndx_spot(date):                                                # VXN-equivalent: own-strip sqrt(V) at 30d
    recs = json.load(gzip.open(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz"))["strikes"]
    ch = expiry(recs, "NDX", 30)
    return float(var_swap(ch, ch[0]["dte"] / 365))


def build_ctx_ndx(date):
    yr = date[:4]
    chain = sanos_chain(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz", ticker="NDX")
    sig_ref = ref_vol(chain)
    kw0 = dict(zip(NAMES, C.X0_MAP["dense"]))
    EV0 = Epi_V(TwoFactorSV(gbar=solve_gbar(kw0, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw0))
    lev = {k: leverage_at(chain, k * DT, EV0, dt=DT) for k in range(1, max(NS) + 1)}
    C._CACHE.clear(); C._CACHE.update(lev)
    LT = {n: [DTt.lev_torch(lev[k + 1].coef, lev[k + 1].zmax, lev[k + 1].safety) for k in range(n)] for n in NS}
    emp, nd = ndx_ssr_cached(yr); vov_d = ndx_vov_cached(yr)
    return dict(date=date, sig_ref=float(sig_ref), LT=LT, spot=ndx_spot(date),
                vdtes=np.array(TENORS, float), vov_d=vov_d, emp=emp, nd=nd, cache_hit=True)


if __name__ == "__main__":
    dates = sys.argv[1:] or ["2018-06-01", "2020-06-01", "2021-06-01", "2022-06-01"]
    anchor = np.asarray(C.X0_MAP["ts"], float)
    print(f"OFF-SPX NDX fit [{DEVICE}] -- own-strip SANOS + realised SSR + realised var-of-var (no VIX)", flush=True)
    t_all = time.time(); rows = []
    for date in dates:
        if not os.path.exists(f"{OUT}/SPX-NDX-RUT-VIX_{date}.json.gz"):
            print(f"  {date}: MISSING"); continue
        tc = time.time(); ctx = build_ctx_ndx(date); t_ctx = time.time() - tc
        res = J.fit_date_multistart(ctx, anchor, J.W_REG, max_nfev=40)
        m = J.model_torch(torch.tensor(res.x, dtype=torch.float32), ctx["LT"], ctx["sig_ref"],
                          ctx["spot"], ctx["vdtes"]).detach().cpu().numpy()
        ns = len(ctx["emp"]); s, v = m[:ns], m[ns:]
        se = 100 * np.sqrt(np.mean(((s - ctx["emp"]) / ctx["emp"]) ** 2))
        ve = 100 * np.sqrt(np.mean(((v - ctx["vov_d"]) / ctx["vov_d"]) ** 2))
        rows.append((date, ctx, se, ve, res, s, v))
        print(f"\n[{date}]  ctx {t_ctx:.1f}s  multistart {res.wall:.0f}s / {res.n_starts} seeds  "
              f"(cost best {min(res.costs):.4f} vs worst {max(res.costs):.4f})", flush=True)
        print(f"   NDX realised SSR {np.round(ctx['emp'],2)}   realised vov {np.round(ctx['vov_d'],2)}"
              f"   (VXN-eq spot {ctx['spot']*100:.1f})", flush=True)
        print(f"   theta " + "  ".join(f"{n}={x:.3f}" for n, x in zip(NAMES, res.x)), flush=True)
        print(f"   model SSR {np.round(s,2)}   model vov {np.round(v,2)}", flush=True)
        print(f"   SSR RMS {se:.1f}%   vov RMS {ve:.1f}%", flush=True)
    print(f"\n=== OFF-SPX NDX: {len(rows)} dates in {time.time()-t_all:.0f}s [{DEVICE}] ===")
    print(f"{'date':>12}{'SSR%':>7}{'vov%':>7}{'fit_s':>7}{'evals':>7}")
    for date, ctx, se, ve, res, s, v in rows:
        print(f"{date:>12}{se:>7.1f}{ve:>7.1f}{res.wall:>7.0f}{res.nfev:>7d}")
