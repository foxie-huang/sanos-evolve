#!/usr/bin/env python3
"""
Joint SSR+VIX calibration with an AUTODIFF Jacobian (torch jacrev), float32, and a cached empirical-SSR
target -- the assembled culmination of the acceleration campaign. One torch residual
    theta8 -> concat[ WREL*(ssr_ts - emp)/emp , wv*(vix_vov - vov_data)/vov_data ]
is differentiated by torch.func.jacrev and fed to scipy least_squares(jac=...). Same objective as
calibrate_joint.py (FD Jacobian), but the Jacobian is AD (~3s float32) not pooled finite differences
(~15s 8-core), and the year's empirical-SSR target is cached (target_cache) instead of recomputed each fit.

gbar is solved per-theta DIFFERENTIABLY (an unrolled fixed point) because it sets the absolute VIX vol
level -- dropping its theta-dependence would bias the VIX block's Jacobian.

    python3 calibrate_joint_torch.py verify [DATE]         # torch value vs numpy _model; jacrev vs torch-FD
    python3 calibrate_joint_torch.py backtest D1 D2 ...     # multi-date fit, warm-started, real wall-time
"""
import sys, os, time
import numpy as np
import torch
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import discslv_slv                                                    # noqa: E402
from slv_fast import propagate_vec                                    # noqa: E402
discslv_slv.propagate = propagate_vec
import discslv_torch as DTt                                           # noqa: E402
import discslv_torch_batched as TB                                    # noqa: E402
import calibrate_slv_exact_ts as C                                    # noqa: E402
from discslv_2f import TwoFactorSV                                    # noqa: E402
from discslv_slv import Epi_V                                         # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at    # noqa: E402
from vix_readout import data_vix                                      # noqa: E402
from target_cache import emp_ssr_cached                               # noqa: E402

torch.set_default_dtype(torch.float32)                                # the fit config: float32 (2.5x on jacrev)
OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"
DT = C.DT; NS = C.NS; NZ = C.NZ; NAMES = C.NAMES; LO = C.LO; HI = C.HI; WREL = C.WREL; LABELS = C.LABELS
MIN_DTE = 7
# Per-param ridge toward the anchor theta -- the fix for the loose-identification railing. WEAK on the
# well-identified params (nu_f,nu_s from VIX; nu_l,lam_skew from SSR), STRONG on the loosely-identified
# nuisance params (lam_f,lam_s,kap_f,kap_s) that otherwise rail to degenerate bounds. Tikhonov on the
# ill-posed directions: it bites only where the data gradient is flat, so informative params still adapt.
#          nu_f  nu_s  nu_l  lam_skew  lam_f  lam_s  kap_f  kap_s
W_REG = np.array([0.03, 0.03, 0.03, 0.03,     0.15,  0.15,  0.15,  0.15])


# ---- differentiable gamma-bar solve (torch port of slv_wire.solve_gbar) ---------------------------------
def _epi_v(ker):
    """E_pi[Vbar] -- stationary-weighted full one-step increment variance (matches discslv_slv.Epi_V)."""
    wl = ker["wl"]
    EVl = (wl * ker["Vl"]).sum(-1); md = (wl * ker["D"]).sum(-1); md2 = (wl * ker["D"] ** 2).sum(-1)
    Vfull = EVl + (md2 - md ** 2)
    return (DTt.stationary_pi(ker) * Vfull).sum()


def solve_gbar_torch(theta8, sig_ref, dt, iters=8):
    target = sig_ref ** 2 * dt
    gbar = torch.log(torch.as_tensor(target, dtype=theta8.dtype, device=theta8.device))
    for _ in range(iters):
        ker = DTt.build_kernel(torch.cat([gbar.reshape(1), theta8]), dt)
        gbar = gbar + torch.log(target / _epi_v(ker))                # E[V] prop e^gbar -> log-correction
    return gbar


def model_torch(theta8, LT, sig_ref, spot, vdtes, nz=NZ):
    """concat[ SSR term structure (batched), VIX vol-of-vol per expiry ] -- differentiable in theta8."""
    th9 = torch.cat([solve_gbar_torch(theta8, sig_ref, DT).reshape(1), theta8])
    ssr = torch.stack([TB.fused_ssr_ts_batched(th9, LT[n], n, DT, nz=nz)[0] for n in NS])
    ker = DTt.build_kernel(th9, DT)
    vov = torch.stack([DTt.vix_ivol(ker, sig_ref, float(d) / 365.0, spot)[1] for d in vdtes])
    return torch.cat([ssr, vov])


# ---- per-date context: numpy statics + torch leverage tensors + (cached) targets -----------------------
def build_date_ctx(date):
    f = OUT + f"/SPX-NDX-RUT-VIX_{date}.json.gz"; yr = date[:4]
    chain = sanos_chain(f); sig_ref = ref_vol(chain)
    kw0 = dict(zip(NAMES, C.X0_MAP["dense"]))                         # theta-invariant EV (gamma-bar reset)
    EV0 = Epi_V(TwoFactorSV(gbar=solve_gbar(kw0, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw0))
    lev = {k: leverage_at(chain, k * DT, EV0, dt=DT) for k in range(1, max(NS) + 1)}
    C._CACHE.clear(); C._CACHE.update(lev)                           # so numpy exact_vec_ts matches (verify path)
    LT = {n: [DTt.lev_torch(lev[k + 1].coef, lev[k + 1].zmax, lev[k + 1].safety) for k in range(n)] for n in NS}
    spot, dv = data_vix(date); dv = [d for d in dv if d[0] >= MIN_DTE]
    vdtes = np.array([d[0] for d in dv], float); vov_d = np.array([d[2] for d in dv])
    emp, nd, hit = emp_ssr_cached(yr, NS, DT)
    return dict(date=date, sig_ref=float(sig_ref), LT=LT, spot=float(spot), vdtes=vdtes,
                vov_d=vov_d, emp=emp, nd=nd, cache_hit=hit)


def _weights(ctx, w_vov):
    n_ssr = len(ctx["emp"]); wv = w_vov * np.sqrt(n_ssr / max(1, len(ctx["vdtes"])))
    return n_ssr, wv


def make_residual(ctx, w_vov=0.8, anchor=None, w_reg=None):
    n_ssr, wv = _weights(ctx, w_vov)
    emp_t = torch.tensor(ctx["emp"], dtype=torch.float32); vov_t = torch.tensor(ctx["vov_d"], dtype=torch.float32)
    wrel_t = torch.tensor(WREL, dtype=torch.float32)
    reg = anchor is not None and w_reg is not None
    if reg:
        anc_t = torch.tensor(np.asarray(anchor, float), dtype=torch.float32)
        rw_t = torch.tensor(w_reg / np.maximum(np.abs(anchor), 0.1), dtype=torch.float32)   # relative ridge

    def resid_t(theta8):
        m = model_torch(theta8, ctx["LT"], ctx["sig_ref"], ctx["spot"], ctx["vdtes"])
        s, v = m[:n_ssr], m[n_ssr:]
        blocks = [wrel_t * (s - emp_t) / emp_t, wv * (v - vov_t) / vov_t]
        if reg:
            blocks.append(rw_t * (theta8 - anc_t))                # ridge toward the anchor (ill-posed directions)
        return torch.cat(blocks)

    return resid_t, n_ssr, wv


def fit_date(ctx, x0, w_vov=0.8, max_nfev=40, anchor=None, w_reg=None):
    resid_t, n_ssr, wv = make_residual(ctx, w_vov, anchor, w_reg)
    jac_fn = torch.func.jacrev(resid_t)

    def resid_np(x):
        return resid_t(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)

    def jac_np(x):
        return jac_fn(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)

    t0 = time.time()
    res = least_squares(resid_np, np.asarray(x0, float), jac=jac_np, bounds=(LO, HI),
                        max_nfev=max_nfev, xtol=1e-6, ftol=1e-6)
    res.wall = time.time() - t0
    return res


def multistart_seeds(anchor):
    """Diverse starts for the loosely-identified objective: the anchor + high/low vol-of-vol (nu_f,nu_s) +
    a steeper leverage skew. The ridge tames railing but not the multi-basin looseness; multi-start picks
    the lowest-cost basin (cost includes the ridge, so it penalises railed basins)."""
    a = np.asarray(anchor, float); seeds = [a.copy()]
    for f in (1.5, 0.65):                                            # high / low vol-of-vol amplitude
        s = a.copy(); s[0] *= f; s[1] *= f; seeds.append(np.clip(s, LO, HI))
    s = a.copy(); s[3] *= 1.8; seeds.append(np.clip(s, LO, HI))      # steeper leverage skew
    return seeds


def fit_date_multistart(ctx, anchor, w_reg, max_nfev=40, w_vov=0.8):
    seeds = multistart_seeds(anchor); best = None; costs = []
    t0 = time.time()
    for x0 in seeds:
        res = fit_date(ctx, x0, w_vov=w_vov, max_nfev=max_nfev, anchor=anchor, w_reg=w_reg)
        costs.append(float(res.cost))
        if best is None or res.cost < best.cost:
            best = res
    best.costs = costs; best.n_starts = len(seeds); best.wall = time.time() - t0
    return best


WARM = {"2015": np.array([0.204, 0.415, 1.101, -0.300, 0.625, 2.455, 0.921, 2.575])}   # held-fit theta (2015 basin)


# ---------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"

    if mode == "verify":
        date = sys.argv[2] if len(sys.argv) > 2 else "2015-06-01"
        ctx = build_date_ctx(date)
        x8 = WARM["2015"].copy()
        # (1) torch model value vs the numpy reference (calibrate_joint._model)
        import calibrate_joint as J
        chain = sanos_chain(OUT + f"/SPX-NDX-RUT-VIX_{date}.json.gz"); sig_ref = ref_vol(chain)
        ref = J._model(x8, chain, sig_ref, ctx["spot"], ctx["vdtes"], nz=NZ)
        mt = model_torch(torch.tensor(x8, dtype=torch.float32), ctx["LT"], ctx["sig_ref"],
                         ctx["spot"], ctx["vdtes"]).detach().numpy()
        print(f"verify {date}  (float32 torch vs float64 numpy)")
        print(f"  numpy _model: {np.round(ref, 4)}")
        print(f"  torch model : {np.round(mt, 4)}")
        print(f"  max|rel diff|: {np.max(np.abs((mt - ref) / ref)) * 100:.3f}%")
        # (2) jacrev vs torch finite-difference (correctness of the AD gradient)
        resid_t, n_ssr, wv = make_residual(ctx)
        xt = torch.tensor(x8, dtype=torch.float32)
        t0 = time.time(); Jad = torch.func.jacrev(resid_t)(xt).detach().numpy(); t_ad = time.time() - t0
        r0 = resid_t(xt).detach().numpy(); Jfd = np.zeros_like(Jad)
        for i in range(len(x8)):
            h = 1e-3 * max(abs(x8[i]), 1.0); xp = x8.copy(); xp[i] += h
            Jfd[:, i] = (resid_t(torch.tensor(xp, dtype=torch.float32)).detach().numpy() - r0) / h
        d = np.abs(Jad - Jfd); sc = np.maximum(np.abs(Jfd), 1e-3)
        print(f"  jacrev vs FD: max|rel|={np.max(d / sc) * 100:.2f}%  (jacrev {t_ad:.2f}s)")
        sys.exit(0)

    if mode == "backtest":
        argv = sys.argv[2:]
        anchored = "anchored" in argv                                  # re-anchor each date to the curated theta
        dates = [a for a in argv if a != "anchored"] or ["2015-06-01"]
        print(f"JOINT SSR+VIX backtest (AD jacrev, float32, cached target, RIDGE)"
              f"{' [ANCHORED]' if anchored else ' [chained]'} -- {len(dates)} dates", flush=True)
        t_all = time.time(); prev = None; rows = []
        for i, date in enumerate(dates):
            tc = time.time(); ctx = build_date_ctx(date); t_ctx = time.time() - tc
            anchor = np.asarray(WARM.get(date[:4], C.X0_MAP["ts"]), float)   # ridge always pulls toward the anchor
            x0 = anchor if (anchored or prev is None) else prev
            res = fit_date(ctx, x0, anchor=anchor, w_reg=W_REG)
            m = model_torch(torch.tensor(res.x, dtype=torch.float32), ctx["LT"], ctx["sig_ref"],
                            ctx["spot"], ctx["vdtes"]).detach().numpy()
            s = m[:len(ctx["emp"])]; v = m[len(ctx["emp"]):]
            ssr_err = 100 * np.sqrt(np.mean(((s - ctx["emp"]) / ctx["emp"]) ** 2))
            vix_err = 100 * np.sqrt(np.mean(((v - ctx["vov_d"]) / ctx["vov_d"]) ** 2))
            prev = res.x.copy()
            rows.append((date, ctx, res, s, v, ssr_err, vix_err, t_ctx))
            print(f"\n[{i+1}/{len(dates)}] {date}  ctx {t_ctx:.1f}s (cache_hit={ctx['cache_hit']})  "
                  f"fit {res.wall:.1f}s / {res.nfev} evals", flush=True)
            print(f"   theta: " + "  ".join(f"{n}={val:.3f}" for n, val in zip(NAMES, res.x)), flush=True)
            print(f"   SSR RMS {ssr_err:.1f}%   VIX RMS {vix_err:.1f}%", flush=True)
        print(f"\n=== backtest done: {len(dates)} dates in {time.time()-t_all:.0f}s "
              f"({(time.time()-t_all)/len(dates):.0f}s/date) ===")
        print(f"{'date':>12}{'SSR%':>7}{'VIX%':>7}{'fit_s':>7}{'evals':>7}{'ctx_s':>7}")
        for date, ctx, res, s, v, se, ve, tctx in rows:
            print(f"{date:>12}{se:>7.1f}{ve:>7.1f}{res.wall:>7.1f}{res.nfev:>7d}{tctx:>7.1f}")
        sys.exit(0)

    print(__doc__)
