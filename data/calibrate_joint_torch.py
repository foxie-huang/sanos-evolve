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

# Sets the GLOBAL default, overriding discslv_torch.py's float64 -- which works only because that
# module is imported above, i.e. by import order. See the warning there.
torch.set_default_dtype(torch.float32)                                # the fit config: float32 (2.5x on jacrev)
OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
DT = C.DT; NS = C.NS; NZ = C.NZ; NAMES = C.NAMES; LO = C.LO; HI = C.HI; WREL = C.WREL; LABELS = C.LABELS
VOV_FRONT = False   # if True, make_residual front-weights the vov tenors toward 30d (else uniform); default = no change
MONO_PEN = 0.0      # if >0, penalise a RISING model SSR (theory: R(T) monotone non-increasing); default = off
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


# ---- NORMALISED PATH (SANOS_REFERENCE.md 16.9/16.10). Set USE_NORM=1 to fit the fixed kernel.
# theta is 7 params (kap_s FIXED, see calibrate_slv_exact_ts.KAP_S_FIXED) instead of 8, and bounds
# come from LO_N/HI_N -- the old kap_s range [0.5, 4.0] is ILLEGAL here (|kap| >= 1 has no stationary
# law and would produce a silently diverging factor).
USE_NORM = os.environ.get("USE_NORM", "0") == "1"
# n_p = 5, raised from 3. Two measurements, not one: ATM skew error vs MC is +2.57% at n_p=5 vs
# +3.35% at n_p=3; AND the martingale error at the high-variance corners of the sigma_ATM grid is
# 60x smaller (-4.2e-06 vs -2.6e-04 after 13 steps). The second was found late -- the price-width
# quadrature integrates exp(SG*z), exact to degree 5 at n_p=3, so its error scales like SG^6 and
# blows up where the variance state is extreme. n_p=3 was chosen on the skew number alone.
N_P = int(os.environ.get("N_P", "5"))       # price sub-abscissas per component (16.10)
N_X = int(os.environ.get("N_X", "3"))       # innovation quadrature (16.9); converged at 3
# NA, NB: abscissa counts for the QUADRATURE over the stationary bivariate law of (u_f, u_s) in
# `_stat_nodes_n` -- the pi-average that replaced the old `stationary_pi` linear solve.
#
# NB WAS 3 AND THAT WAS WRONG. na=5, nb=3 were inherited from the OLD kernel's n_f=5, n_s=3 "so cost
# is unchanged" -- but those were CARRIED abscissa counts for the factors, a different role, and the
# pair was never convergence-tested in its own right. The integrand is not polynomial: the factors
# enter as exp(nu_f u_f + nu_s u_s) (variance is lognormal), so the rule must resolve a function whose
# SCALE is nu, and the error term goes like (nu^2/2)^n / n!.
# Measured against the closed form E[exp(a u_f + b u_s)] = exp((a^2 + 2 c a b + b^2)/2), worst case
# over the nine fitted SPX years:
#     (5,3)  27.9%      (5,5)  2.63%      (5,7)  0.118%      (5,9)  0.034%
# and raising na alone does NOTHING -- (7,3), (9,3), (11,3) are all still 27.9%. The error lives
# entirely on the SLOW axis, because nu_f is 0.69-1.12 while nu_s is 1.16-2.13.
#     nb:      3          5          7          9
#     mean:  15.543%    1.038%    0.049%    0.016%
#     worst: 27.889%    2.626%    0.118%    0.034%
#     cost:    1.0x      1.7x      2.3x      3.0x
# SHIPPING nb = 5. nb = 7 is the accuracy answer but the MODEL (not the rule) overflows there on the
# panel's own theta: the martingale re-lock computes log(sum W exp(MU + 0.5 SG^2)) and at SG ~ 20
# that needs exp(194), past float32's 3.4e38 -> inf -> NaN. nb = 5 reaches only |z| = 2.86 against
# nb = 7's 3.75 and stays clear. Fixing the re-lock with a log-sum-exp (exactly equivalent, cannot
# overflow) is the prerequisite for nb = 7; until then nb = 5 takes the worst case 27.9% -> 2.6%.
NA, NB = int(os.environ.get("NA", "5")), int(os.environ.get("NB", "5"))   # stationary-law quadrature


def _th9_n(theta, gbar):
    """[gbar, nu_f, nu_s, nu_l, lam_skew, rho_f, rho_s, kap_f] + kap_s.

    Dispatches on LENGTH, so both paths share every downstream call:
      len 7 -> kap_s appended from the module constant C.KAP_S_FIXED (the pinned path)
      len 8 -> kap_s is the last FITTED entry (C.NAMES_N8)
    """
    if theta.shape[0] == 8:
        return torch.cat([gbar.reshape(1), theta])
    ks = torch.as_tensor(C.KAP_S_FIXED, dtype=theta.dtype, device=theta.device)
    return torch.cat([gbar.reshape(1), theta, ks.reshape(1)])


def _epi_v_n(ker):
    """E_pi[full one-step increment variance] under the stationary bivariate law -- the normalised
    replacement for `_epi_v`'s `(stationary_pi(ker) * Vfull).sum()`. No linear solve."""
    c = DTt.stationary_n(ker)
    vXs = ker["nu_f"] ** 2 + 2 * ker["nu_f"] * ker["nu_s"] * c + ker["nu_s"] ** 2
    z0 = torch.zeros((), dtype=ker["zl"].dtype, device=ker["zl"].device)
    return DTt._vfull_n(ker, z0, vXs)


def solve_gbar_torch_n(theta7, sig_ref, dt, iters=8):
    """`target` MUST be a tensor, not a Python float.

    A Python float is a DOUBLE. Under forward-mode AD, `double / float32-dual` keeps the PRIMAL
    float32 (weak-scalar promotion) while materialising the scalar as float64 in the TANGENT. So
    gbar's tangent is float32 at iteration 1 and float64 from iteration 2 on, and every downstream
    tensor inherits it. `.dtype` reports only the primal, so the mismatch is invisible to every
    guard -- until `recompress_n`'s index_add, which sees self=Float, source=Double and dies.

    Forward and jacrev are unaffected; only jacfwd. JAC defaults to jacfwd when USE_NORM=1, which
    is why VOVLEV=1 could never be fitted at all.
    """
    tgt = torch.as_tensor(sig_ref * sig_ref * dt, dtype=theta7.dtype, device=theta7.device)
    gbar = torch.log(tgt)
    for _ in range(iters):
        ker = DTt.build_kernel_n(_th9_n(theta7, gbar), dt, n_x=N_X)
        gbar = gbar + torch.log(tgt / _epi_v_n(ker))
    return gbar


def model_torch_n(theta7, LT, sig_ref, spot, vdtes, nz=NZ):
    """concat[ SSR term structure, VIX vol-of-vol per expiry ] on the FIXED kernel, differentiable.

    No batched multi-tenor path yet (the old `fused_ssr_ts_multi` snapshot trick has no `_n` twin),
    so each tenor is a separate call -- measured ~26 s for all five at na=5/nb=3/n_p=5. PROFILE ONE
    WORKER TO PEAK BEFORE SIZING ANY POOL.
    """
    th9 = _th9_n(theta7, solve_gbar_torch_n(theta7, sig_ref, DT))
    ssr = DTt.fused_ssr_ts_multi_n(th9, LT[max(NS)], NS, DT, nz=nz, n_x=N_X, n_p=N_P,
                                   na=NA, nb=NB)
    ker = DTt.build_kernel_n(th9, DT, n_x=N_X)
    # SSR above stays on LT[max(NS)]; the VIX takes the LONGEST ladder available. With LADDER=37
    # that is LT[37], whose first 13 slices ARE LT[13] (both are lev[k+1]), so SSR is untouched --
    # only the leveraged VIX, whose slice index saturated at 12 and produced a FLAT vov tail past
    # 91 days, sees the new weeks. Verified bit-identical SSR under both ladders.
    _lf = LT[max(LT)] if _VOVLEV else None
    vov = torch.stack([DTt.vix_ivol_n(ker, sig_ref, float(d) / 365.0, spot,
                                      lam_fns=_lf, n_p=N_P)[1] for d in vdtes])
    return torch.cat([ssr, vov])


def model_torch(theta8, LT, sig_ref, spot, vdtes, nz=NZ):
    """concat[ SSR term structure (batched), VIX vol-of-vol per expiry ] -- differentiable in theta8.
    SSR via fused_ssr_ts_multi: ONE propagation to max(NS), snapshotted at every tenor, instead of a
    separate chain per tenor. Bit-exact (the per-tenor chains are prefixes), sum(NS)=28 -> max(NS)=13
    propagation steps, measured 1.86x (nk=16) / 2.16x (nk=24).
    NOTE: (zmax,Q,nk,n_f,n_s,n_l) are read off fused_ssr_ts_batched.__defaults__ and passed through, so
    the existing stage-1/stage-2 node patching (dd[4]=7 / 17 in ndx_more.py and ~10 sibling scripts)
    keeps controlling this path exactly as before. Do not hard-code them here."""
    th9 = torch.cat([solve_gbar_torch(theta8, sig_ref, DT).reshape(1), theta8])
    _, _zmax, _Q, _nk, _nf, _ns, _nl = TB.fused_ssr_ts_batched.__defaults__
    ssr = TB.fused_ssr_ts_multi(th9, LT[max(NS)], NS, DT, nz=nz, zmax=_zmax, Q=_Q, nk=_nk,
                                n_f=_nf, n_s=_ns, n_l=_nl)
    ker = DTt.build_kernel(th9, DT)
    # VOVLEV=1 routes the VIX readout through the LEVERAGED joint (log-price, regime) law. The
    # default path uses ker["Vl"] RAW while propagate() levers it -- the same variance is levered
    # on the SSR path and not on the vov path. See discslv_torch.vix_ivol.
    _lf = LT[max(NS)] if _VOVLEV else None
    vov = torch.stack([DTt.vix_ivol(ker, sig_ref, float(d) / 365.0, spot, lam_fns=_lf)[1] for d in vdtes])
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
    _vdt = np.asarray(ctx["vdtes"], float)                                      # per-tenor vov weight:
    _vw = 1.0 / (1.0 + ((_vdt - 30) / 40) ** 2) if VOV_FRONT else np.ones_like(_vdt)   # front-30d if VOV_FRONT
    vov_w_t = torch.tensor(_vw / _vw.mean(), dtype=torch.float32)               # mean-normalised (overall vov weight ~unchanged)
    reg = anchor is not None and w_reg is not None
    if reg:
        anc_t = torch.tensor(np.asarray(anchor, float), dtype=torch.float32)
        rw_t = torch.tensor(w_reg / np.maximum(np.abs(anchor), 0.1), dtype=torch.float32)   # relative ridge

    def resid_t(theta8):
        m = (model_torch_n if USE_NORM else model_torch)(
            theta8, ctx["LT"], ctx["sig_ref"], ctx["spot"], ctx["vdtes"])
        s, v = m[:n_ssr], m[n_ssr:]
        blocks = [wrel_t * (s - emp_t) / emp_t, wv * vov_w_t * (v - vov_t) / vov_t]
        if MONO_PEN > 0:
            blocks.append(MONO_PEN * torch.relu((s[1:] - s[:-1]) / s[:-1]))   # penalise rising SSR (relative), one-sided
        if reg:
            blocks.append(rw_t * (theta8 - anc_t))                # ridge toward the anchor (ill-posed directions)
        return torch.cat(blocks)

    return resid_t, n_ssr, wv


_VOVLEV = os.environ.get("VOVLEV", "0") == "1"   # leveraged VIX readout (off by default)
# jacfwd (forward mode, VMAPPED) not the hand-rolled jvp LOOP: measured 35.8s vs 227.0s on the
# normalised path -- 6.3x -- at IDENTICAL peak RSS (0.56 GB both). The jvp loop exists because
# jacREV blew memory (3920M -> 1066M, worker cap 9 -> 36), but that is reverse mode vmapping 14
# OUTPUTS; jacfwd vmaps 7 TANGENTS and never had that problem. Old path left on "jvp" -- its jacfwd
# memory is unmeasured and changing it would alter deployed behaviour unverified.
JAC_MODE = os.environ.get("JAC", "jacfwd" if os.environ.get("USE_NORM", "0") == "1" else "jvp")     # jvp | jacfwd | jacrev  -- see vix_joint_refit/jac_probe.py


def _jac_fn(resid_t, n):
    """dr/dtheta for an R^n -> R^m map with m ~ 25, evaluated through 13 sequential propagation steps.

    jacrev keeps the whole 13-step tape, so its memory is cumulative in DEPTH: measured 3012 MB for
    the Jacobian alone (3920 MB process peak), which caps a 39 GB box at ~9 workers and is what jammed
    it. Forward mode needs no tape, and with n=8 < m it is also the cheaper direction. Running the 8
    tangents SEQUENTIALLY (jvp per basis vector) instead of vmap'd together (jacfwd) drops the
    Jacobian to 141 MB -- a 21x cut for 1.35x wall -- so memory stops binding and CPU count does.

    Do NOT use finite differences here: the model evaluates in float32, so scipy's '2-point' returns
    max relative error ~8e2 against exact AD. Measured, not assumed.
    """
    if JAC_MODE == "jacrev":
        return torch.func.jacrev(resid_t)
    if JAC_MODE == "jacfwd":
        return torch.func.jacfwd(resid_t)

    def jac_jvp(x):
        cols = []
        for i in range(n):
            v = torch.zeros_like(x)
            v[i] = 1.0
            cols.append(torch.func.jvp(resid_t, (x,), (v,))[1])
        return torch.stack(cols, dim=1)

    return jac_jvp


def fit_date(ctx, x0, w_vov=0.8, max_nfev=40, anchor=None, w_reg=None):
    resid_t, n_ssr, wv = make_residual(ctx, w_vov, anchor, w_reg)
    jac_fn = _jac_fn(resid_t, len(np.asarray(x0, float)))

    def resid_np(x):
        return resid_t(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)

    def jac_np(x):
        return jac_fn(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)

    t0 = time.time()
    n_th = len(np.asarray(x0, float))
    if USE_NORM:
        lo, hi = (C.LO_N8, C.HI_N8) if n_th == 8 else (C.LO_N, C.HI_N)
    else:
        lo, hi = LO, HI                                  # old kap_s range [0.5,4.0] is ILLEGAL here
    res = least_squares(resid_np, np.asarray(x0, float), jac=jac_np, bounds=(lo, hi),
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
