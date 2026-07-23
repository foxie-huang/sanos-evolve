#!/usr/bin/env python3
"""
FULL eq:objective calibration with a TORCH AUTODIFF Jacobian (torch.func.jacrev), float32, optional MPS,
and MULTI-START -- the accelerated + globalized counterpart of calibrate_full.py.

calibrate_joint_torch already fits (SSR + VoV) with jacrev/multi-start; the ONLY missing piece of the
paper's 3-term eq:objective (disc_SLV.tex Sec850 / alg:calib) is term (1), the MARGINAL digital band-loss:
the Gyongy-fused from-spot PROPAGATED marginal G^S_theta vs the SANOS marginals as survival (digital)
distances. This file ports that block to torch (nub, initial_state, E[nu|z], propagate, digitals -- all
differentiable in theta), concatenates it AHEAD of calibrate_joint_torch's SSR+VoV, and differentiates
the whole residual with jacrev. So the full objective now gets: analytic gradient (1 reverse pass vs 8 FD
model evals), GPU (MPS) tensors, and multi-start over the non-convex basin (reuses JT.multistart_seeds).

The forward value MUST match numpy calibrate_full._model (that is what `verify` checks); the fit MUST land
the same basin as calibrate_full but faster. Records wall-time (per start + total).

    OOS_DATE=YYYY-MM-DD python3 calibrate_full_torch.py verify           # torch value vs numpy, jacrev vs FD, timing
    OOS_DATE=YYYY-MM-DD python3 calibrate_full_torch.py fit  [w_marg] [w_vov] [max_nfev]   # multi-start fit
    OOS_DATE=YYYY-MM-DD python3 calibrate_full_torch.py fit1 [w_marg] [w_vov] [max_nfev]   # single-start (anchor only)
  env: DEV=cpu (default). MPS is NOT worth it here: tensors are tiny (<=240 comps, 5x3x5 grid) inside a
       python regime-loop -> many small ops, launch overhead dominates; and the shared DTt._merge/recompress
       create CPU tensors (searchsorted/arange/zeros) so MPS errors without pervasive device plumbing. The
       AD (jacfwd) is the speedup, not the GPU.
"""
import sys, os, time, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from scipy.optimize import least_squares                                 # noqa: E402
import calibrate_joint_torch as JT                                       # noqa: E402  (solve_gbar_torch, build_date_ctx, multistart_seeds, model_torch)
import discslv_torch as DTt                                              # noqa: E402  (build_kernel, propagate, stationary_pi, lev_torch)
import calibrate_slv_exact_ts as C                                       # noqa: E402
import calibrate_slv_exact_ts_par as P                                   # noqa: E402  (OOS_DATE, build_ctx, YR, OUT)
import calibrate_full as CF                                              # noqa: E402  (numpy reference: digitals, model_marginals, _model, KS)
from discslv_2f import TwoFactorSV                                       # noqa: E402
from discslv_slv import Epi_V                                            # noqa: E402
from slv_wire import sanos_chain, solve_gbar, leverage_at                # noqa: E402
from slv_interp import interp_marginal                                   # noqa: E402

torch.set_default_dtype(torch.float32)
DEV = os.environ.get("DEV", "cpu")
DT = C.DT; NS = C.NS; NAMES = C.NAMES; NZ = C.NZ; LO = C.LO; HI = C.HI; WREL = C.WREL
NMAX = max(NS)
KS_T = torch.tensor(CF.KS, dtype=torch.float32, device=DEV)              # SAME log-moneyness grid as numpy CF.digitals
_SQRT2 = 2.0 ** 0.5; _SQRT2PI = (2.0 * np.pi) ** 0.5


# ---- differentiable marginal block (torch port of calibrate_full.model_marginals + digitals) -------------
def _Phi(x): return 0.5 * (1.0 + torch.erf(x / _SQRT2))
def _phi(x): return torch.exp(-0.5 * x ** 2) / _SQRT2PI


def nub_torch(ker):
    """nu_bar[f,s] = Vfull_raw / E_pi[Vfull]  (matches discslv_slv.nu_bar), differentiable in theta."""
    wl = ker["wl"]
    EVl = (wl * ker["Vl"]).sum(-1); md = (wl * ker["D"]).sum(-1); md2 = (wl * ker["D"] ** 2).sum(-1)
    Vfull = EVl + (md2 - md ** 2)                                        # (nf,ns)  E_l[V] + Var_l(d)
    return Vfull / (DTt.stationary_pi(ker) * Vfull).sum()


def initial_state_torch(ker):
    """Point mass at z=0 spread over the stationary (f,s) regimes (matches discslv_slv.initial_state)."""
    nf, ns = ker["n_f"], ker["n_s"]
    W = DTt.stationary_pi(ker).reshape(-1)                               # pi[f*ns+s]
    MU = torch.zeros(nf * ns, device=W.device)
    SG = torch.full((nf * ns,), 1e-4, device=W.device)
    F = torch.arange(nf, device=W.device).repeat_interleave(ns)
    S = torch.arange(ns, device=W.device).repeat(nf)
    return W, MU, SG, F.long(), S.long()


def E_nu_given_z_vec(query, state, nub):
    """E[nu | z=query_i] under the current mixture, vectorized over the component means (matches E_nu_given_z)."""
    W, MU, SG, F, S = state
    d = (query[:, None] - MU[None, :]) / SG[None, :]                     # (N, M)
    dens = W[None, :] * _phi(d) / SG[None, :]                            # (N, M)
    numer = (dens * nub[F, S][None, :]).sum(1)
    return numer / dens.sum(1).clamp_min(1e-30)


def digitals_t(mu):
    """Survival Pr(x>k)=sum_i W_i Phi((MU_i-k)/SG_i) at each k in KS (matches calibrate_full.digitals)."""
    W, MU, SG = mu
    d = (MU[None, :] - KS_T[:, None]) / SG[None, :]                      # (nK, M)
    return (W[None, :] * _Phi(d)).sum(1)


def model_marginals_torch(theta8, sig_ref, LTm, nk=16):
    """From-spot leveraged propagation -> {n: (W,MU,SG)} at NS -- the fused G^S_theta, differentiable in theta."""
    th9 = torch.cat([JT.solve_gbar_torch(theta8, sig_ref, DT).reshape(1), theta8])
    ker = DTt.build_kernel(th9, DT)
    nub = nub_torch(ker)
    state = initial_state_torch(ker); out = {}
    for k in range(1, NMAX + 1):
        lf = LTm[k - 1]                                                  # theta-independent leverage at maturity k*DT

        def lam_eff(mu, st=state, l=lf):                                 # l(mu) / sqrt(clip(E[nu|z],0.3,3)) = the Gyongy correction
            env = E_nu_given_z_vec(mu, st, nub).clamp(0.3, 3.0)
            return l(mu) / torch.sqrt(env)

        state = DTt.propagate(state, ker, lam_eff, nk)
        if k in NS:
            out[k] = (state[0], state[1], state[2])
    return out


def model_full_torch(theta8, ctx):
    """concat[ marginal digitals (NS x KS), SSR (5), VIX vov ] -- the full eq:objective model, one theta."""
    mm = model_marginals_torch(theta8, ctx["sig_ref"], ctx["LTm"])
    marg = torch.cat([digitals_t(mm[n]) for n in NS])
    ssr_vov = JT.model_torch(theta8, ctx["LT"], ctx["sig_ref"], ctx["spot"], ctx["vdtes"])   # [ssr(5), vov]
    return torch.cat([marg, ssr_vov])


# ---- per-date context: JT's SSR/VoV/LT context + the marginal leverage (LTm) + SANOS digital target -------
def build_full_ctx(date):
    ctx = JT.build_date_ctx(date)                                       # LT, sig_ref, spot, vdtes, vov_d, emp (+ C._CACHE)
    chain = sanos_chain(JT.OUT + f"/SPX-NDX-RUT-VIX_{date}.json.gz")
    kw0 = dict(zip(NAMES, C.X0_MAP["dense"]))
    EV0 = Epi_V(TwoFactorSV(gbar=solve_gbar(kw0, ctx["sig_ref"], dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw0))
    lev = {k: leverage_at(chain, k * DT, EV0, dt=DT) for k in range(1, NMAX + 1)}   # SAME objects as JT's lev
    ctx["LTm"] = [DTt.lev_torch(lev[k].coef, lev[k].zmax, lev[k].safety) for k in range(1, NMAX + 1)]
    ctx["sanos_dig"] = np.concatenate([CF.digitals(interp_marginal(chain, n * DT)) for n in NS])  # numpy target, identical grid
    ctx["chain"] = chain
    return ctx


def make_full_residual(ctx, w_marg, w_vov):
    """Residual matching calibrate_full.resid_parts EXACTLY: [w_marg*(marg-sanos), WREL*(ssr-emp)/emp, w_vov*(vov-vov_d)/vov_d]."""
    n_marg = len(ctx["sanos_dig"]); n_ssr = len(ctx["emp"])
    dd = dict(dtype=torch.float32, device=DEV)
    sanos_t = torch.tensor(ctx["sanos_dig"], **dd); emp_t = torch.tensor(ctx["emp"], **dd)
    vov_t = torch.tensor(ctx["vov_d"], **dd); wrel_t = torch.tensor(WREL, **dd)

    def resid_t(theta8):
        full = model_full_torch(theta8, ctx)
        marg = full[:n_marg]; s = full[n_marg:n_marg + n_ssr]; v = full[n_marg + n_ssr:]
        return torch.cat([w_marg * (marg - sanos_t), wrel_t * (s - emp_t) / emp_t, w_vov * (v - vov_t) / vov_t])

    return resid_t, n_marg, n_ssr


def fit_full(ctx, x0, w_marg=8.0, w_vov=0.8, max_nfev=60):
    resid_t, n_marg, n_ssr = make_full_residual(ctx, w_marg, w_vov)
    jac_fn = torch.func.jacfwd(resid_t)              # forward-mode: 8 inputs << 49 outputs -> ~5.5x faster than jacrev

    def resid_np(x):
        return resid_t(torch.tensor(x, dtype=torch.float32, device=DEV)).detach().cpu().numpy().astype(np.float64)

    def jac_np(x):
        return jac_fn(torch.tensor(x, dtype=torch.float32, device=DEV)).detach().cpu().numpy().astype(np.float64)

    t0 = time.time()
    res = least_squares(resid_np, np.asarray(x0, float), jac=jac_np, bounds=(LO, HI),
                        max_nfev=max_nfev, xtol=1e-6, ftol=1e-6)
    res.wall = time.time() - t0; res.n_marg = n_marg; res.n_ssr = n_ssr
    return res


def fit_full_multistart(ctx, anchor, w_marg=8.0, w_vov=0.8, max_nfev=60):
    seeds = JT.multistart_seeds(anchor); best = None; rows = []
    t0 = time.time()
    for i, x0 in enumerate(seeds):
        r = fit_full(ctx, x0, w_marg, w_vov, max_nfev)
        rows.append((i, float(r.cost), r.nfev, r.wall))
        print(f"  start {i}: cost {r.cost:.5g}  ({r.nfev} evals, {r.wall:.0f}s)  theta0={np.round(x0,2)}", flush=True)
        if best is None or r.cost < best.cost:
            best = r
    best.rows = rows; best.n_starts = len(seeds); best.wall_total = time.time() - t0
    return best


def _report(res, ctx, w_marg, w_vov, tag="fit"):
    v = model_full_torch(torch.tensor(res.x, dtype=torch.float32, device=DEV), ctx).detach().cpu().numpy()
    nm = res.n_marg; ns = res.n_ssr; emp = ctx["emp"]; sanos = ctx["sanos_dig"]; vov_d = ctx["vov_d"]
    marg = v[:nm]; ssr = v[nm:nm + ns]; vov = v[nm + ns:]
    print(f"\n{tag.upper()}  ({res.nfev} evals, {getattr(res,'wall_total',res.wall):.0f}s"
          + (f", {res.n_starts} starts" if hasattr(res, 'n_starts') else "") + f", DEV={DEV})")
    print("theta: " + "  ".join(f"{n}={val:.3f}" for n, val in zip(NAMES, res.x)) + "\n")
    print(f"  MARGINAL survival RMS {np.sqrt(np.mean((marg - sanos) ** 2)) * 100:.2f}%")
    print(f"  SSR: model {np.round(ssr,3)} vs emp {np.round(emp,3)}  "
          + "  ".join(f"{100*(ssr[i]-emp[i])/emp[i]:.0f}%" for i in range(ns)))
    print(f"  VoV RMS {np.sqrt(np.mean(((vov - vov_d) / vov_d) ** 2)) * 100:.0f}%")
    print(f"  cost {res.cost:.5g}")


# ---------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    date = P.OOS_DATE
    print(f"calibrate_full_torch  date={date}  DEV={DEV}  torch {torch.__version__}", flush=True)

    if mode == "verify":
        ctx = build_full_ctx(date)
        chain, sig = P.build_ctx()                                       # numpy reference context (repopulates C._CACHE identically)
        spot, vdtes, vov_d = CF._vix_targets(sig)
        x0 = C.X0_MAP["ts"].copy()
        t = time.time(); v_np = CF._model(x0, chain, sig, spot, vdtes); t_np = time.time() - t
        t = time.time(); v_t = model_full_torch(torch.tensor(x0, dtype=torch.float32, device=DEV), ctx).detach().cpu().numpy(); t_t = time.time() - t
        nm = len(ctx["sanos_dig"]); ns = len(ctx["emp"])
        for name, a, b in [("MARGINAL", v_np[:nm], v_t[:nm]),
                           ("SSR", v_np[nm:nm + ns], v_t[nm:nm + ns]),
                           ("VoV", v_np[nm + ns:], v_t[nm + ns:])]:
            print(f"  {name:9} torch-vs-numpy  max|d| {np.max(np.abs(a-b)):.2e}  RMS {np.sqrt(np.mean((a-b)**2)):.2e}   (n={len(a)})")
        print(f"  forward wall:  numpy {t_np:.2f}s   torch {t_t:.2f}s")
        # jacrev vs torch-FD (one column) sanity + timing
        resid_t, _, _ = make_full_residual(ctx, 8.0, 0.8)
        t = time.time(); Jad = torch.func.jacrev(resid_t)(torch.tensor(x0, dtype=torch.float32, device=DEV)).detach().cpu().numpy(); t_ad = time.time() - t
        h = 1e-3; x1 = x0.copy(); x1[0] += h
        r0 = resid_t(torch.tensor(x0, dtype=torch.float32, device=DEV)).detach().cpu().numpy()
        r1 = resid_t(torch.tensor(x1, dtype=torch.float32, device=DEV)).detach().cpu().numpy()
        fd0 = (r1 - r0) / h
        print(f"  jacrev: {Jad.shape} in {t_ad:.2f}s   col0 AD-vs-FD max|d| {np.max(np.abs(Jad[:,0]-fd0)):.2e}")
        sys.exit(0)

    w_marg = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    w_vov = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    max_nfev = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    ctx = build_full_ctx(date)
    anchor = C.X0_MAP["ts"].copy()
    print(f"  blocks: marginal {len(ctx['sanos_dig'])} + SSR {len(ctx['emp'])} + VoV {len(ctx['vdtes'])}   "
          f"emp SSR {np.round(ctx['emp'],3)}", flush=True)

    if mode == "fit1":
        res = fit_full(ctx, anchor, w_marg, w_vov, max_nfev)
        _report(res, ctx, w_marg, w_vov, tag="fit1")
    else:
        res = fit_full_multistart(ctx, anchor, w_marg, w_vov, max_nfev)
        _report(res, ctx, w_marg, w_vov, tag="fit")
