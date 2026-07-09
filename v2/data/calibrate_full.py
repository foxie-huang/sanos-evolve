#!/usr/bin/env python3
"""
FULL eq:objective calibration (disc_SLV.tex §850 / alg:calib) -- the THREE terms of the paper's loss,
where calibrate_joint had only the last two:
  (1) MARGINAL digital band-loss: the Gyongy-fused PROPAGATED marginal G^S_theta (from-spot leveraged
      propagation) vs the SANOS/market marginals, as survival-probability (digital) distances on a
      log-moneyness grid  ->  "fit the kernel to reproduce the T_j->T_{j+1} transition";
  (2) beta_SSR * SSR term-structure (exact beta + term-structure leverage);
  (3) beta_vov * VIX vol-of-vol readout.
Fitting the SSR alone (calibrate_slv_exact_ts) lets theta drift off the marginal (the de-event bridge's
variance over-production -- NOT a real "SSR-vs-marginal tension"). Adding (1) forces the kernel to reproduce
the marginal by construction: the faithful alg:calib, and the correct theta for the de-eventing bridge.
Pooled Jacobian (reuses calibrate_joint's SSR+VoV machinery). Records wall-time.
    OOS_DATE=YYYY-MM-DD python3 calibrate_full.py [test | n_workers] [w_marg] [w_vov]
"""
import sys, os, time, glob
import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import calibrate_slv_exact_ts as C                                     # noqa: E402
import calibrate_slv_exact_ts_par as P                                 # noqa: E402
import discslv_slv                                                     # noqa: E402
from slv_fast import propagate_vec                                     # noqa: E402
discslv_slv.propagate = propagate_vec
from discslv_slv import (propagate, marginal, initial_state, E_nu_given_z_vec,  # noqa: E402
                         Epi_V, nu_bar, raw_increment)
from discslv_2f import TwoFactorSV                                     # noqa: E402
from slv_wire import solve_gbar, leverage_at                          # noqa: E402
from slv_interp import interp_marginal                                # noqa: E402
from vix_readout import model_vix_ivol, data_vix                       # noqa: E402
from empirical_ssr import empirical_ssr                                # noqa: E402

DT = C.DT; NS = C.NS; NAMES = C.NAMES; MIN_DTE = 7
KS = np.linspace(-0.08, 0.08, 7)                                       # log-moneyness grid for the digital band-loss


def digitals(mu):
    """Survival Pr(x > k) = sum_i W_i * Phi((MU_i - k)/SG_i) at each k in KS, for a GM marginal (W,MU,SG)."""
    W, MU, SG = mu
    return np.array([np.sum(W * norm.cdf((MU - k) / SG)) for k in KS])


def model_marginals(chain, sig, theta):
    """From-spot leveraged propagation -> {n: (W,MU,SG)} at the NS maturities (the fused G^S_theta)."""
    kw0 = dict(zip(NAMES, C.X0_MAP["dense"]))
    EV0 = Epi_V(TwoFactorSV(gbar=solve_gbar(kw0, sig, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw0))
    kw = dict(zip(NAMES, np.asarray(theta, float)))
    K = TwoFactorSV(gbar=solve_gbar(kw, sig, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    st = initial_state(K); out = {}
    for k in range(1, max(NS) + 1):
        lf = leverage_at(chain, k * DT, EV0)
        st, _ = propagate(K, st, lambda mm, l=lf, cur=st: l(mm) ** 2 / np.clip(E_nu_given_z_vec(mm, cur, nub), 0.3, 3.0),
                          EV, nub, Vlr, tiltr, 16)
        if k in NS:
            out[k] = marginal(st)
    return out


def _vix_targets(sig_ref):
    spot, dv = data_vix(P.OOS_DATE); dv = [d for d in dv if d[0] >= MIN_DTE]
    return spot, np.array([d[0] for d in dv], float), np.array([d[2] for d in dv])


def _model(theta, chain, sig, spot, vdtes, nz=C.NZ):
    """concat[ marginal digitals (NS x KS), SSR (5), VIX vov (n_vix) ] for one theta."""
    mm = model_marginals(chain, sig, theta)
    marg = np.concatenate([digitals(mm[n]) for n in NS])
    ssr = C.exact_vec_ts(np.asarray(theta, float), chain, sig, nz=nz)
    kw = dict(zip(NAMES, np.asarray(theta, float)))
    K = TwoFactorSV(gbar=solve_gbar(kw, sig, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    vov = np.array([model_vix_ivol(K, sig, d / 365.0, spot=spot)[1] for d in vdtes])
    return np.concatenate([marg, ssr, vov])


_W = {}


def _jinit():
    _W["chain"], _W["sig"] = P.build_ctx()                             # also populates C._CACHE (SSR leverage)
    _W["spot"], _W["vdtes"], _ = _vix_targets(_W["sig"])


def _jeval(theta):
    return _model(theta, _W["chain"], _W["sig"], _W["spot"], _W["vdtes"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "8"
    W_MARG = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    W_VOV = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    date = P.OOS_DATE
    chain, sig = P.build_ctx()
    spot, vdtes, vov_d = _vix_targets(sig)
    emp, nd = empirical_ssr(sorted(glob.glob(f"{P.OUT}/SPX-NDX-RUT-VIX_{P.YR}-*.json.gz")), ns=NS, dt=DT)
    sanos_dig = np.concatenate([digitals(interp_marginal(chain, n * DT)) for n in NS])   # SANOS survival target
    n_marg = len(sanos_dig); n_ssr = len(emp); n_vix = len(vdtes)
    x0 = C.X0_MAP["ts"].copy()

    def resid_parts(v):
        marg = v[:n_marg]; ssr = v[n_marg:n_marg + n_ssr]; vov = v[n_marg + n_ssr:]
        return (W_MARG * (marg - sanos_dig),
                C.WREL * (ssr - emp) / emp,
                W_VOV * (vov - vov_d) / vov_d)

    if mode == "test":                                                 # one eval: check the block magnitudes before fitting
        v = _model(x0, chain, sig, spot, vdtes, nz=9)
        m, s, vv = resid_parts(v)
        print(f"date={date}  x0=ts")
        print(f"  MARGINAL block: {n_marg} resid, RMS(raw survival dist) {np.sqrt(np.mean(((v[:n_marg]-sanos_dig))**2)):.4f}  weighted RMS {np.sqrt(np.mean(m**2)):.4f}")
        print(f"  SSR block: model {np.round(v[n_marg:n_marg+n_ssr],3)} vs emp {np.round(emp,3)}  weighted RMS {np.sqrt(np.mean(s**2)):.4f}")
        print(f"  VOV block: {n_vix} pts, weighted RMS {np.sqrt(np.mean(vv**2)):.4f}")
        sys.exit(0)

    NW = int(mode)
    print(f"FULL eq:objective  date={date}  NW={NW}  w_marg={W_MARG} w_vov={W_VOV}", flush=True)
    print(f"  blocks: marginal {n_marg} (5 mat x {len(KS)} strikes) + SSR {n_ssr} + VoV {n_vix}", flush=True)
    print(f"  emp SSR {np.round(emp,3)}  VIX dtes {vdtes.astype(int)}", flush=True)

    _last = {}
    def ev(x):
        key = np.asarray(x, float).tobytes()
        if _last.get("k") != key:
            _last["k"] = key; _last["v"] = _model(x, chain, sig, spot, vdtes)
        return _last["v"]

    def resid(x):
        return np.concatenate(resid_parts(ev(x)))

    pool = ProcessPoolExecutor(max_workers=NW, initializer=_jinit)

    def jac(x, *a):
        x = np.asarray(x, float); f0 = ev(x)
        h = 5e-2 * np.maximum(np.abs(x), 1.0)
        steps = np.where(x + h <= C.HI, h, -h)
        xs = [x + steps[i] * np.eye(len(x))[i] for i in range(len(x))]
        fps = list(pool.map(_jeval, xs))
        m = n_marg + n_ssr + n_vix
        J = np.zeros((m, len(x)))
        for i in range(len(x)):
            J[:, i] = np.concatenate(resid_parts((fps[i] - f0) / steps[i]))
        return J

    t0 = time.time()
    res = least_squares(resid, x0, jac=jac, bounds=(C.LO, C.HI), max_nfev=80, xtol=1e-6, ftol=1e-6, verbose=2)
    pool.shutdown()
    v = _model(res.x, chain, sig, spot, vdtes, nz=15); m, s, vv = resid_parts(v)
    print(f"\nFULL fit  ({res.nfev} evals, {time.time()-t0:.0f}s)")
    print("theta: " + "  ".join(f"{n}={val:.3f}" for n, val in zip(NAMES, res.x)) + "\n")
    print(f"  MARGINAL survival RMS {np.sqrt(np.mean(((v[:n_marg]-sanos_dig))**2))*100:.2f}%")
    print(f"  SSR: model {np.round(v[n_marg:n_marg+n_ssr],3)} vs emp {np.round(emp,3)}  ({'  '.join(f'{100*(v[n_marg+i]-emp[i])/emp[i]:.0f}%' for i in range(n_ssr))})")
    print(f"  VoV RMS {np.sqrt(np.mean(((vv/W_VOV))**2))*100:.0f}%")
