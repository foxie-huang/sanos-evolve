#!/usr/bin/env python3
"""
slv_fast.py -- vectorized drop-in for discslv_slv.propagate (the fused/leveraged forward map).

The original is a Python quadruple-loop (components x nodes x n_f x n_s with list-appends) written
for validation, ~30x slower than the bare kernel's vectorized propagate. This reproduces its exact
semantics with numpy broadcasting (mirrors discslv_2f.propagate + the leveraged increment). Requires
lam2_of to accept an ARRAY of component means (slv_wire.leverage_at's lam is now vectorized).

Run `python3 slv_fast.py` to verify it matches the original bit-for-bit before swapping it in.
"""
import sys, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import recompress_2f                                  # noqa: E402


def propagate_vec(K, state, lam2_of, EV, nub, Vlr, tiltr, nk):
    """Vectorized fused one step: leveraged increment (sigma->lam*sigma, martingale re-locked per
    component) expanded over (l, f', s') and recompressed. Same output as discslv_slv.propagate."""
    W, MU, SG, F, S = state
    N = W.shape[0]; nl, nf, ns_ = K.n_l, K.n_f, K.n_s
    lam = np.sqrt(np.maximum(np.asarray(lam2_of(MU), float), 1e-12))    # (N,) leverage at component means
    Vl = lam[:, None] ** 2 * Vlr[F, S]                                 # (N, nl) leveraged variance
    mtil = -0.5 * Vl + lam[:, None] * tiltr[F, S]                      # (N, nl)
    A = np.log(np.sum(K.wl[None, :] * np.exp(mtil + 0.5 * Vl), axis=1))  # (N,) martingale lock
    Dl = mtil - A[:, None]                                             # (N, nl)
    Tfi = np.transpose(K.Tf[:, F, :], (1, 0, 2))                       # (N, nl, nf)
    Tsi = np.transpose(K.Ts[:, S, :], (1, 0, 2))                       # (N, nl, ns)
    shape = (N, nl, nf, ns_)
    w = W[:, None, None, None] * K.wl[None, :, None, None] * Tfi[:, :, :, None] * Tsi[:, :, None, :]
    mu_b = np.broadcast_to((MU[:, None] + Dl)[:, :, None, None], shape)
    sg_b = np.broadcast_to(np.sqrt(SG[:, None] ** 2 + Vl)[:, :, None, None], shape)
    fp_b = np.broadcast_to(np.arange(nf)[None, None, :, None], shape)
    sp_b = np.broadcast_to(np.arange(ns_)[None, None, None, :], shape)
    W2 = w.ravel(); W2 = W2 / W2.sum()
    out = recompress_2f(W2, mu_b.ravel(), sg_b.ravel(), fp_b.ravel().astype(np.intp),
                        sp_b.ravel().astype(np.intp), nk, nf, ns_)
    return out, lam


def fused_ssr_readout_ts(K, lam_fns, n, EV, nub, Vlr, tiltr, nk, dt, delta=5e-3):
    """Term-structure variant of discslv_slv.fused_ssr_readout: lam_fns is a list of n leverage
    functions, lam_fns[k] applied at step k (the leverage at maturity ~(k+1)*dt). Tests whether a
    maturity-varying leverage (steep short horizons, flat long) restores the LV that collapses when
    one frozen T-leverage is used at every step. The one-step return moments use lam_fns[0] (short end)."""
    from discslv_slv import stationary_pi, atm_skew_of, marginal, lev_increment
    pi = stationary_pi(K); nf, ns = K.n_f, K.n_s

    def atm_skew(z0, f, s):
        st = (np.array([1.0]), np.array([float(z0)]), np.array([1e-4]), np.array([f], np.intp), np.array([s], np.intp))
        for k in range(n):
            st, _ = propagate_vec(K, st, (lambda fn: (lambda mc: fn(mc) ** 2))(lam_fns[k]), EV, nub, Vlr, tiltr, nk)
        return atm_skew_of(marginal(st), n * dt)

    sig = np.zeros((nf, ns)); skw = np.zeros((nf, ns)); dsig = np.zeros((nf, ns))
    for f in range(nf):
        for s in range(ns):
            sig[f, s], skw[f, s] = atm_skew(0.0, f, s)
            ap, _ = atm_skew(delta, f, s); am, _ = atm_skew(-delta, f, s)
            dsig[f, s] = (ap - am) / (2 * delta)
    skew = float(np.sum(pi * skw))

    lam0 = float(np.asarray(lam_fns[0](0.0)))                          # short-end leverage for the one-step return
    svcov = lvcov = var = 0.0
    for f in range(nf):
        for s in range(ns):
            Dl, Vl = lev_increment(Vlr[f, s], tiltr[f, s], K.wl, lam0)
            mean_r = float(np.sum(K.wl * Dl))
            var += pi[f, s] * (float(np.sum(K.wl * (Dl ** 2 + Vl))) - mean_r ** 2)
            csv = clv = 0.0
            for l in range(K.n_l):
                Pf, Ps = K.trans_f(l, f), K.trans_s(l, s)
                jump = sum(Pf[fp] * Ps[sp] * (sig[fp, sp] - sig[f, s]) for fp in range(nf) for sp in range(ns))
                roll = sum(Pf[fp] * Ps[sp] * dsig[fp, sp] for fp in range(nf) for sp in range(ns))
                csv += K.wl[l] * Dl[l] * jump; clv += K.wl[l] * (Dl[l] ** 2 + Vl[l]) * roll
            svcov += pi[f, s] * csv; lvcov += pi[f, s] * clv
    sv = svcov / var; lv = lvcov / var
    return (lv + sv) / skew, lv / skew, sv / skew


def fused_ssr_exact(K, lam_fn, n, EV, nub, Vlr, tiltr, nk, dt, nz=15, zmax=0.12, Q=5):
    """EXACT closed-form fused SSR (de-sampled fused_ssr_mc). Precompute sigma_ATM(z,f,s) on a
    z-grid per regime, then form beta = Cov(dsigma, r)/Var(r) as an ANALYTIC sum over the one-step
    transition: return nodes l x destination regimes (f',s') x a within-node Gauss-Hermite
    sub-quadrature (Q points) for the Gaussian innovation. No sampling, no roll.r linearization ->
    captures the curvature and cross terms the response-form readout drops. Matches fused_ssr_mc
    to grid resolution. Returns (SSR, beta, skew)."""
    from discslv_slv import stationary_pi, atm_skew_of, marginal, lev_increment
    from numpy.polynomial.hermite_e import hermegauss
    nf, ns, nl = K.n_f, K.n_s, K.n_l
    zg = np.linspace(-zmax, zmax, nz); iz0 = nz // 2
    sigg = np.zeros((nf, ns, nz)); skw0 = np.zeros((nf, ns))          # sigma_ATM(z,f,s) grid + ATM skew at z=0
    for f in range(nf):
        for s in range(ns):
            for iz, z in enumerate(zg):
                st = (np.array([1.0]), np.array([float(z)]), np.array([1e-4]), np.array([f], np.intp), np.array([s], np.intp))
                for _ in range(n):
                    st, _ = propagate_vec(K, st, (lambda mc: lam_fn(mc) ** 2), EV, nub, Vlr, tiltr, nk)
                a, sk = atm_skew_of(marginal(st), n * dt)
                sigg[f, s, iz] = a
                if iz == iz0:
                    skw0[f, s] = sk
    pi = stationary_pi(K); lam0 = float(np.asarray(lam_fn(0.0)))
    zq, wq = hermegauss(Q); wq = wq / wq.sum()                        # within-node Gaussian sub-quadrature
    beta_num = var = 0.0
    for f in range(nf):
        for s in range(ns):
            Dl, Vl = lev_increment(Vlr[f, s], tiltr[f, s], K.wl, lam0)
            mean_r = float(np.sum(K.wl * Dl))
            var += pi[f, s] * (float(np.sum(K.wl * (Dl ** 2 + Vl))) - mean_r ** 2)
            sig0 = sigg[f, s, iz0]; cov = 0.0
            for l in range(nl):
                Pf, Ps = K.trans_f(l, f), K.trans_s(l, s); sd = np.sqrt(max(Vl[l], 1e-16))
                dest = np.array([[Pf[fp] * Ps[sp] for sp in range(ns)] for fp in range(nf)])   # (nf,ns) transition wts
                for q in range(Q):
                    r = Dl[l] + sd * zq[q]                            # return at node l, sub-point q
                    sig_dest = sum(dest[fp, sp] * np.interp(r, zg, sigg[fp, sp]) for fp in range(nf) for sp in range(ns))
                    cov += K.wl[l] * wq[q] * (sig_dest - sig0) * (r - mean_r)
            beta_num += pi[f, s] * cov
    beta = beta_num / var; skew = float(np.sum(pi * skw0))
    return beta / skew, beta, skew


def fused_ssr_exact_ts(K, lam_fns, n, EV, nub, Vlr, tiltr, nk, dt, nz=15, zmax=0.12, Q=5):
    """Term-structure EXACT fused SSR: fused_ssr_exact with a PER-STEP leverage. lam_fns is a list of
    n leverage functions, lam_fns[k] applied at step k (the leverage at maturity ~(k+1)*dt) -- steep at
    short horizons, flat at long. This is the PRINCIPLED + ACCURATE term-structure fix: exact beta (no
    readout truncation, no leading-order approx) with the maturity-varying leverage the local-vol
    surface actually has. The sigma_ATM(z,f,s) grid is propagated with the term structure; the one-step
    return moments use lam_fns[0] (short end). Same readout convention as fused_ssr_exact -> an
    apples-to-apples single-vs-term-structure comparison isolating the leverage-application effect."""
    from discslv_slv import stationary_pi, atm_skew_of, marginal, lev_increment
    from numpy.polynomial.hermite_e import hermegauss
    nf, ns, nl = K.n_f, K.n_s, K.n_l
    lam2_fns = [(lambda g: (lambda mc: np.asarray(g(mc), float) ** 2))(fn) for fn in lam_fns]
    zg = np.linspace(-zmax, zmax, nz); iz0 = nz // 2
    sigg = np.zeros((nf, ns, nz)); skw0 = np.zeros((nf, ns))
    for f in range(nf):
        for s in range(ns):
            for iz, z in enumerate(zg):
                st = (np.array([1.0]), np.array([float(z)]), np.array([1e-4]), np.array([f], np.intp), np.array([s], np.intp))
                for k in range(n):
                    st, _ = propagate_vec(K, st, lam2_fns[k], EV, nub, Vlr, tiltr, nk)   # step k -> its own leverage
                a, sk = atm_skew_of(marginal(st), n * dt)
                sigg[f, s, iz] = a
                if iz == iz0:
                    skw0[f, s] = sk
    pi = stationary_pi(K); lam0 = float(np.asarray(lam_fns[0](0.0)))    # first-step leverage for the one-step return
    zq, wq = hermegauss(Q); wq = wq / wq.sum()
    beta_num = var = 0.0
    for f in range(nf):
        for s in range(ns):
            Dl, Vl = lev_increment(Vlr[f, s], tiltr[f, s], K.wl, lam0)
            mean_r = float(np.sum(K.wl * Dl))
            var += pi[f, s] * (float(np.sum(K.wl * (Dl ** 2 + Vl))) - mean_r ** 2)
            sig0 = sigg[f, s, iz0]; cov = 0.0
            for l in range(nl):
                Pf, Ps = K.trans_f(l, f), K.trans_s(l, s); sd = np.sqrt(max(Vl[l], 1e-16))
                dest = np.array([[Pf[fp] * Ps[sp] for sp in range(ns)] for fp in range(nf)])
                for q in range(Q):
                    r = Dl[l] + sd * zq[q]
                    sig_dest = sum(dest[fp, sp] * np.interp(r, zg, sigg[fp, sp]) for fp in range(nf) for sp in range(ns))
                    cov += K.wl[l] * wq[q] * (sig_dest - sig0) * (r - mean_r)
            beta_num += pi[f, s] * cov
    beta = beta_num / var; skew = float(np.sum(pi * skw0))
    return beta / skew, beta, skew


_GP = {}                                                             # grid-parallel worker globals


def _gp_init(K, lam_fns, EV, nub, Vlr, tiltr, nk, dt, n, zg, iz0):
    """Worker init: store shared kernel/increment data + build the per-step lam^2 closures (the leverage
    arrives as picklable _Lev instances; the squaring wrappers are built locally, not pickled)."""
    _GP.update(K=K, EV=EV, nub=nub, Vlr=Vlr, tiltr=tiltr, nk=nk, dt=dt, n=n, zg=zg, iz0=iz0)
    _GP["lam2"] = [(lambda g: (lambda mc: np.asarray(g(mc), float) ** 2))(fn) for fn in lam_fns]


def _gp_chain(fs):
    """One regime (f,s): propagate each z-grid point n steps (term-structure leverage) -> sigma_ATM row."""
    from discslv_slv import atm_skew_of, marginal
    f, s = fs; K = _GP["K"]; zg = _GP["zg"]; n = _GP["n"]
    sig = np.zeros(len(zg)); skw = 0.0
    for iz, z in enumerate(zg):
        st = (np.array([1.0]), np.array([float(z)]), np.array([1e-4]), np.array([f], np.intp), np.array([s], np.intp))
        for k in range(n):
            st, _ = propagate_vec(K, st, _GP["lam2"][k], _GP["EV"], _GP["nub"], _GP["Vlr"], _GP["tiltr"], _GP["nk"])
        a, sk = atm_skew_of(marginal(st), n * _GP["dt"])
        sig[iz] = a
        if iz == _GP["iz0"]:
            skw = sk
    return f, s, sig, skw


def fused_ssr_exact_ts_gridpar(K, lam_fns, n, EV, nub, Vlr, tiltr, nk, dt, nz=15, zmax=0.12, Q=5, nworkers=8):
    """GRID-PARALLEL fused_ssr_exact_ts: the nf*ns independent sigma_ATM(z,f,s) chains (the expensive
    propagation, ~the whole cost) are computed across a process pool over the (f,s) regimes; the analytic
    covariance sum (cheap) stays serial. Bit-identical to fused_ssr_exact_ts, ~min(nf*ns, nworkers)x on the
    propagation. For a STANDALONE eval / per-date backtest -- do NOT nest inside the Jacobian pool (a pool
    worker cannot spawn its own pool)."""
    from discslv_slv import stationary_pi, lev_increment
    from numpy.polynomial.hermite_e import hermegauss
    nf, ns, nl = K.n_f, K.n_s, K.n_l
    zg = np.linspace(-zmax, zmax, nz); iz0 = nz // 2
    sigg = np.zeros((nf, ns, nz)); skw0 = np.zeros((nf, ns))
    tasks = [(f, s) for f in range(nf) for s in range(ns)]
    with ProcessPoolExecutor(max_workers=nworkers, initializer=_gp_init,
                             initargs=(K, lam_fns, EV, nub, Vlr, tiltr, nk, dt, n, zg, iz0)) as pool:
        for f, s, sig, skw in pool.map(_gp_chain, tasks):
            sigg[f, s, :] = sig; skw0[f, s] = skw
    pi = stationary_pi(K); lam0 = float(np.asarray(lam_fns[0](0.0)))
    zq, wq = hermegauss(Q); wq = wq / wq.sum()
    beta_num = var = 0.0
    for f in range(nf):
        for s in range(ns):
            Dl, Vl = lev_increment(Vlr[f, s], tiltr[f, s], K.wl, lam0)
            mean_r = float(np.sum(K.wl * Dl))
            var += pi[f, s] * (float(np.sum(K.wl * (Dl ** 2 + Vl))) - mean_r ** 2)
            sig0 = sigg[f, s, iz0]; cov = 0.0
            for l in range(nl):
                Pf, Ps = K.trans_f(l, f), K.trans_s(l, s); sd = np.sqrt(max(Vl[l], 1e-16))
                dest = np.array([[Pf[fp] * Ps[sp] for sp in range(ns)] for fp in range(nf)])
                for q in range(Q):
                    r = Dl[l] + sd * zq[q]
                    sig_dest = sum(dest[fp, sp] * np.interp(r, zg, sigg[fp, sp]) for fp in range(nf) for sp in range(ns))
                    cov += K.wl[l] * wq[q] * (sig_dest - sig0) * (r - mean_r)
            beta_num += pi[f, s] * cov
    beta = beta_num / var; skew = float(np.sum(pi * skw0))
    return beta / skew, beta, skew


if __name__ == "__main__":
    import time
    import discslv_slv as S
    from discslv_slv import Epi_V, nu_bar, raw_increment, propagate as propagate_loop, initial_state

    K = S.kernel(); EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K); nk = 16
    lam_fn = lambda z: np.exp(-2.0 * np.asarray(z, float))              # smooth vectorized leverage
    lam2 = lambda mc: lam_fn(mc) ** 2

    # spread a few bare steps, then compare one leveraged step from both implementations
    st = initial_state(K)
    for _ in range(3):
        st, _ = propagate_loop(K, st, lambda mc: 1.0, EV, nub, Vlr, tiltr, nk)

    (Wl, Ml, Sl, Fl, SLl), _ = propagate_loop(K, st, lam2, EV, nub, Vlr, tiltr, nk)
    (Wv, Mv, Sv, Fv, SLv), _ = propagate_vec(K, st, lam2, EV, nub, Vlr, tiltr, nk)
    print(f"components: loop={len(Wl)} vec={len(Wv)}")
    print(f"max |dW|={np.max(np.abs(Wl-Wv)):.2e}  |dMU|={np.max(np.abs(Ml-Mv)):.2e}  "
          f"|dSG|={np.max(np.abs(Sl-Sv)):.2e}  F match={np.array_equal(Fl,Fv)}  S match={np.array_equal(SLl,SLv)}")

    # timing: 26-step propagation (a 6m readout leg), both ways
    for name, prop in [("loop", propagate_loop), ("vec", propagate_vec)]:
        st2 = initial_state(K); t0 = time.time()
        for _ in range(26):
            st2, _ = prop(K, st2, lam2, EV, nub, Vlr, tiltr, nk)
        print(f"{name}: 26 steps in {time.time()-t0:.3f}s")
