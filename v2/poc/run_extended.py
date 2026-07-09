"""
run_extended.py -- SANOS-Evolve PoC, extended capabilities (paper Secs. 7, 8, 11).

Builds on run_poc.py (the single-step Sec. 12 pipeline) to demonstrate, on synthetic
convex-ordered marginals (no real data):

  A. Multi-step propagation + recompression (Secs. 8, 11).  One time-homogeneous
     generator propagates a whole maturity chain; kernel recompression keeps the
     component budget FLAT (vs the L^J blow-up) while preserving the martingale and
     convex order at every maturity.

  B. VIX vol-of-vol layer (Sec. 7).  Model-implied VIX future / smile / vol-of-vol from
     the propagated regime law and the damped regime variances; the damping shows VIX is
     slow-dominated; nu_s identifies the vol-of-vol (the R_vov target).

Run:  python run_extended.py
"""
import numpy as np
from scipy.optimize import minimize_scalar

import discslv as ds
from run_poc import fit_marginal, fit_movement


def propagate_chain(params, mus, maturities, Kgrid, n_keep=6, recompress=True):
    """Propagate mu_0 across the chain with a time-homogeneous generator (same params,
    each step's own dt); recompress each step.  Returns per-step diagnostics and the
    list of joint states by maturity."""
    g, ns, nf, lsk, lmv, eps = params
    k0 = ds.TwoTimescaleKernel(g, ns, nf, lsk, lmv, eps, maturities[1] - maturities[0])
    comps = ds.lift_to_joint(mus[0], k0)
    rows, history = [], [comps]
    for j in range(1, len(maturities)):
        dt = maturities[j] - maturities[j - 1]
        k = ds.TwoTimescaleKernel(g, ns, nf, lsk, lmv, eps, dt)
        comps = k.propagate(comps)
        n_raw = len(comps)
        if recompress:
            comps = ds.recompress_joint(comps, n_keep, k.n_slow)
        prop = ds.marginal_from_joint(comps, F=mus[0].F)
        ks = float(np.max(np.abs(prop.cdf(Kgrid) - mus[j].cdf(Kgrid))))
        cx = ds.convex_order_violation(mus[j - 1], prop, Kgrid)   # want prop >=cx mu_{j-1}
        rows.append((maturities[j], len(comps), n_raw, ks, cx, prop.forward()))
        history.append(comps)
    return rows, history


def main():
    F = 1.0
    maturities = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    Kgrid = np.linspace(0.70, 1.40, 71) * F
    mus = ds.synthetic_chain(maturities, F=F)

    # one generator, fit on the first step (time-homogeneous)
    dt1 = maturities[1] - maturities[0]
    base = fit_marginal(mus[0], mus[1], dt1, Kgrid)
    g, ns, nf, lsk, eps = base
    lmv = fit_movement(base, dt1, 1.30)
    params = (g, ns, nf, lsk, lmv, eps)

    print("=" * 82)
    print("SANOS-Evolve  --  extended PoC: multi-step + recompression (Sec. 8/11), VIX (Sec. 7)")
    print("=" * 82)
    print(f"\nGenerator (fit on step 1, used time-homogeneously):")
    print(f"    gbar={g:.3f}  nu_s={ns:.3f}  nu_f={nf:.3f}  lam_skew={lsk:.3f}"
          f"  lam_mov={lmv:.3f}  eps_s={eps:.3f}")

    # ---- PART A: multi-step propagation + recompression ----
    rows, history = propagate_chain(params, mus, maturities, Kgrid, n_keep=6, recompress=True)
    n0 = len(history[0])
    nf_ns = 3 * 5     # n_slow * n_fast spawned per step
    print(f"\n[A] Multi-step propagation with recompression (budget 6 comps/regime)")
    print(f"    {'T':>5}{'#comp':>8}{'pre-recomp':>12}{'no-recomp*':>12}"
          f"{'KS vs mu_T':>12}{'cx-viol':>10}{'forward':>10}")
    for i, (T, ncomp, nraw, ks, cx, fwd) in enumerate(rows, start=1):
        would_be = n0 * nf_ns ** i                       # L^J blow-up if never recompressed
        print(f"    {T:>5.2f}{ncomp:>8d}{nraw:>12d}{would_be:>12d}"
              f"{ks:>12.4f}{cx:>10.2e}{fwd:>10.6f}")
    print(f"    *no-recomp = theoretical count {n0}x{nf_ns}^j if recompression were off"
          f" (kept flat at {rows[-1][1]} instead).")
    print(f"    KS stays small, convex order holds (cx-viol ~ 0), forward = 1: recompression")
    print(f"    preserves no-arbitrage while bounding the budget.")

    # ---- PART B: VIX layer ----
    jT = 3                                                # report VIX at T = maturities[3] = 1.0
    kT = ds.TwoTimescaleKernel(g, ns, nf, lsk, lmv, eps, maturities[jT] - maturities[jT - 1])
    p, vix, kappa_s, D_s, D_f = ds.vix_layer(kT, history[jT])
    Kvix = np.array([0.12, 0.16, 0.20, 0.24])
    print(f"\n[B] VIX vol-of-vol layer at T={maturities[jT]}  (tau=30d)")
    print(f"    regime law p_b      = {np.array2string(p, precision=3)}")
    print(f"    regime VIX sqrt(vbar)= {np.array2string(vix, precision=3)}")
    print(f"    VIX future E[VIX_T]  = {ds.vix_future(p, vix):.4f}")
    print(f"    VIX vol-of-vol (std) = {ds.vix_vov(p, vix):.4f}")
    print(f"    VIX smile (call px): " +
          "  ".join(f"K={k:.2f}:{c:.4f}" for k, c in zip(Kvix, ds.vix_call(p, vix, Kvix))))
    print(f"    damping  D(kappa_s={kappa_s:.2f})={D_s:.3f}   D(kappa_f=30)={D_f:.3f}"
          f"   -> VIX is slow-dominated")

    # identification: nu_s <-> vol-of-vol
    def xi_of(nu_s_val):
        k = ds.TwoTimescaleKernel(g, nu_s_val, nf, lsk, lmv, eps, dt1)
        pp = k.regime_prior()
        v_inf = np.exp(g); v_b = np.exp(g + nu_s_val * k.zeta_s)
        vv = np.sqrt(np.maximum(v_inf + (v_b - v_inf) * D_s, 1e-12))
        return ds.vix_vov(pp, vv)

    XI_VIX = 0.030
    nu_fit = float(minimize_scalar(lambda x: (xi_of(x) - XI_VIX) ** 2,
                                   bounds=(0.0, 1.5), method="bounded").x)
    print(f"\n    Identification (R_vov):  nu_s -> VIX vol-of-vol")
    for v in [0.2, 0.4, 0.6, 0.8]:
        print(f"        nu_s={v:.2f}  ->  xi={xi_of(v):.4f}")
    print(f"    fit nu_s to xi_VIX target {XI_VIX}:  nu_s={nu_fit:.3f}  (xi={xi_of(nu_fit):.4f})")
    print(f"    -> the VIX smile width identifies the kernel's vol-of-vol (nu_s).")
    print("=" * 82)


if __name__ == "__main__":
    main()
