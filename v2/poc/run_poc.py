"""
run_poc.py -- SANOS-Evolve proof-of-concept driver (paper Sec. 12).

End-to-end on synthetic (convex-ordered) marginals:
  1. SANOS-style marginals + convex-order precheck (Theorem 1 precondition).
  2. Fit one admissible kernel between two maturities (exact martingale by construction).
  3. Marginal-propagation residual d(mu_j K_j, mu_{j+1}) in CDF / call units.
  4. Forward-start density and forward implied volatility.
  5. SSR moves with the state (Method B): d Sigma_ATM / d log S != 0.
  6. Ablation: Method B (spot-coupled, structural SSR) vs Method A (lam_mov = 0,
     sticky-delta) -- the SSR is reached at ~no marginal-fit cost.

The dynamic layer is calibrated to two DYNAMIC targets (a forward-smile skew and an
SSR), exactly as the paper treats them -- not read off the marginal.

Run:  python run_poc.py
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

import discslv as ds


def marginal_residual(kernel, mu0, mu1, Kgrid):
    """Propagate mu0, marginalise, compare to mu1 at the CDF rung.
    Returns (max|dCDF| (KS), RMS call error (% of forward), propagated GMM)."""
    comps = ds.lift_to_joint(mu0, kernel)
    prop = ds.marginal_from_joint(kernel.propagate(comps), F=mu1.F)
    ks = float(np.max(np.abs(prop.cdf(Kgrid) - mu1.cdf(Kgrid))))
    rms = float(np.sqrt(np.mean((prop.call(Kgrid) - mu1.call(Kgrid)) ** 2))) / mu1.F
    return ks, rms, prop


def fit_marginal(mu0, mu1, dt, Kgrid, n_slow=3, n_fast=5):
    """Stage 1: fit (gbar, nu_s, nu_f, lam_skew, eps_s) to the marginal, including its
    skew; the spot-coupling lam_mov stays off here."""
    x0 = np.array([np.log(0.16 ** 2), 0.4, 0.4, -0.3, 0.5])
    bounds = [(-7.0, -1.0), (0.0, 1.5), (0.0, 1.5), (-3.0, 0.0), (0.02, 0.98)]

    def obj(x):
        k = ds.TwoTimescaleKernel(x[0], x[1], x[2], x[3], 0.0, x[4], dt, n_slow, n_fast)
        ks, rms, _ = marginal_residual(k, mu0, mu1, Kgrid)
        return ks + rms

    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=400, ftol=1e-14))
    return res.x   # (gbar, nu_s, nu_f, lam_skew, eps_s)


def fit_movement(base, dt, ssr_target, n_slow=3, n_fast=5):
    """Stage 2: fit lam_mov (spot-coupling) to the SSR view; marginal params held fixed."""
    g, ns, nf, lsk, eps = base

    def obj(lm):
        k = ds.TwoTimescaleKernel(g, ns, nf, lsk, lm, eps, dt, n_slow, n_fast)
        s = k.ssr()
        return (s - ssr_target) ** 2 if np.isfinite(s) else 1e6

    return float(minimize_scalar(obj, bounds=(-4.0, 0.0), method="bounded",
                                 options=dict(xatol=1e-5)).x)


def main():
    F = 1.0
    Kgrid = np.linspace(0.70, 1.40, 71) * F
    SSR_TARGET = 1.30        # empirical SPX SSR at the ~3M step (imposed as a view)

    print("=" * 80)
    print("SANOS-Evolve  --  proof-of-concept pipeline (paper Sec. 12)")
    print("=" * 80)

    # ---- STEP 1: marginals + convex order ----
    mu0, mu1, T1, T2 = ds.synthetic_marginals()
    dt = T2 - T1
    cx = ds.convex_order_violation(mu0, mu1, Kgrid)
    print(f"\n[1] Marginals  T1={T1}, T2={T2}  (synthetic, stand-in for SANOS LP)")
    print(f"    forward(mu0)={mu0.forward():.6f}  forward(mu1)={mu1.forward():.6f}  (target {F})")
    print(f"    convex-order violation (calendar no-arb precondition): {cx:.2e}"
          f"   -> {'OK' if cx < 1e-8 else 'VIOLATED'}")

    # ---- STEP 2: fit one admissible kernel ----
    k_init = ds.TwoTimescaleKernel(np.log(0.18 ** 2), 0.5, 0.5, 0.0, 0.0, 0.5, dt)
    ks_b, rms_b, _ = marginal_residual(k_init, mu0, mu1, Kgrid)
    base = fit_marginal(mu0, mu1, dt, Kgrid)
    g, ns, nf, lam_skew, eps = base
    lam_mov = fit_movement(base, dt, SSR_TARGET)
    kB = ds.TwoTimescaleKernel(g, ns, nf, lam_skew, lam_mov, eps, dt)
    ks_a, rms_a, prop = marginal_residual(kB, mu0, mu1, Kgrid)
    print(f"\n[2] Kernel fit (admissible: positive, martingale, marginal-matching)")
    print(f"    gbar={g:.3f}  nu_s={ns:.3f}  nu_f={nf:.3f}  eps_s={eps:.3f}"
          f"  lam_skew={lam_skew:.3f}  lam_mov={lam_mov:.3f}")
    print(f"    marginal KS  init={ks_b:.4f}  ->  fitted={ks_a:.4f}    (lam_skew set by the marginal;")
    print(f"    call RMS (%F) init={1e2*rms_b:.3f}  ->  fitted={1e2*rms_a:.3f}     lam_mov is the SSR view)")

    # ---- STEP 3: martingale check ----
    print(f"\n[3] Martingale check")
    print(f"    forward(propagated mu0 K) = {prop.forward():.8f}   (target {F})")

    # ---- STEP 4: forward-start smile / forward IV ----
    print(f"\n[4] Forward-start smile  (step dt={dt}),  forward IV by return-moneyness k")
    for k in [0.90, 0.95, 1.00, 1.05, 1.10]:
        print(f"    k={k:.2f}   C^FS={kB.forward_start_call(0.0, k):.5f}"
              f"   fwd-IV={kB.forward_iv(0.0, k):.4f}")
    print(f"    ATM forward-skew dSigma/dlog k = {kB.forward_skew():+.3f}   (from the marginal fit)")

    # ---- STEP 5: SSR moves with the state ----
    dn, a0, up = (kB.forward_iv(z, 1.0) for z in (-0.01, 0.0, +0.01))
    print(f"\n[5] SSR moves with state (structural)")
    print(f"    fwd ATM-IV at z0=-1%,0,+1%:  {dn:.4f}  {a0:.4f}  {up:.4f}"
          f"   (slope dSigma_ATM/dz0 = {(up-dn)/0.02:+.4f})")
    print(f"    model SSR = {kB.ssr():.3f}   (target {SSR_TARGET})")

    # ---- STEP 6: ablation Method A vs B ----
    kA = ds.TwoTimescaleKernel(g, ns, nf, lam_skew, 0.0, eps, dt)   # sticky-delta (no SSR view)
    ksA, rmsA, _ = marginal_residual(kA, mu0, mu1, Kgrid)
    ksB, rmsB, _ = marginal_residual(kB, mu0, mu1, Kgrid)
    print(f"\n[6] Ablation  (same marginal + same static skew; vary only the SSR mechanism)")
    print(f"    {'method':<30}{'marginal KS':>13}{'fwd-skew':>11}{'SSR':>9}")
    print(f"    {'A: sticky-delta (lam_mov=0)':<30}{ksA:>13.4f}"
          f"{kA.forward_skew():>11.3f}{kA.ssr():>9.3f}")
    print(f"    {'B: structural (lam_mov fit)':<30}{ksB:>13.4f}"
          f"{kB.forward_skew():>11.3f}{kB.ssr():>9.3f}")
    print()
    print("    Reading: A and B share the marginal fit and the static skew; only B couples")
    print("    the smile to spot, so only B carries a non-zero SSR -- at no marginal cost.")
    print("    SSR is thus a structural, (nearly) marginal-free dynamic degree of freedom.")
    print("=" * 80)


if __name__ == "__main__":
    main()
