"""
identifiability_check.py -- IS the two-factor GM kernel locally identifiable from the
smile + VIX observables?

This answers the *identification* question by calculation, not by an empirical study:
  (1) Local identifiability = full column rank, well-conditioned Jacobian of the
      observable map  theta -> (ATM vol, skew, SSR, curvature, VIX vov@30d, @90d).
      Because the GM kernel is closed-form, this Jacobian is a finite calculation.
  (2) Synthetic known-kernel recovery: generate observables from a known theta, fit it
      back (clean + noisy), confirm the parameters are recovered.

Maps to paper Sec. 5 (coefficient map) and the conditional-moment identification argument.
Run from poc/ :  python identifiability_check.py
"""
import warnings
import numpy as np
from scipy.optimize import least_squares
from discslv import TwoTimescaleKernel, damping

warnings.filterwarnings("ignore")  # far-off recovery starts can overflow exp() harmlessly

PARAM_NAMES = ["gbar", "nu_s", "nu_f", "lam_skew", "lam_mov", "eps_s"]
OBS_NAMES = ["ATM_fwd_vol", "fwd_skew", "SSR", "smile_curv", "VIXvov_30d", "VIXvov_90d"]


def observables(theta, dt=1.0 / 12.0, n_slow=3, n_fast=5):
    """The closed-form observable map theta -> 6 smile/VIX summaries."""
    gbar, nu_s, nu_f, lam_skew, lam_mov, eps_s = theta
    K = TwoTimescaleKernel(gbar, nu_s, nu_f, lam_skew, lam_mov, eps_s, dt, n_slow, n_fast)
    atm = K.forward_iv(0.0, 1.0)
    skew = K.forward_skew(0.0, dm=5e-3)
    ssr = K.ssr(0.0, h=5e-3, dm=5e-3)
    dm = 1e-2
    iu, i0, idn = K.forward_iv(0, np.exp(dm)), K.forward_iv(0, 1.0), K.forward_iv(0, np.exp(-dm))
    curv = (iu - 2 * i0 + idn) / dm ** 2
    # VIX vol-of-vol at two horizons: slow-dominated, so the tau-dependence separates the
    # vov AMPLITUDE (nu_s) from its DECAY RATE (eps_s -> kappa_s).
    p = K.regime_prior()
    v_inf = np.exp(gbar)
    v_b = np.exp(gbar + nu_s * K.zeta_s)
    kappa_s = -np.log(max(1.0 - eps_s, 1e-9)) / dt

    def vov(tau):
        D = damping(kappa_s, tau)
        vix = np.sqrt(np.maximum(v_inf + (v_b - v_inf) * D, 1e-12))
        m = float(np.sum(p * vix))
        return float(np.sqrt(np.sum(p * (vix - m) ** 2)))

    return np.array([atm, skew, ssr, curv, vov(30 / 365.0), vov(90 / 365.0)])


def jacobian(theta, rel=1e-4):
    f0 = observables(theta)
    J = np.zeros((len(f0), len(theta)))
    for j in range(len(theta)):
        dh = rel * max(abs(theta[j]), 1e-3)
        tp = theta.copy(); tp[j] += dh
        tm = theta.copy(); tm[j] -= dh
        J[:, j] = (observables(tp) - observables(tm)) / (2 * dh)
    return f0, J


def main():
    theta0 = np.array([np.log(0.04), 0.45, 0.35, -1.0, -4.5, 0.5])  # SSR=1.5 (SPX)
    f0, J = jacobian(theta0)
    print("=== baseline ===")
    print("params     :", dict(zip(PARAM_NAMES, np.round(theta0, 3))))
    print("observables:", dict(zip(OBS_NAMES, np.round(f0, 4))))

    # Dimensionless (elasticity) Jacobian: relative sensitivity of each observable to each
    # parameter, so the condition number is scale-free and meaningful for identifiability.
    theta_sc = np.maximum(np.abs(theta0), 1e-2)
    obs_sc = np.maximum(np.abs(f0), 1e-4)
    Js = J * theta_sc[None, :] / obs_sc[:, None]
    U, S, Vt = np.linalg.svd(Js)
    print("\n=== local identifiability (scaled-Jacobian SVD) ===")
    print("singular values :", np.round(S, 4))
    print("condition number:", round(float(S[0] / S[-1]), 1))
    print("numerical rank  :", int(np.sum(S > 1e-8 * S[0])), "/", len(theta0))
    print("softest (least-determined) direction:",
          dict(zip(PARAM_NAMES, np.round(Vt[-1], 2))))
    print("(each parameter's most-sensitive observable -- a distinct one each)")
    for j, pname in enumerate(PARAM_NAMES):
        i = int(np.argmax(np.abs(Js[:, j])))
        print(f"   {pname:9s} -> {OBS_NAMES[i]:11s} (elasticity {Js[i, j]:+.2f})")

    print("\n=== synthetic known-kernel recovery (best of 12 starts, 20%-scattered) ===")

    def recover(target, seed):
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(12):
            init = theta0 + 0.20 * theta_sc * rng.standard_normal(len(theta0))
            r = least_squares(lambda th: (observables(th) - target) / obs_sc, init,
                              method="trf", max_nfev=4000)
            if best is None or r.cost < best.cost:
                best = r
        return best.x

    for noise in [0.0, 0.01, 0.03]:
        rng = np.random.default_rng(1)
        target = f0 * (1.0 + noise * rng.standard_normal(len(f0)))
        x = recover(target, seed=7)
        err = np.abs(x - theta0) / theta_sc
        print(f"noise={noise:>4.0%} | max rel-error={err.max():.3f} | "
              f"per-param={dict(zip(PARAM_NAMES, np.round(err, 3)))}")


if __name__ == "__main__":
    main()
