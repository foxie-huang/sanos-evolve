"""
implied_vol_taylor.py -- EXACT ATM IV Taylor coefficients by implicit differentiation.

The conditional forward-return law is a log-normal (Gaussian) mixture, so the GLIDE
implicit-differentiation formulas (Cao-Huang, Sec. 6.1 / App. B) give the ATM implied-vol
level, skew and curvature in CLOSED FORM -- no Black-Scholes inversion, and EXACT (not the
leading-order cumulant proxy). The implicit identity is C_mix(m) = C_BS(m, sigma(m)); the
level is one probit, the higher coefficients are price-derivatives over BS Greeks.

This computes the exact coefficients + the exact SSR and checks them against (a) the
bisection-inverted IV (ground truth) and (b) the leading-order cumulant proxy.

Run from poc/ :  python3 implied_vol_taylor.py
"""
import numpy as np
from scipy.stats import norm
from discslv import TwoTimescaleKernel, bs_call
from cumulant_features import ssr_cumulant, atm_features


def cond_gm(K, z0):
    """Conditional forward-return GM (omega, d, v) at conditioning spot z0 (martingale: sum omega F = 1)."""
    pb = K.slow_weights(z0)
    om, d, v = [], [], []
    for b in range(K.n_slow):
        wl, dd, sig2 = K._increment(z0, b)
        for l in range(K.n_fast):
            om.append(pb[b] * wl[l]); d.append(dd[l]); v.append(sig2[l])
    return np.array(om), np.array(d), np.array(v)


def atm_call(K, z0):
    om, d, v = cond_gm(K, z0)
    return float(np.sum(om * bs_call(np.exp(d + 0.5 * v), 1.0, np.sqrt(v))))


def exact_coeffs(K, z0):
    """Exact ATM IV level, skew, curvature (annualised) by implicit differentiation -- no inversion.
    Works in total-vol u = sigma*sqrt(dt); the BS forward is 1 (martingale return), strike e^x, x=log k."""
    om, d, v = cond_gm(K, z0)
    dt = K.dt
    delta = d / np.sqrt(v)                                   # ATM d2 of each component
    Phi, phi = norm.cdf, norm.pdf

    C0 = atm_call(K, z0)
    u0 = 2.0 * norm.ppf(0.5 + 0.5 * C0)                      # total ATM IV  (one probit, eq 68)
    vega = phi(u0 / 2)

    # first/second log-strike derivatives of the closed-form mixture price at ATM
    dC = -np.sum(om * Phi(delta))                            # d C_mix / d x
    d2C = np.sum(om * (phi(delta) / np.sqrt(v) - Phi(delta)))  # d^2 C_mix / d x^2
    # same for the BS reference at (x=0, u0)
    dCbs = -Phi(-u0 / 2)
    d2Cbs = -Phi(-u0 / 2) + phi(u0 / 2) / u0
    vanna, volga = phi(u0 / 2) / 2, -u0 * phi(u0 / 2) / 4    # total-vol BS cross-Greeks at ATM

    u1 = (dC - dCbs) / vega                                  # d u / d x         (eq 76)
    u2 = (d2C - d2Cbs - 2 * vanna * u1 - volga * u1 ** 2) / vega  # d^2 u / d x^2 (eq 77)

    rt = np.sqrt(dt)
    return u0 / rt, u1 / rt, u2 / rt                         # annualised level, skew, curvature


def exact_ssr(K, z0, h=1e-4):
    """Exact SSR = (d Sigma_ATM / d z) / skew, both from the closed-form implicit-diff coeffs."""
    up, _, _ = exact_coeffs(K, z0 + h)
    dn, _, _ = exact_coeffs(K, z0 - h)
    _, skew, _ = exact_coeffs(K, z0)
    return ((up - dn) / (2 * h)) / skew


def main():
    K = TwoTimescaleKernel(np.log(0.04), 0.45, 0.35, -1.0, -4.5, 0.5, dt=1.0 / 12.0)
    dt, dm = K.dt, 1e-2
    print("Exact implicit-diff coefficients vs bisection-inverted IV (ground truth):\n")
    print(f"{'z0':>6} {'ATM ex':>8} {'ATM bis':>8} | {'skew ex':>8} {'skew bis':>8} | "
          f"{'curv ex':>8} {'curv bis':>8}")
    for z in [-0.05, 0.0, 0.05]:
        sig0, skew, curv = exact_coeffs(K, z)
        atm_b = K.forward_iv(z, 1.0)
        skew_b = K.forward_skew(z, dm)
        iu, i0, idn = K.forward_iv(z, np.exp(dm)), K.forward_iv(z, 1.0), K.forward_iv(z, np.exp(-dm))
        curv_b = (iu - 2 * i0 + idn) / dm ** 2
        print(f"{z:>+6.2f} {sig0:>8.5f} {atm_b:>8.5f} | {skew:>+8.5f} {skew_b:>+8.5f} | "
              f"{curv:>+8.4f} {curv_b:>+8.4f}")

    print("\nSSR: exact (implicit-diff) vs bisection vs leading-order cumulant proxy:\n")
    print(f"{'z0':>6} {'SSR exact':>10} {'SSR bisect':>11} {'SSR cumul':>10}   skew: ex vs cumul")
    for z in [-0.05, 0.0, 0.05]:
        se, sb, sc = exact_ssr(K, z), K.ssr(z), ssr_cumulant(K, z)
        _, skew_ex, _ = exact_coeffs(K, z)
        skew_cum = atm_features(K, z)["u1"] / np.sqrt(dt)
        print(f"{z:>+6.2f} {se:>10.4f} {sb:>11.4f} {sc:>10.4f}   "
              f"{skew_ex:+.4f} vs {skew_cum:+.4f} ({abs(skew_cum/skew_ex-1):.0%} off)")


if __name__ == "__main__":
    main()
