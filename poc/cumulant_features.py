"""
cumulant_features.py -- density-native ATM features and SSR from the kernel cumulants.

The conditional forward-return density of the two-timescale kernel is a Gaussian mixture, so
its cumulants are closed form; the ATM IV level/skew/curvature (Lee) and the SSR are then
analytic functions of the generator. This computes them in closed form and checks the
cumulant-route SSR against the existing finite-difference (IV-inversion) SSR.

Run from poc/ :  python3 cumulant_features.py
"""
import numpy as np
from discslv import TwoTimescaleKernel


def cond_components(K, z0):
    """Flat (omega, mu, v) of the conditional forward-return GM at conditioning spot z0."""
    pb = K.slow_weights(z0)
    om, mu, v = [], [], []
    for b in range(K.n_slow):
        wl, d, sig2 = K._increment(z0, b)
        for l in range(K.n_fast):
            om.append(pb[b] * wl[l]); mu.append(d[l]); v.append(sig2[l])
    return np.array(om), np.array(mu), np.array(v)


def cumulants(K, z0):
    """kappa_1..4 of the conditional forward-return density (closed form for a GM)."""
    om, mu, v = cond_components(K, z0)
    M1 = np.sum(om * mu)
    M2 = np.sum(om * (mu ** 2 + v))
    M3 = np.sum(om * (mu ** 3 + 3 * mu * v))
    M4 = np.sum(om * (mu ** 4 + 6 * mu ** 2 * v + 3 * v ** 2))
    k1 = M1
    k2 = M2 - M1 ** 2
    k3 = M3 - 3 * M1 * M2 + 2 * M1 ** 3
    k4 = M4 - 4 * M1 * M3 - 3 * M2 ** 2 + 12 * M1 ** 2 * M2 - 6 * M1 ** 4
    return k1, k2, k3, k4


def atm_features(K, z0):
    """Lee cumulant->IV: total ATM level u0, skew u1, curvature u2 + standardized moments."""
    _, k2, k3, k4 = cumulants(K, z0)
    s = np.sqrt(k2)
    return dict(s=s,
                gamma=k3 / k2 ** 1.5,         # skewness
                eta=k4 / k2 ** 2,             # excess kurtosis
                u0=s + k3 / (4 * s),          # total ATM IV
                u1=k3 / (6 * s ** 3),         # total ATM skew (per log-moneyness)
                u2=k4 / (12 * s ** 5))        # total ATM curvature


def ssr_cumulant(K, z0, h=1e-3):
    """Analytic (cumulant-route) SSR = (d sqrt(k2)/d z0)/u1 = 3 k2 (d k2/d z0)/k3.
    Leading-order Lee (ATM level ~ s, ATM skew = k3/6 s^3); the level and skew leading-order
    errors partly cancel in the ratio, so this beats keeping the k3/(4s) level correction."""
    _, k2, k3, _ = cumulants(K, z0)
    _, k2p, _, _ = cumulants(K, z0 + h)
    _, k2m, _, _ = cumulants(K, z0 - h)
    return 3.0 * k2 * (k2p - k2m) / (2 * h) / k3


def main():
    K = TwoTimescaleKernel(np.log(0.04), 0.45, 0.35, -1.0, -4.5, 0.5, dt=1.0 / 12.0)  # SSR~1.5
    z0, dt = 0.0, K.dt
    k1, k2, k3, k4 = cumulants(K, z0)
    feat = atm_features(K, z0)
    print("=== conditional cumulants at z0=0 (closed form) ===")
    print(f"k2={k2:.5e}  k3={k3:+.5e}  k4={k4:+.5e}")
    print(f"skewness gamma={feat['gamma']:+.4f}   excess-kurt eta={feat['eta']:+.4f}")

    print("\n=== Lee ATM features vs kernel finite-difference (total, x sqrt(dt)) ===")
    atm_fd = K.forward_iv(z0, 1.0) * np.sqrt(dt)
    skew_fd = K.forward_skew(z0) * np.sqrt(dt)
    print(f"ATM level: Lee u0={feat['u0']:.5f}  vs fd={atm_fd:.5f}  (rel {abs(feat['u0']/atm_fd-1):.1%})")
    print(f"ATM skew : Lee u1={feat['u1']:+.5f}  vs fd={skew_fd:+.5f}  (rel {abs(feat['u1']/skew_fd-1):.1%})")

    print("\n=== SSR: cumulant-route (analytic) vs finite-difference (IV inversion) ===")
    for z in [-0.05, 0.0, 0.05]:
        sc, sf = ssr_cumulant(K, z), K.ssr(z)
        print(f"z0={z:+.2f}:  SSR_cum={sc:+.4f}   SSR_fd={sf:+.4f}   rel={abs(sc/sf-1):.1%}")


if __name__ == "__main__":
    main()
