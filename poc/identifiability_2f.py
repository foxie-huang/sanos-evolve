"""
identifiability_2f.py -- local identifiability of the 9 two-factor knobs (Step 6, current kernel).

Observable map O(theta) (9 closed-form observables at the calibrated theta):
  ATM vol @ 1m,1y ; ATM skew @ 1m,1y ; curvature @ 1m,1y ; SSR @ 1m,6m,1y.
Finite-difference Jacobian, elasticity-scale  J~_ij = J_ij |theta_j| / |O_i|, SVD -> rank / cond /
softest direction / leading observable per knob.

Run from poc/ :  python3 identifiability_2f.py
"""
import numpy as np
import warnings
from discslv_2f import TwoFactorSV, smile_2f, stationary_pi

warnings.filterwarnings("ignore")
DT = 1.0 / 52.0
NK = 12
NAMES = ["gbar", "nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
XSTAR = np.array([-5.30, 0.43, 0.50, 0.14, -1.48, 0.98, 1.65, 1.00, 2.34])   # calibrated theta
OBS = ["atm1m", "atm1y", "sk1m", "sk1y", "cv1m", "ssr1m", "ssr3m", "ssr6m", "ssr1y"]


def kernel(x):
    return TwoFactorSV(gbar=x[0], nu_f=x[1], nu_s=x[2], lam_skew=x[4], lam_f=x[5], lam_s=x[6],
                       kap_f=x[7], kap_s=x[8], dt=DT, nu_l=x[3], n_f=5, n_s=3, n_l=5)


def smile_stats(g, T, dm=6e-3):
    F = g.forward(); iv = lambda k: float(g.implied_vol(F * k, T)[0])
    a, up, dn = iv(1.0), iv(np.exp(dm)), iv(np.exp(-dm))
    return a, (up - dn) / (2 * dm), (up - 2 * a + dn) / dm ** 2          # atm, skew, curvature


def obs(x):
    K = kernel(x); pi = stationary_pi(K); nf, ns = K.n_f, K.n_s
    res = {}
    for n in (4, 13, 26, 52):
        T = n * K.dt
        A = np.zeros((nf, ns)); S = np.zeros((nf, ns)); C = np.zeros((nf, ns))
        for f in range(nf):
            for s in range(ns):
                A[f, s], S[f, s], C[f, s] = smile_stats(smile_2f(K, n, f, s, NK), T)
        atm = float((pi * A).sum()); skew = float((pi * S).sum()); curv = float((pi * C).sum())
        cov = 0.0; var = 0.0
        for f in range(nf):
            for s in range(ns):
                d, V = K.incr(f, s); mean_r = float((K.wl * d).sum())
                var_fs = float((K.wl * (d ** 2 + V)).sum() - mean_r ** 2)
                c = 0.0
                for l in range(K.n_l):
                    Pf, Ps = K.trans_f(l, f), K.trans_s(l, s)
                    resp = sum(Pf[fp] * Ps[sp] * (A[fp, sp] - A[f, s]) for fp in range(nf) for sp in range(ns))
                    c += K.wl[l] * d[l] * resp
                cov += pi[f, s] * c; var += pi[f, s] * var_fs
        res[n] = (atm, skew, curv, (cov / var) / skew)
    return np.array([res[4][0], res[52][0], res[4][1], res[52][1], res[4][2],
                     res[4][3], res[13][3], res[26][3], res[52][3]])


def main():
    O0 = obs(XSTAR)
    J = np.zeros((9, 9))
    for j in range(9):
        h = 0.03 * max(abs(XSTAR[j]), 0.1)
        xp = XSTAR.copy(); xp[j] += h; xm = XSTAR.copy(); xm[j] -= h
        J[:, j] = (obs(xp) - obs(xm)) / (2 * h)
    Jt = J * np.abs(XSTAR)[None, :] / (np.abs(O0)[:, None] + 1e-9)        # elasticity scale
    U, sv, Vt = np.linalg.svd(Jt)
    rank = int((sv > 1e-8 * sv[0]).sum()); cond = sv[0] / sv[-1]
    print("O(theta*) =", dict(zip(OBS, np.round(O0, 3))))
    print("\nsingular values:", "  ".join(f"{s:.3f}" for s in sv))
    print(f"rank {rank}/9   cond {cond:.1f}")
    print("\nleading observable per knob (|elasticity|):")
    for j, nm in enumerate(NAMES):
        i = int(np.argmax(np.abs(Jt[:, j])))
        print(f"  {nm:>9} -> {OBS[i]:>6}   ({Jt[i, j]:+.2f})")
    print("\nsoftest singular vector (loadings):")
    for nm, v in sorted(zip(NAMES, Vt[-1]), key=lambda t: -abs(t[1])):
        print(f"  {nm:>9}: {v:+.2f}")


if __name__ == "__main__":
    main()
