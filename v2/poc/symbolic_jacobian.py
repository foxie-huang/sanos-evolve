"""
symbolic_jacobian.py -- the identifiability Jacobian, SYMBOLICALLY.

identifiability_check.py concluded "rank 6/6, cond 19" from a CENTRAL-DIFFERENCE Jacobian.
Because the conditional forward-return law is a Gaussian mixture, the whole observable map is
closed form, so here we build it in sympy and differentiate it SYMBOLICALLY -- no finite
differences. We then (a) evaluate the exact symbolic Jacobian at the SPX baseline and report
rank / conditioning / soft direction, (b) cross-check it against a complex-step (machine-exact)
Jacobian of the same map, and (c) print the leading-order analytic sensitivity structure that
explains the rank and the soft pair.

Run from poc/ :  python3 symbolic_jacobian.py
"""
import numpy as np
import sympy as sp
from numpy.polynomial.hermite_e import hermegauss

DT_V = 1.0 / 12.0
N_SLOW, N_FAST = 3, 5
PARAM_NAMES = ["gbar", "nu_s", "nu_f", "lam_skew", "lam_mov", "eps_s"]
OBS_NAMES = ["ATM(u0)", "skew(u1)", "curv(u2)", "SSR", "VIXvov30", "VIXvov90"]

# Gauss-Hermite (probabilists') nodes/weights -> standardized factor nodes + base log-weights
zs, ws = hermegauss(N_SLOW)
zf, wf = hermegauss(N_FAST)
ZS = [sp.Float(float(x)) for x in zs]
ZF = [sp.Float(float(x)) for x in zf]
ES = [sp.Float(float(x)) for x in np.log(ws / ws.sum())]
EF = [sp.Float(float(x)) for x in np.log(wf / wf.sum())]
DT = sp.Float(DT_V)

gbar, nu_s, nu_f, lam_skew, lam_mov, eps_s, z = sp.symbols(
    "gbar nu_s nu_f lam_skew lam_mov eps_s z", real=True)
PARAMS = [gbar, nu_s, nu_f, lam_skew, lam_mov, eps_s]


def _softmax(a):
    e = [sp.exp(x) for x in a]
    S = sum(e)
    return [ei / S for ei in e]


def fast_w(zz):
    return _softmax([EF[l] + lam_mov * ZF[l] * zz for l in range(N_FAST)])


def slow_w(zz):
    return _softmax([ES[b] + lam_mov * ZS[b] * zz for b in range(N_SLOW)])


def components(zz):
    """Conditional forward-return GM (omega, mu, v) at conditioning spot zz -- mirrors
    TwoTimescaleKernel._increment with the exact per-fibre martingale lock A_b(z)."""
    wl, pb = fast_w(zz), slow_w(zz)
    omega, mu, v = [], [], []
    for b in range(N_SLOW):
        sig2, mtil = [], []
        for l in range(N_FAST):
            g = gbar + nu_s * ZS[b] + nu_f * ZF[l]
            s2 = sp.exp(g) * DT
            sig2.append(s2)
            mtil.append(-s2 / 2 + lam_skew * ZF[l] * sp.sqrt(s2))
        A = sp.log(sum(wl[l] * sp.exp(mtil[l] + sig2[l] / 2) for l in range(N_FAST)))
        for l in range(N_FAST):
            omega.append(pb[b] * wl[l]); mu.append(mtil[l] - A); v.append(sig2[l])
    return omega, mu, v


def cumulants(zz):
    om, mu, v = components(zz)
    n = len(om)
    M1 = sum(om[i] * mu[i] for i in range(n))
    M2 = sum(om[i] * (mu[i] ** 2 + v[i]) for i in range(n))
    M3 = sum(om[i] * (mu[i] ** 3 + 3 * mu[i] * v[i]) for i in range(n))
    M4 = sum(om[i] * (mu[i] ** 4 + 6 * mu[i] ** 2 * v[i] + 3 * v[i] ** 2) for i in range(n))
    k2 = M2 - M1 ** 2
    k3 = M3 - 3 * M1 * M2 + 2 * M1 ** 3
    k4 = M4 - 4 * M1 * M3 - 3 * M2 ** 2 + 12 * M1 ** 2 * M2 - 6 * M1 ** 4
    return k2, k3, k4


def build_observables():
    """Six closed-form observables: ATM level, skew, curvature, SSR, VIX-vov @30/90d."""
    k2z, k3z, _ = cumulants(z)              # z-dependent (for the SSR spot-derivative)
    k2, k3, k4 = (c.subs(z, 0) for c in cumulants(z))
    dk2 = sp.diff(k2z, z).subs(z, 0)
    u0 = sp.sqrt(k2)
    u1 = k3 / (6 * k2 ** sp.Rational(3, 2))
    u2 = k4 / (12 * k2 ** sp.Rational(5, 2))
    ssr = 3 * k2 * dk2 / k3                 # cumulant-route SSR

    kappa_s = -sp.log(1 - eps_s) / DT
    pb0 = slow_w(0)
    v_inf = sp.exp(gbar)

    def vov(tau):
        D = (1 - sp.exp(-kappa_s * sp.Float(tau))) / (kappa_s * sp.Float(tau))
        vix = [sp.sqrt(v_inf + (sp.exp(gbar + nu_s * ZS[b]) - v_inf) * D) for b in range(N_SLOW)]
        m = sum(pb0[b] * vix[b] for b in range(N_SLOW))
        return sp.sqrt(sum(pb0[b] * (vix[b] - m) ** 2 for b in range(N_SLOW)))

    return sp.Matrix([u0, u1, u2, ssr, vov(30 / 365.0), vov(90 / 365.0)])


def main():
    print("building symbolic observable map (closed form in theta)...")
    obs = build_observables()
    print("differentiating symbolically (sympy Jacobian, 6x6)...")
    J = obs.jacobian(sp.Matrix(PARAMS))

    base = [float(np.log(0.04)), 0.45, 0.35, -1.0, -4.5, 0.5]  # SPX baseline, SSR~1.5
    f = sp.lambdify(PARAMS, obs, "numpy")
    Jf = sp.lambdify(PARAMS, J, "numpy")
    f0 = np.asarray(f(*base), float).ravel()
    Jn = np.asarray(Jf(*base), float)

    print("\n=== baseline observables (cumulant-form map) ===")
    print(dict(zip(OBS_NAMES, np.round(f0, 4))))

    # complex-step cross-check: exact (machine-precision) Jacobian of the same closed-form map
    fc = sp.lambdify(PARAMS, obs, "numpy")
    Jcs = np.zeros((6, 6))
    for j in range(6):
        hp = [complex(v) for v in base]; hp[j] += 1e-30j
        Jcs[:, j] = np.asarray(fc(*hp), complex).ravel().imag / 1e-30
    print(f"\nsymbolic vs complex-step Jacobian: max abs diff = {np.max(np.abs(Jn - Jcs)):.2e}"
          "  (confirms the symbolic derivative is exact)")

    # elasticity scaling (same convention as identifiability_check) + SVD
    tsc = np.maximum(np.abs(base), 1e-2)
    osc = np.maximum(np.abs(f0), 1e-4)
    Js = Jn * tsc[None, :] / osc[:, None]
    U, S, Vt = np.linalg.svd(Js)
    print("\n=== local identifiability from the SYMBOLIC Jacobian ===")
    print("singular values :", np.round(S, 4))
    print("condition number:", round(float(S[0] / S[-1]), 1))
    print("rank            :", int(np.sum(S > 1e-8 * S[0])), "/ 6")
    print("softest direction:", dict(zip(PARAM_NAMES, np.round(Vt[-1], 2))))
    print("\nparameter -> most-sensitive observable (symbolic elasticities):")
    for j, pn in enumerate(PARAM_NAMES):
        i = int(np.argmax(np.abs(Js[:, j])))
        print(f"   {pn:9s} -> {OBS_NAMES[i]:9s} ({Js[i, j]:+.2f})")

    # leading-order analytic structure (closed form, quadrature-exact MGF E[e^{a zeta}]=e^{a^2/2})
    print("\n=== leading-order analytic structure (closed form) ===")
    k2_lo = sp.exp(gbar + (nu_s ** 2 + nu_f ** 2) / 2) * DT
    print("  kappa2 ~", sp.simplify(k2_lo), "  => dlog k2/d gbar=1, /d nu_s=nu_s, /d nu_f=nu_f")
    print("  d k2/dz ~ lam_mov * Cov_w(zeta_f, v)  => SSR proportional to lam_mov")
    print("  kappa3  ~ lam_skew & nu_f (both enter the fast-scale skew)  => the SOFT PAIR")
    print("  kappa4  ~ nu_s^2 + nu_f^2 (vov)        => curvature")
    print("  VIXvov(tau) ~ nu_s * D(kappa_s,tau)    => nu_s (amp) vs eps_s (decay) via tau")


if __name__ == "__main__":
    main()
