"""
prop_gh.py -- collocation fix (Item 3c) for the closed-form propagation.

The MC cross-check showed the plain propagate() under-states the multi-step skew/SSR: it evaluates the
state-dependent leverage at each blob's MEAN (collocation), missing the within-blob variation of
w_l(z), the slow transition, and the lock.  Here we integrate those over the blob N(mu, s^2) with a
Q-node Gauss-Hermite rule: replace the blob by atoms at  z_q = mu + s*zeta_q  (weights w_q), apply the
exact increment at each atom, and let the atom spacing carry the blob's variance.  Q -> infinity is exact.
"""
import numpy as np
import warnings
from numpy.polynomial.hermite_e import hermegauss
from discslv import recompress_joint, marginal_from_joint

warnings.filterwarnings("ignore")


def propagate_gh(K, comps, Q=3):
    """One kernel step, integrating the leverage over each blob with a Q-node GH rule."""
    zq, wq = hermegauss(Q); wq = wq / wq.sum()
    out = []
    for (w, mu, s, b) in comps:
        for q in range(Q):
            z0 = mu + s * zq[q]
            wl, d, sig2 = K._increment(z0, b)
            base = np.zeros(K.n_slow); base[b] = 1.0
            Pbp = (1.0 - K.eps_s) * base + K.eps_s * K.slow_weights(z0)
            wqb = w * wq[q]
            for l in range(K.n_fast):
                nm = z0 + d[l]; ns = float(np.sqrt(sig2[l]))
                for bp in range(K.n_slow):
                    ww = wqb * wl[l] * Pbp[bp]
                    if ww > 1e-14:
                        out.append((ww, nm, ns, bp))
    tot = sum(c[0] for c in out)
    return [(c[0] / tot, c[1], c[2], c[3]) for c in out]


def horizon_smile_gh(K, n, z0=0.0, nk=10, Q=3):
    pb = K.slow_weights(z0)
    comps = [(pb[b], z0, 1e-4, b) for b in range(K.n_slow)]
    for _ in range(n):
        comps = recompress_joint(propagate_gh(K, comps, Q), nk, K.n_slow)
    return marginal_from_joint(comps)


def ssr_at_gh(K, n, nk=10, Q=3, h=6e-3, dm=6e-3):
    T = n * K.dt
    iv = lambda g, k: float(g.implied_vol(g.forward() * k, T)[0])
    g0 = horizon_smile_gh(K, n, 0.0, nk, Q)
    gu = horizon_smile_gh(K, n, +h, nk, Q); gd = horizon_smile_gh(K, n, -h, nk, Q)
    skew = (iv(g0, np.exp(dm)) - iv(g0, np.exp(-dm))) / (2 * dm)
    return (iv(gu, 1.0) - iv(gd, 1.0)) / (2 * h) / skew


def _stats(K, g, T, dm=6e-3):
    iv = lambda k: float(g.implied_vol(g.forward() * k, T)[0])
    return iv(1.0), (iv(np.exp(dm)) - iv(np.exp(-dm))) / (2 * dm)


if __name__ == "__main__":
    from discslv import TwoTimescaleKernel
    from ssr_demo import BASE, LAM_MOV_0
    K = TwoTimescaleKernel(lam_mov=LAM_MOV_0, **BASE); n = 13; T = n * K.dt
    print("MC truth (n=13):     atm 0.3434   skew -0.7716   ssr 2.80")
    print("Q  nk |  atm      skew     ssr      (GH-corrected collocation)")
    for Q in [1, 3, 5, 7]:
        for nk in [10, 20]:
            a, s = _stats(K, horizon_smile_gh(K, n, 0.0, nk, Q), T)
            r = ssr_at_gh(K, n, nk, Q)
            print(f"{Q}  {nk:>2} |  {a:.4f}  {s:+.4f}  {r:.3f}")
