"""
discslv_gsv.py -- genuine-SV kernel, v2: a discrete OU vol factor (Gap A done right), CLOSED-FORM.

v1 (regime-JUMP leverage, prob eps_s) broke the local-vol floor and was closed-form, but the leverage
was eps_s-diluted -> SSR capped ~0.5 (too low).  v2 makes the regime a genuine mean-reverting (OU)
factor whose leverage acts EVERY step, via a softmax transition:

    P(b' | l, b) = softmax( eta_s  +  kappa_rev * zeta_s[b] * zeta_s[b']  +  lam_lev * zeta_f[l] * zeta_s[b'] )
                            \____/    \_________________________/          \________________________/
                          revert-to-center    persistence (AR1)            leverage (per-step, return-coupled)

  - eta_s  : pulls toward the center (the GH stationary) = mean reversion
  - kappa_rev : pulls toward the current regime b        = persistence / AR(1) timescale
  - lam_lev : pulls toward high vol when the return-node zeta_f[l] is high (down move) = leverage, EVERY step

Still spot(z)-independent -> EXACT closed-form propagation (no collocation, no MC); the realized SSR is
the same finite sum as v1.  Structural knobs (6): (gbar, nu_s, nu_f, lam_skew, kappa_rev, lam_lev).

Run from poc/ :  python3 discslv_gsv.py
"""
import numpy as np
import warnings
from discslv import TwoTimescaleKernel, softmax, recompress_joint, marginal_from_joint

warnings.filterwarnings("ignore")


class GenuineSVKernel(TwoTimescaleKernel):
    def __init__(self, gbar, nu_s, nu_f, lam_skew, lam_lev, kappa_rev, dt, n_slow=3, n_fast=5):
        super().__init__(gbar, nu_s, nu_f, lam_skew, 0.0, 0.0, dt, n_slow, n_fast)  # lam_mov, eps_s unused
        self.lam_lev = lam_lev
        self.kappa_rev = kappa_rev

    def incr(self, b):
        """Spot-independent increment mixture for regime b: (w_l, d_l, sigma2_l)."""
        g = self.gbar + self.nu_s * self.zeta_s[b] + self.nu_f * self.zeta_f
        sig2 = np.exp(g) * self.dt; sig = np.sqrt(sig2)
        wl = softmax(self.eta_f)
        mtil = -0.5 * sig2 + self.lam_skew * self.zeta_f * sig
        A = np.log(np.sum(wl * np.exp(mtil + 0.5 * sig2)))
        return wl, mtil - A, sig2

    def trans(self, l, b):
        """OU regime transition: revert-to-center + persistence(b) + leverage(return-node l). Per step."""
        return softmax(self.eta_s + self.kappa_rev * self.zeta_s[b] * self.zeta_s
                       + self.lam_lev * self.zeta_f[l] * self.zeta_s)

    def propagate(self, comps):
        out = []
        for (w, mu, s, b) in comps:
            wl, d, sig2 = self.incr(b)
            for l in range(self.n_fast):
                Pbp = self.trans(l, b)
                nm = mu + d[l]; ns = float(np.sqrt(s * s + sig2[l]))
                for bp in range(self.n_slow):
                    ww = w * wl[l] * Pbp[bp]
                    if ww > 1e-14:
                        out.append((ww, nm, ns, bp))
        tot = sum(c[0] for c in out)
        return [(c[0] / tot, c[1], c[2], c[3]) for c in out]


def smile_regime(K, n, b, nk=8):
    comps = [(1.0, 0.0, 1e-4, b)]
    for _ in range(n):
        comps = recompress_joint(K.propagate(comps), nk, K.n_slow)
    return marginal_from_joint(comps)


def _atm_skew(g, T, dm=6e-3):
    F = g.forward(); iv = lambda k: float(g.implied_vol(F * k, T)[0])
    return iv(1.0), (iv(np.exp(dm)) - iv(np.exp(-dm))) / (2 * dm)


def stationary_pi(K, iters=300):
    wl, _, _ = K.incr(0)
    Pbar = np.array([sum(wl[l] * K.trans(l, b) for l in range(K.n_fast)) for b in range(K.n_slow)])
    pi = np.ones(K.n_slow) / K.n_slow
    for _ in range(iters):
        pi = pi @ Pbar
    return pi / pi.sum()


def ssr_closed_form(K, n, nk=8, dm=6e-3):
    """Closed-form realized SSR(T): averaged over stationary regime, and per current regime b0."""
    T = n * K.dt
    sm = [_atm_skew(smile_regime(K, n, b, nk), T, dm) for b in range(K.n_slow)]
    sig_atm = np.array([x[0] for x in sm]); skew = np.array([x[1] for x in sm])
    pi = stationary_pi(K)
    cov_b = np.zeros(K.n_slow); var_b = np.zeros(K.n_slow)
    for b in range(K.n_slow):
        wl, d, sig2 = K.incr(b)
        mean_r = float(np.sum(wl * d))
        var_b[b] = float(np.sum(wl * (d ** 2 + sig2)) - mean_r ** 2)
        c = 0.0
        for l in range(K.n_fast):
            c += wl[l] * d[l] * float(np.sum(K.trans(l, b) * (sig_atm - sig_atm[b])))
        cov_b[b] = c
    ssr_b = (cov_b / var_b) / skew
    cov = float(np.sum(pi * cov_b)); var = float(np.sum(pi * var_b)); sk = float(np.sum(pi * skew))
    return (cov / var) / sk, ssr_b, sig_atm, skew


def main():
    base = dict(gbar=np.log(0.04), nu_s=0.5, nu_f=0.45, lam_skew=-1.2, dt=1.0 / 52.0)
    mats = [(1, "1w"), (4, "1m"), (13, "3m"), (26, "6m"), (52, "1y")]
    print("Genuine-SV v2 (OU regime, per-step leverage): CLOSED-FORM SSR.  Target band ~0.9-1.6\n")
    print(f"{'kappa':>6}{'lev':>5} |  " + "  ".join(f"{l:>4}" for _, l in mats))
    for kap in [0.3, 0.6, 1.0]:
        for lev in [1.0, 2.0, 4.0]:
            K = GenuineSVKernel(lam_lev=lev, kappa_rev=kap, **base)
            row = [ssr_closed_form(K, n)[0] for n, _ in mats]
            print(f"{kap:>6.2f}{lev:>5.1f} |  " + "  ".join(f"{v:4.2f}" for v in row), flush=True)


if __name__ == "__main__":
    main()
