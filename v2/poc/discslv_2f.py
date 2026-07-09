"""
discslv_2f.py -- genuine TWO-FACTOR SV kernel (Gap B), CLOSED-FORM.

Single OU factor (discslv_gsv v2) breaks the local-vol floor and is closed-form, but one timescale
can't give a HIGH short end AND a SUSTAINED long end.  Here we carry TWO persistent OU vol regimes:

    variance       V(f,s) = exp(gbar + nu_f zeta_f[f] + nu_s zeta_s[s]) dt           # fast + slow factors
    within-step    return mixture over nodes l with skew tilt lam_skew (short-dated skew)
    fast factor    P(f'|l,f) = softmax(eta_f + kap_f zeta_f[f] zeta_f + lam_f zeta_l[l] zeta_f)
    slow factor    P(s'|l,s) = softmax(eta_s + kap_s zeta_s[s] zeta_s + lam_s zeta_l[l] zeta_s)

kap_f small (fast reversion) -> high, DECAYING short-end SSR; kap_s large (persistent) -> sustained
long-end floor.  Both leverage-coupled to the same return-node l -> spot-vol correlation.  Spot(z)-
independent -> EXACT closed-form: sigma(T;f,s) depends only on the regime pair, realized SSR is a
finite sum.  No Monte Carlo.

Run from poc/ :  python3 discslv_2f.py
"""
import numpy as np
import warnings
from numpy.polynomial.hermite_e import hermegauss
from discslv import softmax, merge_gmm, GMM, bs_implied_vol

warnings.filterwarnings("ignore")


class TwoFactorSV:
    def __init__(self, gbar, nu_f, nu_s, lam_skew, lam_f, lam_s, kap_f, kap_s, dt, nu_l=0.5, n_f=5, n_s=3, n_l=5):
        self.gbar, self.dt = gbar, dt
        self.nu_f, self.nu_s, self.nu_l, self.lam_skew = nu_f, nu_s, nu_l, lam_skew
        self.lam_f, self.lam_s, self.kap_f, self.kap_s = lam_f, lam_s, kap_f, kap_s
        self.n_f, self.n_s, self.n_l = n_f, n_s, n_l
        zf, wf = hermegauss(n_f); self.zf, self.ef = zf, np.log(wf / wf.sum())
        zs, ws = hermegauss(n_s); self.zs, self.es = zs, np.log(ws / ws.sum())
        zl, wl = hermegauss(n_l); self.zl, self.wl = zl, wl / wl.sum()
        # --- precompute: PER-l variances (within-step vol-of-vol nu_l -> instantaneous skew) + increments ---
        self.Vl = np.zeros((n_f, n_s, n_l)); self.D = np.zeros((n_f, n_s, n_l))
        for f in range(n_f):
            for s in range(n_s):
                Vl = np.exp(gbar + nu_f * zf[f] + nu_s * zs[s] + nu_l * zl) * dt   # variance varies across l
                sigl = np.sqrt(Vl)
                mtil = -0.5 * Vl + lam_skew * zl * sigl                            # mean tilt correlated with vol -> skew
                A = np.log(np.sum(self.wl * np.exp(mtil + 0.5 * Vl)))
                self.Vl[f, s] = Vl; self.D[f, s] = mtil - A
        self.Tf = np.array([[softmax(self.ef + kap_f * zf[f] * zf + lam_f * zl[l] * zf)
                             for f in range(n_f)] for l in range(n_l)])   # (n_l, n_f, n_f)
        self.Ts = np.array([[softmax(self.es + kap_s * zs[s] * zs + lam_s * zl[l] * zs)
                             for s in range(n_s)] for l in range(n_l)])   # (n_l, n_s, n_s)
        self.wTf = self.wl[:, None, None] * self.Tf                       # w_l * Tf, for the joint weight

    def incr(self, f, s):
        return self.D[f, s], self.Vl[f, s]                  # d_l, Vl_l (per-l variance)

    def trans_f(self, l, f):
        return self.Tf[l, f]

    def trans_s(self, l, s):
        return self.Ts[l, s]

    def propagate(self, W, MU, SG, F, S):
        """Vectorised one step. State = parallel arrays (W, MU, SG, F, S). Returns the expanded state
        (N * n_l * n_f * n_s components) as arrays. Spot-independent, so no z enters."""
        N = W.shape[0]; nl, nf, ns_ = self.n_l, self.n_f, self.n_s
        Di = self.D[F, S]                                  # (N, n_l)
        Vi = self.Vl[F, S]                                 # (N, n_l)
        new_mu = MU[:, None] + Di                          # (N, n_l)
        new_sg = np.sqrt(SG[:, None] ** 2 + Vi)            # (N, n_l)
        Tfi = np.transpose(self.Tf[:, F, :], (1, 0, 2))    # (N, n_l, n_f)
        Tsi = np.transpose(self.Ts[:, S, :], (1, 0, 2))    # (N, n_l, n_s)
        w_new = (W[:, None, None, None] * self.wl[None, :, None, None]
                 * Tfi[:, :, :, None] * Tsi[:, :, None, :])              # (N, n_l, n_f, n_s)
        shape = (N, nl, nf, ns_)
        mu_b = np.broadcast_to(new_mu[:, :, None, None], shape)
        sg_b = np.broadcast_to(new_sg[:, :, None, None], shape)
        fp_b = np.broadcast_to(np.arange(nf)[None, None, :, None], shape)
        sp_b = np.broadcast_to(np.arange(ns_)[None, None, None, :], shape)
        W2 = w_new.ravel()
        return (W2 / W2.sum(), mu_b.ravel(), sg_b.ravel(),
                fp_b.ravel().astype(np.intp), sp_b.ravel().astype(np.intp))


def _fast_merge(w, mu, sg, n_keep):
    """Moment-preserving recompression in O(n log n): sort by mean, split into n_keep equal-weight
    contiguous chunks, match each chunk's 0/1/2 moments to one Gaussian. No mass dropped."""
    order = np.argsort(mu)
    w, mu, s2 = w[order], mu[order], sg[order] ** 2
    cw = np.cumsum(w)
    edges = np.searchsorted(cw, np.linspace(0.0, cw[-1], n_keep + 1)[1:-1], side="right")
    W, M, S = [], [], []
    for ch in np.split(np.arange(len(w)), edges):
        if ch.size == 0:
            continue
        wc = w[ch]; pc = wc.sum()
        m = float((wc * mu[ch]).sum() / pc)
        v = float((wc * (s2[ch] + mu[ch] ** 2)).sum() / pc - m ** 2)
        W.append(pc); M.append(m); S.append(np.sqrt(max(v, 1e-12)))
    return np.array(W), np.array(M), np.array(S)


def recompress_2f(W, MU, SG, F, S, n_keep, n_f, n_s):
    """Array I/O. Moment-preserving recompression to n_keep components per (f,s) regime."""
    Wo, MUo, SGo, Fo, So = [], [], [], [], []
    for f in range(n_f):
        for s in range(n_s):
            m = (F == f) & (S == s)
            cnt = int(m.sum())
            if cnt == 0:
                continue
            w, mu, sg = W[m], MU[m], SG[m]
            if cnt > n_keep:
                w, mu, sg = _fast_merge(w, mu, sg, n_keep)
            Wo.append(w); MUo.append(mu); SGo.append(sg)
            Fo.append(np.full(w.shape[0], f)); So.append(np.full(w.shape[0], s))
    W = np.concatenate(Wo); MU = np.concatenate(MUo); SG = np.concatenate(SGo)
    F = np.concatenate(Fo).astype(np.intp); S = np.concatenate(So).astype(np.intp)
    A = np.log(np.sum(W * np.exp(MU + 0.5 * SG ** 2)))                    # global re-lock
    return W, MU - A, SG, F, S


def smile_2f(K, n, f0, s0, nk=8):
    W = np.array([1.0]); MU = np.array([0.0]); SG = np.array([1e-4])
    F = np.array([f0], dtype=np.intp); S = np.array([s0], dtype=np.intp)
    for _ in range(n):
        W, MU, SG, F, S = K.propagate(W, MU, SG, F, S)
        W, MU, SG, F, S = recompress_2f(W, MU, SG, F, S, nk, K.n_f, K.n_s)
    return GMM(W, MU, SG, F=1.0)


def _atm_skew(g, T, dm=6e-3):
    F = g.forward(); iv = lambda k: float(g.implied_vol(F * k, T)[0])
    return iv(1.0), (iv(np.exp(dm)) - iv(np.exp(-dm))) / (2 * dm)


def stationary_pi(K, iters=400):
    nfs = K.n_f * K.n_s
    P = np.zeros((nfs, nfs))
    for f in range(K.n_f):
        for s in range(K.n_s):
            i = f * K.n_s + s
            for l in range(K.n_l):
                Pf = K.trans_f(l, f); Ps = K.trans_s(l, s); wl = K.wl[l]
                for fp in range(K.n_f):
                    for sp in range(K.n_s):
                        P[i, fp * K.n_s + sp] += wl * Pf[fp] * Ps[sp]
    pi = np.ones(nfs) / nfs
    for _ in range(iters):
        pi = pi @ P
    return (pi / pi.sum()).reshape(K.n_f, K.n_s)


def ssr_2f(K, n, nk=5, dm=6e-3):
    T = n * K.dt
    sig = np.zeros((K.n_f, K.n_s)); skew = np.zeros((K.n_f, K.n_s))
    for f in range(K.n_f):
        for s in range(K.n_s):
            sig[f, s], skew[f, s] = _atm_skew(smile_2f(K, n, f, s, nk), T, dm)
    pi = stationary_pi(K)
    cov = 0.0; var = 0.0; sk = 0.0
    for f in range(K.n_f):
        for s in range(K.n_s):
            d, V = K.incr(f, s)
            mean_r = float(np.sum(K.wl * d))
            var_fs = float(np.sum(K.wl * (d ** 2 + V)) - mean_r ** 2)
            c = 0.0
            for l in range(K.n_l):
                Pf = K.trans_f(l, f); Ps = K.trans_s(l, s)
                resp = 0.0
                for fp in range(K.n_f):
                    for sp in range(K.n_s):
                        resp += Pf[fp] * Ps[sp] * (sig[fp, sp] - sig[f, s])
                c += K.wl[l] * d[l] * resp
            cov += pi[f, s] * c; var += pi[f, s] * var_fs; sk += pi[f, s] * skew[f, s]
    atm = float(np.sum(pi * sig))                          # stationary-averaged ATM vol
    return (cov / var) / sk, atm, sk                       # (SSR, ATM vol, ATM skew)


def main():
    base = dict(gbar=np.log(0.04), nu_f=0.55, nu_s=0.55, lam_skew=-0.8, dt=1.0 / 52.0, n_f=5, n_s=3, n_l=5)
    mats = [(1, "1w"), (4, "1m"), (13, "3m"), (26, "6m"), (52, "1y")]
    print("Two-factor genuine SV: CLOSED-FORM SSR term structure (no MC).\n")
    print("fast factor: kap_f small (fast revert); slow factor: kap_s large (persistent)\n")
    print(f"{'kap_f':>6}{'kap_s':>6}{'lam_f':>6}{'lam_s':>6} |  " + "  ".join(f"{l:>4}" for _, l in mats))
    for kf, ks, lf, ls in [(0.2, 2.0, 3.0, 3.0), (0.2, 2.0, 4.0, 2.0), (0.4, 3.0, 4.0, 3.0), (0.1, 1.5, 4.0, 4.0)]:
        K = TwoFactorSV(lam_f=lf, lam_s=ls, kap_f=kf, kap_s=ks, **base)
        row = [ssr_2f(K, n, nk=16)[0] for n, _ in mats]
        print(f"{kf:>6.1f}{ks:>6.1f}{lf:>6.1f}{ls:>6.1f} |  " + "  ".join(f"{v:4.2f}" for v in row), flush=True)
    print("\ntarget: ~1.5 short decaying to ~1.0-1.5 long (flat/contango)")


if __name__ == "__main__":
    main()
