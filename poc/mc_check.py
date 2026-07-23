"""
mc_check.py -- Monte-Carlo cross-check of the closed-form readouts (Item 2).

The closed-form readout (horizon_smile / ssr_at) propagates the kernel with two approximations:
(i) COLLOCATION -- state-dependent terms (leverage weights, slow transition) are evaluated at each
component's MEAN, not the true log-price; (ii) RECOMPRESSION -- the mixture is merged back to NK
components each step.  Here we simulate the EXACT kernel dynamics by Monte Carlo (each path carries
its own log-price; no collocation, no recompression) and compare ATM vol, skew, and SSR.  If the
closed-form sits within the MC noise band, the whole closed-form layer is validated.

Kernel law per step from state (z, regime b):
  fast weights      w_l(z) = softmax(eta_f + lam_mov * zeta_f * z)
  node variances    sig2_l = exp(gbar + nu_s zeta_s[b] + nu_f zeta_f + delta z) * dt
  leverage tilt     mtil_l = -1/2 sig2_l + lam_skew zeta_f sig_l ;  lock A = log sum_l w_l e^{mtil_l+sig2_l/2}
  increment         z' = z + (mtil_l - A) + sig_l * N(0,1),   l ~ w_l(z)
  regime            b' ~ (1-eps_s) e_b + eps_s softmax(eta_s + lam_mov zeta_s z)
Initial regime ~ softmax(eta_s + lam_mov zeta_s z0) (matches horizon_smile).  SSR uses common random
numbers (same seed for the z0 +/- h sims) so the ATM-vol difference is low-variance.

Run from poc/ :  python3 mc_check.py
"""
import numpy as np
import warnings
from discslv import bs_implied_vol, TwoTimescaleKernel
from ssr_demo import BASE, LAM_MOV_0, ssr_at, horizon_smile

warnings.filterwarnings("ignore")
NPATHS = 500_000
NBATCH = 10
SEED = 20260626


def _softmax_rows(X):
    X = X - X.max(axis=1, keepdims=True)
    E = np.exp(X)
    return E / E.sum(axis=1, keepdims=True)


def _choice_rows(P, u):
    return (u[:, None] > np.cumsum(P, axis=1)).sum(axis=1).clip(0, P.shape[1] - 1)


def simulate(K, n, z0, seed, npaths=NPATHS, init_b=None):
    """Exact MC of the kernel for n steps from spot z0; returns log-price samples.
    init_b=None -> initial regime ~ slow_weights(z0) (comparative-statics, regime tied to spot).
    init_b=int  -> initial regime FIXED at that value (realized SSR: regime held, then spot bumped)."""
    rng = np.random.default_rng(seed)
    zf, ef, zs, es = K.zeta_f, K.eta_f, K.zeta_s, K.eta_s
    idx = np.arange(npaths)
    z = np.full(npaths, float(z0))
    if init_b is None:
        b = _choice_rows(_softmax_rows(es + K.lam_mov * zs * z[:, None]), rng.random(npaths))
    else:
        b = np.full(npaths, int(init_b))
    for _ in range(n):
        wl = _softmax_rows(ef + K.lam_mov * zf * z[:, None])
        g = K.gbar + K.nu_s * zs[b][:, None] + K.nu_f * zf + K.delta * z[:, None]
        sig2 = np.exp(g) * K.dt; sig = np.sqrt(sig2)
        mtil = -0.5 * sig2 + K.lam_skew * zf * sig
        A = np.log((wl * np.exp(mtil + 0.5 * sig2)).sum(1))
        d = mtil - A[:, None]
        l = _choice_rows(wl, rng.random(npaths))
        sw = _softmax_rows(es + K.lam_mov * zs * z[:, None])            # collocation at OLD z
        Pbp = K.eps_s * sw; Pbp[idx, b] += (1.0 - K.eps_s)
        bnew = _choice_rows(Pbp, rng.random(npaths))
        z = z + d[idx, l] + sig[idx, l] * rng.standard_normal(npaths)
        b = bnew
    return z


def _iv(S, F, k, T):
    return bs_implied_vol(float(np.maximum(S - k, 0.0).mean()), F, k, T)


def mc_stats(K, n, seed=SEED, npaths=NPATHS, nbatch=NBATCH, h=6e-3, dm=6e-3):
    """Batched MC (mean, stderr) for ATM vol, skew, SSR; CRN across the +/-h sims."""
    T = n * K.dt
    z0 = simulate(K, n, 0.0, seed, npaths)
    zp = simulate(K, n, +h, seed, npaths)
    zm = simulate(K, n, -h, seed, npaths)
    S0, Sp, Sm = np.exp(z0), np.exp(zp), np.exp(zm)
    bs = npaths // nbatch
    atm, skew, ssr = [], [], []
    for i in range(nbatch):
        s = slice(i * bs, (i + 1) * bs)
        F0 = S0[s].mean()
        a = _iv(S0[s], F0, F0, T)
        sk = (_iv(S0[s], F0, F0 * np.exp(dm), T) - _iv(S0[s], F0, F0 * np.exp(-dm), T)) / (2 * dm)
        Fp, Fm = Sp[s].mean(), Sm[s].mean()
        num = (_iv(Sp[s], Fp, Fp, T) - _iv(Sm[s], Fm, Fm, T)) / (2 * h)
        atm.append(a); skew.append(sk); ssr.append(num / sk)
    f = lambda x: (float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))))
    return f(atm), f(skew), f(ssr)


def cf_stats(K, n, dm=6e-3):
    """Closed-form ATM vol, skew, SSR (collocation + recompression)."""
    T = n * K.dt
    g0 = horizon_smile(K, n, 0.0)
    iv = lambda g, k: float(g.implied_vol(g.forward() * k, T)[0])
    atm = iv(g0, 1.0); skew = (iv(g0, np.exp(dm)) - iv(g0, np.exp(-dm))) / (2 * dm)
    return atm, skew, ssr_at(K, n)              # ssr_at = verified closed-form SSR


def main():
    K = TwoTimescaleKernel(lam_mov=LAM_MOV_0, **BASE)
    mats = [(4, "1m"), (13, "3m"), (26, "6m"), (52, "1y")]
    print(f"Monte-Carlo cross-check of the closed-form  (paths={NPATHS:,}, seed={SEED}, dt=1/52)\n")
    print(f"{'mat':>4} {'quantity':>9} | {'closed-form':>11} | {'MC':>9} +/- {'se':<7} | {'diff':>7} {'(sigma)':>8}")
    print("-" * 72)
    for n, lab in mats:
        cf = cf_stats(K, n)
        (a, ae), (s, se_), (r, re_) = mc_stats(K, n)
        for name, c, (m, e) in [("ATM vol", cf[0], (a, ae)), ("skew", cf[1], (s, se_)), ("SSR", cf[2], (r, re_))]:
            scale = 1e4 if name != "SSR" else 1.0
            diff = (c - m) * scale; sig = (c - m) / e if e > 0 else float("nan")
            unit = "bp" if name != "SSR" else "  "
            print(f"{lab:>4} {name:>9} | {c:>11.4f} | {m:>9.4f} +/- {e:<7.4f} | {diff:>5.1f}{unit} {sig:>+7.1f}")
        print()


if __name__ == "__main__":
    main()
