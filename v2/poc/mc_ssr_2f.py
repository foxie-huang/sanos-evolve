"""
mc_ssr_2f.py -- Monte-Carlo cross-check of the CURRENT two-factor closed-form SSR (ssr_2f).

The realized SSR is the one-step regression beta = Cov(d sigma_hat, r) / Var(r), then /skew.
Because the kernel is z-invariant, sigma_hat depends only on the regime, so we precompute
sigma_hat[f,s] once (= what ssr_2f uses) and simulate one-step transitions:
   draw (f,s) ~ pi ;  l ~ w_l ;  r = D[f,s,l] + sqrt(V[f,s,l]) * xi ;  f' ~ Tf[l,f] ; s' ~ Ts[l,s].
If the closed-form finite sum is correct, MC SSR == closed-form SSR up to MC noise.

Run from poc/ :  python3 mc_ssr_2f.py
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from discslv_2f import TwoFactorSV, smile_2f, stationary_pi, ssr_2f, _atm_skew

DT = 1.0 / 52.0


def kernel():
    return TwoFactorSV(gbar=-5.30, nu_f=0.43, nu_s=0.50, lam_skew=-1.48, lam_f=0.98, lam_s=1.65,
                       kap_f=1.00, kap_s=2.34, dt=DT, nu_l=0.14, n_f=5, n_s=3, n_l=5)


def precompute_sig(K, n, nk):
    nf, ns = K.n_f, K.n_s
    sig = np.zeros((nf, ns)); skw = np.zeros((nf, ns))
    for f in range(nf):
        for s in range(ns):
            sig[f, s], skw[f, s] = _atm_skew(smile_2f(K, n, f, s, nk), n * DT)
    return sig, skw


def mc_ssr(K, sig, skw, N, seed):
    rng = np.random.default_rng(seed)
    nf, ns, nl = K.n_f, K.n_s, K.n_l
    pi = stationary_pi(K)
    idx = rng.choice(nf * ns, N, p=pi.ravel()); f0 = idx // ns; s0 = idx % ns
    l = rng.choice(nl, N, p=K.wl)
    r = K.D[f0, s0, l] + np.sqrt(K.Vl[f0, s0, l]) * rng.standard_normal(N)
    fp = (np.cumsum(K.Tf[l, f0, :], axis=1) < rng.random(N)[:, None]).sum(1)   # f' ~ Tf[l,f0]
    sp = (np.cumsum(K.Ts[l, s0, :], axis=1) < rng.random(N)[:, None]).sum(1)   # s' ~ Ts[l,s0]
    dsig = sig[fp, sp] - sig[f0, s0]
    beta = np.cov(dsig, r)[0, 1] / np.var(r)
    skew = float(np.sum(pi * skw))
    return beta / skew, beta, np.var(r)


def main():
    K = kernel(); nk = 20; N = 600_000
    print(f"MC cross-check of the closed-form SSR (current 2f kernel), N={N:,} per maturity:\n", flush=True)
    print(f"{'T':>4}{'closed-form':>13}{'MC (2 seeds)':>20}{'|diff|':>9}", flush=True)
    for n, lab in [(4, "1m"), (13, "3m"), (26, "6m"), (52, "1y")]:
        sig, skw = precompute_sig(K, n, nk)
        cf = ssr_2f(K, n, nk=nk)[0]
        m0 = mc_ssr(K, sig, skw, N, 0)[0]; m1 = mc_ssr(K, sig, skw, N, 1)[0]
        print(f"{lab:>4}{cf:>13.3f}{m0:>11.3f},{m1:>7.3f}{max(abs(cf-m0),abs(cf-m1)):>9.3f}", flush=True)
    print("\nIf |diff| ~ MC noise (a few e-3), the z-invariant finite-sum SSR is confirmed.", flush=True)


if __name__ == "__main__":
    main()
