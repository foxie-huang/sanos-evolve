#!/usr/bin/env python3
"""
Is the SANOS-Evolve model sticky-moneyness? A direct conditional-smile-dynamics test.

STICKY-MONEYNESS := after a spot move, the smile in log-moneyness k=ln(K/S) is UNCHANGED --
level, skew, AND curvature frozen at each fixed k. To first order in the one-step return r the
model's conditional smile is
      sigma(k | r) = sigma_0(k) + r * [ beta + (d_skew) k + (d_curv) k^2 + ... ]
so sticky-moneyness  <=>  beta = d_skew = d_curv = 0, where
      beta   = d sigma_ATM / d lnS = SSR * skew   (level response)
      d_skew = d(skew)     / d lnS                (skew response)
      d_curv = d(curv)     / d lnS                (curvature response)
Same coefficients under the reference dynamics:
      sticky-moneyness :  beta = 0,       d_skew = 0,     ...        (SSR = 0)
      sticky-strike    :  beta = skew,    d_skew = 2*curv, ...       (SSR = 1)  smile translates in K
      local vol        :  beta = 2*skew,  ...                        (SSR = 2)  ATM moves at 2x skew

We compute the model's (beta, d_skew, d_curv) with the EXACT-beta transition sum (the de-sampled MC
of slv_fast.fused_ssr_exact_ts), extended from the ATM level to the ATM skew & curvature of the
conditional smile -- no sampling, no roll linearisation. Term-structure leverage + theta_ts (the
faithful ~4%-fit 2015 model). Reports each response as its position on the sticky map:
      level:  SSR      = beta / skew        (0 sticky-money | 1 sticky-K | 2 local-vol)
      skew : skew-SR   = d_skew / (2*curv)  (0 sticky-money | 1 sticky-K)
Records wall-time.
"""
import sys, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from discslv_2f import TwoFactorSV                                          # noqa: E402
import discslv_slv                                                         # noqa: E402
from discslv_slv import (Epi_V, nu_bar, raw_increment, stationary_pi,      # noqa: E402
                         lev_increment, marginal, iv_at)
from slv_fast import propagate_vec                                         # noqa: E402
discslv_slv.propagate = propagate_vec
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at         # noqa: E402
from numpy.polynomial.hermite_e import hermegauss                          # noqa: E402

DT = 1.0 / 52.0
NS = [1, 4, 8, 13]; LABELS = ["1wk", "1m", "2m", "3m"]
THETA = dict(nu_f=0.696, nu_s=0.290, nu_l=0.999, lam_skew=-0.462,          # theta_ts (2015 flat ~4% fit)
             lam_f=0.439, lam_s=2.465, kap_f=0.903, kap_s=2.780)
DATE = os.path.join(os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod")), "SPX-NDX-RUT-VIX_2015-06-01.json.gz")


def atm_moments_of(mu, T, half=0.015, npts=9):
    """(level, skew, curvature) of a GM marginal's smile in the ATM neighbourhood, via a deg-2 fit of
    the implied vol over log-moneyness k in [-half, half].  p[2]=level, p[1]=skew, 2*p[0]=curvature."""
    ks = np.linspace(-half, half, npts)
    iv = np.asarray(iv_at(mu, T, list(ks)))
    p = np.polyfit(ks, iv, 2)
    return float(p[2]), float(p[1]), float(2 * p[0])


def smile_dynamics(K, lam_fns, n, EV, nub, Vlr, tiltr, nk, dt, nz=13, zmax=0.12, Q=5, half=0.015, npts=9):
    """EXACT conditional-smile dynamics: beta_level/skew/curv = Cov(d moment, r)/Var(r) over the one-step
    transition, extending fused_ssr_exact_ts from the ATM level to the whole ATM neighbourhood."""
    nf, ns, nl = K.n_f, K.n_s, K.n_l
    lam2 = [(lambda g: (lambda mc: np.asarray(g(mc), float) ** 2))(fn) for fn in lam_fns]
    zg = np.linspace(-zmax, zmax, nz); iz0 = nz // 2
    L = np.zeros((nf, ns, nz)); S = np.zeros((nf, ns, nz)); C = np.zeros((nf, ns, nz))
    base = np.zeros((nf, ns, 3))
    for f in range(nf):
        for s in range(ns):
            for iz, z in enumerate(zg):
                st = (np.array([1.0]), np.array([float(z)]), np.array([1e-4]),
                      np.array([f], np.intp), np.array([s], np.intp))
                for k in range(n):
                    st, _ = propagate_vec(K, st, lam2[k], EV, nub, Vlr, tiltr, nk)
                lv, sk, cv = atm_moments_of(marginal(st), n * dt, half, npts)
                L[f, s, iz] = lv; S[f, s, iz] = sk; C[f, s, iz] = cv
                if iz == iz0:
                    base[f, s] = [lv, sk, cv]
    pi = stationary_pi(K); lam0 = float(np.asarray(lam_fns[0](0.0)))
    zq, wq = hermegauss(Q); wq = wq / wq.sum()
    var = 0.0; covL = covS = covC = 0.0
    for f in range(nf):
        for s in range(ns):
            Dl, Vl = lev_increment(Vlr[f, s], tiltr[f, s], K.wl, lam0)
            mean_r = float(np.sum(K.wl * Dl))
            var += pi[f, s] * (float(np.sum(K.wl * (Dl ** 2 + Vl))) - mean_r ** 2)
            l0, s0, c0 = base[f, s]; cL = cS = cC = 0.0
            for l in range(nl):
                Pf, Ps = K.trans_f(l, f), K.trans_s(l, s); sd = np.sqrt(max(Vl[l], 1e-16))
                dest = np.array([[Pf[fp] * Ps[sp] for sp in range(ns)] for fp in range(nf)])
                for q in range(Q):
                    r = Dl[l] + sd * zq[q]; w = K.wl[l] * wq[q] * (r - mean_r)
                    Ld = sum(dest[fp, sp] * np.interp(r, zg, L[fp, sp]) for fp in range(nf) for sp in range(ns))
                    Sd = sum(dest[fp, sp] * np.interp(r, zg, S[fp, sp]) for fp in range(nf) for sp in range(ns))
                    Cd = sum(dest[fp, sp] * np.interp(r, zg, C[fp, sp]) for fp in range(nf) for sp in range(ns))
                    cL += w * (Ld - l0); cS += w * (Sd - s0); cC += w * (Cd - c0)
            covL += pi[f, s] * cL; covS += pi[f, s] * cS; covC += pi[f, s] * cC
    skew0 = float(np.sum(pi * base[:, :, 1])); curv0 = float(np.sum(pi * base[:, :, 2]))
    bl, bs, bc = covL / var, covS / var, covC / var
    return dict(beta=bl, d_skew=bs, d_curv=bc, skew0=skew0, curv0=curv0,
                SSR=bl / skew0, skewSR=(bs / (2 * curv0) if abs(curv0) > 1e-9 else float("nan")))


if __name__ == "__main__":
    t0 = time.time()
    half = float(sys.argv[1]) if len(sys.argv) > 1 else 0.015                # ATM-fit half-window (log-moneyness)
    npts = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    chain = sanos_chain(DATE); sig_ref = ref_vol(chain)
    K = TwoFactorSV(gbar=solve_gbar(THETA, sig_ref, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **THETA)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    lam_cache = {k: leverage_at(chain, k * DT, EV, dt=DT) for k in range(1, max(NS) + 1)}

    print(f"Is the model sticky-moneyness?  Conditional-smile response to a spot move (SPX 2015, theta_ts)")
    print(f"ATM window +-{half:.3f} log-moneyness, {npts} pts, deg-2\n")
    print("reference:  sticky-moneyness SSR=0, skew-SR=0  |  sticky-strike SSR=1, skew-SR=1  |  local-vol SSR=2\n")
    print(f"{'mat':>4} | {'SSR':>6}{'(level)':>8} | {'base skew':>10}{'d_skew':>9}{'skew-SR':>9} | {'base curv':>10}{'d_curv':>9}")
    res = {}
    for n, lab in zip(NS, LABELS):
        lam_fns = [lam_cache[k + 1] for k in range(n)]
        d = smile_dynamics(K, lam_fns, n, EV, nub, Vlr, tiltr, 16, DT, nz=13, half=half, npts=npts)
        res[lab] = d
        print(f"{lab:>4} | {d['SSR']:>6.2f}{'':>8} | {d['skew0']:>10.3f}{d['d_skew']:>9.2f}{d['skewSR']:>9.2f} | "
              f"{d['curv0']:>10.2f}{d['d_curv']:>9.1f}")
    print(f"\nlevel (SSR): distance from sticky-moneyness = |SSR-0|;  shape (skew-SR): 0=frozen-in-moneyness, 1=translates-in-K")

    import json
    json.dump({k: {m: v[m] for m in ("SSR", "beta", "d_skew", "d_curv", "skew0", "curv0", "skewSR")}
               for k, v in res.items()}, open(os.path.join(HERE, "sticky_check_results.json"), "w"), indent=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt   # noqa: E402
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    C0, C1, C2 = "#1f6f8b", "#b02a37", "#9a6b00"
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    wks = [1, 4, 8, 13]
    ax[0].plot(wks, [res[l]["SSR"] for l in LABELS], "o-", c=C0, zorder=3, label="model")
    for y, lab, c in [(0, "sticky-moneyness", C2), (1, "sticky-strike", "gray"), (2, "local vol", C1)]:
        ax[0].axhline(y, ls="--", c=c, lw=1); ax[0].text(13.2, y, lab, va="center", fontsize=7.5, color=c)
    ax[0].set(title="(a) Level dynamics: SSR term structure", xlabel="maturity (weeks)", ylabel="SSR",
              xlim=(0, 18), ylim=(-0.2, 2.3)); ax[0].legend(frameon=False, loc="center right")
    d = res["1m"]; k = np.linspace(-0.05, 0.05, 101); r = -0.01
    ax[1].plot(k * 100, r * (d["beta"] + d["d_skew"] * k + 0.5 * d["d_curv"] * k ** 2) * 100, c=C0, lw=1.8, label="model")
    ax[1].axhline(0, ls="--", c=C2, lw=1.2, label="sticky-moneyness")
    ax[1].plot(k * 100, r * (d["skew0"] + d["curv0"] * k) * 100, ls="--", c="gray", lw=1.2, label="sticky-strike")
    ax[1].set(title="(b) Shape dynamics: 1m smile shift on $-1\\%$ spot", xlabel="log-moneyness $k$ (%)",
              ylabel=r"$\Delta\sigma(k)$ (vol pts)"); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "sticky_check.png"), dpi=150, bbox_inches="tight")
    print(f"wrote sticky_check.png, sticky_check_results.json")
    print(f"wall {time.time()-t0:.0f}s")
