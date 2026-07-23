"""
band_fig_single5.py -- regenerate figs/synthetic_main.png as the SINGLE decoupling-band panel,
5-observable hold (vol@1m,6m,1y + skew@1m + curvature@1m) -- the honest 'fixed statics' band that
produced the paper's [0.93,1.02]/0.08 (residual_band.py method, NK=20). Persists
band_results_5obs.json with achieved hold errors + wall time. Run from poc/ (~8-10 min).
"""
import json, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from discslv_2f import TwoFactorSV, ssr_2f, smile_2f, stationary_pi

T0 = time.time()
DT = 1.0 / 52.0; NK = 20
NS = [4, 13, 26, 52]; T = np.array(NS) * DT
B = np.array([-5.30, 0.43, 0.50, 0.14, -1.48, 0.98, 1.65, 1.00, 2.34])
H = np.array([0.08, 0.05, 0.05, 0.04, 0.10, 0.08, 0.10, 0.08, 0.12])
LO = np.array([np.log(0.002), 0.05, 0.05, 0.05, -3.0, 0.0, 0.0, 0.05, 0.5])
HI = np.array([np.log(0.15), 1.2, 1.2, 1.0, 0.0, 8.0, 8.0, 2.0, 4.0])
TOL = np.array([12e-4, 12e-4, 12e-4, 0.02, 0.6])      # vol1m, vol6m, vol1y, skew1m, curv1m


def kern(x):
    return TwoFactorSV(gbar=x[0], nu_f=x[1], nu_s=x[2], nu_l=x[3], lam_skew=x[4],
                       lam_f=x[5], lam_s=x[6], kap_f=x[7], kap_s=x[8], dt=DT, n_f=5, n_s=3, n_l=5)


def sstats(g, Tn, dm=6e-3):
    F = g.forward(); iv = lambda k: float(g.implied_vol(F * k, Tn)[0])
    a, up, dn = iv(1.0), iv(np.exp(dm)), iv(np.exp(-dm))
    return a, (up - dn) / (2 * dm), (up - 2 * a + dn) / dm ** 2


def obs(x):
    K = kern(x); pi = stationary_pi(K)
    A = np.zeros((K.n_f, K.n_s)); S = np.zeros_like(A); C = np.zeros_like(A)
    for f in range(K.n_f):
        for s in range(K.n_s):
            A[f, s], S[f, s], C[f, s] = sstats(smile_2f(K, 4, f, s, NK), 4 * DT)
    v1 = (pi * A).sum(); k1 = (pi * S).sum(); c1 = (pi * C).sum()
    _, v6, _ = ssr_2f(K, 26, nk=NK); r4, v4, _ = ssr_2f(K, 52, nk=NK)
    return np.array([v1, v6, v4, k1, c1]), r4


def curve(x):
    K = kern(x); return [ssr_2f(K, n, nk=NK)[0] for n in NS]


def main():
    st0, ssr0 = obs(B); sc = np.abs(st0)
    print(f"target: vol1m={st0[0]:.4f} vol6m={st0[1]:.4f} vol1y={st0[2]:.4f} "
          f"sk1m={st0[3]:.3f} cv1m={st0[4]:.2f} | SSR1y={ssr0:.3f}  [{time.time()-T0:.0f}s]", flush=True)
    Js = np.zeros((5, 9)); gS = np.zeros(9)
    for j in range(9):
        xp = B.copy(); xp[j] += H[j]; xm = B.copy(); xm[j] -= H[j]
        sp, rp = obs(xp); sm, rm = obs(xm)
        Js[:, j] = ((sp - sm) / sc) / (2 * H[j]); gS[j] = (rp - rm) / (2 * H[j])
        print(f"  jac col {j}  [{time.time()-T0:.0f}s]", flush=True)
    Jpinv = np.linalg.pinv(Js); P = np.eye(9) - Jpinv @ Js
    rate = np.linalg.norm(P @ gS); u = (P @ gS) / rate
    print(f"||proj grad|| = {rate:.4f}", flush=True)

    def hold(a):
        x = np.clip(B + a * u, LO, HI)
        for _ in range(4):
            st, _ = obs(x); x = np.clip(x - Jpinv @ ((st - st0) / sc), LO, HI)
        return x

    recs = []
    for a in [-1.4, -1.0, -0.6, -0.3, 0.3, 0.6, 1.0, 1.4]:
        x = hold(a); st, r = obs(x); err = np.abs(st - st0)
        ok = bool(np.all(err < TOL))
        recs.append({"alpha": a, "ssr1y": float(r), "held": ok,
                     "err_vol_bp": [round(float(e) * 1e4, 1) for e in err[:3]],
                     "err_skew": round(float(err[3]), 4), "err_curv": round(float(err[4]), 3)})
        if ok:
            recs[-1]["x"] = x.tolist()
        print(f"  a={a:+.2f}: SSR1y={r:.3f} held={ok} errs vol {err[0]*1e4:.0f}/{err[1]*1e4:.0f}/"
              f"{err[2]*1e4:.0f}bp sk {err[3]:.3f} cv {err[4]:.2f}  [{time.time()-T0:.0f}s]", flush=True)
    held = [r for r in recs if r["held"]]
    band = [ssr0] + [r["ssr1y"] for r in held]
    lo_r, hi_r = min(band), max(band)
    print(f"5-obs band SSR(1y): [{lo_r:.3f},{hi_r:.3f}] width {hi_r-lo_r:.3f} ({len(held)}/8 held)", flush=True)

    SSR0 = curve(B)
    curves = [SSR0]
    if held:
        xlo = np.array(min(held, key=lambda t: t["ssr1y"])["x"])
        xhi = np.array(max(held, key=lambda t: t["ssr1y"])["x"])
        curves += [curve(xlo), curve(xhi)]
    arr = np.array(curves); LOb = arr.min(0); HIb = arr.max(0); w1y = HIb[3] - LOb[3]

    wall = time.time() - T0
    json.dump({"method": "5-obs hold (vol@1m,6m,1y + skew@1m + curv@1m), null-space walk",
               "nk": NK, "theta0": B.tolist(), "tol": {"vol_bp": 12, "skew": 0.02, "curv": 0.6},
               "ssr1y_band": [round(lo_r, 3), round(hi_r, 3)], "width_1y": round(hi_r - lo_r, 3),
               "T_years": T.tolist(), "SSR0": [float(v) for v in SSR0],
               "LOband": LOb.tolist(), "HIband": HIb.tolist(),
               "walk": recs, "wall_seconds": round(wall, 1)},
              open("band_results_5obs.json", "w"), indent=1)

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(T, SSR0, "o-", color="#1f6f8b", label="baseline SSR")
    ax.fill_between(T, LOb, HIb, alpha=.30, color="#b02a37", label="reachable, full smile held")
    ax.set_xlabel("maturity (y)"); ax.set_ylabel("SSR")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(0.03, 0.05, "held: vol@1m,6m,1y + skew@1m + curv@1m\n"
            f"band(1y) $\\approx$ {w1y:.2f}",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.85))
    fig.tight_layout()
    fig.savefig("../../figs/synthetic_main.png", bbox_inches="tight")
    print(f"wrote figs/synthetic_main.png; wall {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
