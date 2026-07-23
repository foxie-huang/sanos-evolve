"""
band_fig_single.py -- regenerate figs/synthetic_main.png as the SINGLE decoupling-band panel
(paper fig:synth after the surgical cut: the fit demo panels 1-3 are dropped; the band survives).
Same computation as final_band_fig.py (steepest static-preserving direction, Newton-held walk),
plus: persist the band arrays + wall time to band_results.json. Run from poc/ (~5-6 min).
"""
import io, json, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from discslv_2f import TwoFactorSV, ssr_2f

T0 = time.time()
DT = 1.0 / 52.0; NK = 24
NS = [4, 13, 26, 52]; T = np.array(NS) * DT
B = np.array([-5.30, 0.43, 0.50, 0.14, -1.48, 0.98, 1.65, 1.00, 2.34])
H = np.array([0.08, 0.05, 0.05, 0.04, 0.10, 0.08, 0.10, 0.08, 0.12])
LO = np.array([np.log(0.002), 0.05, 0.05, 0.05, -3.0, 0.0, 0.0, 0.05, 0.5])
HI = np.array([np.log(0.15), 1.2, 1.2, 1.0, 0.0, 8.0, 8.0, 2.0, 4.0])


def kern(x):
    return TwoFactorSV(gbar=x[0], nu_f=x[1], nu_s=x[2], nu_l=x[3], lam_skew=x[4],
                       lam_f=x[5], lam_s=x[6], kap_f=x[7], kap_s=x[8], dt=DT, n_f=5, n_s=3, n_l=5)


def stat_ssr(x):
    K = kern(x); _, v1, k1 = ssr_2f(K, 4, nk=NK); r4, v4, _ = ssr_2f(K, 52, nk=NK)
    return np.array([v1, v4, k1]), r4


def full_curve(x):
    K = kern(x); return [ssr_2f(K, n, nk=NK)[0] for n in NS]


def main():
    st0, _ = stat_ssr(B)
    SSR0 = full_curve(B)
    Js = np.zeros((3, 9)); gS = np.zeros(9)
    for j in range(9):
        xp = B.copy(); xp[j] += H[j]; xm = B.copy(); xm[j] -= H[j]
        sp, rp = stat_ssr(xp); sm, rm = stat_ssr(xm)
        Js[:, j] = (sp - sm) / (2 * H[j]); gS[j] = (rp - rm) / (2 * H[j])
    Jpinv = np.linalg.pinv(Js); u = (np.eye(9) - Jpinv @ Js) @ gS; u /= np.linalg.norm(u)
    print(f"u ready  [{time.time()-T0:.0f}s]", flush=True)

    def hold(alpha):
        x = np.clip(B + alpha * u, LO, HI)
        for _ in range(4):
            st, _ = stat_ssr(x); x = np.clip(x - Jpinv @ (st - st0), LO, HI)
        return x

    curves_loose = [SSR0]; curves_tight = [SSR0]; held = [(0.0, 0.0, 0.0)]
    for a in [-0.7, -0.5, -0.35, -0.2, 0.15, 0.30]:
        x = hold(a); st, _ = stat_ssr(x)
        verr = max(abs(st[0] - st0[0]), abs(st[1] - st0[1])) * 1e4; kerr = abs(st[2] - st0[2])
        loose = verr < 15 and kerr < 0.02; tight = verr < 13 and kerr < 0.012
        if loose:
            c = full_curve(x); curves_loose.append(c); held.append((a, verr, kerr))
            if tight: curves_tight.append(c)
            print(f"  alpha={a:+.2f} held verr={verr:.1f}bp kerr={kerr:.3f} "
                  f"({'tight' if tight else 'loose'}); SSR={np.round(c,3)}  [{time.time()-T0:.0f}s]", flush=True)
        else:
            print(f"  alpha={a:+.2f} REJECT verr={verr:.1f}bp kerr={kerr:.3f}", flush=True)
    aL = np.array(curves_loose); aT = np.array(curves_tight)
    LOband, HIband = aT.min(0), aT.max(0); w1y = HIband[3] - LOband[3]     # figure/paper = TIGHT
    LOl, HIl = aL.min(0), aL.max(0)
    print(f"TIGHT band (13bp/0.012): 1y[{LOband[3]:.3f},{HIband[3]:.3f}] (w1y {w1y:.3f})", flush=True)
    print(f"loose band (15bp/0.02):  1y[{LOl[3]:.3f},{HIl[3]:.3f}] (w1y {HIl[3]-LOl[3]:.3f})", flush=True)

    wall = time.time() - T0
    json.dump({"T_years": T.tolist(), "SSR0": list(map(float, SSR0)),
               "tight_LOband": LOband.tolist(), "tight_HIband": HIband.tolist(),
               "tight_width_1y": float(w1y), "tight_tol": "vol<=13bp skew<=0.012",
               "loose_LOband": LOl.tolist(), "loose_HIband": HIl.tolist(),
               "loose_width_1y": float(HIl[3] - LOl[3]), "loose_tol": "vol<=15bp skew<=0.02",
               "alphas_held_verr_kerr": held, "theta0": B.tolist(), "nk": NK,
               "note": "post-Jul-2 discslv_2f code; supersedes the Jun-27 [0.93,1.02] w=0.08 run",
               "wall_seconds": round(wall, 1)},
              open("band_results.json", "w"), indent=1)

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(T, SSR0, "o-", color="#1f6f8b", label="baseline SSR")
    ax.fill_between(T, LOband, HIband, alpha=.30, color="#b02a37", label="reachable at fixed statics")
    ax.set_xlabel("maturity (y)"); ax.set_ylabel("SSR")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(0.03, 0.05, f"statics held $\\leq$13bp / $\\leq$0.012\nband(1y) $\\approx$ {w1y:.2f}",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.85))
    fig.tight_layout()
    out = "../../figs/synthetic_main.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}; wall {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
