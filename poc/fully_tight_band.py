"""
fully_tight_band.py -- the precise statics/dynamics decoupling band at nk=24.

Scan all FOUR dynamics knobs (lam_f, lam_s, kap_f, kap_s); at each config hold the static smile
(vol1m, vol1y, skew1m) FIXED by a Newton step on the 3 static knobs (gbar, nu_s, lam_skew) using a
precomputed nk=24 static-Jacobian; keep only configs whose statics are held to tight tolerance.
The reachable SSR range over the held configs is the precise decoupling band.  Renders + splices the
band into ../discussions/math_flow.html.  Run from poc/  (slow, ~8-12 min).
"""
import base64, io, re, warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from discslv_2f import TwoFactorSV, ssr_2f

DT = 1.0 / 52.0; NK = 24
NS = [4, 13, 26, 52]; T = np.array(NS) * DT
TGT_SSR = [1.45, 1.30, 1.15, 1.05]; TGT_VOL = [0.17] * 4; TGT_SKEW = [-0.50, -0.40, -0.34, -0.28]
B = dict(gbar=-5.30, nu_f=0.43, nu_s=0.50, nu_l=0.14, lam_skew=-1.48, lam_f=0.98, lam_s=1.65, kap_f=1.00, kap_s=2.34)
SLO = np.array([np.log(0.002), 0.05, -3.0]); SHI = np.array([np.log(0.15), 1.2, 0.0])
TOL_VOL = 12e-4; TOL_SK = 0.02          # "held" tolerance: vol <=12 bp, skew <=0.02


def kern(s, d):
    return TwoFactorSV(gbar=s[0], nu_f=B["nu_f"], nu_s=s[1], nu_l=B["nu_l"], lam_skew=s[2],
                       lam_f=d[0], lam_s=d[1], kap_f=d[2], kap_s=d[3], dt=DT, n_f=5, n_s=3, n_l=5)


def statics(s, d):                       # (vol1m, vol1y, skew1m) at nk=24
    _, v1, k1 = ssr_2f(kern(s, d), 4, nk=NK); _, v4, _ = ssr_2f(kern(s, d), 52, nk=NK)
    return np.array([v1, v4, k1])


def main():
    s0 = np.array([B["gbar"], B["nu_s"], B["lam_skew"]]); d0 = np.array([B["lam_f"], B["lam_s"], B["kap_f"], B["kap_s"]])
    TGT = statics(s0, d0)
    print(f"target statics (nk={NK}): vol1m={TGT[0]:.4f} vol1y={TGT[1]:.4f} sk1m={TGT[2]:.3f}", flush=True)
    base_curve = [ssr_2f(kern(s0, d0), n, nk=NK) for n in NS]
    SSR0 = [b[0] for b in base_curve]; VOL0 = [b[1] for b in base_curve]; SKEW0 = [b[2] for b in base_curve]

    hs = np.array([0.08, 0.05, 0.12]); J = np.zeros((3, 3))      # static Jacobian d(statics)/d(s)
    for j in range(3):
        sp = s0.copy(); sp[j] += hs[j]; sm = s0.copy(); sm[j] -= hs[j]
        J[:, j] = (statics(sp, d0) - statics(sm, d0)) / (2 * hs[j])
    Jinv = np.linalg.inv(J)
    print("static Jacobian ready", flush=True)

    def hold(d, s_init, steps=3):
        s = s_init.copy()
        for _ in range(steps):
            s = np.clip(s - Jinv @ (statics(s, d) - TGT), SLO, SHI)
        return s, statics(s, d)

    LO = dict(lam_f=0.5, lam_s=1.0, kap_f=0.5, kap_s=1.5); HI = dict(lam_f=1.6, lam_s=2.8, kap_f=1.6, kap_s=3.2)
    names = ["lam_f", "lam_s", "kap_f", "kap_s"]
    configs = [("base", d0)]
    for i, nm in enumerate(names):
        for lab, val in [("lo", LO[nm]), ("hi", HI[nm])]:
            d = d0.copy(); d[i] = val; configs.append((f"{nm}_{lab}", d))
    for a, b in [(LO["lam_s"], LO["kap_s"]), (LO["lam_s"], HI["kap_s"]), (HI["lam_s"], LO["kap_s"]), (HI["lam_s"], HI["kap_s"])]:
        d = d0.copy(); d[1] = a; d[3] = b; configs.append((f"ls{a}_ks{b}", d))

    rows = []; s_warm = s0.copy()
    for nm, d in configs:
        s_warm, held = hold(d, s_warm)
        ok = abs(held[0] - TGT[0]) < TOL_VOL and abs(held[1] - TGT[1]) < TOL_VOL and abs(held[2] - TGT[2]) < TOL_SK
        ssr = [ssr_2f(kern(s_warm, d), n, nk=NK)[0] for n in (4, 26, 52)]
        rows.append((nm, ssr, held, ok))
        print(f"  {nm:12s} SSR(1m,6m,1y)={np.round(ssr,3)}  held: vol1m={ (held[0]-TGT[0])*1e4:+.0f} vol1y={(held[1]-TGT[1])*1e4:+.0f}bp sk={held[2]-TGT[2]:+.3f}  {'OK' if ok else 'reject'}", flush=True)
        if not ok:
            s_warm = s0.copy()                                   # reset warm-start after a reject

    held_rows = [r for r in rows if r[3]]
    s1m = [r[1][0] for r in held_rows]; s6m = [r[1][1] for r in held_rows]; s1y = [r[1][2] for r in held_rows]
    print(f"\n{len(held_rows)}/{len(rows)} configs hold statics (vol<=12bp, skew<=0.02).", flush=True)
    print(f"BAND @ held statics (nk=24, 4 dynamics knobs):  SSR(1y)=[{min(s1y):.3f},{max(s1y):.3f}]  "
          f"SSR(6m)=[{min(s6m):.3f},{max(s6m):.3f}]  SSR(1m)=[{min(s1m):.3f},{max(s1m):.3f}]", flush=True)

    SSR_LO = [min(s1m), min(s6m), min(s1y)]; SSR_HI = [max(s1m), max(s6m), max(s1y)]; TB = np.array([4, 26, 52]) * DT
    w1y = max(s1y) - min(s1y)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.1))
    ax[0].plot(T, SSR0, "o-", label="SANOS-Evolve"); ax[0].plot(T, TGT_SSR, "s--", label="SPX target")
    ax[0].set_title("SSR term structure"); ax[0].legend(fontsize=7)
    ax[1].plot(T, VOL0, "o-"); ax[1].plot(T, TGT_VOL, "s--"); ax[1].set_title("ATM vol"); ax[1].set_ylim(.10, .22)
    ax[2].plot(T, SKEW0, "o-"); ax[2].plot(T, TGT_SKEW, "s--"); ax[2].set_title("ATM skew")
    ax[3].plot(T, SSR0, "o-", color="C0", label="baseline SSR")
    ax[3].fill_between(TB, SSR_LO, SSR_HI, alpha=.25, color="C3", label="reachable @ fixed statics")
    ax[3].plot(TB, SSR_LO, "_", color="C3"); ax[3].plot(TB, SSR_HI, "_", color="C3")
    ax[3].set_title("controllable SSR @ fixed statics"); ax[3].legend(fontsize=6, loc="upper right")
    ax[3].text(0.04, 0.06, f"vol+skew held ≤12bp/0.02\n4 dyn. knobs · nk=24\nband(1y)≈{w1y:.2f}",
               transform=ax[3].transAxes, fontsize=6.3, va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.85))
    for a_ in ax:
        a_.grid(alpha=.3); a_.set_xlabel("maturity (y)")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()

    P = "../discussions/math_flow.html"; html = open(P, encoding="utf-8").read()
    newfig = ('<figure><img alt="2f calibrated fit" src="data:image/png;base64,' + b64 + '">\n'
              '<figcaption>The calibrated two-factor genuine-SV kernel (closed form, no Monte&nbsp;Carlo; nk=24). '
              'Panels 1&ndash;3: the <strong>SSR</strong>, ATM&nbsp;vol and ATM&nbsp;skew term structures, model vs SPX target. '
              'Panel&nbsp;4: <strong>statics/dynamics decoupling</strong>, precise band &mdash; scanning all four dynamics knobs '
              '$(\\lambda_f,\\lambda_s,\\kappa_f,\\kappa_s)$ while a Newton step on $(\\bar\\gamma,\\nu_s,\\lambda_{\\rm skew})$ '
              'holds the static smile to $\\le12$&nbsp;bp / $\\le0.02$ (all at nk=24), the SSR is freely movable inside the shaded band &mdash; '
              f'the long-end spans $\\approx{min(s1y):.2f}\\text{{--}}{max(s1y):.2f}$' + ' at <em>fixed statics</em>. '
              'The reachable set is a <em>bounded region</em> (the Step-6 soft direction made concrete), not a monotone single-knob dial; '
              'wider SSR moves at fixed statics are the role of the full SANOS marginal layer in the integrated model. '
              '$\\theta_0=(\\bar\\gamma\\,{-}5.30,\\nu_f\\,0.43,\\nu_s\\,0.50,\\nu_l\\,0.14,\\lambda_{\\rm skew}\\,{-}1.48,'
              '\\lambda_f\\,0.98,\\lambda_s\\,1.65,\\kappa_f\\,1.00,\\kappa_s\\,2.34)$.</figcaption></figure>')
    html2 = re.sub(r'<figure><img alt="2f calibrated fit".*?</figure>', lambda m: newfig, html, count=1, flags=re.S)
    assert html2 != html, "figure not found"
    open(P, "w", encoding="utf-8").write(html2)
    print(f"spliced; SSR(1y) band [{min(s1y):.3f},{max(s1y):.3f}] width {w1y:.3f}; HTML {len(html2)} bytes", flush=True)


if __name__ == "__main__":
    main()
