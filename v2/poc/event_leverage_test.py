"""
Stage 1 -- synthetic event-leverage NON-REDUNDANCY test.

The make-or-break question: can a scheduled event carry DYNAMICS (a jump<->vol leverage =
an event-conditional SSR) that is INVISIBLE in the static event-horizon marginal?  If yes,
static de-eventing (Zhong/Vola: a per-expiry marginal jump density) provably cannot see it,
and the forward-start SSR is where the signal lives -- which is exactly Zhong's empirical
"SPX-macro event = variance, not directional skew": the static channel is flat, the dynamics
channel is non-empty.

Construction (on the calibrated diffusive kernel K_diff(theta_0)):
  The EVENT is a one-step modification at the event step that bumps the REGIME-TRANSITION
  leverage (lam_f, lam_s) -- the spot<->vol coupling -- WITHOUT touching the within-step
  return (D, V).  Provable fact in this model (discslv_2f.py): lam_f/lam_s enter ONLY the
  transition P(f'|l,f)=softmax(.. + lam_f zeta_l[l] zeta_f), and Sum_{f'} P(f'|l,f)=1, so the
  one-step SPOT marginal -- a mixture over l of N(D[f,s,l], V[f,s,l]) -- does NOT depend on
  lam_f/lam_s.  Hence the event-horizon marginal is UNCHANGED, but the regime arriving at T3
  is now correlated with the event return, so the forward-start SSR across the event carries
  the leverage.

  Scenario A (pure leverage): static marginal EXACTLY unchanged, dynamics SSR rises.
  Scenario B (Zhong-realistic): + a symmetric martingale jump N(-J/2, J) -> static gains
  VARIANCE but zero added asymmetry (third central moment), dynamics SSR still rises.

Run from poc/ :  python3 event_leverage_test.py
"""
import sys, json
POC = "/Users/foxie/Documents/Research/2026/SANOS_Evolve/disc_SLV/poc"
sys.path.insert(0, POC)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from discslv_2f import TwoFactorSV, smile_2f, stationary_pi, _atm_skew

DT = 1 / 52.0
# the paper's calibrated theta_0 (fig:synth)
TH = dict(gbar=-5.30, nu_f=0.43, nu_s=0.50, lam_skew=-1.48, lam_f=0.98, lam_s=1.65,
          kap_f=1.00, kap_s=2.34, dt=DT, nu_l=0.14, n_f=5, n_s=3, n_l=5)
NFWD = 26   # post-event forward horizon (6m) for the forward-start smile
NK = 16

def build(lam_mult):
    th = dict(TH); th["lam_f"] = TH["lam_f"] * lam_mult; th["lam_s"] = TH["lam_s"] * lam_mult
    return TwoFactorSV(**th)

Kd = build(1.0)                 # diffusive / de-evented baseline kernel
PI = stationary_pi(Kd)          # pre-event regime mix (set by the CLEAN pre-event windows)

# ---- forward smiles per post-event regime (post-event reverts to diffusive => from K_diff) ----
SIG = np.zeros((Kd.n_f, Kd.n_s)); SKW = np.zeros((Kd.n_f, Kd.n_s))
for f in range(Kd.n_f):
    for s in range(Kd.n_s):
        SIG[f, s], SKW[f, s] = _atm_skew(smile_2f(Kd, NFWD, f, s, NK), NFWD * DT)


def static_moments(addJ=0.0):
    """Event-horizon (one-step) return marginal: mixture over regimes (~PI) and nodes l of
    N(D[f,s,l], V[f,s,l]), optionally convolved with a symmetric martingale jump N(-J/2, J).
    Depends ONLY on (D, V, PI) -- NOT on lam_f/lam_s -- so it is identical for K_diff and K_event.
    Returns (variance, skewness, third_central_moment)."""
    ws, mus, vs = [], [], []
    for f in range(Kd.n_f):
        for s in range(Kd.n_s):
            d, V = Kd.incr(f, s)
            for l in range(Kd.n_l):
                ws.append(PI[f, s] * Kd.wl[l]); mus.append(d[l] - 0.5 * addJ); vs.append(V[l] + addJ)
    w = np.array(ws); w /= w.sum(); mu = np.array(mus); v = np.array(vs)
    m = float((w * mu).sum())
    c2 = float((w * (v + (mu - m) ** 2)).sum())                      # variance
    c3 = float((w * ((mu - m) ** 3 + 3 * (mu - m) * v)).sum())       # third central moment
    return c2, c3 / c2 ** 1.5, c3


def event_ssr(Kev):
    """Event-conditional forward-start SSR: diffusive within-step return (Kd.incr) + Kev's
    transitions (the event leverage) + diffusive post-event forward smiles (SIG, SKW)."""
    cov = var = sk = 0.0
    for f in range(Kd.n_f):
        for s in range(Kd.n_s):
            d, V = Kd.incr(f, s)
            mean_r = float(np.sum(Kd.wl * d)); var_fs = float(np.sum(Kd.wl * (d ** 2 + V)) - mean_r ** 2)
            c = 0.0
            for l in range(Kd.n_l):
                Pf = Kev.trans_f(l, f); Ps = Kev.trans_s(l, s); resp = 0.0
                for fp in range(Kd.n_f):
                    for sp in range(Kd.n_s):
                        resp += Pf[fp] * Ps[sp] * (SIG[fp, sp] - SIG[f, s])
                c += Kd.wl[l] * d[l] * resp
            cov += PI[f, s] * c; var += PI[f, s] * var_fs; sk += PI[f, s] * SKW[f, s]
    return (cov / var) / sk


# ----------------------------- baseline -----------------------------
ssr_d = event_ssr(Kd)
var_d, skew_d, c3_d = static_moments(0.0)
print("=" * 78)
print("STAGE 1: event-leverage NON-REDUNDANCY  (calibrated theta_0, closed-form)")
print("=" * 78)
print(f"baseline (de-evented diffusive):  forward SSR = {ssr_d:.4f}")
print(f"                                  static 1-step marginal: var={var_d:.5e}  skew={skew_d:+.4f}\n")

# --------------- Scenario A: PURE leverage (no variance bump) ---------------
print("--- Scenario A: PURE event-leverage (lam_f,lam_s bumped; D,V untouched) ---")
print(f"{'lam x':>6} | {'fwd SSR':>9} {'dSSR':>8} | {'static var':>12} {'static skew':>12} "
      f"{'|dVar|':>10} {'|dSkew|':>10}")
mults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
A = []
for m in mults:
    Kev = build(m)
    ssr = event_ssr(Kev)
    var, skew, _ = static_moments(0.0)   # static is lam-independent by construction
    A.append(dict(mult=m, ssr=ssr, dssr=ssr - ssr_d, var=var, skew=skew,
                  dvar=abs(var - var_d), dskew=abs(skew - skew_d)))
    print(f"{m:>6.2f} | {ssr:>9.4f} {ssr-ssr_d:>+8.4f} | {var:>12.5e} {skew:>+12.4f} "
          f"{abs(var-var_d):>10.2e} {abs(skew-skew_d):>10.2e}", flush=True)

# --------------- Scenario B: Zhong-realistic (symmetric variance jump + leverage) ---------------
print("\n--- Scenario B: symmetric variance jump J + leverage (Zhong 'variance, not skew') ---")
Jbump = 0.0009   # ~ a 3% 1-day event move's variance (0.03**2), in variance units
print(f"event variance jump J = {Jbump:.4f}  (adds variance, ZERO third central moment)")
var_J, skew_J, c3_J = static_moments(Jbump)
print(f"static WITH jump:  var={var_J:.5e} (dVar=+{var_J-var_d:.2e} = J)  "
      f"skew={skew_J:+.4f}  3rd-central c3={c3_J:+.3e} vs baseline {c3_d:+.3e} (event asymmetry = {c3_J-c3_d:+.2e})")
B = []
for m in mults:
    Kev = build(m); ssr = event_ssr(Kev)
    B.append(dict(mult=m, ssr=ssr, dssr=ssr - ssr_d, var=var_J, skew=skew_J))
    print(f"  lam x{m:.2f}: fwd SSR={ssr:.4f} (dSSR={ssr-ssr_d:+.4f}); static var={var_J:.5e}, "
          f"event 3rd-moment={c3_J-c3_d:+.2e}", flush=True)

# ----------------------------- verdict -----------------------------
print("\n" + "=" * 78)
maxdskew = max(r["dskew"] for r in A); maxdvar = max(r["dvar"] for r in A)
dssr_range = A[-1]["dssr"]
print("VERDICT")
print(f"  Static channel under pure leverage: max |dVar|={maxdvar:.2e}, max |dSkew|={maxdskew:.2e} "
      f"-> EXACTLY FLAT (machine precision).")
print(f"  Dynamics channel: forward SSR moves {ssr_d:.3f} -> {A[-1]['ssr']:.3f} "
      f"(dSSR={dssr_range:+.3f}) as event leverage rises.")
print(f"  Scenario B: the event adds variance (J={Jbump}) with event asymmetry {c3_J-c3_d:+.1e} (~0) "
      f"-- matches Zhong 'variance, not skew' -- yet dSSR={B[-1]['dssr']:+.3f}.")
nonredundant = (maxdskew < 1e-9 and maxdvar < 1e-9 and abs(dssr_range) > 0.05)
print(f"  => NON-REDUNDANT: {nonredundant}  "
      f"(the dynamics channel carries signal the static marginal provably cannot see).")
print("=" * 78)

# ----------------------------- plot -----------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
ms = [r["mult"] for r in A]
a1.axhline(0, color="#bbb", lw=.8)
a1.plot(ms, [r["dvar"] for r in A], "-o", color="#1b7837", ms=4, label="|dVar| (static)")
a1.plot(ms, [r["dskew"] for r in A], "-s", color="#762a83", ms=4, label="|dSkew| (static)")
a1.set(title="Static event-horizon marginal\n(pure-leverage event)", xlabel="event leverage  (lam multiple)",
       ylabel="deviation from de-evented", ylim=(-1e-3, 5e-3))
a1.legend(fontsize=8, frameon=False); a1.grid(alpha=.25)
a1.text(1.05, 3.2e-3, "FLAT at 0 (machine precision):\nthe static channel is blind\nto the event leverage",
        fontsize=8, color="#555")
a2.plot(ms, [r["dssr"] for r in A], "-o", color="#b2182b", ms=5, label="dSSR (dynamics)")
a2.axhline(0, color="#bbb", lw=.8)
a2.set(title="Forward-start SSR across the event\n(the dynamics channel)", xlabel="event leverage  (lam multiple)",
       ylabel="dSSR vs de-evented")
a2.legend(fontsize=8, frameon=False); a2.grid(alpha=.25)
fig.suptitle("Stage 1: event-leverage is INVISIBLE to the static marginal but recovered by the forward-start SSR\n"
             "(non-redundancy of the dynamics channel -- the make-or-break for de-eventing)", fontsize=10)
fig.tight_layout()
PNG = POC + "/event_leverage_nonredundancy.png"
fig.savefig(PNG, dpi=130, bbox_inches="tight")
print("plot saved:", PNG)
json.dump({"baseline_ssr": ssr_d, "baseline_var": var_d, "baseline_skew": skew_d,
           "scenarioA": A, "scenarioB": B, "Jbump": Jbump, "event_asymmetry_c3": c3_J - c3_d,
           "nonredundant": bool(nonredundant)},
          open(POC + "/event_leverage_results.json", "w"), indent=2)
print("json saved:", POC + "/event_leverage_results.json")
