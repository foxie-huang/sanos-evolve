#!/usr/bin/env python3
"""
FAITHFUL clean-flank bridge on the disc_SLV PAPER's kernel (two-factor GM + discrete-Dupire leverage) --
NOT GLIDE's generic kernel. GLIDE gave the STRUCTURE (density-first marginal chain, kernel propagation,
de-eventing = stationarity break); the actual kernel / leverage / propagation is the paper's own machinery
(`sanos_chain` marginals + `TwoFactorSV` + `leverage_at` + `propagate_vec`). Crucially the LEVERAGE generates
the diffusive put-skew that GLIDE's Level-1 kernel lacked (-> the control artifact), so the residual can
isolate the REAL event skew.

  mu3      = SANOS marginal at the event-spanning expiry T3 (event-contaminated), interp_marginal(chain, T3)
  mu3^diff = leveraged-kernel forward prediction propagated from spot to T3 (model chain interp to T3)
  J_n      = kappa_n(mu3) - kappa_n(mu3^diff):  n=2 variance (sqrt=move), n=3 skew, n=4 kurtosis.
Same interface (build) as the retired GLIDE version, so the pool/control drivers just re-point here.
    python3 deevent_bridge_slv.py [EVENT=2024-05-01,...] [TICKER=SPX] [BACKDAYS=9]
"""
import sys, os, time
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from datetime import date as D, timedelta

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import discslv_slv                                                         # noqa: E402
from slv_fast import propagate_vec                                         # noqa: E402
discslv_slv.propagate = propagate_vec
from discslv_slv import (propagate, marginal, initial_state, E_nu_given_z_vec,  # noqa: E402
                         Epi_V, nu_bar, raw_increment)
from discslv_2f import TwoFactorSV, stationary_pi                          # noqa: E402
from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at         # noqa: E402
from slv_interp import interp_marginal                                     # noqa: E402
import calibrate_slv_exact_ts as C                                        # noqa: E402

DT = 1.0 / 52.0; NAMES = C.NAMES
OUT = "/Users/foxie/Documents/Research/2026/US_equity_data/orats_eod"
# Full eq:objective fit (calibrate_full_torch, 2019-06-03, jacfwd multi-start, fixed per-component E[nu|z]):
# marginal survival RMS 2.86% (best of 4 starts, cost 1.3211). Bridge mu3^diff ivol 15.1% vs SANOS 15.8% (variance
# matched); skew -0.83 vs -2.36 = from-spot flattening (needs one-step-from-flank). NOT the SSR-only fit (over-produces).
THETA = {"SPX": [0.112, 0.406, 0.189, -2.982, 0.337, 2.581, 0.367, 2.656]}


def cumulants(mu):
    """Central cumulants (k2,k3,k4) of the log-return under a GM marginal (W, MU, SG)."""
    W, MU, SG = mu; m = np.sum(W * MU); d = MU - m
    k2 = np.sum(W * (d ** 2 + SG ** 2))
    k3 = np.sum(W * (d ** 3 + 3 * d * SG ** 2))
    k4 = np.sum(W * (d ** 4 + 6 * d ** 2 * SG ** 2 + 3 * SG ** 4))
    return float(k2), float(k3), float(k4 - 3 * k2 ** 2)


def model_chain(chain, sig, theta, nmax):
    """Model marginals at each weekly step 1..nmax: the leveraged two-factor kernel propagated from spot
    (paper's from-spot fusion, E[nu|z]-corrected leverage), as in curv_diag.model_marg but saving each step."""
    kw0 = dict(zip(NAMES, C.X0_MAP["dense"]))
    EV0 = Epi_V(TwoFactorSV(gbar=solve_gbar(kw0, sig, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw0))
    kw = dict(zip(NAMES, theta))
    K = TwoFactorSV(gbar=solve_gbar(kw, sig, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    st = initial_state(K); mc = []
    for k in range(1, nmax + 1):
        lf = leverage_at(chain, k * DT, EV0)
        st, _ = propagate(K, st, lambda mm, l=lf, cur=st: l(mm) ** 2 / np.clip(E_nu_given_z_vec(mm, cur, nub), 0.3, 3.0),
                          EV, nub, Vlr, tiltr, 16)
        mc.append((k * DT, marginal(st)))
    return mc


def snapshot(event, backdays):
    ev = D.fromisoformat(event)
    for d in range(backdays, backdays + 6):
        p = f"{OUT}/SPX-NDX-RUT-VIX_{(ev - timedelta(d)).isoformat()}.json.gz"
        if os.path.exists(p):
            return p, (ev - timedelta(d))
    return None, None


def build(event="2024-05-01", ticker="SPX", backdays=9):
    p, snap = snapshot(event, backdays)
    if not p:
        return {"event": event, "err": "no snapshot"}
    chain = sanos_chain(p, ticker=ticker)
    if len(chain) < 4:
        return {"event": event, "err": f"short chain {len(chain)}"}
    sig = ref_vol(chain)
    ev_T = ((D.fromisoformat(event) - snap).days + 1) / 365.0             # ORATS dte = calendar-days + 1
    Ts = [c[0] for c in chain]
    post = [T for T in Ts if T >= ev_T]
    if not post or Ts[0] >= ev_T:                                         # need >=1 clean pre-event expiry
        return {"event": event, "err": f"no bracket (min T {Ts[0]:.4f}, ev_T {ev_T:.4f})"}
    T3 = post[0]
    mu3 = interp_marginal(chain, T3)                                      # actual SANOS marginal at T3 (event-contaminated)
    mc = model_chain(chain, sig, THETA.get(ticker, THETA["SPX"]), max(2, int(np.ceil(T3 / DT)) + 1))
    mu3d = interp_marginal(mc, T3)                                        # de-evented model prediction at T3
    k2a, k3a, k4a = cumulants(mu3); k2d, k3d, k4d = cumulants(mu3d)
    J2, J3, J4 = k2a - k2d, k3a - k3d, k4a - k4d
    if os.environ.get("DBG"):
        print(f"  [dbg] {event}: mu3 ivol={np.sqrt(k2a/T3)*100:.1f}% sk={k3a/k2a**1.5:+.2f} | "
              f"mu3diff ivol={np.sqrt(max(k2d,0)/T3)*100:.1f}% sk={k3d/k2d**1.5:+.2f}")
    return dict(event=event, snap=snap.isoformat(), T3=T3, J2=J2, J3=J3, J4=J4,
                move=np.sqrt(max(J2, 0)), skew=(J3 / J2 ** 1.5 if J2 > 0 else float("nan")),
                exkurt=(J4 / J2 ** 2 if J2 > 0 else float("nan")))


def lift_marginal(mu, K):
    """Lift a plain GM marginal (W,MU,SG) -- no regime tags -- into a kernel state by spreading EACH
    component over the stationary (f,s) regimes (the analog of initial_state, but at the mu shape instead
    of a point mass). Regime-agnostic: assumes the SV state sits at its stationary law at the anchor -- the
    clean default for a single step; the leverage/regime correlation then develops over that one step."""
    W2, MU2, SG2 = mu
    pi = stationary_pi(K).ravel(); ns = K.n_s
    W, MU, SG, F, S = [], [], [], [], []
    for i in range(len(W2)):
        for f in range(K.n_f):
            for s in range(K.n_s):
                W.append(W2[i] * pi[f * ns + s]); MU.append(MU2[i]); SG.append(SG2[i]); F.append(f); S.append(s)
    return (np.array(W), np.array(MU), np.array(SG), np.array(F, np.intp), np.array(S, np.intp))


def onestep_expand(state, K, lam2_of, Vlr, tiltr):
    """Exact ONE-step marginal (W,MU,SG) from the full (l,f',s') expansion -- NO recompress, so the anchor's
    skew is preserved. One recompress flattens a high-skew anchor badly (mu2 @-2.06 -> nk16 -1.29 vs
    no-recompress -1.76; controls go from ~0.15 to ~0.03 skew-gap). Valid for a SINGLE step ONLY -- the
    expansion is ~n_l*n_f*n_s = 75x the input, so never chain it; the marginal (ignoring regime tags) is
    all we need for the residual cumulants (matches slv_fast.propagate_vec pre-recompress)."""
    W, MU, SG, F, S = state; wl = K.wl; nl, nf, ns = K.n_l, K.n_f, K.n_s
    lam = np.sqrt(np.maximum(lam2_of(MU), 1e-12))                         # (N,) leverage at component means
    Vl = lam[:, None] ** 2 * Vlr[F, S]; mtil = -0.5 * Vl + lam[:, None] * tiltr[F, S]
    A = np.log(np.sum(wl[None, :] * np.exp(mtil + 0.5 * Vl), axis=1)); Dl = mtil - A[:, None]   # martingale lock
    Tfi = np.transpose(K.Tf[:, F, :], (1, 0, 2)); Tsi = np.transpose(K.Ts[:, S, :], (1, 0, 2))
    w = W[:, None, None, None] * wl[None, :, None, None] * Tfi[:, :, :, None] * Tsi[:, :, None, :]
    mu_b = np.broadcast_to((MU[:, None] + Dl)[:, :, None, None], (len(W), nl, nf, ns))
    sg_b = np.broadcast_to(np.sqrt(SG[:, None] ** 2 + Vl)[:, :, None, None], (len(W), nl, nf, ns))
    W2 = w.ravel()
    return (W2 / W2.sum(), mu_b.ravel().copy(), sg_b.ravel().copy())


def event_convolve(mu, J):
    """Load a scheduled variance lump J into a GM marginal: convolve the log-return with N(-J/2, J), which is
    martingale-preserving (each component's forward e^{m+s^2/2} is unchanged). In GM space this is trivial --
    shift every mean by -J/2, add J to every variance -- and adds EXACTLY J to kappa_2 with ZERO skew. This is
    the event(J) kernel: a pure scheduled variance jump, no asymmetry."""
    W, MU, SG = mu
    return (W, MU - 0.5 * J, np.sqrt(SG ** 2 + J))


def _digitals(mu, KS):
    """Survival Pr(x>k) = sum_i W_i Phi((MU_i-k)/SG_i) at each k in KS, for a GM marginal (W,MU,SG)."""
    W, MU, SG = mu
    return np.array([np.sum(W * norm.cdf((MU - k) / SG)) for k in KS])


def fit_event_J(mu3, mu3d, nK=15, span=2.5):
    """STAGE-2 event-vol fit: with the clean transition mu3d fixed, fit the scalar event variance J so that
    mu3d (x) event(J) matches the CONTAMINATED T3 SMILE (digitals on a +-span*SD grid), NOT just its variance.
    Returns (J, smile_rms, Jvar, resid_skew):
      J          = the smile-fit event variance (the Stage-2 answer);
      smile_rms  = digital RMS a PURE variance lump can't explain (= the event's non-variance content);
      Jvar       = kappa2(mu3)-kappa2(mu3d), the moment read-off (J that matches variance ONLY -- for compare);
      resid_skew = put-minus-call asymmetry of the leftover digitals (flags an event SKEW the lump misses).
    If the event is a pure variance lump: J~Jvar, smile_rms~0, resid_skew~0."""
    sd = np.sqrt(max(cumulants(mu3)[0], 1e-12))
    KS = np.linspace(-span * sd, span * sd, nK)
    tgt = _digitals(mu3, KS)
    Jvar = cumulants(mu3)[0] - cumulants(mu3d)[0]
    obj = lambda J: float(np.sum((_digitals(event_convolve(mu3d, J), KS) - tgt) ** 2))
    hi = max(4 * abs(Jvar), 1e-4)
    J = float(minimize_scalar(obj, bounds=(0.0, hi), method="bounded").x)
    resid = _digitals(event_convolve(mu3d, J), KS) - tgt
    smile_rms = float(np.sqrt(np.mean(resid ** 2)))
    resid_skew = float(np.sum(resid * np.sign(KS)))                      # put-side minus call-side leftover (at fitted J)
    # MATCHED-VARIANCE skew probe: at J=Jvar the variance is EXACTLY equal to mu3's, so the leftover digital
    # asymmetry is PURE shape (skew/kurt) -- the un-confounded event-skew test (no -J/2 mean-shift artifact).
    resid_mv = _digitals(event_convolve(mu3d, max(Jvar, 0.0)), KS) - tgt
    resid_skew_mv = float(np.sum(resid_mv * np.sign(KS)))
    return J, smile_rms, Jvar, resid_skew, resid_skew_mv


def build_flank(event="2024-05-01", ticker="SPX", backdays=9):
    """ONE-STEP-FROM-FLANK bridge (fixes the from-spot cold-start skew lag).
      mu2      = SANOS marginal at the LAST CLEAN expiry T2 < event (already carries the market skew);
      mu3^diff = mu2 lifted to a regime state, propagated ONE time-rescaled step (dt = Delta = T3-T2) with
                 the EVENT-FREE flank leverage (evaluated at the clean T2), to T3;
      mu3      = SANOS marginal at T3 (event-contaminated, interp_marginal(chain,T3)).
    J_n = kappa_n(mu3) - kappa_n(mu3^diff): the event's marginal effect, with the skew reference CORRECT
    (mu3^diff inherits mu2's skew instead of building it from a point mass over many steps)."""
    p, snap = snapshot(event, backdays)
    if not p:
        return {"event": event, "err": "no snapshot"}
    chain = sanos_chain(p, ticker=ticker)
    if len(chain) < 4:
        return {"event": event, "err": f"short chain {len(chain)}"}
    sig = ref_vol(chain)
    ev_T = ((D.fromisoformat(event) - snap).days + 1) / 365.0             # ORATS dte = calendar-days + 1
    Ts = [c[0] for c in chain]
    pre = [T for T in Ts if T < ev_T]; post = [T for T in Ts if T >= ev_T]
    if not pre or not post:                                              # need a clean flank T2 AND an event-spanning T3
        return {"event": event, "err": f"no flank bracket (ev_T {ev_T:.4f}, Ts {Ts[0]:.3f}..{Ts[-1]:.3f})"}
    T2, T3 = max(pre), min(post); Delta = T3 - T2
    mu2 = interp_marginal(chain, T2); mu3 = interp_marginal(chain, T3)   # clean anchor / contaminated target
    kw = dict(zip(NAMES, THETA.get(ticker, THETA["SPX"])))
    K = TwoFactorSV(gbar=solve_gbar(kw, sig, dt=Delta), dt=Delta, n_f=5, n_s=3, n_l=5, **kw)   # time-rescaled kernel
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    kw0 = dict(zip(NAMES, C.X0_MAP["dense"]))
    EV0 = Epi_V(TwoFactorSV(gbar=solve_gbar(kw0, sig, dt=Delta), dt=Delta, n_f=5, n_s=3, n_l=5, **kw0))   # EV0 at dt=Delta
    lf = leverage_at(chain, T2, EV0, dt=Delta)                           # event-free leverage from the clean flank
    st = lift_marginal(mu2, K)
    mu3d = onestep_expand(st, K, lambda mm, l=lf, cur=st: l(mm) ** 2 / np.clip(E_nu_given_z_vec(mm, cur, nub), 0.3, 3.0),
                          Vlr, tiltr)                                     # exact one-step marginal, NO recompress (preserves skew)
    k2a, k3a, k4a = cumulants(mu3); k2d, k3d, k4d = cumulants(mu3d); k2m, k3m, _ = cumulants(mu2)
    J2, J3, J4 = k2a - k2d, k3a - k3d, k4a - k4d
    Jfit, smile_rms, Jvar, resid_skew, resid_skew_mv = fit_event_J(mu3, mu3d)   # STAGE-2: fit event variance to the T3 smile
    if os.environ.get("DBG"):
        print(f"  [dbg] {event}: T2={T2:.4f} T3={T3:.4f} d={Delta:.4f} | mu3 iv={np.sqrt(k2a/T3)*100:.1f}% sk={k3a/k2a**1.5:+.2f}"
              f" | mu3diff iv={np.sqrt(max(k2d,0)/T3)*100:.1f}% sk={k3d/max(k2d,1e-12)**1.5:+.2f}"
              f" | Jfit->move {np.sqrt(max(Jfit,0))*100:.2f}% (Jvar {np.sqrt(max(Jvar,0))*100:.2f}%)  smileRMS {smile_rms*100:.2f}%  residSk {resid_skew*100:+.2f}")
    return dict(event=event, snap=snap.isoformat(), T2=T2, T3=T3, Delta=Delta, J2=J2, J3=J3, J4=J4,
                Jfit=Jfit, smile_rms=smile_rms, Jvar=Jvar, resid_skew=resid_skew, resid_skew_mv=resid_skew_mv,
                move=np.sqrt(max(J2, 0)), skew=(J3 / J2 ** 1.5 if J2 > 0 else float("nan")),
                exkurt=(J4 / J2 ** 2 if J2 > 0 else float("nan")))


if __name__ == "__main__":
    ev = sys.argv[1] if len(sys.argv) > 1 else "2024-05-01"
    tk = sys.argv[2] if len(sys.argv) > 2 else "SPX"
    bd = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    method = sys.argv[4] if len(sys.argv) > 4 else "flank"                # flank (default) | spot
    fn = build_flank if method == "flank" else build
    t0 = time.time()
    print(f"{method.upper()} bridge on the disc_SLV kernel (two-factor + leverage) -- {tk}\n")
    print(f"{'event':>11} {'snap':>11} | {'move%':>6} {'ev-skew':>8} {'ev-exkurt':>9} | {'J2':>10} {'J3':>11}")
    for e in ev.split(","):
        r = fn(e, tk, bd)
        if "err" in r:
            print(f"{e:>11}  {r['err']}"); continue
        print(f"{r['event']:>11} {r['snap']:>11} | {r['move']*100:>6.2f} {r['skew']:>8.2f} {r['exkurt']:>9.1f} | "
              f"{r['J2']:>10.2e} {r['J3']:>11.2e}")
    print(f"\nwall {time.time()-t0:.1f}s")
