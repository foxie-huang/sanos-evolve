#!/usr/bin/env python3
"""
Real-data proof-of-concept, steps 3 & 4 (paper Sec.12 real-data protocol), end-to-end on real SANOS SPX
marginals with the calibrated per-date kernel. NO new model code -- reuses the validated fusion
(discslv_slv) + real chain/leverage (slv_wire) + OU interp (slv_interp).

  (3) Marginal-propagation residual d(mu_j K_j, mu_{j+1}), per ADJACENT real-expiry pair, re-seeded from
      the real mu_j (stationary-regime lift), propagated with the exact discrete-Dupire fusion leverage
      (sigma^2_Dupire/E[nu|z], the structural Gyongy match). Reported in three units:
        IV-bp (max |dIV| on +-2.5 SD),  call-bp (RMS |dC| / F),  CDF-KS (sup |dCDF|).
  (4) Forward-start return density (law of S_T2/S_T1) + forward IV smile: propagate spot->T1, read the T1
      regime law, seed a fresh z=0 state with it, propagate the forward window T1->T2 with the forward
      leverage. Validate mass=1 and martingale fwd=1; report forward ATM vol + forward skew vs the real
      spot skews at T1,T2 (the skew roll-down).

    python3 real_pipeline_34.py            # all dates, writes real_pipeline_34_results.json
    python3 real_pipeline_34.py 2015-06-01 # one date
"""
import sys, os, json
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
sys.path.insert(0, POC); sys.path.insert(0, HERE)
import discslv_slv as S                                              # noqa: E402
from discslv_slv import (Epi_V, nu_bar, raw_increment, propagate, marginal, initial_state,   # noqa: E402
                         gm_call, gm_density_S, iv_at, atm_skew_of, sd_of, iv_err_bp, E_nu_given_z)
from discslv_2f import TwoFactorSV, recompress_2f, stationary_pi     # noqa: E402
from slv_wire import sanos_chain, leverage_at, ref_vol, solve_gbar   # noqa: E402
from slv_interp import interp_marginal, total_var                  # noqa: E402

DT = 1.0 / 52.0; NK = 16
CORRECT = True                                                       # E[nu|z] second-order Gyongy leverage correction
NAMES = ["nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]
OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
# ridge-fitted per-date theta from the 2015 joint SSR+VIX backtest
THETA = {
    "2015-06-01": [0.224, 0.410, 1.078, -0.303, 0.625, 2.184, 0.933, 2.781],
    "2015-06-15": [0.342, 0.504, 0.989, -0.283, 0.651, 2.684, 0.988, 2.375],
    "2015-07-01": [0.397, 0.677, 1.219, -0.384, 0.427, 3.143, 1.000, 1.555],
    "2015-07-15": [0.264, 0.423, 1.165, -0.325, 0.668, 2.314, 0.919, 2.745],
    "2015-08-03": [0.234, 0.425, 1.169, -0.268, 0.610, 2.268, 0.874, 2.851],
}


def gm_cdf(K, mu):
    W, MU, SG = mu
    return float(np.sum(W * norm.cdf((np.log(K) - MU) / SG)))         # P(S<=K) for the GM in S


def call_rms_bp(ma, mb, Kg, F=1.0):
    d = np.array([gm_call(K, ma) - gm_call(K, mb) for K in Kg])
    return float(np.sqrt(np.mean(d ** 2)) / F * 1e4)


def ks_cdf(ma, mb, Kg):
    return float(np.max([abs(gm_cdf(K, ma) - gm_cdf(K, mb)) for K in Kg]))


def build_kernel(date):
    chain = sanos_chain(OUT + f"/SPX-NDX-RUT-VIX_{date}.json.gz")
    sig = ref_vol(chain); kw = dict(zip(NAMES, THETA[date]))
    K = TwoFactorSV(gbar=solve_gbar(kw, sig, dt=DT), dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    return chain, sig, K, EV, nub, Vlr, tiltr


def lift_to_state(mu, K):
    """Lift a GM marginal (no regime info) to the joint state by splitting each component across regimes
    by the stationary law; recompress to NK. The natural 'structural Gyongy' seed (marginal matched)."""
    W, MU, SG = mu; pi = stationary_pi(K).ravel(); ns = K.n_s
    oW, oMU, oSG, oF, oS = [], [], [], [], []
    for c in range(len(W)):
        for f in range(K.n_f):
            for s in range(K.n_s):
                oW.append(W[c] * pi[f * ns + s]); oMU.append(MU[c]); oSG.append(SG[c]); oF.append(f); oS.append(s)
    W2 = np.array(oW); W2 /= W2.sum()
    return recompress_2f(W2, np.array(oMU), np.array(oSG),
                         np.array(oF, np.intp), np.array(oS, np.intp), NK, K.n_f, K.n_s)


def _lev_cache(chain, EV, nmax):
    """Benign discrete-Dupire leverage per dt-maturity (leverage_at: central slope + density-weighted
    smooth log-lambda + tail-freeze + safety clamp; leading order E[nu|z]~1). Built once, theta-invariant."""
    return {k: leverage_at(chain, k * DT, EV) for k in range(1, nmax + 1)}


def _lev2(lf, cur, nub):
    """Squared leverage lambda^2(mc) fed to propagate. With CORRECT, apply the E[nu|z] SECOND-ORDER Gyongy
    correction: sigma^2_LV = sigma^2_Dupire/E[nu|z], i.e. lambda^2 = leverage_at^2 / E[nu|z] (leverage_at is
    the leading order E[nu|z]~1). E[nu|0]~0.7<1 along the from-spot ATM path (leverage skew concentrates
    high-vol regimes at down-moves), so the correction lifts the under-accumulated level. Clamped for stability."""
    if not CORRECT:
        return lambda mc: lf(mc) ** 2
    return lambda mc: float(np.clip(lf(mc) ** 2 / np.clip(E_nu_given_z(mc, cur, nub), 0.3, 3.0), 1e-3, 9.0))


def step3(date):
    chain, sig, K, EV, nub, Vlr, tiltr = build_kernel(date)
    nmax = int(round(chain[-1][0] / DT)); LEV = _lev_cache(chain, EV, nmax)
    res = []
    for j in range(len(chain) - 1):
        Tj, muj = chain[j]; Tjp1, mujp1 = chain[j + 1]
        kj, kjp1 = int(round(Tj / DT)), int(round(Tjp1 / DT))
        if kjp1 <= kj:
            continue                                                  # same dt-bin: no propagation step
        st = lift_to_state(muj, K)                                    # re-seed from the real mu_j
        for k in range(kj + 1, kjp1 + 1):
            lf = LEV.get(k) or leverage_at(chain, k * DT, EV)
            st, _ = propagate(K, st, _lev2(lf, st, nub), EV, nub, Vlr, tiltr, NK)
        m = marginal(st); Kg = np.exp(sd_of(mujp1) * np.linspace(-2.5, 2.5, 41))
        av_m, sk_m = atm_skew_of(m, Tjp1); av_r, sk_r = atm_skew_of(mujp1, Tjp1)
        res.append(dict(Tj=round(Tj, 4), Tjp1=round(Tjp1, 4), dte=round(Tjp1 * 365), nsteps=kjp1 - kj,
                        atm_bp=round((av_m - av_r) * 1e4, 1), skew_m=round(sk_m, 3), skew_r=round(sk_r, 3),
                        iv_bp=round(iv_err_bp(m, mujp1, Tjp1), 2),
                        call_bp=round(call_rms_bp(m, mujp1, Kg), 2), ks=round(ks_cdf(m, mujp1, Kg), 5)))
    return res


def _propagate_window(K, st, chain, EV, nub, Vlr, tiltr, k_start, nsteps, LEV):
    for i in range(nsteps):
        k = k_start + i + 1
        lf = LEV.get(k) or leverage_at(chain, k * DT, EV)
        st, _ = propagate(K, st, _lev2(lf, st, nub), EV, nub, Vlr, tiltr, NK)
    return st


def step4(date, windows):
    chain, sig, K, EV, nub, Vlr, tiltr = build_kernel(date)
    nmax = max(int(round(T2 / DT)) for _, T2 in windows); LEV = _lev_cache(chain, EV, nmax)
    out = []
    for (T1, T2) in windows:
        n1 = int(round(T1 / DT)); nf = int(round((T2 - T1) / DT))
        if nf < 1 or n1 < 1:
            continue
        st = _propagate_window(K, initial_state(K), chain, EV, nub, Vlr, tiltr, 0, n1, LEV)  # spot -> T1
        W, MU, SG, F, Sreg = st                                        # regime law at T1
        reg = np.zeros((K.n_f, K.n_s))
        for c in range(len(W)):
            reg[F[c], Sreg[c]] += W[c]
        reg /= reg.sum()
        fW, fMU, fSG, fF, fS = [], [], [], [], []                     # fresh z=0 state, T1 regime law
        for f in range(K.n_f):
            for s in range(K.n_s):
                fW.append(reg[f, s]); fMU.append(0.0); fSG.append(1e-4); fF.append(f); fS.append(s)
        fst = (np.array(fW), np.array(fMU), np.array(fSG), np.array(fF, np.intp), np.array(fS, np.intp))
        fst = _propagate_window(K, fst, chain, EV, nub, Vlr, tiltr, n1, nf, LEV)   # forward window T1 -> T2
        fm = marginal(fst); Tw = T2 - T1
        W2, MU2, SG2 = fm
        mass = float(W2.sum()); fwd = float(np.sum(W2 * np.exp(MU2 + 0.5 * SG2 ** 2)))
        atm, skew = atm_skew_of(fm, Tw)
        fv_model = total_var(fm)                                      # model forward-return variance
        fv_real = max(total_var(interp_marginal(chain, T2)) - total_var(interp_marginal(chain, T1)), 1e-9)
        anch_atm = atm * np.sqrt(fv_real / max(fv_model, 1e-12))      # LEVEL pinned to the real variance curve
        real_fwd_vol = np.sqrt(fv_real / Tw)                          # real forward-variance vol (statics)
        ks = np.round(np.linspace(-0.08, 0.08, 9), 3)
        smile = iv_at(fm, Tw, ks)
        sk1 = atm_skew_of(interp_marginal(chain, T1), T1)             # real spot skews for context
        sk2 = atm_skew_of(interp_marginal(chain, T2), T2)
        out.append(dict(T1=round(T1, 4), T2=round(T2, 4), Tw=round(Tw, 4),
                        mass=round(mass, 6), fwd=round(fwd, 6),
                        fwd_atm_vol=round(atm, 4), anch_atm_vol=round(anch_atm, 4),
                        real_fwd_vol=round(real_fwd_vol, 4), fwd_skew=round(skew, 3),
                        spot_skew_T1=round(sk1[1], 3), spot_skew_T2=round(sk2[1], 3),
                        ks=ks.tolist(), smile=[round(float(v), 4) for v in smile]))
    return out


if __name__ == "__main__":
    dates = [sys.argv[1]] if len(sys.argv) > 1 else list(THETA)
    WK = 1.0 / 52.0
    windows = [(4 * WK, 8 * WK), (4 * WK, 13 * WK), (13 * WK, 26 * WK), (26 * WK, 52 * WK)]  # 1m->2m,1m->3m,3m->6m,6m->1y
    allres = {}
    for date in dates:
        print(f"\n{'='*70}\n{date}\n{'='*70}", flush=True)
        s3 = step3(date)
        print(" STEP 3  marginal-propagation residual  d(mu_j K_j, mu_{j+1})")
        print(f"   {'Tj->Tjp1':>14}{'dte':>5}{'IV-bp':>8}{'call-bp':>9}{'CDF-KS':>9}")
        for r in s3:
            print(f"   {r['Tj']:.3f}->{r['Tjp1']:.3f} {r['dte']:>5}{r['iv_bp']:>8.1f}{r['call_bp']:>9.1f}{r['ks']:>9.4f}")
        ivs = [r["iv_bp"] for r in s3]
        print(f"   IV-bp: median {np.median(ivs):.1f}  mean {np.mean(ivs):.1f}  max {np.max(ivs):.1f}")
        s4 = step4(date, windows)
        print(" STEP 4  forward-start density + forward IV")
        print(f"   {'window':>12}{'mass':>7}{'fwd':>7}{'rawATM':>8}{'anchATM':>8}{'realfwd':>8}{'fwdskew':>8}{'spotsk1/2':>14}")
        for r in s4:
            lab = f"{r['T1']*52:.0f}w->{r['T2']*52:.0f}w"
            print(f"   {lab:>12}{r['mass']:>7.4f}{r['fwd']:>7.4f}{r['fwd_atm_vol']:>8.4f}{r['anch_atm_vol']:>8.4f}"
                  f"{r['real_fwd_vol']:>8.4f}{r['fwd_skew']:>8.2f}  {r['spot_skew_T1']:>5.2f}/{r['spot_skew_T2']:<5.2f}")
        allres[date] = dict(theta=dict(zip(NAMES, THETA[date])), step3=s3, step4=s4)
    with open(os.path.join(HERE, "real_pipeline_34_results.json"), "w") as f:
        json.dump(allres, f, indent=1)
    print(f"\nwrote real_pipeline_34_results.json ({len(allres)} dates)")
