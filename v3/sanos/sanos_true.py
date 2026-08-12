#!/usr/bin/env python3
"""SANOS as specified, not the per-expiry variant: the joint LP of Buehler et al. section 4.2.

The implementation in v2/poc/sanos_lp.py fits each expiry independently with only
{q>=0, 1'q=1, K'q=1}. SANOS's production program (MDL) is a SINGLE linear program over all
expiries at once, carrying the calendar constraint

    u_j >= r_j ,      u_j := U_j q_j ,   r_j := R_j q_{j-1}

with U, R the discrete call transforms. Taking the paper's omega = 0 (the setting it says ensures
the marginals constitute a martingale density, i.e. strict absence of arbitrage in time), the
transform is the intrinsic map U^{l,i} = (K^i - K^l)^+, so on a common model-strike grid the
constraint reads U q_j >= U q_{j-1}: the discrete call curve is non-decreasing in maturity. That IS
convex order for atomic measures, imposed by construction rather than checked afterwards.

Other things the source specifies that the per-expiry code drops, all restored here:
  * model strikes are NOT the market strikes -- SANOS adds strikes where market strikes are further
    apart than dx, plus boundary strikes outside the observed range;
  * implied variances must be increasing, 0 < V_1 < ... < V_M (Section 4.2 states this as an
    ASSUMPTION on the inputs and gives no remedy; screen_monotone drops violators);
  * bid and ask are QUOTED PRICES, with A > B assumed strictly;
  * objective = band violation + eps*|mid error| with eps = 1e-8 -- Remark 4.1's third option,
    which the paper calls "our recommended default setting".

Two places where the paper offers a choice, and which one is taken:
  * WEIGHTS. Section 4.2: "typically the inverse of the prevailing bid/ask spread, or the inverse of
    Vega to approximate a fit in implied volatilities." We use INVERSE VEGA. It is bounded here by
    construction, since the screen already enforces Vega/sqrt(T) >= 0.1%. The spread weighting (34)
    is not usable without invented repair: deep-OTM price spreads reach ~1e-6, so 1/(A-B) spans six
    decades, the objective spans ~14 against eps=1e-8, and the solve returns optimal_inaccurate.
  * ETA. 0.11, one of the paper's own Figure-8 values, chosen on SANOS's own acceptance metric:
    99% of options in-band here against 69% at the Section-5 example value of 0.25. THIS MODULE IS
    THE SINGLE SOURCE OF TRUTH for eta -- sanos_leverage, static_payload and the probes all re-export
    it rather than carrying their own literal, because they did carry their own and disagreed: the
    module default was 0.25 while every production path passed 0.11, so the LP probes were
    characterising a bandwidth production never used.

Not applied: the p.2 illustration screen (>=100 lots traded, >=20 active options per expiry), which
belongs to a different figure; and the numerical example's cap of "1000 options chosen by closeness
to ATM".

Model expiries are taken to coincide with the admitted market expiries, which is the simplification
the paper itself makes in section 4.3, so the alpha-blend matrices collapse to a single C_j.

    python3 sanos_true.py NDX 2020-06-01
"""
import os, sys, time
import numpy as np
import cvxpy as cp
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths as _P                        # noqa: E402
HERE = _P.DATA                             # code moved; fits/caches/records did not

sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "v2", "poc")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "v2", "data")))
ORATS = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))

from sanos_lp import forward, bs_call                              # noqa: E402
from orats_sanos import orats_chain_df                             # noqa: E402

ETA = 0.11          # THE eta. See the docstring: 99% in-band vs 69% at SANOS's 0.25 example value.
                    # Every other module re-exports this; do not re-declare it anywhere else.
DX = 0.01           # max model-strike spacing (SANOS: "adds strikes when market strikes are
                    # further apart than some maximum dx")
PAD = 0.15          # boundary strikes K_min/K_max this far outside the observed range
VEGA_MIN = 1e-3     # SANOS: Vega/sqrt(T) >= 0.1%
N_ATM_SD = 2.0      # SANOS section 5: options within 2 ATM implied-vol standard deviations
EPS = 1e-8          # SANOS's recommended mid-fit weight
SOLVER = getattr(cp, os.environ.get("LPSOLVER", "HIGHS"))   # the paper only says "an LP"
TOL_C = 1e-12       # drop pricing-matrix entries below this fraction of the row max


def forward_df(g):
    """(F, DF) by the same parity regression as sanos_lp.forward, which returns only F.

    DF is needed because SANOS works in normalised undiscounted prices with C(T,0)=1 and "pure"
    strikes k = K/F: a quoted call enters as C_mkt/(DF*F), a quoted put by parity as
    P_mkt/(DF*F) + (1-k).
    """
    calls = g[g.type == "call"].groupby("strike").first()
    puts = g[g.type == "put"].groupby("strike").first()
    common = calls.index.intersection(puts.index)
    K = common.values.astype(float)
    d = (0.5 * (calls.loc[common, "bid"] + calls.loc[common, "ask"])
         - 0.5 * (puts.loc[common, "bid"] + puts.loc[common, "ask"])).values
    lo, hi = np.quantile(K, [0.25, 0.75]); m = (K >= lo) & (K <= hi)
    if m.sum() >= 3:
        K, d = K[m], d[m]
    b0, a0 = np.linalg.lstsq(np.vstack([np.ones_like(K), K]).T, d, rcond=None)[0]
    DF = -a0
    if not (1e-3 < DF <= 1.0 + 1e-6):
        DF = 1.0
    F = float(b0 / DF) if DF > 0 else float(K[np.argmin(np.abs(d))])
    return F, float(DF)


def market_slice(g):
    """One expiry's admitted quotes, following SANOS as written.

    Screen = the paper's numerical-example screen: OTM, Vega/sqrt(T) >= 0.1%, within 2 ATM implied
    -volatility standard deviations. The p.2 illustration screen (>=100 lots traded, >=20 active
    options per expiry) belongs to a different figure and is NOT applied. NOTE the example also
    caps the set at "1000 options ... chosen by closeness to ATM"; that cap is not applied here.

    The band is the QUOTED PRICE band -- Section 4.2 supplies bid B and ask A as prices and assumes
    A > B strictly -- mapped into normalised undiscounted call units, calls directly and puts by
    parity. An earlier version built the band from ivBid/ivAsk, which then forced dropping any
    quote lacking a two-sided IV band; neither of those is in the paper.
    """
    F, DF = forward_df(g); tau = float(g.dte.iloc[0]) / 365.0
    sc = DF * F
    rows = []
    for _, r in g.iterrows():
        otm = (r.type == "call" and r.strike >= F) or (r.type == "put" and r.strike < F)
        if not (otm and r.bid > 0 and r.ask > r.bid and np.isfinite(r.impliedVolatility)
                and 0.02 < r.impliedVolatility < 1.5):
            continue                                   # A > B strictly, per Section 4.2
        k = float(r.strike) / F
        shift = 0.0 if r.type == "call" else (1.0 - k)  # parity C = P + DF*(F-K)
        rows.append((k, float(r.impliedVolatility),
                     float(r.bid) / sc + shift, float(r.ask) / sc + shift))
    if not rows:
        return None
    rows = sorted(set(rows))
    kap = np.array([x[0] for x in rows]); iv = np.array([x[1] for x in rows])
    c_lo = np.array([x[2] for x in rows]); c_hi = np.array([x[3] for x in rows])
    lk = np.log(kap); w0 = iv * np.sqrt(tau)
    vega = (np.asarray(bs_call(1.0, kap, w0 + 1e-4)) - np.asarray(bs_call(1.0, kap, w0))) / 1e-4
    atm0 = iv[np.argmin(np.abs(lk))]
    keep = (vega >= VEGA_MIN) & (np.abs(lk) <= N_ATM_SD * atm0 * np.sqrt(tau))
    if keep.sum() < 4:
        return None
    kap, iv, lk, vega = kap[keep], iv[keep], lk[keep], vega[keep]
    c_lo, c_hi = c_lo[keep], c_hi[keep]
    atm = iv[np.abs(lk) < 0.05]
    atm_iv = float(np.median(atm)) if len(atm) else float(iv[np.argmin(np.abs(lk))])
    return dict(tau=tau, kappa=kap, atm_iv=atm_iv, F=F, DF=DF, vega=vega,
                c_mid=0.5 * (c_lo + c_hi), c_lo=c_lo, c_hi=c_hi)


def model_grid(slices):
    """One common model-strike grid at spacing dx, spanning the observed range with padding.

    Deliberately NOT the union of market strikes: SANOS's model strikes "may or may not include any
    market strikes", and market strikes enter only through the pricing matrix C_j. Unioning them in
    makes the grid the size of the whole cross-section (2909 strikes on NDX 2020, 81k LP variables,
    a U with 1.1e8 nonzeros) for no modelling gain.
    """
    allk = np.concatenate([s["kappa"] for s in slices])
    lo, hi = max(allk.min() - PAD, 1e-3), allk.max() + PAD
    return np.linspace(lo, hi, int(np.ceil((hi - lo) / DX)) + 1)


def screen_monotone(slices, margin=1.001):
    """SANOS requires 0 < V_1 < ... < V_M of the admitted expiries. Where the extracted ATM total
    variance breaks it, DROP the offending expiries rather than coerce V upward.

    Coercion (the previous behaviour, V[j] = max(V[j], V[j-1]*margin)) keeps a quote set that fails
    SANOS's own precondition and hides that it did; worse, each coerced value feeds the next
    comparison, so the distortion compounds along the term structure. On SPX 2023-06-01 that meant
    six coercions of up to 18% of total variance, concentrated at 16-61 days, and a fit that
    collapsed at long maturities. Monotonicity is a screen on the data, so failing it is grounds for
    exclusion.

    Returns the indices of the LARGEST subset satisfying the condition (longest strictly increasing
    subsequence), so the fewest expiries are lost -- a greedy forward pass would discard a whole run
    of good expiries after one high outlier.
    """
    V = np.array([s["atm_iv"] ** 2 * s["tau"] for s in slices], float)
    M = len(V)
    best = [1] * M; prev = [-1] * M
    for j in range(M):
        for i in range(j):
            if V[j] > V[i] * margin and best[i] + 1 > best[j]:
                best[j] = best[i] + 1; prev[j] = i
    j = int(np.argmax(best)); keep = []
    while j >= 0:
        keep.append(j); j = prev[j]
    return np.array(keep[::-1], dtype=int)


def fit_joint(slices, eta=ETA, verbose=True):
    """The MDL program: one LP over all expiries with u_j >= r_j.

    Expiries failing the ATM-total-variance monotonicity screen are dropped, not coerced.
    Returns (K, V, qs, keep) with `keep` the surviving indices into the input `slices`.
    """
    keep = screen_monotone(slices)
    if verbose and len(keep) < len(slices):
        lost = sorted(set(range(len(slices))) - set(keep.tolist()))
        print(f"  monotonicity screen: dropped {len(lost)}/{len(slices)} expiries "
              f"(tau*365 = {[round(slices[i]['tau']*365) for i in lost]})")
    slices = [slices[i] for i in keep]
    K = model_grid(slices)
    M = len(slices)
    V = np.array([s["atm_iv"] ** 2 * s["tau"] for s in slices], float)
    # omega=0 makes the transform the intrinsic map U^{l,i} = (K^i - K^l)^+, so
    #     (U q)_l = sum_{i>l} (K^i - K^l) q_i = S2_l - K^l * S1_l ,
    #     S1_l = sum_{i>l} q_i ,   S2_l = sum_{i>l} K^i q_i .
    # S1 and S2 obey one-step recursions, so carrying them as variables expresses exactly the same
    # calendar constraint with O(n) nonzeros per expiry instead of O(n^2). Forming U densely put
    # ~3.9M of 6.1M nonzeros into the program and HiGHS returned "Model status: Not Set" on the
    # three largest cross-sections. The feasible set is unchanged.
    n = len(K)
    q = [cp.Variable(n, nonneg=True) for _ in range(M)]
    S1 = [cp.Variable(n) for _ in range(M)]
    S2 = [cp.Variable(n) for _ in range(M)]
    cons, obj = [], 0
    call = []
    for j in range(M):
        cons += [S1[j][n - 1] == 0, S2[j][n - 1] == 0,
                 S1[j][:-1] == S1[j][1:] + q[j][1:],
                 S2[j][:-1] == S2[j][1:] + cp.multiply(K[1:], q[j][1:])]
        call.append(S2[j] - cp.multiply(K, S1[j]))
    for j, s in enumerate(slices):
        cons += [cp.sum(q[j]) == 1, K @ q[j] == 1]
        if j > 0:
            cons += [call[j] >= call[j - 1]]             # <-- the calendar constraint
        Cj = np.asarray(bs_call(K[None, :], s["kappa"][:, None], np.sqrt(eta * V[j])))
        # C_j is effectively BANDED: with a component variance of eta*V_j, a basis anchor far from
        # a market strike prices in the denormal range (values to 1e-314 were observed). Handing
        # those to the solver is what broke the large cross-sections -- HiGHS reported 1.18M of
        # 6.1M nonzeros at |value| <= 1e-9 and returned "Model status: Not Set" on SPX 2022/2024
        # and NDX 2024. Dropping entries that cannot affect any price to exact zero is a storage
        # decision, not a modelling one: the largest discarded entry is below 1e-12 of that
        # strike's own price scale.
        Cj[Cj < TOL_C * np.maximum(Cj.max(axis=1, keepdims=True), 1e-300)] = 0.0
        c = sparse.csr_matrix(Cj) @ q[j]
        # Section 4.2 offers two weightings: "the inverse of the prevailing bid/ask spread, or
        # the inverse of Vega to approximate a fit in implied volatilities". Use INVERSE VEGA: it
        # is the paper's own alternative, and it is bounded by construction because the screen
        # already enforces Vega/sqrt(T) >= 0.1%. The spread weighting (34) is unusable here without
        # invented repair -- deep-OTM price spreads reach ~1e-6, so 1/(A-B) spans six decades, the
        # objective spans ~14 against eps=1e-8, and the solve returned optimal_inaccurate in 170s.
        w = 1.0 / np.maximum(s["vega"], VEGA_MIN)
        obj += cp.sum(cp.multiply(w, EPS * cp.abs(s["c_mid"] - c)
                                  + cp.pos(c - s["c_hi"]) + cp.pos(s["c_lo"] - c)))
    t0 = time.time()
    prob = cp.Problem(cp.Minimize(obj), cons)
    # This is a pure LP; a simplex/barrier LP solver handles it far better than a conic
    # interior-point one (CLARABEL: 40s and optimal_inaccurate on NDX 2020).
    # Solver choice is ours -- Section 4.2 says only "a linear or quadratic programming framework".
    # HiGHS is fast and exact on most cross-sections but returns "Model status: Not Set" on the
    # largest (SPX 2022/2024, NDX 2024). That is a solver limitation, NOT a property of the data:
    # the feasible set here depends on no market data at all, and the band enters only a convex
    # piecewise-linear objective, so a finite minimiser always exists. Inconsistent quotes simply
    # raise the optimal value. Fall back rather than change the program.
    _err = None
    for _sv in (SOLVER, cp.CLARABEL):
        try:
            prob.solve(solver=_sv, verbose=bool(os.environ.get("LPVERBOSE")))
            if q[0].value is not None:
                break
        except Exception as e:                      # noqa: BLE001 - try the next solver
            _err = e
    if q[0].value is None and _err is not None:
        raise _err
    if verbose:
        print(f"  LP: {M} expiries x {len(K)} model strikes = {M*len(K)} vars, "
              f"status {prob.status}, {time.time()-t0:.1f}s")
    if q[0].value is None:
        return None
    return K, V, [np.asarray(x.value) for x in q], keep


if __name__ == "__main__":
    tk = sys.argv[1] if len(sys.argv) > 1 else "NDX"
    date = sys.argv[2] if len(sys.argv) > 2 else "2020-06-01"
    df = orats_chain_df(f"{ORATS}/SPX-NDX-RUT-VIX_{date}.json.gz", tk)
    slices, dtes = [], []
    for dte, g in sorted(df.groupby("dte")):
        s = market_slice(g)
        if s is not None:
            slices.append(s); dtes.append(int(dte))
    print(f"{tk} {date}: {len(slices)} expiries admitted (dte {dtes})")
    out = fit_joint(slices)
    if out is None:
        print("  LP failed"); sys.exit(1)
    K, V, qs, keep = out
    slices = [slices[i] for i in keep]; dtes = [dtes[i] for i in keep]   # realign to the survivors
    print(f"\n{'dte':>5s} {'n_mkt':>6s} {'band viol':>11s} {'eff comps':>10s} {'V_j':>10s}")
    print("-" * 50)
    for j, (s, dte, qj) in enumerate(zip(slices, dtes, qs)):
        Cj = np.asarray(bs_call(K[None, :], s["kappa"][:, None], np.sqrt(ETA * V[j])))
        c = Cj @ qj
        viol = float(np.max(np.maximum(np.maximum(s["c_lo"] - c, c - s["c_hi"]), 0))) * 1e4
        print(f"{dte:5d} {len(s['kappa']):6d} {viol:10.2f}bp {int((qj > 1e-4).sum()):10d} {V[j]:10.5f}")
    # SANOS's own acceptance metric (section 5): share of options fitted inside bid/ask, and for
    # those outside, the median miss as a fraction of half-spread. It reports 91.4% and 21%.
    n_in = n_tot = 0; miss = []
    for j, (s_, dte, qj) in enumerate(zip(slices, dtes, qs)):
        Cj = np.asarray(bs_call(K[None, :], s_["kappa"][:, None], np.sqrt(ETA * V[j])))
        c = Cj @ qj
        over = np.maximum(np.maximum(s_["c_lo"] - c, c - s_["c_hi"]), 0.0)
        half = np.maximum((s_["c_hi"] - s_["c_lo"]) / 2.0, 1e-12)
        n_tot += len(c); n_in += int((over <= 0).sum())
        miss += list(over[over > 0] / half[over > 0])
    print(f"\nSANOS acceptance metric: {100*n_in/n_tot:.1f}% of options inside bid/ask "
          f"(SANOS reports 91.4%)")
    if miss:
        print(f"  of those outside, median miss = {100*float(np.median(miss)):.0f}% of half-spread "
              f"(SANOS reports 21%)")
    U = np.maximum(K[None, :] - K[:, None], 0.0)
    worst = min(float((U @ (qs[j] - qs[j - 1])).min()) for j in range(1, len(qs)))
    tol = 1e-7          # LP solver tolerance; the constraint is imposed exactly in the program
    print(f"\nconvex order across all {len(qs)-1} adjacent pairs: worst U(q_j - q_j-1) = {worst:.3e}"
          f"   ({'SATISFIED to solver tolerance' if worst > -tol else 'VIOLATED'})")
