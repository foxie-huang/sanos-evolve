#!/usr/bin/env python3
"""Refit SPX dates on `kernel_fast`, MPS, VOVLEV=1. Mirrors the reference objective exactly.

Objective is `calibrate_joint_torch.make_residual` term for term: equal-percent SSR residuals,
vov residuals at wv = w_vov*sqrt(n_ssr/n_vov), and the relative ridge toward the anchor. MONO_PEN
is 0, as fit_norm_panel.py sets it. The ONLY thing that differs from a reference fit is the kernel
backend -- so a difference in the answer is attributable to the backend and nothing else.

RUN IT UNBUFFERED. Every iteration prints a timestamped line; without -u nothing appears until the
process exits, which has already cost two blind hours in this project.

    python3 -u refit.py 2016-06-01 2018-06-01 2022-06-01 2024-06-03
    DEV=cpu python3 -u refit.py 2022-06-01           # CPU comparison leg
    FTOL=5e-3 XTOL=1e-4 TAG=_tol python3 -u refit.py # tolerances at the measured staircase scale
    KAPS_FREE=1 TAG=_ks python3 -u refit.py          # kap_s as an 8th parameter (see KAPS_FREE)
"""
import os
import sys
import json
import time

DATES = [a for a in sys.argv[1:] if a[:2] == "20"] or \
        ["2016-06-01", "2018-06-01", "2022-06-01", "2024-06-03"]
DEV = os.environ.get("DEV", "mps")
MAXNFEV = int(os.environ.get("MAXNFEV", "40"))
W_VOV = float(os.environ.get("W_VOV", "0.8"))
# SEW=1 weights the SSR residuals by the inverse of their MEASURED relative HAC error, instead of
# treating all five tenors as equally reliable.
#
# WHY. The objective builds `(s - emp)/emp` -- equal weight per tenor -- while `ssr_joint_hac.json`
# has carried a per-tenor `se_joint` all along, unused. At NDX 2017 the 1wk se is 24% of the target
# and the 13wk se is 5.7%, so the fit chases the least-determined cell as hard as the best-determined
# one and pays for it with a 3.2-sigma miss at 13wk (6e.39).
#
# IT REDISTRIBUTES, IT DOES NOT REWEIGHT THE BLOCK. Weights are normalised to mean(w^2)=1, so the SSR
# block's expected squared magnitude is unchanged and the SSR-vs-vov balance is untouched. Without
# that, dividing by se would inflate the block ~10-50x and silently swamp the vov residuals, making
# the A/B two-variable.
#
# CAVEAT: `se_joint` is the OLS estimator's error. Where the FITTED target is the Huber-robust beta
# (NDX 2017 only, 24.9% apart) the weight and the target come from different estimators. It is the
# only uncertainty measure available, and using it beats assuming equal reliability.
SEW = os.environ.get("SEW", "0") == "1"
# VOVLEV=0 uses the UNLEVERED VIX readout (lam_fns=None), i.e. the `_fk` panel's objective. Note
# VOVLAMTEN and LADDER are then irrelevant -- the unlevered path is closed form and never touches
# lambda -- so the frozen static payload is fine and NOFREEZE is unnecessary.
VOVLEV = os.environ.get("VOVLEV", "1") == "1"
# Termination tolerances. Defaults are the production values -- change nothing unless asked.
#
# Both are set far inside the objective's own resolution, which ftol_probe.py measured: perturbing
# one theta coordinate by a relative 1e-8 to 1e-4 moves the cost by 1e-3 to 1.6e-2 RELATIVE, and
# NON-MONOTONICALLY (2016 responds less at 1e-4 than at 1e-5). A smooth objective would give
# dcost proportional to eps; this is a band-flip staircase across the whole tested range.
#
# Consequences, both confirmed: ftol=1e-6 never fires -- all eight recorded fits end status=3, i.e.
# XTOL -- and xtol=1e-6 fires because trf's predicted reduction (from the stop-gradient Jacobian,
# which cannot see the jumps) keeps missing the measured one, so the trust radius collapses. That is
# "stopped moving inside the staircase", not "found an optimum".
#
# TAG names the output so an A/B does not overwrite the baseline: fit_kf{TAG}_{date}.json
# KAPS_FREE=1 fits kap_s as an 8th parameter instead of pinning it at C.KAP_S_FIXED = 0.9956.
#
# WHY THIS EXISTS. The pin is justified on identification grounds -- a 37-week window cannot resolve
# a slow timescale -- but the _fk panel FITTED kap_s on 9 dates and the result is bimodal with very
# tight clusters, not the wandering of an unidentified parameter: 2016/17/18/22/24 land in
# 0.98212-0.98485 (half-life 38-45 wk) and 2012/19/20/21 in 0.93666-0.95058 (11-14 wk). All are
# interior to HI_N8 = 0.998, so none is railing. The PIN at 0.9956 is a 157-week (3.0 yr) half-life:
# 5.5x the mean fitted value and 3.5x the longest one ever fitted. The identification argument rules
# out 157 weeks; it does not obviously rule out 40.
#
# WHY IT MATTERS FOR VOV. kap_s sets the long-end decay of the VIX term structure. Pinned at a
# 157-week half-life the slow factor barely mean-reverts over the 24-37 step propagation VOVLEV=1
# runs, so the term structure has no reason to come down -- and the 2016 VOVLEV=1 fit goes flat at
# ~0.92 from 78 days out while the target decays to 0.62.
#
# The 2x2 (kap_s free/pinned) x (VOVLEV 0/1) had exactly one empty cell, VOVLEV=1 + free, because
# the free-kap_s panel predates VOVLEV=1. This fills it. `th9` and `solve_gbar` already accept a
# length-8 theta, so nothing downstream changes.
KAPS_FREE = os.environ.get("KAPS_FREE", "0") == "1"
FTOL = float(os.environ.get("FTOL", "1e-6"))
XTOL = float(os.environ.get("XTOL", "1e-6"))
TAG = os.environ.get("TAG", "")
WARM = os.environ.get("WARM", "")
PIN = os.environ.get("PIN", "")
BOX = os.environ.get("BOX", "")
# SEED=<name>=<value>[;...] overrides coordinates of the STARTING POINT (cold seed or warm
# start -- applied AFTER WARM) without moving the ridge anchor.
#
# WHY. All three cold seeds (ts/dense/low) carry the SAME kap_s = 0.98 = 34.3wk -- there is no seed
# diversity in that coordinate. At 2016 the full kap_s profile (6 free params refitted per rung) puts
# the optimum at 16-32wk with DATA 0.0261-0.0300, i.e. the seed starts essentially AT the answer --
# and the free fit still walks up to the 0.998 ceiling and ends at 0.04409, 40% WORSE. 5 of 9 dates
# move up from that seed (2016/17/18/22/24 -> 137-343wk), 4 move down.
#
# Applied AFTER `anc = x0.copy()` AND after WARM, so the anchor stays at the published seed: this isolates WHERE THE
# SEARCH STARTS from WHAT IT IS PULLED TOWARD. (The kap_s ridge contributes ~4e-6 either way, so the
# distinction is bookkeeping here, but it is the error that invalidated the first ridge experiment.)
SEED0 = os.environ.get("SEED", "")
# MONOPEN=<w> adds a one-sided SHAPE penalty on the SSR term structure. Two forms:
#
#   default (RELATIVE):  w * relu( (ds_model - ds_target) / target )
#     "the model must not rise MORE than the data does". Where the target falls this is the plain
#     monotonicity penalty; where the target rises it permits the model to rise as far.
#   MONOGATE=1 (literal): w * relu(ds_model / target), applied ONLY on dates whose target is
#     monotone non-increasing.
#
# WHY THE RELATIVE FORM IS THE DEFAULT. R(T) monotone is a SINGLE-timescale property, not a
# requirement -- SSR = beta/skew is a ratio of two decaying quantities and rises wherever skew decays
# faster, and a two-factor model with a frozen slow factor is EXPECTED to turn up at the long end
# (each factor's own ratio runs 2->1 on its own timescale; at long T the slower one, still in its
# short-T regime where its ratio is ~2, dominates). fit_norm_panel.py:15 rejects the plain penalty
# for exactly this reason: SPX realised SSR RISES at 1m->2m in 3 of 9 years, so penalising any rise
# forces the model away from the data. The relative form cannot do that.
#
# WHY NOT GATE ON TARGET MONOTONICITY. The test would run on point estimates whose joint-HAC bands
# are +-0.15..0.30: at 2012 the target rises but the floor is 7.0%, at 2018 it is monotone but the
# floor is 10.6% -- neither is established. A binary switch on that test puts band noise into the
# objective. MONOGATE=1 is provided to measure the difference, not as the recommended setting.
#
# The SSR block's own residuals only constrain the LEVEL; at 2016/2018 the model sits inside the band
# and still bends the wrong way, which is what this term can see and they cannot.
MONOPEN = float(os.environ.get("MONOPEN", "0") or 0)
MONOGATE = os.environ.get("MONOGATE", "0") == "1"
NORIDGE = os.environ.get("NORIDGE", "")
# TICKER=NDX fits the NDX panel instead of SPX. Everything else is unchanged: same objective, same
# design constants, same protocol. Two things differ and both come from the DATA, not the code:
#   * the vov block is 2 tenors (30d, 90d liquid anchors) against SPX's 9-12, so the per-point vov
#     weight wv = W_VOV*sqrt(n_ssr/n_vov) is larger by construction;
#   * `end_to_end.ctx_rebuilt` routes to `calibrate_ndx.build_ctx_ndx`.
# Output is suffixed `_ndx` so an NDX run can NEVER overwrite a shipped SPX fit by tag collision.
TICKER = os.environ.get("TICKER", "SPX").upper()
# Ticker suffix, MODULE level. It used to be computed only inside the save block, so the WARM path
# built `fit_kf<tag>_<date>.json` with no suffix -- which on NDX silently resolved to the SPX file of
# the same tag and seeded an NDX fit from SPX parameters (ns 0.507 vs 0.208, a 2.4x gap). That is what
# the NDX _dw9 run of 6e.29a actually did; its log reads "warm start from fit_kf_n9_2012-06-01.json".
SFX = "" if TICKER == "SPX" else f"_{TICKER.lower()}"
# VOVMNY=1 makes the vov comparison MONEYNESS-MATCHED. Both sides currently take ATM implied vol AT
# THEIR OWN FORWARD, and the model's VIX forward runs ~11.7% below the market's (handoff 6e.27). The
# VIX smile slopes UP (d(IV)/dlogK = +0.35..+0.99, mean +0.6) BECAUSE VOL-OF-VOL RISES WITH VIX LEVEL,
# so the instrument reads the model's dispersion at a displaced point of a level-dependent smile --
# worth -9.1% of ATM vol on average (6e.27e), the same order as the whole vov residual. nu is
# identified through this block, so nu inherits that bias.
#
# The correction translates the MODEL's ATM vol to the market's forward:
#     vov_adj = iv_model + slope * log(F_market / F_model)
# MODEL SIDE, NOT TARGET SIDE, deliberately: reading the market smile AT the model's forward would
# make the TARGET depend on theta, and since the smile slopes up the optimiser could lower F_model to
# drag the target down after it -- a self-referential degeneracy. This leaves the target untouched, so
# the fit is IDENTICAL wherever the forwards agree.
#
# *** DO NOT USE. OFF PERMANENTLY. Kept only so nobody rebuilds it. (2026-08-10) ***
#
# THREE reasons, any one sufficient:
#
# 1. IT IS OUT OF SCOPE. It imports the VIX FORWARD LEVEL into the objective, which makes this a
#    joint SPX/VIX calibration. This project is an SPX model whose spot-vol dynamics are constrained
#    by SSR, with VIX entering ONLY as a softly weighted ATM readout that identifies the vol-of-vol
#    amplitude. The v3 paper says so in its own contributions. VIX futures are not a target.
#
# 2. IT CREATES A FLAT DIRECTION. The correction is delta = slope*log(F_mkt/F(theta)) and F depends
#    on theta. LOWERING nu (smaller iv_raw) and RAISING F (smaller delta) reduce the corrected
#    reading IDENTICALLY, so they are substitutes and the split between them is arbitrary. Measured
#    over 9 dates: |dF| up to +27%, d(nu_s) from -92% to +91%, and corr(dF, d nu_s) = +0.001. The
#    resulting nu_s values are a point on a flat direction, not estimates. Both forms fail this way
#    -- per-tenor (`_mny9`) and level-only (`_lvl9`) -- because both make delta theta-dependent.
#
# 3. IT CHANGES NOTHING MEASURABLE. SSR -- the only block untouched by the correction, and the only
#    one with error bars -- is a dead heat: mean RMS 1.95% -> 1.92% over the panel, everything far
#    inside HAC floors of 5-16%.
#
# The observation that motivated it IS real: the model's VIX forward sits ~11.7% below the market's,
# the VIX smile slopes up (+0.6 mean) because vol-of-vol rises with VIX level, so the two ATM vols
# are read ~9% apart. That is a CAVEAT ON THE INSTRUMENT, to be stated, not a defect to be corrected.
#
# The ONLY non-degenerate form would be a theta-INDEPENDENT constant shift -- which is literally
# "lower the vov target by ~8%", and the SSR evidence above says even that buys nothing.
#
# Runs are archived at _backups/WITHDRAWN-moneyness-20260810.tgz. Do not restore them to ship.
VOVMNY = os.environ.get("VOVMNY", "0") == "1"
# VOVMNYLEVEL=1 makes that correction a PARALLEL SHIFT: ONE gap per date, not one per tenor.
#
# WHY, and the per-tenor form is WITHDRAWN because of it. The gap log(F_mkt/F_model) GROWS WITH TENOR,
# because the model's VIX curve is flat while the market's is in contango (6e.27). So the per-tenor
# correction is ~0% at 8d and +15-17% at 78-169d -- it TILTS the vov term structure, and the only way
# the fit absorbs a steeper long end is to cut long-end dispersion. Measured on 6 dates: nu_s falls
# 21-86% and the slow/fast vov ratio collapses from 0.05-0.56 to 0.04-0.12, i.e. the two-factor
# vol-of-vol degenerates toward single-factor, while SSR degrades ~19%.
#
# That chain runs: model cannot produce VIX contango (OUT OF SCOPE, 6e.27d) -> gap grows with tenor
# -> correction tilts -> nu_s collapses. It imports an out-of-scope defect straight into the in-scope
# dynamic parameter, which is exactly the joint-fit contamination the project excludes.
#
# The LEVEL-dependence of vol-of-vol is real and worth correcting; the TILT is an artefact. Using the
# date-median gap shifts every tenor equally, so the term structure -- and therefore nu_s, which is
# identified by SHAPE -- is left to the data.
VOVMNYLEVEL = os.environ.get("VOVMNYLEVEL", "0") == "1"
sys.argv = [sys.argv[0], "cpu"]

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
torch.set_num_threads(1)
from scipy.optimize import least_squares                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                         # noqa: E402
import consts, fkernel as kernel, readouts, vix as VX       # noqa: E402
import discslv_torch as D                                   # noqa: E402
import calibrate_joint_torch as J                           # noqa: E402
import calibrate_slv_exact_ts as C                          # noqa: E402
import vix_smile as VS                                      # noqa: E402
import end_to_end as E                                      # noqa: E402

# KAPS=<value> overrides the pin for a LADDER SWEEP. In the 7-parameter formulation kap_s is a
# CONSTANT, not a parameter, so there is no ridge term on it -- costs are directly comparable across
# rungs, which is what makes the sweep meaningful. Ignored when KAPS_FREE=1.
KAPS = float(os.environ["KAPS"]) if os.environ.get("KAPS") else C.KAP_S_FIXED

T0 = time.time()
CT = json.load(open(os.path.join(_P.DATA, "corrected_targets.json")))[TICKER]
OUT = _P.DATA


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def rms(m, t):
    m, t = np.asarray(m, float), np.asarray(t, float)
    return 100 * float(np.sqrt(np.mean(((m - t) / t) ** 2)))


def make_model(ctx, K, dev, mny=None):
    # LONGEST available ladder. With LADDER=37 this is LT[37]; LT[37][0..12] IS LT[13][0..12]
    # (both are lev[k+1]), and ssr_ts loops range(K.nmax)=13, so SSR is bit-identical either way.
    # Only the leveraged VIX, whose li saturated at 12, sees the new slices.
    LAM, SIG, SPOT, VD = ctx["LT"][max(ctx["LT"])], ctx["sig_ref"], ctx["spot"], list(ctx["vdtes"])
    n_var = max(1, int(round((30.0 / 365.0) / K.dt)))

    def model(t):
        g = kernel.solve_gbar(t, SIG, K, KAPS)
        kk = kernel.build_kernel(kernel.th9(t, g, K, KAPS), K)
        ssr = readouts.ssr_ts(kk, LAM, D._interp_lin, readouts.atm_skew)
        u0 = VX.solve_us0(kk, SIG, SPOT, n_var)          # tau-independent: solve once, not per tenor
        vx = [VX.vix_ivol(kk, SIG, float(d) / 365.0, SPOT,
                          lam_fns=(LAM if VOVLEV else None), us0=u0) for d in VD]
        vov = torch.stack([o[1] for o in vx])
        if mny is not None:                       # VOVMNY: translate to the MARKET's forward
            Fm, sl = mny
            F = torch.stack([o[0] for o in vx])
            vov = vov + sl * torch.log(Fm / torch.clamp(F, min=1e-8))
        return torch.cat([ssr, vov])
    return model


def fit(date):
    log(f"=== {TICKER} {date}  dev={DEV}{'  PIN='+PIN if PIN else ''}  VOVLEV={int(VOVLEV)}  kap_s={'FREE' if KAPS_FREE else KAPS}  "
        f"max_nfev={MAXNFEV}  ftol={FTOL:.0e} xtol={XTOL:.0e}")
    ctx, _c, _p = E.ctx_rebuilt(date, TICKER)
    K = consts.Consts(DEV, torch.float32)
    mny = None
    if VOVMNY:
        _Fm, _sl = VS.smile_at(date, [int(x) for x in ctx["vdtes"]])
        _ok = np.isfinite(_Fm)
        _F0 = VS.model_fwd(ctx, date, DEV) if VOVMNYLEVEL else None   # model forward at the _n9 theta
        _sl = np.where(_ok, _sl, 0.0); _Fm = np.where(_ok, _Fm, 1.0)   # unusable expiry -> no shift
        if VOVMNYLEVEL:
            # PARALLEL SHIFT: one gap for the whole date. The per-tenor gap is applied at fit time as
            # log(Fm/F(theta)); to make that tenor-independent we hand every tenor the SAME ratio,
            # built from the date MEDIAN of F_mkt and of the model's own forward at the _n9 theta.
            # Median, not mean, so a single bad expiry cannot set the level.
            _r = float(np.median(_Fm[_ok] / _F0[_ok])) if _ok.any() else 1.0
            _Fm = _F0 * _r                     # => log(Fm/F) is the same constant at every tenor
            log(f"    VOVMNYLEVEL on: parallel shift, median F_mkt/F_model = {_r:.4f} "
                f"({100*(_r-1):+.1f}%), mean dIV/dlogK {_sl[_ok].mean():+.3f}")
        else:
            log(f"    VOVMNY on: {int(_ok.sum())}/{len(_ok)} expiries, mean dIV/dlogK {_sl[_ok].mean():+.3f}")
        mny = (torch.tensor(_Fm, dtype=torch.float32, device=DEV),
               torch.tensor(_sl, dtype=torch.float32, device=DEV))
    model = make_model(ctx, K, DEV, mny)
    emp = np.asarray(ctx["emp"], float)
    vov_t = np.asarray(ctx["vov_d"], float)
    n_ssr, n_vov = len(emp), len(vov_t)
    wv = W_VOV * np.sqrt(n_ssr / max(1, n_vov))
    NAMES = C.NAMES_N8 if KAPS_FREE else C.NAMES_N
    LO, HI = (C.LO_N8, C.HI_N8) if KAPS_FREE else (C.LO_N, C.HI_N)
    # BOX=<name>=<lo>,<hi>[;<name>=<lo>,<hi>...] widens or narrows one parameter's bounds for THIS
    # run only. Applied before WARM (which clips the seed into the box) and before PIN.
    #
    # WHY, and it is not cosmetic. trf handles bounds by Coleman-Li scaling: the step in coordinate i
    # is multiplied by v_i = x_i - lb_i whenever the gradient pushes toward the lower bound. AT the
    # bound that factor is EXACTLY ZERO, so the coordinate cannot move however hard the objective
    # pulls. rho_s has lb = 0 and every unridged fit sits at exactly 0.0000 -- which reads as "free
    # and chose to stay" but is only that where the gradient points INWARD. Measured per date on the
    # profile through _prs0: at 2022/2024 cost rises on both sides of 0, so 0 is a genuine stationary
    # point; at 2016/2018 cost FALLS going negative (0.04462 -> 0.04397, 0.01500 -> 0.01363), so the
    # bound is doing the work and those dates never had the option.
    #
    # Deliberately NOT a change to C.LO_N8/C.HI_N8: fit_norm_panel.py, vov_only_fit.py and
    # identify.py all read those, and identify.py normalises its standard errors BY BOX WIDTH, so
    # editing them in place would silently rescale a published diagnostic.
    if BOX:
        LO = np.array(LO, float); HI = np.array(HI, float)
        for part in BOX.split(";"):
            nm, rng = part.split("=")
            lo_, hi_ = (float(v) for v in rng.split(","))
            bi = NAMES.index(nm.strip())
            log(f"    BOX {nm.strip()}: [{LO[bi]:.3f}, {HI[bi]:.3f}] -> [{lo_:.3f}, {hi_:.3f}]")
            LO[bi], HI[bi] = lo_, hi_
    x0 = np.asarray((C.X0_MAP_N8 if KAPS_FREE else C.X0_MAP_N)["ts"], float)
    x0 = np.clip(x0, LO, HI)                      # the cold seed may sit outside a narrowed box
    # The RIDGE ANCHOR is ALWAYS the cold `ts` seed and must be fixed BEFORE any warm
    # start moves x0 -- otherwise WARM changes the OBJECTIVE (the ridge is ~0 at its own
    # anchor), not just the starting point, and the run is not comparable to the pinned
    # baseline. Caught by 2018 returning IDENTICAL SSR/vov with a lower cost.
    anc = x0.copy()
    rw = J.W_REG[:len(x0)] / np.maximum(np.abs(anc), 0.1)
    # NORIDGE=<name>[,<name>...] zeroes the ridge on those coordinates (or "all").
    #
    # WHY. `anc = x0.copy()` makes the ridge anchor the SEED -- an earlier calibration's output, not
    # a belief held independently of this fit. For rho_s that anchor is 0.9267 while the data prefers
    # ~0: pinning rho_s at 0 improved the DATA fit on 4/4 dates (-13.3% total, -68% at 2022) purely
    # by escaping the prior. A regulariser is meant to break ties the likelihood cannot, not to drag
    # the fit away from a preference the likelihood expresses clearly.
    #
    # Where an external prior exists (kap_s = 0.9956 is Bergomi's k2 ~ 0.23/yr) the anchor is
    # defensible. For the per-factor rho's there is no such number, so the honest options are a
    # stated prior or NO ridge -- not the seed. This flag provides the second.
    if NORIDGE:
        names = NAMES if NORIDGE.strip().lower() == "all" else [x.strip() for x in NORIDGE.split(",")]
        rw = np.array(rw, float)
        for nm in names:
            rw[NAMES.index(nm)] = 0.0
        log(f"    ridge DISABLED on: {', '.join(names)}")
    # WARM=<tag> seeds from fit_kf<tag>_<date>.json instead of the cold "ts" anchor.
    #
    # WHY IT IS NEEDED FOR THE kap_s TEST. The 7-parameter model is NESTED in the 8-parameter one,
    # so freeing kap_s cannot raise the attainable minimum -- yet a cold-started KAPS_FREE=1 run
    # scored WORSE on cost on all four dates (0.0806->0.0898, 0.0243->0.0303, 0.0078->0.0122,
    # 0.0137->0.0142). Four of four nesting violations means the 8-parameter fits landed in worse
    # local optima, not that pinning is better; the objective is a discontinuous staircase
    # (ftol_probe: 1e-3 to 1.6e-2 jumps) and every fit stops on xtol inside it. Seeding from the
    # pinned solution starts the larger model AT the smaller model's optimum, so any change it
    # reports is attributable to the extra parameter rather than to where it happened to start.
    if WARM:
        # accept either naming: this package's fit_kf<tag>_<date>.json, or the reference panel's
        # fit_norm_<date><tag>.json (e.g. WARM=_fk seeds from the pre-restructure free-kap_s panel).
        wp = os.path.join(_P.DATA, f"fit_kf{WARM}_{date}{SFX}.json")
        if not os.path.exists(wp) and TICKER == "SPX":
            wp = os.path.join(_P.DATA, f"fit_norm_{date}{WARM}.json")
        if not os.path.exists(wp):
            # NEVER fall back across tickers. A missing NDX seed must stop the run, not quietly
            # become an SPX seed -- that turns a one-variable A/B into a two-variable one.
            raise SystemExit(f"WARM={WARM}: no seed at {os.path.basename(wp)} for TICKER={TICKER}. "
                             f"Run the {WARM} panel for this ticker first.")
        w = json.load(open(wp))
        if TICKER != "SPX" and w.get("ticker") != TICKER:
            raise SystemExit(f"WARM seed {os.path.basename(wp)} has ticker={w.get('ticker')}, "
                             f"expected {TICKER}")
        seed = [w["theta"][n] for n in C.NAMES_N]
        if KAPS_FREE:
            seed.append(w["kap_s"])                    # start kap_s AT the pin it was fixed to
        x0 = np.clip(np.asarray(seed, float), LO, HI)
        log(f"    warm start from {os.path.basename(wp)}  cost there was {w['cost']:.6f}")

    sew = np.ones(n_ssr)
    if SEW:
        _jh = json.load(open(os.path.join(_P.DATA, "ssr_joint_hac.json")))[TICKER]
        _se = np.asarray(_jh[date[:4]]["se_joint"], float)
        if len(_se) != n_ssr:
            raise SystemExit(f"SEW=1: se_joint has {len(_se)} entries, need {n_ssr}")
        _rel = _se / np.abs(emp)                       # RELATIVE, to match the relative residual form
        sew = 1.0 / _rel
        sew = sew / np.sqrt(np.mean(sew ** 2))         # mean(w^2)=1 -> block scale preserved
        log(f"    SEW weights " + " ".join(f"{w:.3f}" for w in sew)
            + f"   (rel se " + " ".join(f"{100*r:.1f}%" for r in _rel) + ")")

    tt = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)          # noqa: E731
    emp_d, vov_d, anc_d, rw_d = tt(emp), tt(vov_t), tt(anc), tt(rw)
    sew_d = tt(sew)
    log(f"    n_ssr={n_ssr} n_vov={n_vov} wv={wv:.4f} n_theta={len(x0)}")

    dtgt_d = emp_d[1:] - emp_d[:-1]                    # target's own step, for the relative form
    tgt_mono = bool((emp[1:] - emp[:-1] <= 0).all())
    if MONOPEN > 0:
        log(f"    MONOPEN={MONOPEN}  mode={'GATE' if MONOGATE else 'relative'}"
            + (f"  target monotone={tgt_mono} -> {'ON' if tgt_mono else 'OFF'}" if MONOGATE else ""))

    def resid_t(th):
        m = model(th)
        s, v = m[:n_ssr], m[n_ssr:]
        blocks = [(s - emp_d) / emp_d * sew_d,
                  (v - vov_d) / vov_d * wv,
                  (th - anc_d) * rw_d]
        if MONOPEN > 0 and (not MONOGATE or tgt_mono):
            ds = s[1:] - s[:-1]
            rise = ds if MONOGATE else (ds - dtgt_d)
            blocks.append(MONOPEN * torch.relu(rise) / emp_d[:-1])
        return torch.cat(blocks)

    # PIN=<name>=<value> holds one parameter fixed by collapsing its BOUNDS, so the profile
    # (fix theta_k, re-optimise the rest) can be run for any coordinate without touching the model.
    # This is the same measurement that settled kap_s: a COLD profile measures which basin each
    # start finds, so pair it with WARM=<tag>. PIN is applied AFTER WARM -- the warm seed supplies
    # the other coordinates, the pin overrides this one.
    if SEED0:
        x0 = np.array(x0, float)
        for part in SEED0.split(";"):
            nm, val = part.split("=")
            si = NAMES.index(nm.strip())
            log(f"    SEED {nm.strip()}: {x0[si]:.4f} -> {float(val):.4f}  (anchor stays {anc[si]:.4f})")
            x0[si] = float(val)
        x0 = np.clip(x0, LO, HI)
    if PIN:
        pn, pv = PIN.split("="); pv = float(pv)
        pi = NAMES.index(pn)
        LO = np.array(LO, float); HI = np.array(HI, float)
        # clamp the epsilon-widened window to the parameter's OWN box -- pinning at a boundary
        # (e.g. rho_s = 0, whose box starts at 0) would otherwise place the lower bound outside it.
        lo0, hi0 = LO[pi], HI[pi]
        LO[pi] = max(pv - 1e-9, lo0); HI[pi] = min(pv + 1e-9, hi0)
        if HI[pi] <= LO[pi]:
            HI[pi] = LO[pi] + 1e-12
        x0 = np.array(x0, float); x0[pi] = pv
        log(f"    PIN {pn}[{pi}] = {pv}  (bounds collapsed; {len(x0)-1} free)")

    st = dict(n=0, j=0, t=time.time())

    def resid_np(x):
        r = resid_t(torch.tensor(x, dtype=torch.float32, device=DEV)).detach().cpu().numpy()
        st["n"] += 1
        dtm = time.time() - st["t"]; st["t"] = time.time()
        if st["n"] % 5 == 1 or dtm > 20:
            log(f"    f#{st['n']:3d}  cost={0.5 * float(np.sum(r ** 2)):.6f}  ({dtm:.1f}s)")
        return r.astype(np.float64)

    def jac_np(x):
        jc = torch.func.jacfwd(resid_t)(torch.tensor(x, dtype=torch.float32, device=DEV))
        st["j"] += 1
        dtm = time.time() - st["t"]; st["t"] = time.time()
        log(f"    J#{st['j']:3d}  ({dtm:.1f}s)")
        return jc.detach().cpu().numpy().astype(np.float64)

    t0 = time.time()
    # XSCALE=jac uses scipy's Jacobian-derived per-coordinate scaling instead of the default 1.0.
    #
    # WHY. trf's trust region and its xtol test live in the RAW coordinate metric, and these
    # coordinates do not share a scale. For kap_s the natural step -- the move that changes the
    # half-life by 1% -- is kap_s*(-log kap_s)/100, which runs 3.06e-03 at kap_s=0.60 down to
    # 2.00e-05 at kap_s=0.998: a 153x variation INSIDE one coordinate, and ~1e4 against nu_f whose
    # natural scale is O(1). With x_scale=1.0 a radius small enough to mean anything for kap_s is
    # useless for everything else, which is a textbook cause of the njev=1..2 collapses that have
    # been blamed on the banding staircase alone.
    #
    # Not a substitute for fixing the parameterisation (kap_s should be fitted as log mean-reversion
    # rate; its box [0.5, 0.998] spends 50% of its measure on 1.0-2.4 week half-lives and 0.8% on
    # 115-346 weeks). This is the cheap half of the fix and it is testable immediately.
    _xs = os.environ.get("XSCALE", "")
    xscale = "jac" if _xs == "jac" else (float(_xs) if _xs else 1.0)
    if _xs:
        log(f"    x_scale = {xscale!r}  (default is 1.0)")
    r = least_squares(resid_np, x0, jac=jac_np, bounds=(LO, HI), x_scale=xscale,
                      max_nfev=MAXNFEV, xtol=XTOL, ftol=FTOL)
    wall = time.time() - t0
    m = model(torch.tensor(r.x, dtype=torch.float32, device=DEV)).detach().cpu().numpy()
    ssr, vov = m[:n_ssr], m[n_ssr:]
    out = dict(date=date, ticker=TICKER, device=DEV, vovlev=int(VOVLEV), backend="kernel_fast",
               seed=WARM or "ts",         # was hardcoded "ts", so warm starts recorded as cold
               theta=dict(zip(NAMES, [float(v) for v in r.x])),
               kap_s=float(r.x[7]) if KAPS_FREE else float(KAPS),
               kap_s_fitted=bool(KAPS_FREE),
               cost=float(r.cost), ftol=FTOL, xtol=XTOL,
               # PROVENANCE. Without these a fit file is identified only by its filename TAG, which
               # is a convention rather than a record -- fit_kf_avg_* and fit_kf_L37_* differ by the
               # lambda configuration and nothing in the file said so. Anything that changes the
               # readout or the static layer belongs here.
               vovlamten=os.environ.get("VOVLAMTEN", "expiry"),
               ladder=int(os.environ.get("LADDER") or 0) or None,
               # THE LAMBDA-SMOOTHING CONFIGURATION. `_sl9` is DEFINED by these and not one of them
               # was recorded -- the run had to be recovered from a shell transcript to learn it also
               # carried LAMKEEP=1. Anything that changes the static layer belongs in the file.
               lam=dict(smooth=int(os.environ.get("LAMSMOOTH") or 0),
                        keep=int(os.environ.get("LAMKEEP") or 0),
                        slope_only=os.environ.get("LAMSLOPE") == "1",
                        sg=int(os.environ.get("LAMSG") or 0),
                        sgdeg=int(os.environ.get("LAMSGDEG") or 2),
                        nofreeze=os.environ.get("NOFREEZE", "") == "1",
                        # the per-date gate (lam_gate.py). `applied` is what the gate DECIDED for
                        # this date, so the file says which of the two models was actually run.
                        # dsg/h change the CALENDAR DERIVATIVE, i.e. the ladder's source, not lambda
                        dsg=int(os.environ.get("LAMDSG") or 0),
                        dsgdeg=int(os.environ.get("LAMDSGDEG") or 2),
                        h=float(os.environ.get("LAMH") or 1.0),
                        pillaraware=os.environ.get("PILLARAWARE", "1") != "0",
                        dates=os.environ.get("LAMDATES") or None,
                        applied=bool(E._lam_on(date)) if os.environ.get("LAMDATES") else None),
               vixfix=os.environ.get("VIXFIX", "1") == "1",
               vovmny=bool(VOVMNY), vovmnylevel=bool(VOVMNYLEVEL),
               # WHICH NDX vov TARGETS: constant-maturity (no roll artefact) or nearest-expiry
               # snapped. Only meaningful for TICKER=NDX; recorded always so a file is never
               # ambiguous about the target set it was fitted to.
               ndxvovcm=(os.environ.get("NDXVOVCM", "0") == "1") if TICKER == "NDX" else None,
               # Q/P object correction (spx_pq_vov.py). Recorded because a _pq9 record is otherwise
               # indistinguishable from a _cm9 one -- same config, silently different targets.
               ndxvovpq=(os.environ.get("NDXVOVPQ", "0") == "1") if TICKER == "NDX" else None,
               # NDXVOVSCR was NOT recorded until 2026-08-12, so `_t9`/`_t9s`/`_h9`/`_c9` carry it as
               # absent even though _c9 was fitted on the screened+churn targets. Their vov_target
               # arrays are stored, so nothing is lost, but the flag itself was invisible.
               ndxvovscr=(os.environ.get("NDXVOVSCR", "0") == "1") if TICKER == "NDX" else None,
               ndxtenors=os.environ.get("NDXTENORS") if TICKER == "NDX" else None,
               blend=os.environ.get("BLEND", "linear"), noridge=NORIDGE or None, box=BOX or None,
               xscale=_xs or None, seed0=SEED0 or None, monopen=MONOPEN or None,
               monogate=bool(MONOGATE) if MONOPEN else None,
               ssr=ssr.tolist(), ssr_target=CT[date[:4]]["corrected"],
               vov=vov.tolist(), vov_target=vov_t.tolist(),
               # WHICH TENORS. Without this a record carrying 8 vov values is indistinguishable from
               # one carrying 2, and any consumer that rebuilds ctx without NDXTENORS silently pairs
               # the wrong axis with the values (this is exactly how the _t9 panel first failed).
               vov_tenor_d=[float(x) for x in ctx["vdtes"]],
               sew=SEW, sew_w=[float(x) for x in sew],
               ssr_rms=rms(ssr, CT[date[:4]]["corrected"]), vov_rms=rms(vov, vov_t),
               wall=wall, nfev=int(r.nfev), njev=int(r.njev), status=int(r.status))
    p = os.path.join(OUT, f"fit_kf{TAG}_{date}{SFX}.json")
    json.dump(out, open(p, "w"), indent=1)
    log(f"    DONE {date}  wall {wall:.0f}s  nfev {r.nfev} njev {r.njev}  "
        f"cost {r.cost:.6f}  SSR RMS {out['ssr_rms']:.2f}%  vov RMS {out['vov_rms']:.2f}%")
    log(f"    -> {os.path.basename(p)}")
    return out


# Guard added 2026-08-10. Without it, `import refit` RUNS ALL NINE FITS -- which it silently did
# during a unit check. Running the script is unchanged; the module is now importable for tests.
if __name__ == "__main__":
    res = [fit(d) for d in DATES]
    log("=" * 72)
    log(f"{'date':12s} {'wall':>7s} {'nfev':>5s} {'njev':>5s} {'cost':>10s} {'SSR RMS':>8s} {'vov RMS':>8s}")
    for o in res:
        log(f"{o['date']:12s} {o['wall']:7.0f} {o['nfev']:5d} {o['njev']:5d} {o['cost']:10.6f} "
            f"{o['ssr_rms']:7.2f}% {o['vov_rms']:7.2f}%")
    log(f"TOTAL {time.time() - T0:.0f}s for {len(res)} fits")
