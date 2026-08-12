"""Batch-first propagation and recompression. State fields are (B, N) throughout; B = 1 is just a
degenerate batch, so there is no separate scalar path to keep in sync.

Contrast with `discslv_torch.recompress_n`, which this replaces:
  * no `bool(m.any())`      -- 20 device syncs per call there, 58,500 per model evaluation
  * no boolean mask indexing -- ~199,000 more implicit syncs, and dynamic shapes
  * no `[keep]`              -- empty cells are retained at weight 0, so the output is a static
                                (B, nc) and never ragged (the reference oscillates 239/240)
Retaining zero-weight cells is sound: such an element never advances the cumulative weight, so it
cannot move any positive-weight element's band; it contributes 0 to every segment sum; and it
branches to zero-weight children.
"""
import torch

import os

from bands import _pref, seg_bands, seg_bands_grouped, seg_bounds

# GROUPED=0 reproduces the reference's element-wise banding (arbitrary among ties); GROUPED=1 bands
# on DISTINCT values so tied components cannot be split. With no ties the two are identical, so the
# switch only bites on `mf`/`ms`. Default 1: the partition should be a function of the data, not of
# an unstable sort. See HANDOFF 6b.
# BANDMODE -- how factor-key ties are ordered. `mf`/`ms` are 85-96% tied (they do not depend on the
# price sub-abscissa) and band boundaries land inside those runs, so this decides the partition.
# ALL THREE are deterministic; the reference's unstable sort is deliberately not reproducible here.
#
#   stable  (default) -- stable sort, ties by ORIGINAL INDEX. Cheapest, and full price resolution
#                        (37 live components at step 1 vs `grouped`'s 27). Not as arbitrary as it
#                        sounds: within a parent the tied run is the n_p price abscissas in
#                        ascending zp, so index order IS MU order there; it only diverges when a
#                        tied group spans parents. 1wk SSR is bit-identical to `mu`.
#   mu                -- ties ordered by MU explicitly. The principled criterion, and correct across
#                        parents too, but costs +64% wall (11.00s vs 6.69s on the SSR block) for a
#                        <=0.14% readout change with NO accuracy gain established. Neither has been
#                        checked against MC, which is the only arbiter. Use if that check happens.
#   grouped           -- a tied state is never split, so it occupies ONE cell. Deterministic but
#                        COARSER where distinct states < cells: at step 1, 4 cells vs 12 and 27 live
#                        components vs 37. Splitting is harmless (pure pieces, MGF ratio exactly
#                        1.000000000) and is how spare cells buy MU resolution, so this gives up
#                        something for nothing. Kept for A/B.
_BANDMODE = os.environ.get("BANDMODE", "stable")
# DETSEG=1 (default): deterministic segment sum, see `seg` below. DETSEG=0 restores the
# plain float scatter_add, which is faster but NOT reproducible on GPU.
_DETSEG = os.environ.get("DETSEG", "1") == "1"
_CENTRE = os.environ.get("CENTRE", "1") == "1"
# SEGFLOOR=<rel> drops segments holding less than this FRACTION of the row weight; 0 restores the
# old `Wj > 1e-15` guard for A/B. See the long note at its use site in `recompress`.
_SEGFLOOR = float(os.environ.get("SEGFLOOR", "1e-9"))


def recompress(state, ker):
    """(B, M) -> (B, nc). Hierarchical: bands on mf, then ms | mf-band, then MU | cell."""
    W, MU, SG, mf, ms, Vff, Vss, Vfs = state
    K = ker["K"]
    B, nc = W.shape[0], K.nc
    with torch.no_grad():
        z0 = torch.zeros_like(W, dtype=torch.long)
        if _BANDMODE == "grouped":
            b1 = seg_bands_grouped(W, mf, z0, K.nb_f, 1)
            b2 = seg_bands_grouped(W, ms, b1, K.nb_s, K.nb_f)
        else:
            # argsort(MU) is the same for both factor levels: compute once.
            to = torch.argsort(MU, dim=1, stable=True) if _BANDMODE == "mu" else None
            b1 = seg_bands(W, mf, z0, K.nb_f, 1, tie_order=to)
            b2 = seg_bands(W, ms, b1, K.nb_s, K.nb_f, tie_order=to)
        cell = b1 * K.nb_s + b2
        key = cell * K.nk + seg_bands(W, MU, cell, K.nk, K.nb_f * K.nb_s)   # MU: no ties

    if _DETSEG:
        # DETERMINISTIC segment sum. Float `scatter_add` lowers to atomics on GPU, so the arrival
        # order of the ~75 adds per output slot is scheduler-dependent; float addition is not
        # associative, so the result varies run to run. Measured on MPS: 6.08e-07 on one call, which
        # the discontinuous banding then amplifies to 8.9e-03 on the cost (CPU gives 0.00e+00).
        #
        # This builds the same sums from primitives that ARE deterministic on MPS -- argsort,
        # cumsum, gather, and INTEGER scatter_add (integer addition is associative, so order cannot
        # matter). Sort by key, take one prefix sum in that fixed order, then read each segment off
        # as a difference of prefix values:
        #     sum[c] = prefix[inc[c]] - prefix[exc[c]]
        # with exc/inc the segment start/end offsets read off the counts, and `_pref` doing the
        # exclusive-prefix lookup. Empty segments give inc == exc and therefore exactly 0.
        #
        # `key` is (B, M) with M up to 18000, so keep everything that touches it O(1) in copies.
        # Prepending a zero column (torch.cat) instead of indexing idx-1 measured 82.57 s vs 0.45 s
        # on the SSR block, MPS, same DETSEG.
        with torch.no_grad():
            sK = torch.argsort(key, dim=1, stable=True)
            exc, inc = seg_bounds(torch.gather(key, 1, sK), nc)

        cnt = (inc - exc).to(W.dtype)

        def seg(v):
            # CENTRED prefix. A segment total is ~4e-3 while an uncentred prefix climbs to ~1, so
            # differencing two of them loses ~2.5e-5 relative where a direct 75-term sum loses
            # ~1e-6. Subtracting the row mean first makes the prefix random-walk about zero instead
            # of climbing, which cuts the cancellation by ~sqrt(M); the mean is added back exactly
            # via the integer segment counts. CENTRE=0 drops it.
            vs = torch.gather(v, 1, sK)
            if not _CENTRE:
                cw = torch.cumsum(vs, 1)
                return _pref(cw, inc) - _pref(cw, exc)
            c = vs.mean(1, keepdim=True)
            cw = torch.cumsum(vs - c, 1)
            return (_pref(cw, inc) - _pref(cw, exc)) + c * cnt
    else:
        def seg(v):
            return torch.zeros(B, nc, dtype=v.dtype, device=v.device).scatter_add(1, key, v)

    Wj = seg(W)
    # SEGMENT-WEIGHT FLOOR. The old guard was `Wj > 1e-15`, which is far below the accuracy to which
    # a prefix-differenced segment weight is actually KNOWN, so a cell holding ~1e-10 of the row was
    # treated as real and its moments computed as (cancellation noise) / 1e-10.
    #
    # MEASURED FAILURE, 2024-06-03 at kap_s = 0.9785720 (clean 5e-7 away on both sides):
    #   step 2  four cells with 0 < Wj < 1e-9   -> their averages become ~1e13 garbage
    #   step 3  those cells merge into a HEALTHY cell (Wj = 3.6e-3); 1e-9 * 1e13 = 1e4
    #           -> max|Vss| 1.36e4 against a physical bound of ~1 for a unit-variance AR(1) factor
    #   step 4  -> 2.0e22 -> exp overflow -> MU NaN in 240 cells -> SSR NaN for the rest of the run
    # DETSEG=0 is clean at the same theta, so this is the 6e.0 prefix-difference cancellation again:
    # the stable-centred-variance repair made it rare, not absent. Note the symptom appears one step
    # AFTER the cause and in a cell whose own weight is fine -- which is why the 1e-15 guard missed it.
    #
    # THE THRESHOLD IS NOT A JUDGEMENT CALL. The weight distribution has a clean gap: cells are
    # either exactly 0 (empty segments give inc == exc, hence exactly 0 by construction), or below
    # 1e-9, or above 1e-6 -- with legitimate cells at ~4e-3. Nothing occupies 1e-9..1e-6 at any step
    # on either theta. Relative to the row total so it survives any renormalisation.
    #
    # THIS IS INERT FOR HEALTHY CONFIGURATIONS, BY CONSTRUCTION. An empty segment already had
    # Ws = 1 and seg(W*v) = 0 exactly, so avg was 0; it is still 0. A cell above the floor is
    # untouched. Only cells in (0, floor) -- where the old code was computing garbage -- change.
    # Verified bit-identical on the nine fitted thetas; see SEGFLOOR=0 to A/B.
    if _SEGFLOOR:
        wfl = _SEGFLOOR * Wj.sum(1, keepdim=True).clamp(min=1e-30)
        okc = Wj > wfl
        Ws = torch.where(okc, Wj, torch.ones_like(Wj))
        zro = torch.zeros_like(Wj)
        avg = lambda v: torch.where(okc, seg(W * v) / Ws, zro)           # noqa: E731
    else:
        Ws = torch.where(Wj > 1e-15, Wj, torch.ones_like(Wj))
        avg = lambda v: seg(W * v) / Ws                                  # noqa: E731
    # STABLE law of total (co)variance: E[V] + Var(m) computed about the CELL MEAN, not the
    # E[X^2] - E[X]^2 form. The two are algebraically identical; numerically they are not.
    #
    # WHY THIS MATTERS HERE. With DETSEG=1 the segment sums come from differencing a prefix, which
    # carries ~2.5e-5 relative cancellation. Fed into E[X^2] - E[X]^2 -- itself a difference of
    # nearly-equal quantities -- that error can survive into the factor variance, and the variance
    # RECURSES: vX -> sX -> x -> gg hits its clamp ceiling -> sig 1.13 -> 55.95 -> the step's
    # exponent reaches +-125.88 -> exp overflows float32 -> inf - inf = NaN. Measured at
    # kap_s = 0.99 on MPS, which killed that ladder rung at f#1 with "Residuals are not finite in
    # the initial point" while 0.98 and 0.9956 ran clean. DETSEG=0 did not fail there.
    #
    # Centring about the cell mean removes the cancellation and makes each term manifestly >= 0, so
    # the clamps below become a formality rather than the only thing holding the recursion together.
    # Costs 4 extra `seg` calls; correctness first, and it is more accurate than the scatter_add
    # form it replaces, not merely equal to it.
    M = avg(MU)
    dMU = MU - torch.gather(M, 1, key)
    Vout = torch.clamp(avg(SG.pow(2)) + avg(dMU.pow(2)), min=1e-12)
    Mf, Ms = avg(mf), avg(ms)
    dmf = mf - torch.gather(Mf, 1, key)
    dms = ms - torch.gather(Ms, 1, key)
    Ff = avg(Vff) + avg(dmf.pow(2))
    Fs = avg(Vss) + avg(dms.pow(2))
    Fc = avg(Vfs) + avg(dmf * dms)
    # RELATIVE martingale re-lock. The absolute form `A = log(post)` forces E[S] = 1 every step and
    # so SUBTRACTS the starting level: lambda(MU) then sees MU ~ 0 instead of z0 and dsigma/dz decays
    # like 1/n (measured 0.100 vs MC at 13wk). Zero-weight cells drop out of both sums.
    pre = torch.log((W * torch.exp(MU + SG.pow(2) * K.half)).sum(1, keepdim=True))
    post = torch.log((Wj * torch.exp(M + Vout * K.half)).sum(1, keepdim=True))
    return (Wj, M - (post - pre), torch.sqrt(Vout), Mf, Ms,
            torch.clamp(Ff, min=1e-12), torch.clamp(Fs, min=1e-12), Fc)


def step(state, ker, lam):
    """One propagation step for a batch of chains. Branch axes (p, x, l) flatten into N."""
    W, MU, SG, mf, ms, Vff, Vss, Vfs = state
    K = ker["K"]
    B = W.shape[0]
    nu_f, nu_s, nu_l = ker["nu_f"], ker["nu_s"], ker["nu_l"]
    mX = nu_f * mf + nu_s * ms
    vX = torch.clamp(nu_f.pow(2) * Vff + nu_f * nu_s * Vfs * K.two + nu_s.pow(2) * Vss, min=1e-16)
    sX = torch.sqrt(vX)
    x = mX[..., None] + sX[..., None] * K.zx                              # (B,N,nx)
    gg = torch.clamp(ker["gbar"] + x[..., None] + nu_l * K.zl, min=K.lo_g, max=K.hi_g)
    sig = torch.sqrt(torch.exp(gg) * K.dt_t)                               # (B,N,nx,nl)
    p = MU[..., None] + SG[..., None] * K.zp                              # (B,N,np)
    lmp = torch.clamp(lam(p), min=1e-6)                                   # lambda AT the price (16.10)
    sg4 = sig[:, :, None, :, :]
    lm4 = lmp[:, :, :, None, None]
    V = (lm4 * sg4).pow(2)
    w3 = K.wx[None, None, None, :, None] * K.wl[None, None, None, None, :]
    # LOG-SUM-EXP, stabilised. `exp` of the raw exponent overflows float32 at |z| ~ 88, and that is
    # reachable: at kap_s = 0.99 on MPS the exponent hit +-125.88 by step 4, the sum went to inf,
    # A to inf, MUn to -inf, and recompress then formed inf - inf = NaN in the variance -- killing
    # the whole fit at f#1 with "Residuals are not finite in the initial point".
    #
    # The trigger was upstream (a spuriously large factor variance from the DETSEG=1 prefix-difference
    # cancellation, which drives gg to its clamp ceiling and sig from 1.13 to 55.95), but this line is
    # what turns a bad value into a NaN instead of a large one. Subtracting the row max is exact --
    # log sum w exp(z) = zmax + log sum w exp(z - zmax) -- and makes the exp argument <= 0 always.
    # zmax is detached: the identity is independent of it, so this is exact for the value and correct
    # for the gradient, and it keeps amax out of the forward-mode graph.
    _z = ker["lam_skew"] * K.zl * lm4 * sg4
    _zm = _z.amax(dim=4, keepdim=True).amax(dim=3, keepdim=True).detach()
    A = _zm[..., 0, 0] + torch.log((w3 * torch.exp(_z - _zm)).sum((3, 4)))
    MUn = p[..., None, None] + (V * K.neg_half + lm4 * ker["lam_skew"] * K.zl * sg4 - A[..., None, None])
    SGn = torch.sqrt(V)
    Wn = W[..., None, None, None] * K.wp[None, None, :, None, None] * w3
    # factor posterior given X = x, then the exact AR(1) update (independent of the price abscissa)
    bf = (nu_f * Vff + nu_s * Vfs) / vX
    bs = (nu_f * Vfs + nu_s * Vss) / vX
    d = sX[..., None] * K.zx
    cmf, cms = mf[..., None] + bf[..., None] * d, ms[..., None] + bs[..., None] * d
    cVff = Vff - bf * (nu_f * Vff + nu_s * Vfs)
    cVss = Vss - bs * (nu_f * Vfs + nu_s * Vss)
    cVfs = Vfs - bf * (nu_f * Vfs + nu_s * Vss)
    mfn = (ker["kap_f"] * cmf[..., None] + ker["s_f"] * ker["rho_f"] * K.zl)[:, :, None, :, :]
    msn = (ker["kap_s"] * cms[..., None] + ker["s_s"] * ker["rho_s"] * K.zl)[:, :, None, :, :]
    sh = MUn.shape
    lead = lambda v: v[..., None, None, None].expand(sh)                  # noqa: E731
    out = (Wn.expand(sh), MUn.expand(sh), SGn.expand(sh), mfn.expand(sh), msn.expand(sh),
           lead(ker["kap_f"].pow(2) * cVff + ker["q_f"]),
           lead(ker["kap_s"].pow(2) * cVss + ker["q_s"]),
           lead(ker["kap_f"] * ker["kap_s"] * cVfs))
    flat = [t.reshape(B, -1) for t in out]
    flat[0] = flat[0] / flat[0].sum(1, keepdim=True)
    return recompress(tuple(flat), ker)
