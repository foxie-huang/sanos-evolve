"""Segmented equal-weight banding: sort + cumsum + closed form. No masks, no ragged output.

`discslv_torch._bands_n` assigns, for sorted position i with cumulative weight cw_i and group total
T,  band(i) = #{j : e_j <= i}  with  e_j = #{k : cw_k <= T*j/nb}. Because cw is non-decreasing,

    e_j <= i   <=>   cw_i > q_j          (q_j = T*j/nb)

so band(i) = #{j in 1..nb-1 : cw_i > T*j/nb} = clamp(ceil(nb*cw_i/T) - 1, 0, nb-1). That form
depends only on the cumulative weight, not on the position, which is what lets it be segmented and
what makes zero-weight padding inert (a zero-weight element does not advance cw, so it cannot move
any positive-weight element's band).

Verified against `_bands_n` on identical inputs: 0 of 239 components differ.
"""
import torch


def _pref(cw, idx):
    """Exclusive prefix at `idx` (0..M) given the INCLUSIVE cumsum `cw`, without a `torch.cat`.

    cwx[idx] == cw[idx-1] for idx>=1 and 0 for idx==0. Prepending a zero column instead would copy
    the whole (B, M) tensor on every call. Matched A/B on the SSR block, MPS, DETSEG=0, changing
    nothing else: torch.cat 82.57 s -> this 0.45 s. It is the single most expensive mistake measured
    in this file.
    """
    g = torch.gather(cw, 1, (idx - 1).clamp(min=0))
    return torch.where(idx > 0, g, torch.zeros((), dtype=cw.dtype, device=cw.device))


def _grp_tot(cw, gs, G):
    """Per-group weight totals, deterministically, from sorted group ids `gs` and prefix `cw`.

    `gs` is already sorted ascending, so each group is one contiguous run and `searchsorted` gives
    its start; the total is the difference of the prefix at consecutive starts. No float scatter_add,
    hence no atomics and no run-to-run variation.

    Cost matters here and the obvious route is a trap. Measured on MPS, (B, M, G)=(225,18000,240):
        searchsorted           9.7 ms
        scatter_add fp32       9.0 ms   <- what this replaces
        scatter_add int64     83.6 ms   <- "just count them with an exact integer add" is 9x WORSE
        scatter_add int32     37.2 ms      than the float scatter. int64 is a bad dtype on MPS.
    """
    exc, inc = seg_bounds(gs, G)
    return _pref(cw, inc) - _pref(cw, exc)


def seg_bounds(ks, G):
    """(start, end) offsets of each of the G segments in the ascending key vector `ks`, (B, M).

    Deterministic by construction: binary search, no accumulation. Shared with `propagate.recompress`
    so both the banding and the moment sums use one definition of "where does segment c live".
    """
    b = torch.searchsorted(ks, torch.arange(G + 1, device=ks.device, dtype=ks.dtype)
                           .expand(ks.shape[0], G + 1).contiguous())
    return b[:, :-1], b[:, 1:]



def seg_bands(w, x, grp, nb, G, tie=None, tie_order=None):
    """(B,M) equal-weight banding of `x` within each (row, grp) group -> band ids in [0, nb).

    `tie` is a SECONDARY sort key used to order elements with equal `x`. Pass it whenever `x` can
    tie -- which for `mf`/`ms` is 85-96% of the array, because the factor means do not depend on the
    price sub-abscissa. Without it the order among ties is whatever an unstable sort returns, and
    since band boundaries land inside those runs, an arbitrary ordering decides the partition:
    flipping only that tie-break moves the reference's own SSR by 0.1214%.

    Passing `tie = MU` fixes it in the useful direction. Splitting a tied factor group is NOT a
    defect -- each piece stays pure, E[exp(nu.u)] is preserved to 1.000000000, and the split lets a
    state occupy several cells and so claim more MU sub-bands. What was wrong was only that the
    split was arbitrary. Ordering by MU makes it deterministic (MU has no ties) AND divides the
    group along the axis that carries the price information.

    Stop-gradient throughout, matching the reference: membership is a discrete decision and is
    deliberately not differentiated.
    """
    B, M = w.shape
    with torch.no_grad():
        if tie is None and tie_order is None:
            o1 = torch.argsort(x, dim=1, stable=True)
        else:
            # lexicographic (x, tie): stable sorts, MINOR key first. `tie_order` lets the caller
            # hoist argsort(tie) -- it is identical at both factor levels, and recomputing it there
            # measured 12.83s vs 6.86s on the SSR block.
            ot = torch.argsort(tie, dim=1, stable=True) if tie_order is None else tie_order
            o1 = torch.gather(ot, 1, torch.argsort(torch.gather(x, 1, ot), dim=1, stable=True))
        o2 = torch.argsort(torch.gather(grp, 1, o1), dim=1, stable=True)
        order = torch.gather(o1, 1, o2)                       # (grp, x) lexicographic per row
        ws = torch.gather(w, 1, order)
        gs = torch.gather(grp, 1, order)
        cw = torch.cumsum(ws, dim=1)
        # Group totals from the PREFIX SUM, not a float scatter_add. Float scatter lowers to atomics
        # on GPU, so its accumulation order varies run to run; the band boundary then moves, a
        # component flips cell, and the discontinuity turns 1e-7 of rounding into ~1e-3 of readout.
        tot = _grp_tot(cw, gs, G)
        off = torch.cumsum(tot, dim=1) - tot                  # exclusive cumsum over groups
        loc = (cw - torch.gather(off, 1, gs)) / torch.gather(tot, 1, gs).clamp(min=1e-30)
        band = torch.clamp(torch.ceil(loc * nb) - 1.0, 0.0, nb - 1.0).to(torch.long)
        return torch.empty(B, M, dtype=torch.long, device=w.device).scatter_(1, order, band)


def seg_bands_grouped(w, x, grp, nb, G):
    """Equal-weight banding on the DISTINCT values of `x` within each (row, grp) group.

    Same contract as `seg_bands`, with one guarantee added: elements sharing an x-value ALWAYS land
    in the same band. That removes the tie-break question rather than answering it.

    WHY. `mf`/`ms` do not depend on the price sub-abscissa, so each parent's children arrive in runs
    of n_p with identical factor state -- 85-96% of the banded array is tied. `seg_bands` (and the
    reference `_bands_n`) then let an UNSTABLE sort decide which side of a quantile boundary each
    tied element falls on. Measured: flipping only that tie-break moves the reference's own SSR by
    0.1214%. Banding on distinct values makes the partition a function of the data alone.

    NOT an accuracy fix, and the docstring should not be read as claiming otherwise. Splitting a
    tied group is harmless -- each piece stays PURE, and E[exp(nu.u)] is preserved to 1.000000000
    (measured). The real recompression error is cross-STATE merging, which is forced by the cell
    budget (240 groups, 0 pure, by step 4) and is untouched by this. See HANDOFF 6b.

    Mechanics: sort by (grp, x); mark where the (grp, x) pair changes; dense-rank those runs; take
    each run's cumulative weight at its LAST element, so a run straddling a boundary goes entirely
    to the upper band; then apply the same closed form as `seg_bands`.
    """
    B, M = w.shape
    with torch.no_grad():
        o1 = torch.argsort(x, dim=1, stable=True)
        o2 = torch.argsort(torch.gather(grp, 1, o1), dim=1, stable=True)
        order = torch.gather(o1, 1, o2)
        ws = torch.gather(w, 1, order)
        gs = torch.gather(grp, 1, order)
        xs = torch.gather(x, 1, order)

        # dense rank of each (grp, x) run, per row
        new = torch.ones(B, M, dtype=torch.bool, device=w.device)
        new[:, 1:] = (gs[:, 1:] != gs[:, :-1]) | (xs[:, 1:] != xs[:, :-1])
        gid = torch.cumsum(new.to(torch.long), dim=1) - 1              # 0..nruns-1, sorted

        cw = torch.cumsum(ws, dim=1)
        # each run's cumulative weight at its LAST element: cw is increasing within a run, so amax
        run_end = torch.zeros(B, M, dtype=w.dtype, device=w.device)
        run_end = run_end.scatter_reduce(1, gid, cw, reduce="amax", include_self=False)
        cw_run = torch.gather(run_end, 1, gid)                         # broadcast back to elements

        tot = _grp_tot(cw, gs, G)   # prefix-sum group totals, same reason as in seg_bands
        off = torch.cumsum(tot, dim=1) - tot
        loc = (cw_run - torch.gather(off, 1, gs)) / torch.gather(tot, 1, gs).clamp(min=1e-30)
        band = torch.clamp(torch.ceil(loc * nb) - 1.0, 0.0, nb - 1.0).to(torch.long)
        return torch.empty(B, M, dtype=torch.long, device=w.device).scatter_(1, order, band)
