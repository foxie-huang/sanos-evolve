"""Readouts on the batched kernel: the sigma_ATM grid and the SSR term structure.

Phase 1 (the grid) is ~99% of the cost and is fully batched -- 13 sequential steps on a leading
B = na*nb*nz axis, instead of 2,925 dispatch-bound calls. Phases 2 and 3 mirror `_ssr_core_n`.
"""
import torch

from consts import scalar
from fkernel import stat_nodes
from propagate import step


_SQRT2 = 2.0 ** 0.5
_ISQRT2PI = 1.0 / (2.0 * 3.141592653589793) ** 0.5


def _Phi(z):
    return (torch.erf(z / scalar(z, _SQRT2)) + scalar(z, 1.0)) * scalar(z, 0.5)


def _gm_call(Kstrike, W, MU, SG):
    """Batched Gaussian-mixture call at a scalar strike, unit forward. (B,)."""
    Fw = torch.exp(MU + SG.pow(2) * scalar(SG, 0.5))
    d1 = (torch.log(Fw / scalar(Fw, Kstrike)) + SG.pow(2) * scalar(SG, 0.5)) / SG
    return (W * (Fw * _Phi(d1) - _Phi(d1 - SG) * scalar(Fw, Kstrike))).sum(-1)


def atm_skew(W, MU, SG, Tmat, dm=6e-3, iters=8):
    """MPS-clean port of `discslv_torch_batched.atm_skew_batch`: probit guess + unrolled Newton at
    three strikes, skew by central difference. Same algorithm, same iteration count.

    The reference is NOT MPS-safe: it uses `np.log(Kk)` and `(2*np.pi)**0.5`, which are numpy
    float64 scalars, and MPS refuses the promotion. Here every scalar is a Python float (a weak
    type torch folds into the tensor's own dtype) and none sits to the LEFT of a tensor. Bit-identity
    against the reference is checked in verify.py, stage 0.
    """
    sqrtT = float(Tmat) ** 0.5
    lKlo, lKhi = -float(dm), float(dm)
    Klo, Khi = float(torch.exp(torch.tensor(-dm))), float(torch.exp(torch.tensor(dm)))
    Ca = _gm_call(1.0, W, MU, SG)
    Cl = _gm_call(Klo, W, MU, SG)
    Ch = _gm_call(Khi, W, MU, SG)
    u0 = torch.erfinv(torch.clamp(Ca, -0.999, 0.999)) * scalar(Ca, 2.0 * _SQRT2)

    def inv(logK, Kk, C):
        u = u0
        for _ in range(iters):
            d1 = (u.pow(2) * scalar(u, 0.5) - scalar(u, logK)) / u
            vega = torch.clamp(torch.exp(d1.pow(2) * scalar(d1, -0.5)) * scalar(d1, _ISQRT2PI), min=1e-10)
            u = torch.clamp(u - (_Phi(d1) - _Phi(d1 - u) * scalar(u, Kk) - C) / vega, min=1e-6)
        return u
    ua = inv(0.0, 1.0, Ca)
    ul = inv(lKlo, Klo, Cl)
    uh = inv(lKhi, Khi, Ch)
    return ua / scalar(ua, sqrtT), (uh - ul) / scalar(ua, sqrtT * 2.0 * dm)


def atm_vsc(W, MU, SG, Tmat, dm=6e-3, iters=8):
    """(ATM vol, ATM skew, ATM curvature) from the SAME three-strike inversion as `atm_skew`.

    `atm_skew` already inverts at k = -dm, 0, +dm and forms the skew as the central difference; the
    curvature is the second difference of those same three implied vols and costs nothing extra --
    no additional propagation, no additional Newton iterations. It is a separate function rather than
    an extra return value because `atm_skew` is checked for BIT-IDENTITY against the reference in
    verify.py stage 0, and changing its signature would break that check and its callers.
    """
    sqrtT = float(Tmat) ** 0.5
    lKlo, lKhi = -float(dm), float(dm)
    Klo, Khi = float(torch.exp(torch.tensor(-dm))), float(torch.exp(torch.tensor(dm)))
    Ca = _gm_call(1.0, W, MU, SG)
    Cl = _gm_call(Klo, W, MU, SG)
    Ch = _gm_call(Khi, W, MU, SG)
    u0 = torch.erfinv(torch.clamp(Ca, -0.999, 0.999)) * scalar(Ca, 2.0 * _SQRT2)

    def inv(logK, Kk, C):
        u = u0
        for _ in range(iters):
            d1 = (u.pow(2) * scalar(u, 0.5) - scalar(u, logK)) / u
            vega = torch.clamp(torch.exp(d1.pow(2) * scalar(d1, -0.5)) * scalar(d1, _ISQRT2PI), min=1e-10)
            u = torch.clamp(u - (_Phi(d1) - _Phi(d1 - u) * scalar(u, Kk) - C) / vega, min=1e-6)
        return u
    ua = inv(0.0, 1.0, Ca)
    ul = inv(lKlo, Klo, Cl)
    uh = inv(lKhi, Khi, Ch)
    v = ua / scalar(ua, sqrtT)
    s_ = (uh - ul) / scalar(ua, sqrtT * 2.0 * dm)
    c = (uh - ua * scalar(ua, 2.0) + ul) / scalar(ua, sqrtT * dm * dm)
    return v, s_, c


def smile_grid(ker, lam_fns):
    """(nT, na, nb, nz) ATM vol, ATM skew AND ATM curvature -- the z-dependence RETAINED.

    `sigma_grid` computes the skew for the whole batch, spot nodes included, and then slices it to
    the central node on the way out (`[..., K.iz0]`). The smile-motion decomposition needs exactly
    what that slice discards: how the skew and the curvature move with SPOT. Same chain, same cost;
    only the output is kept whole, plus the curvature that `atm_vsc` gets free from the inversion
    `atm_skew` already performs.
    """
    K = ker["K"]
    uf, us, PI, za, zb = stat_nodes(ker)
    na, nb, nz = K.na, K.nb, K.nz
    B = na * nb * nz
    one = torch.ones(B, 1, dtype=K.dtype, device=K.device)
    MU0 = K.zg[None, None, :].expand(na, nb, nz).reshape(B, 1).contiguous()
    mf0 = uf[:, :, None].expand(na, nb, nz).reshape(B, 1).contiguous()
    ms0 = us[:, :, None].expand(na, nb, nz).reshape(B, 1).contiguous()
    zer = torch.zeros(B, 1, dtype=K.dtype, device=K.device)
    st = (one, MU0, torch.full((B, 1), 1e-4, dtype=K.dtype, device=K.device),
          mf0, ms0, zer, zer.clone(), zer.clone())
    V = [None] * len(K.NS); S = [None] * len(K.NS); C = [None] * len(K.NS)
    for k in range(K.nmax):
        st = step(st, ker, lam_fns[k])
        t = K.snap.get(k + 1)
        if t is None:
            continue
        W = st[0] / st[0].sum(1, keepdim=True)
        lF = torch.log((W * torch.exp(st[1] + st[2].pow(2) * K.half)).sum(1, keepdim=True))
        v, sk, cv = atm_vsc(W, st[1] - lF, st[2], (k + 1) * K.dt)
        V[t] = v.reshape(na, nb, nz); S[t] = sk.reshape(na, nb, nz); C[t] = cv.reshape(na, nb, nz)
    return torch.stack(V), torch.stack(S), torch.stack(C), PI


def sigma_grid(ker, lam_fns, atm_skew_fn):
    """(nT, na, nb, nz) sigma_ATM and (nT, na, nb) ATM skew, in ONE batched chain."""
    K = ker["K"]
    uf, us, PI, za, zb = stat_nodes(ker)
    na, nb, nz = K.na, K.nb, K.nz
    B = na * nb * nz
    one = torch.ones(B, 1, dtype=K.dtype, device=K.device)
    MU0 = K.zg[None, None, :].expand(na, nb, nz).reshape(B, 1).contiguous()
    mf0 = uf[:, :, None].expand(na, nb, nz).reshape(B, 1).contiguous()
    ms0 = us[:, :, None].expand(na, nb, nz).reshape(B, 1).contiguous()
    zer = torch.zeros(B, 1, dtype=K.dtype, device=K.device)
    st = (one, MU0, torch.full((B, 1), 1e-4, dtype=K.dtype, device=K.device),
          mf0, ms0, zer, zer.clone(), zer.clone())
    sig = [None] * len(K.NS)
    skw = [None] * len(K.NS)
    for k in range(K.nmax):
        st = step(st, ker, lam_fns[k])
        t = K.snap.get(k + 1)
        if t is None:                       # static: depends on the step index, not on data
            continue
        W = st[0] / st[0].sum(1, keepdim=True)
        lF = torch.log((W * torch.exp(st[1] + st[2].pow(2) * K.half)).sum(1, keepdim=True))
        v, s = atm_skew_fn(W, st[1] - lF, st[2], (k + 1) * K.dt)
        sig[t] = v.reshape(na, nb, nz)
        skw[t] = s.reshape(na, nb, nz)[..., K.iz0]
    return torch.stack(sig), torch.stack(skw)


def ssr_ts(ker, lam_fns, interp_lin, atm_skew_fn):
    """SSR term structure. `interp_lin` and `atm_skew_fn` are injected so the gate can run this
    against the reference's own helpers and isolate the grid as the only thing that changed."""
    K = ker["K"]
    uf, us, PI, za, zb = stat_nodes(ker)
    sig, skw0 = sigma_grid(ker, lam_fns, atm_skew_fn)

    lam0 = lam_fns[0](torch.zeros(1, dtype=K.dtype, device=K.device))[0]
    X = ker["nu_f"] * uf + ker["nu_s"] * us
    gg = torch.clamp(ker["gbar"] + X[..., None] + ker["nu_l"] * K.zl, min=K.lo_g, max=K.hi_g)
    Vl = lam0.pow(2) * torch.exp(gg) * K.dt_t
    sdl = torch.sqrt(torch.clamp(Vl, min=1e-16))
    mtil = Vl * K.neg_half + ker["lam_skew"] * K.zl * sdl
    A = torch.log((K.wl * torch.exp(mtil + Vl * K.half)).sum(-1, keepdim=True))
    Dl = mtil - A
    mean_r = (K.wl * Dl).sum(-1)
    var = (PI * ((K.wl * (Dl.pow(2) + Vl)).sum(-1) - mean_r.pow(2))).sum()

    r = Dl[..., None] + sdl[..., None] * K.zq
    ef = ker["s_f"] * torch.sqrt(torch.clamp(K.one - ker["rho_f"].pow(2), min=0.0))
    es = ker["s_s"] * torch.sqrt(torch.clamp(K.one - ker["rho_s"].pow(2), min=0.0))
    ufn = ker["kap_f"] * uf[..., None, None] + ker["s_f"] * ker["rho_f"] * K.zl[:, None] + ef * K.ze
    usn = ker["kap_s"] * us[..., None, None] + ker["s_s"] * ker["rho_s"] * K.zl[:, None] + es * K.ze
    ia0, ia1, ta = interp_lin(za, ufn)
    ib0, ib1, tb = interp_lin(zb, usn)
    iz0_, iz1_, tz = interp_lin(K.zg, r)

    w4 = (K.wl[:, None, None, None] * K.we[None, :, None, None]
          * K.we[None, None, :, None] * K.wq[None, None, None, :])
    out = []
    for t in range(len(K.NS)):
        bn = torch.zeros((), dtype=K.dtype, device=K.device)
        for a in range(K.na):
            for b in range(K.nb):
                A0, A1, TA = ia0[a, b], ia1[a, b], ta[a, b]
                B0, B1, TB = ib0[a, b], ib1[a, b], tb[a, b]
                Z0, Z1, TZ = iz0_[a, b], iz1_[a, b], tz[a, b]
                sg = sig[t]
                g = lambda A_, B_, Z_: sg[A_[:, :, None, None], B_[:, None, :, None],
                                          Z_[:, None, None, :]]          # noqa: E731
                wA, wB, wZ = TA[:, :, None, None], TB[:, None, :, None], TZ[:, None, None, :]
                mA, mB, mZ = K.one - wA, K.one - wB, K.one - wZ      # never `1 - tensor`
                sd = (mA * (mB * (mZ * g(A0, B0, Z0) + wZ * g(A0, B0, Z1))
                            + wB * (mZ * g(A0, B1, Z0) + wZ * g(A0, B1, Z1)))
                      + wA * (mB * (mZ * g(A1, B0, Z0) + wZ * g(A1, B0, Z1))
                              + wB * (mZ * g(A1, B1, Z0) + wZ * g(A1, B1, Z1))))
                bn = bn + PI[a, b] * (w4 * (sd - sig[t, a, b, K.iz0])
                                      * (r[a, b][:, None, None, :] - mean_r[a, b])).sum()
        out.append((bn / var) / (PI * skw0[t]).sum())
    return torch.stack(out)


def smile_response(ker, lam_fns, interp_lin):
    """The smile's response to a spot move, decomposed by moment: (beta, d_skew, d_curv, skew0,
    curv0, SSR) per tenor.

    This is `ssr_ts`'s machinery applied to three fields instead of one. `ssr_ts` already forms
    beta = cov(dsigma_ATM, r)/var(r) -- it is `bn/var`, divided by the skew only on the last line to
    give the SSR -- so the level response comes free. Running the identical trilinear interpolation
    and PI-weighted covariance against the SKEW and the CURVATURE fields gives the shape response,
    which is what distinguishes sticky-moneyness (d_skew = 0) from sticky-strike (d_skew = 2*curv0).

    Conventions are taken from `ssr_ts` verbatim rather than re-derived: same destination
    interpolation, same w4 quadrature weights, same stationary weighting, same origin node K.iz0.
    """
    K = ker["K"]
    uf, us, PI, za, zb = stat_nodes(ker)
    SIG, SKW, CRV, _ = smile_grid(ker, lam_fns)

    lam0 = lam_fns[0](torch.zeros(1, dtype=K.dtype, device=K.device))[0]
    X = ker["nu_f"] * uf + ker["nu_s"] * us
    gg = torch.clamp(ker["gbar"] + X[..., None] + ker["nu_l"] * K.zl, min=K.lo_g, max=K.hi_g)
    Vl = lam0.pow(2) * torch.exp(gg) * K.dt_t
    sdl = torch.sqrt(torch.clamp(Vl, min=1e-16))
    mtil = Vl * K.neg_half + ker["lam_skew"] * K.zl * sdl
    A = torch.log((K.wl * torch.exp(mtil + Vl * K.half)).sum(-1, keepdim=True))
    Dl = mtil - A
    mean_r = (K.wl * Dl).sum(-1)
    var = (PI * ((K.wl * (Dl.pow(2) + Vl)).sum(-1) - mean_r.pow(2))).sum()

    r = Dl[..., None] + sdl[..., None] * K.zq
    ef = ker["s_f"] * torch.sqrt(torch.clamp(K.one - ker["rho_f"].pow(2), min=0.0))
    es = ker["s_s"] * torch.sqrt(torch.clamp(K.one - ker["rho_s"].pow(2), min=0.0))
    ufn = ker["kap_f"] * uf[..., None, None] + ker["s_f"] * ker["rho_f"] * K.zl[:, None] + ef * K.ze
    usn = ker["kap_s"] * us[..., None, None] + ker["s_s"] * ker["rho_s"] * K.zl[:, None] + es * K.ze
    ia0, ia1, ta = interp_lin(za, ufn)
    ib0, ib1, tb = interp_lin(zb, usn)
    iz0_, iz1_, tz = interp_lin(K.zg, r)
    w4 = (K.wl[:, None, None, None] * K.we[None, :, None, None]
          * K.we[None, None, :, None] * K.wq[None, None, None, :])

    out = []
    for t in range(len(K.NS)):
        acc = [torch.zeros((), dtype=K.dtype, device=K.device) for _ in range(3)]
        for a in range(K.na):
            for b in range(K.nb):
                A0, A1, TA = ia0[a, b], ia1[a, b], ta[a, b]
                B0, B1, TB = ib0[a, b], ib1[a, b], tb[a, b]
                Z0, Z1, TZ = iz0_[a, b], iz1_[a, b], tz[a, b]
                wA, wB, wZ = TA[:, :, None, None], TB[:, None, :, None], TZ[:, None, None, :]
                mA, mB, mZ = K.one - wA, K.one - wB, K.one - wZ
                dr = (r[a, b][:, None, None, :] - mean_r[a, b])
                for i, F in enumerate((SIG, SKW, CRV)):
                    fd = F[t]
                    g = lambda A_, B_, Z_: fd[A_[:, :, None, None], B_[:, None, :, None],
                                              Z_[:, None, None, :]]      # noqa: E731
                    sd = (mA * (mB * (mZ * g(A0, B0, Z0) + wZ * g(A0, B0, Z1))
                                + wB * (mZ * g(A0, B1, Z0) + wZ * g(A0, B1, Z1)))
                          + wA * (mB * (mZ * g(A1, B0, Z0) + wZ * g(A1, B0, Z1))
                                  + wB * (mZ * g(A1, B1, Z0) + wZ * g(A1, B1, Z1))))
                    acc[i] = acc[i] + PI[a, b] * (w4 * (sd - F[t, a, b, K.iz0]) * dr).sum()
        beta, dsk, dcv = (x / var for x in acc)
        sk0 = (PI * SKW[t, :, :, K.iz0]).sum()
        cv0 = (PI * CRV[t, :, :, K.iz0]).sum()
        out.append(torch.stack([beta, dsk, dcv, sk0, cv0, beta / sk0]))
    return torch.stack(out)
