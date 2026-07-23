#!/usr/bin/env python3
"""
discslv_torch.py (v2) -- the fused forward map in PyTorch, differentiable in theta for AD gradients.

STAGE 1: kernel build + fused propagate + recompress, ported from v1's numpy (TwoFactorSV + slv_fast.
propagate_vec + discslv_2f.recompress_2f). The recompress keeps v1's EXACT adaptive equal-weight-chunk
assignment but STOP-GRADIENTs it (a detached membership matrix); only the moment-matching is
differentiable. Forward values match v1 bit-close; gradients flow through theta -> (kernel, increments)
-> the propagated marginal. float64 to match numpy.

Validate:  python3 discslv_torch.py   (compares kernel arrays + one propagate step to v1)
"""
import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss

torch.set_default_dtype(torch.float64)
PNAMES = ["gbar", "nu_f", "nu_s", "nu_l", "lam_skew", "lam_f", "lam_s", "kap_f", "kap_s"]


def lev_torch(coef, zmax, safety):
    """Torch leverage lambda(MU) = clip(exp(polyval(coef, clip(MU,-zmax,zmax))), safety). coef from v1's
    _Lev (theta-invariant statics), so this is differentiable in MU (the state) but constant in theta."""
    c = np.asarray(coef, float); lo, hi = safety

    def lam(mu):
        z = torch.clamp(mu, -zmax, zmax)
        p = torch.zeros_like(z)
        for a in c:                                        # Horner polyval (a scalars cast to z's dtype/device)
            p = p * z + float(a)
        return torch.clamp(torch.exp(p), lo, hi)
    return lam


def build_kernel(theta, dt, n_f=5, n_s=3, n_l=5):
    """theta: torch tensor [gbar,nu_f,nu_s,nu_l,lam_skew,lam_f,lam_s,kap_f,kap_s]. Returns a dict of torch
    kernel arrays (differentiable in theta): Vl,D (n_f,n_s,n_l), tilt (n_f,n_s,n_l), Tf (n_l,n_f,n_f),
    Ts (n_l,n_s,n_s), and the fixed nodes/weights."""
    gbar, nu_f, nu_s, nu_l, lam_skew, lam_f, lam_s, kap_f, kap_s = theta
    dd = dict(dtype=theta.dtype, device=theta.device)                # kernel follows theta's dtype/device
    zf = torch.as_tensor(hermegauss(n_f)[0], **dd); wf = torch.as_tensor(hermegauss(n_f)[1], **dd); wf = wf / wf.sum(); ef = torch.log(wf)
    zs = torch.as_tensor(hermegauss(n_s)[0], **dd); ws = torch.as_tensor(hermegauss(n_s)[1], **dd); ws = ws / ws.sum(); es = torch.log(ws)
    zl = torch.as_tensor(hermegauss(n_l)[0], **dd); wl = torch.as_tensor(hermegauss(n_l)[1], **dd); wl = wl / wl.sum()
    g = gbar + nu_f * zf[:, None, None] + nu_s * zs[None, :, None] + nu_l * zl[None, None, :]   # (nf,ns,nl)
    Vl = torch.exp(g) * dt
    sigl = torch.sqrt(Vl)
    tilt = lam_skew * zl[None, None, :] * sigl
    mtil = -0.5 * Vl + tilt
    A = torch.log((wl[None, None, :] * torch.exp(mtil + 0.5 * Vl)).sum(-1, keepdim=True))
    D = mtil - A
    Tf = torch.stack([torch.stack([torch.softmax(ef + kap_f * zf[f] * zf + lam_f * zl[l] * zf, 0)
                                   for f in range(n_f)]) for l in range(n_l)])   # (nl,nf,nf)
    Ts = torch.stack([torch.stack([torch.softmax(es + kap_s * zs[s] * zs + lam_s * zl[l] * zs, 0)
                                   for s in range(n_s)]) for l in range(n_l)])   # (nl,ns,ns)
    return dict(Vl=Vl, D=D, tilt=tilt, Tf=Tf, Ts=Ts, wl=wl, zf=zf, zs=zs, zl=zl, n_f=n_f, n_s=n_s, n_l=n_l, dt=dt)


def _merge(w, mu, sg, nk):
    """Detached-membership recompression: v1's sort -> equal-weight chunks (STOP-GRADIENT) -> per-chunk
    0/1/2 moment-match (differentiable). Same forward value as _fast_merge; gradient = the within-cell part."""
    n = w.shape[0]
    with torch.no_grad():                                  # discrete assignment: no grad
        order = torch.argsort(mu)
        cw = torch.cumsum(w[order], 0)
        q = torch.linspace(0.0, float(cw[-1]), nk + 1)[1:-1]
        edges = torch.searchsorted(cw, q, right=True)
        chunk = torch.searchsorted(edges, torch.arange(n), right=True)          # chunk id per sorted position
        memb = torch.zeros(n, nk); memb[order, chunk] = 1.0                      # original component -> its chunk
    Wj = memb.t() @ w
    keep = Wj > 1e-15
    Ws = torch.where(keep, Wj, torch.ones_like(Wj))
    Mj = (memb.t() @ (w * mu)) / Ws
    E2 = (memb.t() @ (w * (sg ** 2 + mu ** 2))) / Ws
    Vj = torch.clamp(E2 - Mj ** 2, min=1e-12)
    return Wj[keep], Mj[keep], torch.sqrt(Vj)[keep]


def recompress(W, MU, SG, F, S, nk, n_f, n_s):
    """Per-(f,s)-regime detached-membership recompression + global martingale re-lock (matches v1)."""
    Wo, MUo, SGo, Fo, So = [], [], [], [], []
    for f in range(n_f):
        for s in range(n_s):
            m = (F == f) & (S == s)
            if int(m.sum()) == 0:
                continue
            w, mu, sg = W[m], MU[m], SG[m]
            if w.shape[0] > nk:
                w, mu, sg = _merge(w, mu, sg, nk)
            Wo.append(w); MUo.append(mu); SGo.append(sg)
            Fo.append(torch.full((w.shape[0],), f)); So.append(torch.full((w.shape[0],), s))
    W = torch.cat(Wo); MU = torch.cat(MUo); SG = torch.cat(SGo)
    F = torch.cat(Fo).long(); S = torch.cat(So).long()
    A = torch.log((W * torch.exp(MU + 0.5 * SG ** 2)).sum())                     # global re-lock
    return W, MU - A, SG, F, S


def propagate(state, ker, lam, nk):
    """One fused step (ports slv_fast.propagate_vec): leveraged increment -> expand over (l,f',s') -> recompress."""
    W, MU, SG, F, S = state
    nl, nf, ns_ = ker["n_l"], ker["n_f"], ker["n_s"]; wl = ker["wl"]
    Vlr = ker["Vl"][F, S]                                  # (N, nl)
    lm = torch.clamp(lam(MU), min=1e-6)                    # (N,) leverage at component means
    Vl = lm[:, None] ** 2 * Vlr
    mtil = -0.5 * Vl + lm[:, None] * ker["tilt"][F, S]
    A = torch.log((wl[None, :] * torch.exp(mtil + 0.5 * Vl)).sum(-1))            # (N,)
    Dl = mtil - A[:, None]
    Tfi = ker["Tf"][:, F, :].permute(1, 0, 2)              # (N, nl, nf)
    Tsi = ker["Ts"][:, S, :].permute(1, 0, 2)              # (N, nl, ns)
    w = W[:, None, None, None] * wl[None, :, None, None] * Tfi[:, :, :, None] * Tsi[:, :, None, :]
    mu_b = (MU[:, None] + Dl)[:, :, None, None].expand(-1, -1, nf, ns_)
    sg_b = torch.sqrt(SG[:, None] ** 2 + Vl)[:, :, None, None].expand(-1, -1, nf, ns_)
    fp = torch.arange(nf)[None, None, :, None].expand(w.shape)
    sp = torch.arange(ns_)[None, None, None, :].expand(w.shape)
    W2 = w.reshape(-1); W2 = W2 / W2.sum()
    return recompress(W2, mu_b.reshape(-1), sg_b.reshape(-1), fp.reshape(-1), sp.reshape(-1), nk, nf, ns_)


def stationary_pi(ker):
    """Stationary (f,s) distribution by a differentiable linear solve (pi P = pi, sum pi = 1)."""
    nf, ns = ker["n_f"], ker["n_s"]; nfs = nf * ns
    P = torch.einsum("l,lac,lbd->abcd", ker["wl"], ker["Tf"], ker["Ts"]).reshape(nfs, nfs)
    dd = dict(dtype=P.dtype, device=P.device)
    A = torch.cat([(P.t() - torch.eye(nfs, **dd))[:-1], torch.ones(1, nfs, **dd)], 0)
    b = torch.zeros(nfs, **dd); b[-1] = 1.0
    return torch.linalg.solve(A, b).reshape(nf, ns)


def lev_increment(Vlr_fs, tilt_fs, wl, lam0):
    """One-step return moments (Dl, Vl) at the ATM leverage lam0 (ports lev_increment)."""
    Vl = lam0 ** 2 * Vlr_fs
    mtil = -0.5 * Vl + lam0 * tilt_fs
    A = torch.log((wl * torch.exp(mtil + 0.5 * Vl)).sum())
    return mtil - A, Vl


def interp_batch(query, xp, yp):
    """Linear interp of EACH row of yp (M, nz) at all `query` points, edge-clipped like np.interp.
    Fully vectorized (shared brackets since xp is common) -- no .item() sync. Differentiable in yp, query."""
    q = query.reshape(-1)
    idx = torch.searchsorted(xp, q).clamp(1, xp.shape[0] - 1)
    x0 = xp[idx - 1]; x1 = xp[idx]
    t = (q - x0) / (x1 - x0)                                          # (Nq,)
    out = yp[:, idx - 1] + t[None, :] * (yp[:, idx] - yp[:, idx - 1]) # (M, Nq)
    out = torch.where(q[None, :] < xp[0], yp[:, :1], out)
    out = torch.where(q[None, :] > xp[-1], yp[:, -1:], out)
    return out.reshape(yp.shape[0], *query.shape)


def fused_ssr_ts(theta, lam_fns, n, dt, nz=13, zmax=0.12, Q=5, nk=16, n_f=5, n_s=3, n_l=5):
    """Full exact-beta term-structure SSR in torch (ports slv_fast.fused_ssr_exact_ts). Differentiable in
    theta -> SSR. sigma_ATM(z,f,s) grid via propagate+atm_skew; analytic covariance sum over the one-step
    transition (nodes l x GH sub-quadrature q x dest regimes)."""
    ker = build_kernel(theta, dt, n_f, n_s, n_l)
    zg = torch.linspace(-zmax, zmax, nz); iz0 = nz // 2
    sig_rows = []; skw0 = []
    for f in range(n_f):
        for s in range(n_s):
            row = []
            for iz in range(nz):
                st = (torch.ones(1), zg[iz].reshape(1).clone(), torch.full((1,), 1e-4), torch.tensor([f]), torch.tensor([s]))
                for k in range(n):
                    st = propagate(st, ker, lam_fns[k], nk)
                a, sk = atm_skew(st[0], st[1], st[2], n * dt)
                row.append(a)
                if iz == iz0:
                    skw0.append(sk)
            sig_rows.append(torch.stack(row))
    sigg = torch.stack(sig_rows).reshape(n_f, n_s, nz)                 # (nf,ns,nz)
    skw0 = torch.stack(skw0).reshape(n_f, n_s)
    pi = stationary_pi(ker); lam0 = lam_fns[0](torch.zeros(1))[0]
    zq_np, wq_np = hermegauss(Q); wq_np = wq_np / wq_np.sum()
    zq = torch.tensor(zq_np); wq = torch.tensor(wq_np); wl = ker["wl"]
    wlq = wl[:, None] * wq[None, :]                                   # (nl, Q)
    siggflat = sigg.reshape(n_f * n_s, nz)
    beta_num = torch.zeros(()); var = torch.zeros(())
    for f in range(n_f):                                             # per starting regime; l/q/dest all vectorized
        for s in range(n_s):
            Dl, Vl = lev_increment(ker["Vl"][f, s], ker["tilt"][f, s], wl, lam0)   # (nl,)
            mean_r = (wl * Dl).sum()
            var = var + pi[f, s] * ((wl * (Dl ** 2 + Vl)).sum() - mean_r ** 2)
            sd = torch.sqrt(torch.clamp(Vl, min=1e-16))
            r = Dl[:, None] + sd[:, None] * zq[None, :]              # (nl, Q) one-step return at node l, sub-point q
            SIG = interp_batch(r, zg, siggflat).reshape(n_f, n_s, n_l, Q)          # sigma_ATM at r per dest (fp,sp)
            P = ker["Tf"][:, f, :][:, :, None] * ker["Ts"][:, s, :][:, None, :]    # (nl, nf, ns) dest weights
            sig_dest = torch.einsum("lab,ablq->lq", P, SIG)          # (nl, Q)
            cov = (wlq * (sig_dest - sigg[f, s, iz0]) * (r - mean_r)).sum()
            beta_num = beta_num + pi[f, s] * cov
    beta = beta_num / var; skew = (pi * skw0).sum()
    return beta / skew, beta, skew


_SQRT2 = 2.0 ** 0.5
_SQRT2PI = (2.0 * np.pi) ** 0.5


def _Phi(x):
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _phi(x):
    return torch.exp(-0.5 * x ** 2) / _SQRT2PI


def gm_call(Kstrike, W, MU, SG):
    """Closed-form GM call price (sum of Black calls), unit-forward. Differentiable. Ports gm_call."""
    F = torch.exp(MU + 0.5 * SG ** 2)
    d1 = (torch.log(F / Kstrike) + 0.5 * SG ** 2) / SG
    return (W * (F * _Phi(d1) - Kstrike * _Phi(d1 - SG))).sum()


def _bs_call(Kstrike, u):
    d1 = (-torch.log(Kstrike) + 0.5 * u ** 2) / u
    return _Phi(d1) - Kstrike * _Phi(d1 - u)                          # unit-forward Black call, total-vol u


def _implied_u(Kstrike, C, u0, iters=8):
    """Differentiable BS implied total-vol by unrolled Newton (vega = phi(d1)); matches v1's bisection."""
    u = u0
    for _ in range(iters):
        d1 = (-torch.log(Kstrike) + 0.5 * u ** 2) / u
        vega = torch.clamp(_phi(d1), min=1e-10)
        u = torch.clamp(u - (_bs_call(Kstrike, u) - C) / vega, min=1e-6)
    return u


def atm_skew(W, MU, SG, T, dm=6e-3):
    """ATM implied vol + ATM skew of a GM marginal -- CLOSED-FORM + differentiable (ports atm_skew_of).
    ATM guess by probit u=2*sqrt(2)*erfinv(C_atm); the three IVs (0,+-dm) by differentiable Newton;
    skew = central difference of the inverted IVs (same dm as v1)."""
    sqrtT = float(T) ** 0.5
    one = torch.ones((), dtype=MU.dtype)
    Catm = gm_call(one, W, MU, SG)
    u0 = 2.0 * _SQRT2 * torch.erfinv(torch.clamp(Catm, -0.999, 0.999))   # ATM total-vol guess (exact at ATM)
    Klo = torch.exp(torch.tensor(-dm)); Khi = torch.exp(torch.tensor(dm))
    u_a = _implied_u(one, Catm, u0)
    u_lo = _implied_u(Klo, gm_call(Klo, W, MU, SG), u0)
    u_hi = _implied_u(Khi, gm_call(Khi, W, MU, SG), u0)
    sig_atm = u_a / sqrtT
    skew = (u_hi - u_lo) / sqrtT / (2 * dm)
    return sig_atm, skew


def vix_ivol(ker, sig_ref, tau_opt, spot, tau_var=30.0 / 365.0):
    """Torch VIX ATM implied vol (differentiable), ports vix_readout.model_vix_ivol: per-regime forward
    variance v = sig_ref^2 * Vbar/E_pi[Vbar]; 30d forward variance by exact multi-step transition; terminal
    regime law = spot-matched slow regime propagated tau_opt steps; atomic VIX ATM option, probit inversion."""
    wl = ker["wl"]; dt = ker["dt"]; nf, ns = ker["n_f"], ker["n_s"]
    EVl = (wl * ker["Vl"]).sum(-1); md = (wl * ker["D"]).sum(-1); md2 = (wl * ker["D"] ** 2).sum(-1)
    Vfull = EVl + (md2 - md ** 2)                                     # (nf,ns) full one-step increment variance
    pi = stationary_pi(ker); EV = (pi * Vfull).sum()
    v = sig_ref ** 2 * Vfull / EV                                    # forward-variance rate per regime
    Pf = torch.einsum("l,lij->ij", wl, ker["Tf"]); Ps = torch.einsum("l,lij->ij", wl, ker["Ts"])
    n_var = max(1, int(round(tau_var / dt)))
    Y = torch.zeros_like(v); Pfm = torch.eye(nf); Psm = torch.eye(ns)
    for _ in range(n_var):
        Pfm = Pfm @ Pf; Psm = Psm @ Ps; Y = Y + Pfm @ v @ Psm.t()   # 30d forward variance, exact multi-step
    vix = torch.sqrt(Y / n_var)
    n_opt = max(1, int(round(tau_opt / dt))); fc = nf // 2
    with torch.no_grad():
        s0 = int(torch.argmin(torch.abs(vix[fc, :] - spot)))         # today's slow regime = VIX spot
    pf = torch.matrix_power(Pf, n_opt)[fc]; ps = torch.matrix_power(Ps, n_opt)[s0]
    p = torch.outer(pf, ps)                                         # terminal regime law
    F = (p * vix).sum(); Catm = (p * torch.clamp(vix - F, min=0.0)).sum()
    iv = (2.0 / tau_opt ** 0.5) * _SQRT2 * torch.erfinv(torch.clamp(Catm / F, -0.999, 0.999))   # probit ATM inversion
    return F, iv


if __name__ == "__main__":
    import sys, os
    HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.join(HERE, "..", "poc")
    sys.path.insert(0, POC); sys.path.insert(0, HERE)
    import discslv_slv
    from slv_fast import propagate_vec
    discslv_slv.propagate = propagate_vec
    from discslv_2f import TwoFactorSV
    from discslv_slv import Epi_V, nu_bar, raw_increment
    from slv_wire import sanos_chain, ref_vol, solve_gbar, leverage_at

    DT = 1.0 / 52.0
    kw = dict(nu_f=0.208, nu_s=0.411, nu_l=1.070, lam_skew=-0.303, lam_f=0.633, lam_s=2.092, kap_f=0.937, kap_s=2.706)
    OUT = os.environ.get("ORATS_EOD_DIR", os.path.expanduser("~/orats_eod"))
    chain = sanos_chain(OUT + "/SPX-NDX-RUT-VIX_2015-06-01.json.gz"); sig = ref_vol(chain)
    gbar = solve_gbar(kw, sig, dt=DT)
    K = TwoFactorSV(gbar=gbar, dt=DT, n_f=5, n_s=3, n_l=5, **kw)
    EV = Epi_V(K); nub = nu_bar(K, EV); Vlr, tiltr = raw_increment(K)
    lamv1 = leverage_at(chain, 4 * DT, EV, dt=DT)

    theta = torch.tensor([gbar, kw["nu_f"], kw["nu_s"], kw["nu_l"], kw["lam_skew"],
                          kw["lam_f"], kw["lam_s"], kw["kap_f"], kw["kap_s"]], requires_grad=True)
    ker = build_kernel(theta, DT)
    lamt = lev_torch(lamv1.coef, lamv1.zmax, lamv1.safety)

    print("=== kernel-array match (torch vs v1) ===")
    for nm, a, b in [("Vl", ker["Vl"], K.Vl), ("D", ker["D"], K.D), ("Tf", ker["Tf"], K.Tf), ("Ts", ker["Ts"], K.Ts)]:
        print(f"  {nm:>3}  max|diff| = {float((a.detach().numpy() - b).__abs__().max()):.2e}")

    st_np = (np.array([1.0]), np.array([0.0]), np.array([1e-4]), np.array([2], np.intp), np.array([1], np.intp))
    st_t = (torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1e-4]), torch.tensor([2]), torch.tensor([1]))
    for step in range(4):
        st_np, _ = propagate_vec(K, st_np, (lambda mc: lamv1(mc) ** 2), EV, nub, Vlr, tiltr, 16)
        st_t = propagate(st_t, ker, lamt, 16)
    Wn, Mn, Sn, Fn, Sn2 = st_np
    Wt, Mt, St, Ft, St2 = (x.detach().numpy() for x in st_t)
    on = np.lexsort((Mn, Sn2, Fn)); ot = np.lexsort((Mt, St2, Ft))
    print(f"\n=== 4-step marginal match (sorted by regime,mean) ===  n_comp v1={len(Wn)} torch={len(Wt)}")
    print(f"  max|dW|={np.abs(Wn[on]-Wt[ot]).max():.2e}  max|dMU|={np.abs(Mn[on]-Mt[ot]).max():.2e}  "
          f"max|dSG|={np.abs(Sn[on]-St[ot]).max():.2e}  regimes match={np.array_equal(Fn[on],Ft[ot]) and np.array_equal(Sn2[on],St2[ot])}")
