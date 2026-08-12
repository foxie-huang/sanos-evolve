#!/usr/bin/env python3
"""Which parameter directions does the DATA identify, and does VOVLEV change the answer?

The counting condition says SPX is over-determined -- 11 to 18 data residuals against n_theta = 8.
Counting is not identification. What matters is the rank and conditioning of the residual Jacobian:
a direction in which the readouts barely move is one where the fitted value is set by the seed (or
the ridge), not by the data. Measured for kap_s the hard way -- a warm-started 5-rung ladder -- the
cost varies ~30% across a 22x range of half-life while the STARTING POINT alone moves a single rung
44%. This does the same test for every direction at once, and cheaply.

METHOD. J = d(data residual)/d(theta) at the fitted theta, RIDGE EXCLUDED -- identification is about
what the data says, not what the prior says. H = J'J is the Gauss-Newton Hessian; its eigenvalues
are the curvature along each direction, and sqrt(diag(H^-1)) is each parameter's standard error with
correlations to the others accounted for. A parameter whose SE dwarfs its own value is not estimated.

CAVEAT, and it matters here. The recompression banding is stop-gradient, so J sees only the SMOOTH
part of the objective. The true surface is a staircase (ftol_probe: 1e-3 to 1.6e-2 jumps for theta
perturbations from 1e-8 to 1e-4). So these standard errors are a LOWER BOUND on the true uncertainty
-- a direction that looks identified here may still be swamped by band flips. A direction that looks
flat HERE is flat, full stop.

    python3 identify.py                 # VOVLEV=1 and 0, all four dates
    VOVLEV=1 python3 identify.py 2016-06-01
"""
import os
import sys

DATES = [a for a in sys.argv[1:] if a[:2] == "20"] or \
        ["2016-06-01", "2018-06-01", "2022-06-01", "2024-06-03"]
LEVS = [os.environ["VOVLEV"] == "1"] if os.environ.get("VOVLEV") else [True, False]
sys.argv = [sys.argv[0], "cpu"]

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                         # noqa: E402
import consts, fkernel as kernel, readouts, vix as VX       # noqa: E402
import discslv_torch as D                                   # noqa: E402
import calibrate_joint_torch as J                           # noqa: E402
import calibrate_slv_exact_ts as C                          # noqa: E402
import end_to_end as E                                      # noqa: E402

NAMES = list(C.NAMES_N8)
W_VOV = 0.8


def analyse(date, vovlev):
    ctx, _c, _p = E.ctx_rebuilt(date, "SPX")
    LAM = ctx["LT"][max(ctx["LT"])]
    SIG, SPOT, VD = ctx["sig_ref"], ctx["spot"], list(ctx["vdtes"])
    K = consts.Consts("cpu", torch.float64)
    n_var = max(1, int(round((30.0 / 365.0) / K.dt)))
    emp = torch.tensor(np.asarray(ctx["emp"], float), dtype=torch.float64)
    vt = torch.tensor(np.asarray(ctx["vov_d"], float), dtype=torch.float64)
    wv = W_VOV * float(np.sqrt(len(emp) / max(1, len(vt))))

    # the fitted theta for THIS readout: _L37c is VOVLEV=1 (kap_s pinned), _u is VOVLEV=0 (free)
    import json
    # SRC=<tag> reads fit_kf<tag>_<date>.json instead of the hardcoded defaults. NEEDED because the
    # spectrum is a LOCAL quantity at one theta: computed at _L37c/_u it calls rho_s the flattest
    # direction, while in the _prs0 basin rho_s is among the STEEPEST (+1365% at 2022 for a 0.825
    # excursion, 6e.8). So the published table describes the basin the fit happened to land in, not
    # the parameter. Re-run at the basin you actually intend to cite.
    SRC = os.environ.get("SRC", "")
    tag = (f"fit_kf{SRC}_%s.json" if SRC
           else ("fit_kf_L37c_%s.json" if vovlev else "fit_kf_u_%s.json"))
    p = os.path.join(_P.DATA, tag % date)
    if not os.path.exists(p):
        return None
    f = json.load(open(p))
    th = torch.tensor([f["theta"][n] for n in C.NAMES_N] + [f["kap_s"]], dtype=torch.float64)

    def resid(t):
        """DATA residuals only -- no ridge. theta is length 8, kap_s is t[7]."""
        g = kernel.solve_gbar(t, SIG, K)
        kk = kernel.build_kernel(kernel.th9(t, g, K), K)
        ssr = readouts.ssr_ts(kk, LAM, D._interp_lin, readouts.atm_skew)
        u0 = VX.solve_us0(kk, SIG, SPOT, n_var)
        vov = torch.stack([VX.vix_ivol(kk, SIG, float(d) / 365.0, SPOT,
                                       lam_fns=(LAM if vovlev else None), us0=u0)[1] for d in VD])
        return torch.cat([(ssr - emp) / emp, (vov - vt) / vt * wv])

    Jm = torch.func.jacfwd(resid)(th).detach().numpy()
    H = Jm.T @ Jm
    # CALIBRATED NOISE SCALE. Cov(theta_hat) ~= sigma^2 (J'J)^-1, and the earlier version omitted
    # sigma^2 entirely -- i.e. assumed sigma = 1. These residuals are RELATIVE errors, so sigma = 1
    # asserts 100% relative noise per observation when the actual residuals are a few percent, and
    # every SE came out inflated by ~1/sigma_hat (about 8x at 2016). That produced the false headline
    # "only lam_skew is determined to better than its own box width".
    #
    # sigma_hat^2 = RSS/(n-p) is the standard estimate from the fit itself. It is a PER-DATE COMMON
    # FACTOR, so it never changes the ranking of parameters within a date or the eigenvectors -- only
    # whether an absolute claim like "SE exceeds the box" is true. SIGMA=1 restores the old scale.
    r0 = resid(th).detach().numpy()
    rss = float(np.sum(r0 ** 2))
    dof = max(len(r0) - len(th), 1)
    s2 = 1.0 if os.environ.get("SIGMA") == "1" else rss / dof
    ev = np.linalg.eigvalsh(H)
    # pseudo-inverse: a genuinely singular direction has no finite SE, so report it as such
    U, s, Vt = np.linalg.svd(H)
    tol = max(s) * 1e-12
    inv = (Vt.T * np.where(s > tol, 1.0 / np.maximum(s, tol), 0.0)) @ U.T
    se = np.sqrt(s2 * np.clip(np.diag(inv), 0, None))
    return dict(theta=th.numpy(), se=se, ev=ev, H=H, nres=Jm.shape[0], sig=float(np.sqrt(s2)),
                rss=rss, dof=dof,
                cond=float(max(ev) / max(min(ev), 1e-300)))


SUMMARY = {}

for vovlev in LEVS:
    print(f"\n{'='*86}\nVOVLEV={int(vovlev)}   J'J of the DATA residuals at the fitted theta "
          f"(ridge excluded)\n{'='*86}")
    for date in DATES:
        r = analyse(date, vovlev)
        if r is None:
            print(f"  {date}: no fit file for this readout"); continue
        print(f"\n  {date}   {r['nres']} residuals, 8 parameters   cond(J'J) = {r['cond']:.2e}"
              f"   sigma_hat = {r['sig']:.4f}  (RSS {r['rss']:.4f}, dof {r['dof']})")
        # NORMALISE BY THE BOX WIDTH, not by |theta|. Dividing by |theta| flatters any parameter
        # whose value happens to be large: kap_s sits at 0.9956 with SE 0.229, which looks like a
        # 23% relative error but actually spans 46% of its [0.50, 0.998] range -- i.e. useless.
        # That normalisation initially reported kap_s as the BEST determined parameter, contradicting
        # the warm-started ladder which showed it flat across a 22x range of half-life.
        W = np.asarray(C.HI_N8, float) - np.asarray(C.LO_N8, float)
        print(f"    {'param':10s} {'theta':>9s} {'std err':>11s} {'SE/box':>8s}   identified?")
        for n, t, s, w in zip(NAMES, r["theta"], r["se"], W):
            rel = s / w
            flag = "FLAT" if rel > 0.5 else ("weak" if rel > 0.1 else "ok")
            print(f"    {n:10s} {t:9.4f} {s:11.3e} {rel:8.2f}   {flag}")
        e = r["ev"][::-1]
        print(f"    eigenvalues: " + " ".join(f"{x:.2e}" for x in e))
        # effective rank: how many directions the data actually curves in, at a 1e-4 relative cut
        k = int((e / e[0] > 1e-4).sum())
        print(f"    effective rank {k}/8 at 1e-4 of the top eigenvalue  -> the data supports ~{k} "
              f"parameters, not 8")
        w, V = np.linalg.eigh(r["H"])
        for j in (0, 1):                      # the two FLATTEST directions
            v = V[:, j]
            top = np.argsort(-np.abs(v))[:3]
            desc = " + ".join(f"{v[i]:+.2f}*{NAMES[i]}" for i in top)
            print(f"    flat dir {j+1} (eig {w[j]:.2e}): {desc}")
        SUMMARY.setdefault(vovlev, []).append(
            (date, k, np.abs(V[:, 0]), np.abs(V[:, 1]), r["se"] / W, r["cond"]))

for vovlev, rows in SUMMARY.items():
    if len(rows) < 2:
        continue
    print(f"\n{'-'*86}\n  CROSS-DATE SUMMARY, VOVLEV={int(vovlev)}, {len(rows)} dates\n{'-'*86}")
    L1 = np.mean([r[2] for r in rows], 0); L2 = np.mean([r[3] for r in rows], 0)
    SE = np.median([r[4] for r in rows], 0)
    print(f"  {'param':10s} {'|load| flat1':>13s} {'|load| flat2':>13s} {'median SE/box':>14s}")
    for i, n_ in enumerate(NAMES):
        print(f"  {n_:10s} {L1[i]:13.2f} {L2[i]:13.2f} {SE[i]:14.1f}")
    print(f"\n  effective rank by date: " + " ".join(f"{r[0][:4]}:{r[1]}" for r in rows))
    print(f"  cond(J'J) range: {min(r[5] for r in rows):.1e} .. {max(r[5] for r in rows):.1e}")
    print(f"  flattest direction is dominated by: {NAMES[int(np.argmax(L1))]} "
          f"(mean |loading| {L1.max():.2f}); consistent on "
          f"{sum(1 for r in rows if int(np.argmax(r[2])) == int(np.argmax(L1)))}/{len(rows)} dates")
