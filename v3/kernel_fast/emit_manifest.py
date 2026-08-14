#!/usr/bin/env python3
"""Emit the replication manifest: every numerical constant, read from the code, plus a per-date
ladder attribution VERIFIED against each fit record rather than asserted from session notes.

WHY. The shipped fit records store the optimiser settings (ftol, xtol, box, seed, device, wall)
but NOT the leverage-ladder flags. Stage 2 of the SPX protocol mollifies dC/dT at source with
LAMH=4.0 PILLARAWARE=0, and neither appears anywhere in the artifact, so four of the nine SPX rows
of Table 12 cannot be reproduced from the records alone -- a reader who reruns them at the default
ladder gets a different number and has no way to discover why.

HOW THE ATTRIBUTION IS ESTABLISHED. Not from memory. Each record's own theta is re-evaluated under
each candidate ladder and compared elementwise with the record's own stored readout. The right
ladder reproduces it to a few tenths of a percent; the wrong one misses by an order of magnitude
more (measured at 2016: 0.8% against 3.6%, with the 1wk tenor off by 0.214 in SSR units). The
verdict, the observed deviation, and the margin over the runner-up are all written out, so a
reconstructed field is never mistaken for a recorded one.

Writes manifest.json (machine-readable) and manifest_table.tex (\\input-able) side by side.

    DEV=mps python3 emit_manifest.py                # verify + write
    DEV=mps python3 emit_manifest.py --dry-run
"""
import argparse
import ast
import hashlib
import json
import os
import platform
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--table-only", action="store_true",
                help="rewrite manifest_table.tex from the existing manifest.json without "
                     "re-running the ladder attribution (which is a real compute job)")
A = ap.parse_args()

DEV = os.environ.get("DEV", "mps")
os.environ.setdefault("LADDER", "42")
os.environ.setdefault("VOVLAMTEN", "avg")

# NDXTENORS MUST BE SET BEFORE THE IMPORTS. `calibrate_ndx.TENORS` is a module-level constant read
# from the environment at import time (calibrate_ndx.py:48, default "30,90"), so setting it later --
# inside the row builder, as the first version of this script did -- has no effect: the context comes
# back on the 30/90 anchor axis instead of the eight tenors the shipped panel was fitted on. Read it
# from the records rather than transcribing it, and refuse to proceed if they disagree.
_HERE0 = os.path.dirname(os.path.abspath(__file__))
_DATA0 = next((d for d in (os.path.normpath(os.path.join(_HERE0, "..", "artifacts")),
                           os.path.normpath(os.path.join(_HERE0, "..", "..", "vix_joint_refit")))
               if os.path.isdir(d)), "")
_nt = set()
for _fn in os.listdir(_DATA0):
    if _fn.startswith("fit_kf_c9_") and _fn.endswith("_ndx.json"):
        _v = json.load(open(os.path.join(_DATA0, _fn))).get("ndxtenors")
        if _v:
            _nt.add(_v)
if len(_nt) > 1:
    sys.exit(f"off-index records disagree on NDXTENORS: {sorted(_nt)}")
if _nt:
    os.environ["NDXTENORS"] = _nt.pop()

sys.argv = [sys.argv[0], "cpu"]

import torch                                                          # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                                   # noqa: E402
import consts, fkernel as kernel, readouts, vix as VX                 # noqa: E402
import discslv_torch as D                                             # noqa: E402
import calibrate_slv_exact_ts as C                                    # noqa: E402
import end_to_end as E                                                # noqa: E402

SHIPPED = {"2012-06-01": "_dw9", "2016-06-01": "_dw9", "2017-06-01": "_dw9", "2018-06-01": "_dw9",
           "2019-06-03": "_n9", "2020-06-01": "_n9", "2021-06-01": "_n9", "2022-06-01": "_n9",
           "2024-06-03": "_n9"}
# The off-index panel ships one tag on one ladder, so its rows need no attribution search -- but the
# hash and the reproduction check belong in the manifest just as much as SPX's.
SHIPPED_NDX = {d: "_c9" for d in SHIPPED}
# the two candidate ladders the protocol uses; stage 2 is the mollified one
LADDERS = {"stage 1 (cold)": {"LAMH": "1.0", "PILLARAWARE": None},
           "stage 2 (mollified)": {"LAMH": "4.0", "PILLARAWARE": "0"}}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def env_defaults(path, names):
    """Module-level `X = <cast>(os.environ.get("X", "<default>"))` defaults, read from the source.

    Importing refit.py to reach them would run its whole heavy import block and pick up whatever
    is in the current environment -- which is exactly the drift this manifest exists to prevent.
    Parsing gets the DEFAULT, which is what the shipped fits ran under.
    """
    out, tree = {}, ast.parse(open(path).read())
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names):
            continue
        v = node.value
        cast = v.func.id if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) else None
        call = v.args[0] if cast in ("int", "float") and v.args else v
        # unwrap the three shapes refit.py actually uses beyond the plain one:
        #   float(os.environ.get(...) or 0)      -> BoolOp
        #   os.environ.get(...).upper()          -> method call on the get
        #   os.environ.get(...) == "1"           -> Compare
        if isinstance(call, ast.BoolOp) and call.values:
            call = call.values[0]
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr in ("upper", "lower", "strip")):
            call = call.func.value
        if isinstance(call, ast.Compare):
            call, cast = call.left, "bool_eq"
        if not (isinstance(call, ast.Call) and len(call.args) == 2):
            continue
        raw = ast.literal_eval(call.args[1])
        out[node.targets[0].id] = {"int": int, "float": float}.get(cast, lambda x: x)(raw)
    missing = set(names) - set(out)
    if missing:
        sys.exit(f"env_defaults: could not read {sorted(missing)} from {path}")
    return out


def readout(th8, LAM, SIG, SPOT, VD, K, want_sbar=False):
    n_var = max(1, int(round((30.0 / 365.0) / K.dt)))
    with torch.no_grad():
        g = kernel.solve_gbar(th8, SIG, K)
        kk = kernel.build_kernel(kernel.th9(th8, g, K), K)
        if want_sbar:
            # the SSR denominator: Proposition (readout regularity) assumes it bounded away from
            # zero, and a scan between fitted vectors found 0.10 with SSR(1wk) above 7
            _uf, _us, PI, _za, _zb = kernel.stat_nodes(kk)
            _sg, skw0 = readouts.sigma_grid(kk, LAM, readouts.atm_skew)
            return np.array([float((PI * skw0[i]).sum()) for i in range(len(skw0))])
        ssr = readouts.ssr_ts(kk, LAM, D._interp_lin, readouts.atm_skew)
        u0 = VX.solve_us0(kk, SIG, SPOT, n_var)
        vov = torch.stack([VX.vix_ivol(kk, SIG, float(d) / 365.0, SPOT, lam_fns=LAM, us0=u0)[1]
                           for d in VD])
        return torch.cat([ssr, vov]).cpu().numpy()


def attribute(date, tag):
    """Which ladder reproduces this record's own stored readout? Verified, not recalled."""
    f = json.load(open(os.path.join(_P.DATA, f"fit_kf{tag}_{date}.json")))
    ref = np.concatenate([np.asarray(f["ssr"], float), np.asarray(f["vov"], float)])
    th8 = torch.tensor([f["theta"][n] for n in C.NAMES_N] + [f["kap_s"]],
                       dtype=torch.float32, device=DEV)
    K = consts.Consts(DEV, torch.float32)
    devs = {}
    for name, env in LADDERS.items():
        for k, v in env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        E._CTX.clear() if hasattr(E, "_CTX") else None
        ctx, _c, _n = E.ctx_rebuilt(date, "SPX")
        LAM, SIG, SPOT, VD = (ctx["LT"][max(ctx["LT"])], ctx["sig_ref"], ctx["spot"],
                              list(ctx["vdtes"]))
        R = readout(th8, LAM, SIG, SPOT, VD, K)
        devs[name] = float(np.mean(np.abs(R - ref) / np.abs(ref))) if len(R) == len(ref) else np.inf
    best = min(devs, key=devs.get)
    other = min(d for n, d in devs.items() if n != best)
    for k, v in LADDERS[best].items():                 # re-select the winning ladder for s-bar
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    ctx, _c, _n = E.ctx_rebuilt(date, "SPX")
    sbar = readout(th8, ctx["LT"][max(ctx["LT"])], ctx["sig_ref"], ctx["spot"],
                   list(ctx["vdtes"]), K, want_sbar=True)
    return best, devs[best], other, f, sbar


if __name__ == "__main__":
    K = consts.Consts(DEV, torch.float32)
    quad = {"n_l": K.n_l, "n_x": K.n_x, "n_p": K.n_p, "na": K.na, "nb": K.nb, "ne": K.ne,
            "Q": K.Q, "nq": K.nq, "q_vix": K.q_vix, "nk": K.nk, "nz": K.nz,
            "nb_f": K.nb_f, "nb_s": K.nb_s, "zmax": K.zmax, "dt": K.dt,
            "NS_weeks": list(K.NS), "nc": int(K.nc)}
    box = dict(zip(C.NAMES_N8, zip([float(x) for x in C.LO_N8], [float(x) for x in C.HI_N8])))
    box["nu_l"] = (0.1, 3.0)                       # BOX="nu_l=0.1,3.0", the override the fits ran under

    # every one of these was hardcoded here until it was not: read them from the fitter's source
    OPT = env_defaults(os.path.join(HERE, "refit.py"),
                       {"W_VOV", "FTOL", "XTOL", "MAXNFEV", "SEW", "VOVLEV"})
    import calibrate_joint_torch as JT                                        # noqa: E402
    w_reg = [float(x) for x in JT.W_REG[:len(C.NAMES_N8)]]
    ridge = {"weights": dict(zip(C.NAMES_N8, w_reg)), "floor_eps": 0.1,
             "anchor": "the current stage's own starting point",
             "form": "varrho_k (theta_k - theta0_k) / max(|theta0_k|, eps)"}
    clamp = {"log_variance": [float(K.lo_g), float(K.hi_g)],
             "leverage_floor": 1e-6, "variance_floor": 1e-16}
    determinism = {"rng": "none -- quadrature and optimiser are deterministic; no seeded sampling",
                   "starting_point": "fixed cold vector, then stage 1's optimum",
                   "calendar_interpolation": "linear in T (BLEND)",
                   "VIXFIX": VX._VIXFIX, "VOVLAMTEN": VX._LAMTEN,
                   "SEW": {"SPX": False, "NDX": True},
                   "dtype": "float32", "device_sensitive": True}

    def ndx_row(date, tag):
        """Off-index rows: one tag, one ladder, so verify the reproduction rather than search."""
        for k, v in LADDERS["stage 1 (cold)"].items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        fp = os.path.join(_P.DATA, f"fit_kf{tag}_{date}_ndx.json")
        f = json.load(open(fp))
        # THE TENOR AXIS IS NOT A DEFAULT. `build_ctx_ndx` supplies the 30/90-day anchors; the
        # shipped panel was fitted on eight tenors set by NDXTENORS, which the record stores for
        # exactly this reason ("any consumer that rebuilds ctx without NDXTENORS silently pairs the
        # wrong axis with the values" -- refit.py). Restore it from the record, not from the shell.
        if f.get("ndxtenors"):
            os.environ["NDXTENORS"] = f["ndxtenors"]
        if f.get("ndxvovscr") is not None:
            os.environ["NDXVOVSCR"] = "1" if f["ndxvovscr"] else "0"
        ref = np.concatenate([np.asarray(f["ssr"], float), np.asarray(f["vov"], float)])
        th8 = torch.tensor([f["theta"][n] for n in C.NAMES_N] + [f["kap_s"]],
                           dtype=torch.float32, device=DEV)
        ctx, _c, _n = E.ctx_rebuilt(date, "NDX")
        vd = list(ctx["vdtes"])
        want = [float(x) for x in f["vov_tenor_d"]]
        if [float(x) for x in vd] != want:
            sys.exit(f"NDX {date}: rebuilt tenor axis {vd} != the record's {want}; refusing to "
                     f"write a manifest row against a mispaired axis")
        R = readout(th8, ctx["LT"][max(ctx["LT"])], ctx["sig_ref"], ctx["spot"], vd,
                    consts.Consts(DEV, torch.float32))
        if len(R) != len(ref):
            sys.exit(f"NDX {date}: readout has {len(R)} entries, record has {len(ref)}")
        dev = float(np.mean(np.abs(R - ref) / np.abs(ref)))
        return dict(date=date, tag=tag, ticker="NDX", ladder="stage 1 (cold)",
                    deviation_pct=float(f"{100*dev:.3g}"), sew=bool(f.get("sew")),
                    wall_s=round(f["wall"], 1), nfev=f["nfev"], njev=f["njev"],
                    device=f["device"], sha=sha(fp))

    rows = []
    if A.table_only:
        rows = json.load(open(os.path.join(_P.DATA, "manifest.json")))["spx_dates"]
        print(f"  --table-only: reusing {len(rows)} verified rows from manifest.json")
    for date, tag in ([] if A.table_only else SHIPPED.items()):
        name, dev, alt, f, sbar = attribute(date, tag)
        want = LADDERS["stage 2 (mollified)"] if tag == "_dw9" else LADDERS["stage 1 (cold)"]
        expect = "stage 2 (mollified)" if tag == "_dw9" else "stage 1 (cold)"
        rows.append(dict(date=date, tag=tag, ladder=name, agrees=(name == expect),
                         deviation_pct=float(f"{100*dev:.3g}"),
                         alternative_ladder_deviation_pct=round(100 * alt, 2),
                         bit_exact=bool(dev == 0.0),
                         sbar=[round(float(x), 4) for x in sbar],
                         sbar_min=round(float(np.abs(sbar).min()), 4),
                         wall_s=round(f["wall"], 1), nfev=f["nfev"], njev=f["njev"],
                         device=f["device"], sha=sha(os.path.join(_P.DATA,
                                                    f"fit_kf{tag}_{date}.json"))))
        print(f"  {date} {tag:5s} -> {name:22s} "
              f"reproduces to {'BIT-EXACT' if dev == 0 else f'{100*dev:.3g}%':>9s}; "
              f"the other ladder misses by {100*alt:5.1f}%   "
              f"{'OK' if rows[-1]['agrees'] else 'MISMATCH'}", flush=True)

    ndx_rows = []
    if not A.table_only:
        for date, tag in SHIPPED_NDX.items():
            r = ndx_row(date, tag)
            ndx_rows.append(r)
            print(f"  {date} {tag:5s} NDX -> reproduces to {r['deviation_pct']:.3g}%", flush=True)
    else:
        ndx_rows = json.load(open(os.path.join(_P.DATA, "manifest.json"))).get("ndx_dates", [])

    man = {
        "generated_from": "v3_scripts/kernel_fast/emit_manifest.py (values read from the code)",
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "platform": platform.platform(), "device": DEV},
        "quadrature_and_recompression": quad,
        "parameters": {"names": list(C.NAMES_N8), "box": {k: list(v) for k, v in box.items()},
                       "pinned": {"rho_s": 0.0}, "solved_not_fitted": ["gbar"]},
        "objective": {"w_vov": OPT["W_VOV"], "ftol": OPT["FTOL"], "xtol": OPT["XTOL"],
                      "max_nfev": OPT["MAXNFEV"], "ridge": ridge, "blend": "linear"},
        "clamps": clamp,
        "determinism": determinism,
        "other_production_constants": env_defaults(
            os.path.join(HERE, "refit.py"),
            {"KAPS_FREE", "MONOPEN", "MONOGATE", "VOVMNY", "VOVMNYLEVEL", "TICKER", "DEV"}),
        "protocol": {"stage 1": {"seed": "cold", "LAMH": "1.0", "LADDER": 42,
                                 "VOVLEV": 1, "VOVLAMTEN": "avg", "PIN": "rho_s=0",
                                 "KAPS_FREE": 1, "BOX": "nu_l=0.1,3.0"},
                     "stage 2": {"seed": "warm from stage 1", "LAMH": "4.0",
                                 "PILLARAWARE": "0", "accept_if": "data cost lower"}},
        "spx_dates": rows,
        "ndx_dates": ndx_rows,
    }
    bad = [r["date"] for r in rows if not r["agrees"]]
    print(f"\n  {len(rows)-len(bad)}/{len(rows)} dates confirm their expected ladder"
          + (f"; MISMATCH at {bad}" if bad else ""))

    if not A.dry_run:
        jp = os.path.join(_P.DATA, "manifest.json")
        json.dump(man, open(jp, "w"), indent=1)
        # explicit symbol + role per knob: a bare multi-letter italic says nothing about what
        # the constant controls, and the point of the table is that a reader can set them all.
        ROWS = [("n_l", r"$n_{\mathsf L}$", "within-step branch nodes"),
                ("n_x", r"$n_{\mathsf X}$", "combined-factor quadrature"),
                ("n_p", r"$n_{\mathsf P}$", "price sub-abscissas of the leverage"),
                ("na", r"$n_a$", "stationary factor grid, fast"),
                ("nb", r"$n_b$", "stationary factor grid, slow"),
                ("nz", r"$n_z$", "spot-shift grid for the smile readout"),
                ("nb_f", r"$n_{b,\mathsf F}$", "recompression bands, fast"),
                ("nb_s", r"$n_{b,\mathsf S}$", "recompression bands, slow"),
                ("nc", r"$n_c$", "components carried per fibre (the budget)"),
                ("nq", r"$n_q$", "variance-index quadrature"),
                ("q_vix", r"$q_{\mathrm{vix}}$", "leveraged variance-index quadrature")]
        L = [r"\begin{tabular}{lll}", r"\toprule",
             r"Setting & Value & Role \\", r"\midrule",
             r"\multicolumn{3}{l}{\emph{Quadrature and recompression}} \\"]
        for k, sym, role in ROWS:
            L.append(rf"\quad {sym} & ${quad[k]}$ & {role} \\")
        L += [rf"\quad $z_{{\max}}$, $\Delta t$ & ${quad['zmax']}$, $1/52$ & readout half-width, step \\",
              rf"\quad SSR tenors (weeks) & {quad['NS_weeks']} & snapshot grid \\",
              r"\midrule", r"\multicolumn{3}{l}{\emph{Parameters}} \\",
              rf"\quad free & {len(C.NAMES_N8)-1} of {len(C.NAMES_N8)}"
              r" & $\rho_{\mathsf S}$ pinned at $0$ \\",
              r"\quad solved, not fitted & $\bar\gamma$ & from~\eqref{eq:gbar} \\",
              rf"\quad $\nu_{{\mathsf L}}$ box & $[{box['nu_l'][0]}, {box['nu_l'][1]}]$ "
              r"& widened from the default \\",
              rf"\quad log-variance clamp & $[{clamp['log_variance'][0]:.0f}, "
              rf"{clamp['log_variance'][1]:.0f}]$ & before exponentiation; bounds $V_\ell$ \\",
              r"\quad floors $\ell$, $V$ & $10^{-6}$, $10^{-16}$ & leverage and variance \\",
              r"\midrule", r"\multicolumn{3}{l}{\emph{Objective and optimiser}} \\",
              rf"\quad $w_{{\mathrm{{vov}}}}$ & ${OPT['W_VOV']}$ & forward-variance block weight \\",
              r"\quad $\varsigma$ & $1$ (SPX), inverse rel.\ HAC s.e.\ (NDX)"
              r" & per-tenor SSR weights \\",
              rf"\quad ridge $\varrho$ & ${w_reg[0]:.2f}$ ($\nu,\lambda$), ${w_reg[4]:.2f}$ "
              r"($\rho,\kappa$) & relative, toward the stage anchor \\",
              rf"\quad ridge floor $\varepsilon$ & ${ridge['floor_eps']}$ & in "
              r"$\max(|\theta^0_k|,\varepsilon)$ \\",
              rf"\quad ftol, xtol & $10^{{{int(round(np.log10(OPT['FTOL'])))}}}$, "
              rf"$10^{{{int(round(np.log10(OPT['XTOL'])))}}}$ & least-squares stopping \\",
              rf"\quad max nfev & ${OPT['MAXNFEV']}$ & per stage \\",
              r"\midrule", r"\multicolumn{3}{l}{\emph{Protocol}} \\",
              r"\quad stage 1 & cold & LADDER $42$, VOVLEV $1$, $\lambda$ avg, "
              r"PIN $\rho_{\mathsf S}{=}0$ \\",
              r"\quad stage 2 & warm from stage 1 & LAMH $4.0$, PILLARAWARE $0$ \\",
              r"\quad acceptance & per date & stage 2 only where its data cost is lower \\",
              rf"\quad calendar interpolation & {determinism['calendar_interpolation']}"
              r" & between listed expiries \\",
              rf"\quad VIXFIX, VOVLAMTEN & {int(determinism['VIXFIX'])}, "
              rf"{determinism['VOVLAMTEN']} & expiry expansion; $\lambda$ window rule \\",
              r"\midrule", r"\multicolumn{3}{l}{\emph{Environment and determinism}} \\",
              rf"\quad device, dtype & \textsc{{{DEV}}}, {determinism['dtype']}"
              r" & readouts are device-sensitive \\",
              rf"\quad torch, python & {torch.__version__}, {platform.python_version()} & \\",
              r"\quad random seeds & none & quadrature and optimiser are deterministic \\",
              r"\bottomrule", r"\end{tabular}"]
        body = "\n".join(L) + "\n"
        written = []
        for tp in (os.path.join(_P.DATA, "manifest_table.tex"),
                   # written straight into the manuscript so the table cannot drift from the code
                   os.path.normpath(os.path.join(HERE, "..", "..", "manuscript_v3",
                                                 "manifest_table.tex"))):
            if os.path.isdir(os.path.dirname(tp)):
                open(tp, "w").write(body); written.append(tp)
        print(f"  wrote {jp}")
        for t in written:
            print(f"  wrote {t}")
