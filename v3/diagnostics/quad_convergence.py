#!/usr/bin/env python3
"""How much of the fit RMS is DISCRETISATION error, and what would removing it cost?

These counts are not parameters. Raising one cannot make the model represent anything new -- it only
shrinks the gap between the model-as-computed and the model-as-defined. So the honest measure is:
hold theta FIXED at its fitted value, raise one count, and see how far the model OUTPUT moves. That
displacement is an upper bound on the RMS that count could ever buy back.

Compared against the reference point from today: freeing kap_s from 0.9956 to 0.98 moved SPX vov RMS
5.18% -> 4.24% and cost 1.85x, at ZERO extra compute.
"""
import json, os, sys, time
import numpy as np
sys.argv = [sys.argv[0], "cpu"]
import torch; torch.set_num_threads(4)
import sys  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths as _P                        # noqa: E402
HERE = _P.DATA                             # code moved; fits/caches/records did not

sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "v2", "data")))
import calibrate_joint_torch as J
import calibrate_slv_exact_ts as C
import discslv_torch as DTt
import end_to_end as E

DATE = os.environ.get("DATE", "2022-06-01")
r = json.load(open(os.path.join(HERE, f"fit_norm_{DATE}_ks0.98.json")))
C.KAP_S_FIXED = r["kap_s"]
ctx, _c, _n = E.ctx_rebuilt(DATE, "SPX")
th = torch.tensor([r["theta"][k] for k in C.NAMES_N], dtype=torch.float32)
tgt = np.concatenate([np.asarray(r["ssr_target"], float), np.asarray(r["vov_target"], float)])
n_ssr = len(r["ssr_target"])
BASE_NL = DTt.build_kernel_n.__defaults__          # (n_l, n_x)


def run(tag, n_x=None, n_p=None, na=None, nb=None, nz=None, n_l=None):
    sv = (J.N_X, J.N_P, J.NA, J.NB, DTt.build_kernel_n.__defaults__)
    if n_x is not None: J.N_X = n_x
    if n_p is not None: J.N_P = n_p
    if na is not None: J.NA = na
    if nb is not None: J.NB = nb
    if n_l is not None:
        d = list(DTt.build_kernel_n.__defaults__); d[0] = n_l
        DTt.build_kernel_n.__defaults__ = tuple(d)
    t0 = time.time()
    with torch.no_grad():
        m = J.model_torch_n(th, ctx["LT"], ctx["sig_ref"], ctx["spot"], ctx["vdtes"],
                            nz=(nz if nz is not None else J.NZ)).numpy()
    w = time.time() - t0
    J.N_X, J.N_P, J.NA, J.NB, DTt.build_kernel_n.__defaults__ = sv
    return m, w


base, wbase = run("base")
# rms takes the matching target slice explicitly -- `tgt` is the concatenated 17-vector, so
# slicing the model without slicing the target broadcasts wrong.
rms = lambda m, t: 100 * float(np.sqrt(np.mean(((m - t) / t) ** 2)))
TS, TV = tgt[:n_ssr], tgt[n_ssr:]
brs = rms(base[:n_ssr], TS); brv = rms(base[n_ssr:], TV)
print(f"{DATE}  SPX  kap_s={r['kap_s']}  baseline: n_l=5 n_x=3 n_p=5 na=5 nb=3 nz={J.NZ}")
print(f"  baseline  SSR RMS {brs:.3f}%   vov RMS {brv:.3f}%   [{wbase:.0f}s]\n")
print("%-22s %10s %10s %9s %9s %8s" % ("raised count", "max dSSR", "max dvov", "SSR RMS", "vov RMS", "time"))
# na/nb moved the SSR block 16-18%, which is not a convergence tail -- push it out to see whether it
# SETTLES (then 5,3 is simply under-resolved) or keeps drifting (then _stat_nodes_n has a defect).
# NOTE na=5, nb=3 were chosen to MATCH THE OLD n_f=5, n_s=3 "so cost is unchanged" -- those were
# CARRIED abscissa counts for the factors, a different role entirely. There is no record of na/nb
# ever being convergence-tested in their own right.
CFG = [("na,nb 5,3 -> 7,5", dict(na=7, nb=5)), ("na,nb 5,3 -> 9,7", dict(na=9, nb=7)),
       ("na,nb 5,3 -> 11,9", dict(na=11, nb=9)), ("na,nb 5,3 -> 13,11", dict(na=13, nb=11)),
       ("na only 5 -> 9", dict(na=9)), ("nb only 3 -> 7", dict(nb=7))]
rows = []
for tag, kw in CFG:
    m, w = run(tag, **kw)
    ds = 100 * np.abs((m[:n_ssr] - base[:n_ssr]) / base[:n_ssr]).max()
    dv = 100 * np.abs((m[n_ssr:] - base[n_ssr:]) / base[n_ssr:]).max()
    print("  %-20s %9.3f%% %9.3f%% %8.3f%% %8.3f%% %7.1fx"
          % (tag, ds, dv, rms(m[:n_ssr], TS), rms(m[n_ssr:], TV), w / wbase))
    rows.append(dict(cfg=tag, max_dssr=float(ds), max_dvov=float(dv), ssr_rms=rms(m[:n_ssr], TS),
                     vov_rms=rms(m[n_ssr:], TV), time_x=w / wbase))
json.dump(dict(date=DATE, kap_s=r["kap_s"], base_ssr_rms=brs, base_vov_rms=brv,
               base_wall_s=wbase, rows=rows),
          open(os.path.join(HERE, f"quad_convergence_{DATE}.json"), "w"), indent=1)
print(f"\n  REFERENCE: kap_s 0.9956 -> 0.98 moved SPX vov RMS 5.18% -> 4.24% at 1.0x compute.")
