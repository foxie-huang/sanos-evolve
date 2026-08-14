#!/usr/bin/env python3
"""Does the CALIBRATED KERNEL's SSR out-hedge the industry minimum-variance delta?

The benchmark experiment in Section 5.6(B) never tested the kernel. Its `model` leg is
`_dk_forecaster` -- the Doeff-Kamal skew-decay SSR read analytically off the observed smile, with
no kernel anywhere -- and the only place the calibrated kernel appears is claim (A), against Black
(R=0), which every sensible R beats by 2.8-3.4x. So the question the paper implies but does not
answer is whether the fitted kernel's own readout beats the cross-strike Hull-White delta. This
runs it, as one more comparator in the same replay, under the same train-only affine map.

NO LOOK-AHEAD. Each annual fit is calibrated to that calendar year's realised SSR, so the fit dated
June of year Y uses data through December of Y. Using it to hedge inside Y would import roughly
seven months of the future. The forecaster therefore serves, on any date in year Y, the fit from the
most recent calibration year STRICTLY BEFORE Y. That is the strongest feasible version: a desk in
year Y could have had it.

    python3 kernel_hedge_test.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.normpath(os.path.join(HERE, "..", "..", "manuscript_v2", "scripts"))
sys.path.insert(0, V2)
sys.path.insert(0, os.path.dirname(HERE))
import _paths as _P                                                   # noqa: E402
import ssr_forecast_eval as E                                         # noqa: E402
import hullwhite as HW                                                # noqa: E402

SHIPPED = {"2012-06-01": "_dw9", "2016-06-01": "_dw9", "2017-06-01": "_dw9", "2018-06-01": "_dw9",
           "2019-06-03": "_n9", "2020-06-01": "_n9", "2021-06-01": "_n9", "2022-06-01": "_n9",
           "2024-06-03": "_n9"}
WIN, HORIZON = 63, 21

FIT = {}
for d, t in SHIPPED.items():
    FIT[int(d[:4])] = float(json.load(open(os.path.join(_P.DATA,
                                    f"fit_kf{t}_{d}.json")))["ssr"][2])      # 1m readout
YEARS = sorted(FIT)


def kernel_forecaster(dates):
    """R_t = the 1m SSR of the latest fit calibrated STRICTLY BEFORE this date's year."""
    yr = np.asarray(dates, int) // 10000

    def f(panel, t, tenor):
        prior = [y for y in YEARS if y < yr[t]]
        return FIT[max(prior)] if prior else np.nan
    return f


def purged_train(frame, embargo=HORIZON - 1):
    """Train mask with the last `embargo` rows dropped.

    LEAKAGE. `build_frame` splits at a row index and each row's TARGET is the realised comovement
    over (t, t+HORIZON]. The last HORIZON-1 training rows therefore look forward into the first
    test origins, so the affine level map -- which is fitted on the training targets -- is fitted
    partly on the period it is then applied to. Dropping those rows is the standard purge/embargo
    and costs 20 of ~600 training rows. Only `corrected()` consumes the train mask; the test set is
    untouched, so the replay itself is unchanged.
    """
    tr = frame.train_mask.copy()
    idx = np.flatnonzero(tr)
    if embargo > 0 and idx.size:
        tr[idx[-embargo:]] = False
    return tr


def corrected(frame, fc, purge=True):
    tr = purged_train(frame) if purge else frame.train_mask
    te = frame.test_mask
    y_tr = frame.target[tr]
    out = {}
    for k, v in fc.items():
        v = np.asarray(v, float); g = v.copy()
        g[te] = E.affine_bias_correct(y_tr, v[tr], v[te])
        out[k] = g
    return out


def run(y0, y1):
    dates, spot, vol, skew, sgrid = HW.load_all(y0, y1)
    panel = E.Panel(dates=dates, spot=spot,
                    atmiv={HW.TENORS[j]: vol[:, j] for j in range(4)},
                    skew={HW.TENORS[j]: skew[:, j] for j in range(4)})
    tenor = HW.TENORS[0]
    extras = {"kernel": kernel_forecaster(dates),
              "const": (lambda p, t, ten: 1.5),
              "atm_MV": E.RealisedSSRForecaster(window=WIN, as_ssr=True)}
    frame = E.build_frame(panel, tenor, HORIZON,
                          model=HW._dk_forecaster(skew, HW.TENORS),
                          bench=HW._arr_fc(HW.hullwhite_R_series(sgrid, skew[:, 0], spot, WIN)),
                          train_frac=0.5, min_trail=WIN, as_ssr=True, extra_forecasters=extras)
    fc = {"f_model": frame.f_model, "f_bench": frame.f_bench,
          "kernel": frame.extra["kernel"], "const": frame.extra["const"],
          "atm_MV": frame.extra["atm_MV"]}
    raw = E.hedging_replay(panel, frame, tenor, fc, baseline="f_bench")
    cor = E.hedging_replay(panel, frame, tenor, corrected(frame, fc), baseline="f_bench")
    te = frame.test_mask
    k = np.asarray(frame.extra["kernel"], float)[te]
    return frame, raw, cor, (np.nanmin(k), np.nanmax(k), np.nanstd(k),
                             float(np.mean(~np.isfinite(k))))


if __name__ == "__main__":
    print("  fitted 1m SSR by calibration year: " +
          "  ".join(f"{y}:{FIT[y]:.2f}" for y in YEARS) + "\n")
    out = {}
    for y0, y1 in ((2015, 2019), (2015, 2023)):
        frame, raw, cor, kstat = run(y0, y1)
        lo, hi, sd, miss = kstat
        print(f"  {y0}-{y1}   kernel R on test: {lo:.2f}-{hi:.2f} (sd {sd:.2f}), "
              f"{100*miss:.0f}% of test dates have no prior fit")
        for name, r in (("raw      ", raw), ("corrected", cor)):
            m = r["mean_resid_var"]
            print(f"    {name}  " + "  ".join(
                f"{k} {m[k]:.3f}" for k in ("kernel", "f_model", "const", "atm_MV", "f_bench",
                                            "black") if k in m)
                  + f"   DM(model vs HW) p={r['dm_model_vs_baseline']['p']:.3f}")
        out[f"{y0}_{y1}"] = {"raw": raw["mean_resid_var"], "corrected": cor["mean_resid_var"],
                             "kernel_R_range": [lo, hi], "kernel_R_sd": sd,
                             "test_dates_without_prior_fit": miss}
        print()
    jp = os.path.join(_P.DATA, "kernel_hedge_test.json")
    json.dump(out, open(jp, "w"), indent=1, default=float)
    print(f"  wrote {jp}")
