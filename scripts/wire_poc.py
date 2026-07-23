"""
TEMPLATE: wire the SSR-forecast harness to the real POC engine + ORATS data.
============================================================================

This is a fill-in template, NOT a runnable script. It shows the exact calls into
`poc/` (signatures verified) so a real out-of-sample run is a matter of supplying the
ORATS panel and the per-date calibration. Nothing here fabricates results.

Fill in the two TODO blocks (`load_orats_panel`, and the calibration inside
`PocModelForecaster.__call__`), then:

    frame = build_frame(panel, tenor=1/12, horizon=21, model=PocModelForecaster(),
                        bench=RealisedSSRForecaster(window=63, as_ssr=True),
                        train_frac=0.5, as_ssr=True,
                        extra_forecasters={"persistence": persistence_forecaster(21),
                                           "const": lambda p,t,ten: 1.6})
    print(format_report(evaluate(frame), hedging_replay(panel, frame, 1/12, {...})))

Run per tenor T in {1/52, 1/12, 1/4} and report Tables A/B/C from the design doc.
"""
from __future__ import annotations
import os, sys
import numpy as np

# make poc importable ---------------------------------------------------------------
_POC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "poc"))
sys.path.insert(0, _POC)

from ssr_forecast_eval import (Panel, RealisedSSRForecaster, persistence_forecaster,
                               build_frame, evaluate, hedging_replay, format_report)


# ======================================================================================
# 1. DATA  ->  Panel     (TODO: point at the ORATS cache; memory: ORATS EOD SPX+VIX 2010-2026)
# ======================================================================================
def load_orats_panel(tenors=(1/52, 1/12, 1/4)) -> Panel:
    """Build a daily Panel from the ORATS EOD cache.

    Required per trading day: spot S_t, and per tenor T a CONSTANT-MATURITY ATM implied vol
    and ATM skew (d sigma / d k at k=0). Use the SANOS/interpolated surface for the CMV ATM
    series (not raw nearest-strike quotes) so the ATM proxy is consistent day to day.

    Also stash the per-date option cross-section (or its cached SANOS marginals) in
    `Panel.xsec[date]` so PocModelForecaster can calibrate at t.

    TODO: implement using the project's ORATS loader / data_port.load_chain(csv_path).
    Return a Panel with dates sorted ascending.
    """
    raise NotImplementedError(
        "Wire to ORATS: for each date build CMV ATM vol + ATM skew per tenor, and stash the "
        "t-cross-section in xsec[date]. See poc/data_port.load_chain and fetch_data.py schema."
    )


# ======================================================================================
# 2. MODEL forecast hook  (forward-looking, Q):  calibrate at t -> ssr_2f
# ======================================================================================
class PocModelForecaster:
    """Calibrate SANOS-Evolve to the time-t cross-section and read SSR^Q_t(T).

    Uses the exact POC entry points:
        calibrate_2f.kernel(x)          # x = 9 params -> TwoFactorSV
        discslv_2f.ssr_2f(K, n, nk, dm) # -> (ssr, atm_vol, atm_skew); n = round(T / K.dt)
    Calibration mirrors calibrate_2f.py: scipy.optimize.least_squares(residuals, X0, bounds=...).

    Caching is important: calibration is ~60-120 s/date (interface map), so cache theta by date.
    """
    def __init__(self, nk: int = 16, cache: dict | None = None):
        self.nk = nk
        self.cache = {} if cache is None else cache
        from calibrate_2f import kernel                # noqa: verified signature
        from discslv_2f import ssr_2f                  # noqa: verified signature
        self._kernel, self._ssr_2f = kernel, ssr_2f

    def _theta_at(self, panel: Panel, t: int) -> np.ndarray:
        key = panel.dates[t]
        if key in self.cache:
            return self.cache[key]
        # TODO: build the calibration TARGETS from the t cross-section (statics + a dynamic anchor:
        #       forward-starts or VIX where available, else a realised anchor), then:
        #   from scipy.optimize import least_squares
        #   res = least_squares(residuals, X0, bounds=(LO, HI), diff_step=3e-2, max_nfev=160)
        #   theta = res.x
        # NOTE: use ONLY information dated <= t (no look-ahead).
        raise NotImplementedError("Supply per-date calibration targets from panel.xsec[panel.dates[t]].")

    def __call__(self, panel: Panel, t: int, tenor: float) -> float:
        theta = self._theta_at(panel, t)
        K = self._kernel(theta)
        n = max(1, int(round(tenor / K.dt)))
        return float(self._ssr_2f(K, n, nk=self.nk)[0])          # SSR^Q_t(T)


# ======================================================================================
# 3. Driver skeleton
# ======================================================================================
def run(tenor: float = 1/12, horizon: int = 21, trail_w: int = 63,
        const_R: float = 1.6, train_frac: float = 0.5):
    panel = load_orats_panel()
    model = PocModelForecaster()
    bench = RealisedSSRForecaster(window=trail_w, as_ssr=True)     # SSR units (matches ssr_2f)
    extras = {"persistence": persistence_forecaster(horizon, as_ssr=True),
              "const": (lambda p, t, ten: const_R)}
    frame = build_frame(panel, tenor, horizon, model=model, bench=bench,
                        train_frac=train_frac, as_ssr=True, extra_forecasters=extras)
    ev = evaluate(frame, bias_correct=True)
    fc = {"f_model": frame.f_model, "f_bench": frame.f_bench,
          "persistence": frame.extra["persistence"], "const": frame.extra["const"]}
    hr = hedging_replay(panel, frame, tenor, fc)
    print(format_report(ev, hr))
    return ev, hr


if __name__ == "__main__":
    print(__doc__)
    print("Fill in load_orats_panel() and PocModelForecaster._theta_at(), then call run(tenor=...).")
