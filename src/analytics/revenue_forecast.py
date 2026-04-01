"""
Revenue Forecasting — 30 / 60 / 90 day projections.

Uses Holt-Winters Exponential Smoothing (ETS) on the monthly MRR series,
extended with a linear trend extrapolation for short-horizon accuracy.
Provides confidence intervals and scenario bands (optimistic / base / pessimistic).
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats

log = logging.getLogger(__name__)


class RevenueForecaster:
    """
    Forecast MRR 90 days into the future.

    Fitting strategy:
      - Primary: ETS (Holt-Winters additive) for trend + seasonality
      - Fallback: linear regression for very short series
    """

    def __init__(self, seasonal_periods: int = 12):
        self.seasonal_periods = seasonal_periods
        self.model    = None
        self.fit_result = None
        self.history:  Optional[pd.Series] = None
        self.last_date: Optional[pd.Timestamp] = None
        self.fitted:   bool = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, monthly_revenue: pd.DataFrame,
            date_col: str = "month",
            value_col: str = "total_mrr") -> "RevenueForecaster":
        """
        Fit the forecasting model on historical monthly MRR.

        Parameters
        ----------
        monthly_revenue : DataFrame with at least date_col and value_col
        date_col        : Name of the month column
        value_col       : Name of the MRR column
        """
        df = monthly_revenue[[date_col, value_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).dropna()
        df = df[df[value_col] > 0]

        ts = df.set_index(date_col)[value_col]
        ts.index = pd.DatetimeIndex(ts.index).to_period("M")
        self.history   = ts
        self.last_date = df[date_col].max()
        n = len(ts)
        log.info("Fitting forecast model on %d monthly observations …", n)

        if n >= 24:
            try:
                self.model = ExponentialSmoothing(
                    ts,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=self.seasonal_periods,
                    initialization_method="estimated",
                )
                self.fit_result = self.model.fit(optimized=True, remove_bias=True)
                self.fitted = True
                log.info("  ETS model fitted (AIC=%.1f)", self.fit_result.aic)
                return self
            except Exception as e:
                log.warning("  ETS failed: %s — falling back to SARIMAX", e)

        if n >= 12:
            try:
                self.model = SARIMAX(
                    ts,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.fit_result = self.model.fit(disp=False)
                self.fitted = True
                log.info("  SARIMAX model fitted")
                return self
            except Exception as e:
                log.warning("  SARIMAX failed: %s — falling back to linear trend", e)

        # Linear trend fallback
        log.info("  Using linear trend extrapolation")
        x = np.arange(n)
        slope, intercept, r, p, se = stats.linregress(x, ts.values)
        self.model = {"type": "linear", "slope": slope, "intercept": intercept,
                      "se": se, "n": n}
        self.fitted = True
        return self

    # ── Forecasting ───────────────────────────────────────────────────────────

    def forecast(self,
                 horizon_days: int = 90,
                 confidence: float = 0.90) -> pd.DataFrame:
        """
        Produce daily forecast for the next horizon_days.

        Returns
        -------
        DataFrame with columns:
          ds, forecast, lower_bound, upper_bound,
          scenario_optimistic, scenario_pessimistic
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before forecast().")

        horizon_months = max(3, -(-horizon_days // 30))  # ceiling division

        if isinstance(self.model, dict):
            # linear fallback
            n = self.model["n"]
            x_future = np.arange(n, n + horizon_months)
            monthly_forecast = (
                self.model["slope"] * x_future + self.model["intercept"]
            )
            z = stats.norm.ppf((1 + confidence) / 2)
            residual_std = self.model["se"] * np.sqrt(1 + 1/n + (x_future - n/2)**2 / (n**3/12))
            lower = monthly_forecast - z * residual_std
            upper = monthly_forecast + z * residual_std
        else:
            # ETS or SARIMAX
            try:
                if hasattr(self.fit_result, "get_forecast"):
                    # SARIMAX path
                    fc = self.fit_result.get_forecast(steps=horizon_months)
                    fc_summary = fc.summary_frame(alpha=(1 - confidence))
                    monthly_forecast = fc_summary["mean"].values
                    lower = fc_summary["mean_ci_lower"].values
                    upper = fc_summary["mean_ci_upper"].values
                else:
                    # HoltWinters (ETS) — no built-in prediction_intervals in
                    # older statsmodels; derive CI from in-sample residual std.
                    monthly_forecast = self.fit_result.forecast(horizon_months).values
                    resid_std = float(np.std(self.fit_result.resid))
                    z = stats.norm.ppf((1 + confidence) / 2)
                    # widen interval with horizon (uncertainty grows over time)
                    horizon_scale = np.sqrt(np.arange(1, horizon_months + 1))
                    lower = monthly_forecast - z * resid_std * horizon_scale
                    upper = monthly_forecast + z * resid_std * horizon_scale
            except Exception as e:
                log.warning("Forecast extraction failed: %s — using naive method", e)
                last_val = float(self.history.iloc[-1])
                growth = float(self.history.pct_change().mean())
                monthly_forecast = np.array([
                    last_val * (1 + growth) ** i for i in range(1, horizon_months + 1)
                ])
                std = float(self.history.std())
                z = stats.norm.ppf((1 + confidence) / 2)
                lower = monthly_forecast - z * std
                upper = monthly_forecast + z * std

        # Scenario multipliers based on historical volatility
        hist_vals = self.history.values.astype(float)
        growth_rates = np.diff(hist_vals) / hist_vals[:-1]
        optimistic_mult  = 1 + (np.percentile(growth_rates, 75) * horizon_months / 12)
        pessimistic_mult = 1 + (np.percentile(growth_rates, 25) * horizon_months / 12)

        # Interpolate monthly → daily
        start_date = self.last_date + pd.DateOffset(days=1)
        end_date   = self.last_date + pd.DateOffset(days=horizon_days)
        dates      = pd.date_range(start=start_date, end=end_date, freq="D")

        # Create monthly anchor dates for interpolation
        month_ends = pd.date_range(
            start=self.last_date + pd.offsets.MonthEnd(1),
            periods=horizon_months,
            freq="ME",
        )

        # Build daily series via linear interpolation between monthly anchors
        last_actual = float(self.history.iloc[-1])
        monthly_vals = np.concatenate([[last_actual], monthly_forecast])
        lower_vals   = np.concatenate([[last_actual], np.maximum(lower, 0)])
        upper_vals   = np.concatenate([[last_actual], upper])

        monthly_anchors = pd.Series(
            monthly_vals,
            index=pd.DatetimeIndex(
                [self.last_date] + list(month_ends[:horizon_months])
            ),
        ).reindex(
            pd.date_range(self.last_date, end_date, freq="D")
        ).interpolate("time")

        lower_interp = pd.Series(
            lower_vals,
            index=pd.DatetimeIndex([self.last_date] + list(month_ends[:horizon_months]))
        ).reindex(
            pd.date_range(self.last_date, end_date, freq="D")
        ).interpolate("time")

        upper_interp = pd.Series(
            upper_vals,
            index=pd.DatetimeIndex([self.last_date] + list(month_ends[:horizon_months]))
        ).reindex(
            pd.date_range(self.last_date, end_date, freq="D")
        ).interpolate("time")

        daily = pd.DataFrame({
            "ds":                  monthly_anchors.index[1:],
            "forecast":            monthly_anchors.values[1:],
            "lower_bound":         lower_interp.values[1:],
            "upper_bound":         upper_interp.values[1:],
        })
        daily["lower_bound"] = daily["lower_bound"].clip(lower=0)

        # scenario bands
        daily["scenario_optimistic"]  = daily["forecast"] * np.linspace(1.0, optimistic_mult, len(daily))
        daily["scenario_pessimistic"] = daily["forecast"] * np.linspace(1.0, pessimistic_mult, len(daily))
        daily["scenario_pessimistic"] = daily["scenario_pessimistic"].clip(lower=0)

        # Horizon labels
        daily["horizon"] = "90d"
        daily.loc[daily["ds"] <= self.last_date + pd.DateOffset(days=30), "horizon"] = "30d"
        daily.loc[
            (daily["ds"] > self.last_date + pd.DateOffset(days=30)) &
            (daily["ds"] <= self.last_date + pd.DateOffset(days=60)),
            "horizon"
        ] = "60d"

        return daily.reset_index(drop=True)

    def horizon_summary(self,
                        daily_forecast: pd.DataFrame) -> dict:
        """
        Summarise MRR projections at 30, 60, and 90 day marks.
        """
        last_actual = float(self.history.iloc[-1]) if self.history is not None else 0

        def _at_day(n):
            target = self.last_date + pd.DateOffset(days=n)
            row = daily_forecast[daily_forecast["ds"] <= target].iloc[-1]
            return {
                "forecast":      round(float(row["forecast"]), 0),
                "lower":         round(float(row["lower_bound"]), 0),
                "upper":         round(float(row["upper_bound"]), 0),
                "optimistic":    round(float(row["scenario_optimistic"]), 0),
                "pessimistic":   round(float(row["scenario_pessimistic"]), 0),
                "pct_change":    round(
                    100 * (float(row["forecast"]) - last_actual) / max(last_actual, 1), 1
                ),
            }

        return {
            "30d":  _at_day(30),
            "60d":  _at_day(60),
            "90d":  _at_day(90),
            "last_actual_mrr": last_actual,
        }

    def historical_with_forecast(self, daily_forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Combine historical MRR with forecast for a seamless chart.
        """
        hist = self.history.to_timestamp().reset_index()
        hist.columns = ["ds", "actual_mrr"]
        hist["ds"] = pd.to_datetime(hist["ds"])

        fc = daily_forecast.copy()
        fc["actual_mrr"] = np.nan

        combined = pd.concat([hist, fc], ignore_index=True).sort_values("ds")
        return combined
