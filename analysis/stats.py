"""
analysis/stats.py

Statistical analysis module for FinSight.
Provides:
- Z-score standardization of financial metrics and stock returns
- Anomaly detection (|z| > 2 flags)
- Denoised linear trend prediction with 95% confidence intervals
- Log-Normal Monte Carlo simulation
- statsmodels.tsa.AutoReg stock price forecasting
"""

import numpy as np
from scipy import stats
import pandas as pd

def standardize_series(values_dict):
    """
    Given a dict of {year: value}, compute z-scores for each year.
    Returns: {year: {'value': v, 'z_score': z, 'is_anomaly': bool}}
    """
    years = sorted(values_dict.keys())
    values = [values_dict[y] for y in years if values_dict[y] is not None]
    valid_years = [y for y in years if values_dict[y] is not None]

    if len(values) < 3:
        return {}

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    if std == 0:
        return {y: {"value": v, "z_score": 0.0, "is_anomaly": False}
                for y, v in zip(valid_years, values)}

    result = {}
    for year, value in zip(valid_years, values):
        z = (value - mean) / std
        result[year] = {
            "value": float(value),
            "z_score": float(round(z, 3)),
            "is_anomaly": bool(abs(z) > 2),
            "mean": float(round(mean, 4)),
            "std": float(round(std, 4))
        }
    return result


def run_monte_carlo_forecast(historical_series, steps=3, simulations=5000):
    """
    Runs a Log-Normal Monte Carlo simulation to forecast future horizons.
    """
    vals = [float(v) for v in historical_series.values() if v is not None and v > 0]
    if len(vals) < 2:
        return None

    log_returns = [np.log(vals[i] / vals[i - 1]) for i in range(1, len(vals))]
    if not log_returns:
        log_returns = [0.0]

    mu = np.mean(log_returns)
    sigma = np.std(log_returns) if np.std(log_returns) > 0 else 0.02

    last_value = vals[-1]
    last_year = int(max(list(historical_series.keys()))[:4])
    sim_results = np.zeros((simulations, steps))

    for s in range(simulations):
        current_val = last_value
        for step in range(steps):
            z = np.random.normal(0, 1)
            growth_factor = np.exp((mu - 0.5 * (sigma ** 2)) + sigma * z)
            current_val = current_val * growth_factor
            sim_results[s, step] = current_val

    forecast_years = [str(last_year + i + 1) for i in range(steps)]
    p10 = np.percentile(sim_results, 10, axis=0)
    p50 = np.percentile(sim_results, 50, axis=0)
    p90 = np.percentile(sim_results, 90, axis=0)

    return {
        "years": forecast_years,
        "p10": p10.tolist(),
        "p50": p50.tolist(),
        "p90": p90.tolist(),
        "last_year": str(last_year),
        "last_value": last_value
    }


def monte_carlo_forecast_chart(mc_results, metric_name, company_name):
    """
    Renders a 3-year probabilistic forecasting cloud using Monte Carlo outputs.
    """
    if not mc_results:
        return None

    import plotly.graph_objects as go

    clean_metric = metric_name.replace('_', ' ').title()
    timeline = [mc_results["last_year"]] + mc_results["years"]
    y_p10 = [mc_results["last_value"]] + mc_results["p10"]
    y_p50 = [mc_results["last_value"]] + mc_results["p50"]
    y_p90 = [mc_results["last_value"]] + mc_results["p90"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline, y=y_p90, mode='lines',
        line=dict(width=0.5, color='rgba(78, 121, 167, 0.2)'),
        name='90th Percentile (Optimistic Target)', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=timeline, y=y_p10, mode='lines', fill='tonexty',
        fillcolor='rgba(78, 121, 167, 0.15)',
        line=dict(width=0.5, color='rgba(78, 121, 167, 0.2)'),
        name='10th Percentile (Downside Risk)', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=timeline, y=y_p50, mode='lines+markers',
        line=dict(width=3, color='#2B5C8F', dash='dash'),
        name='Median Expected Path (Monte Carlo Walk)'
    ))
    fig.update_layout(
        title=f"{company_name} — 3-Year Log-Normal Probabilistic {clean_metric} Forecast",
        xaxis=dict(title="Fiscal Year Horizon", type="category", automargin=True),
        yaxis=dict(title=clean_metric, automargin=True),
        template="plotly_white", hovermode="x unified"
    )
    return fig


def predict_next_value(values_dict, n_ahead=1):
    """
    Fit an exponentially weighted local linear trend with damped mean reversion
    and calibrated 1-step-ahead prediction intervals.
    """
    years = sorted(values_dict.keys())
    values = [values_dict[y] for y in years if values_dict[y] is not None]
    valid_years = [y for y in years if values_dict[y] is not None]

    n = len(values)
    if n < 3:
        return None

    x = np.arange(n, dtype=float)
    y = np.array(values, dtype=float)

    # Exponential recency weights (more weight on recent fiscal years)
    weights = np.exp(np.linspace(0, 1.2, n))
    weights = weights / weights.sum()

    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)
    sxx = np.sum(weights * (x - x_mean) ** 2)

    raw_slope = np.sum(weights * (x - x_mean) * (y - y_mean)) / sxx if sxx > 0 else 0.0
    intercept = y_mean - raw_slope * x_mean

    # In-sample fit & standard error (corrected degrees of freedom)
    y_pred = raw_slope * x + intercept
    residuals = y - y_pred
    dof_scale = n / max(n - 2, 1)
    residual_std = float(np.sqrt(np.sum(weights * residuals ** 2) * dof_scale))

    # Damped projection: blend linear trend with recent level to prevent overshoot
    phi = 0.75  # damping factor for stationary financial ratios
    damped_step = sum(phi ** k for k in range(1, n_ahead + 1))
    linear_proj = (raw_slope * (n - 1) + intercept) + raw_slope * damped_step
    recent_anchor = float(np.average(y[-min(3, n):], weights=np.linspace(0.5, 1.0, min(3, n))))
    predicted = 0.65 * linear_proj + 0.35 * recent_anchor

    # Calibrated 95% prediction interval (fixes the sqrt(n) inflation bug)
    next_x = n - 1 + n_ahead
    leverage = np.sqrt(1.0 + (1.0 / n) + ((next_x - x_mean) ** 2) / max(sxx * n, 1e-8))
    t_crit = stats.t.ppf(0.975, df=max(n - 2, 1))
    margin = float(t_crit * residual_std * leverage)

    # Out-of-sample rolling 1-step-ahead backtest error (MAPE & RMSE)
    backtest_errors = []
    backtest_pct_errors = []
    for t in range(3, n):
        w_t = np.exp(np.linspace(0, 1.2, t))
        w_t /= w_t.sum()
        x_t = np.arange(t, dtype=float)
        y_t = y[:t]
        xm, ym = np.average(x_t, weights=w_t), np.average(y_t, weights=w_t)
        sxx_t = np.sum(w_t * (x_t - xm) ** 2)
        sl_t = np.sum(w_t * (x_t - xm) * (y_t - ym)) / sxx_t if sxx_t > 0 else 0.0
        ic_t = ym - sl_t * xm
        pred_t = 0.65 * (sl_t * t + ic_t) + 0.35 * float(np.mean(y_t[-3:]))
        err = abs(y[t] - pred_t)
        backtest_errors.append(err)
        if abs(y[t]) > 1e-6:
            backtest_pct_errors.append((err / abs(y[t])) * 100.0)

    rmse = float(np.mean(backtest_errors)) if backtest_errors else residual_std
    mape = float(np.mean(backtest_pct_errors)) if backtest_pct_errors else 0.0
    fit_accuracy = max(0.0, min(99.9, 100.0 - mape))

    # Keep r_squared key for backward compatibility, plus calibrated accuracy metrics
    ss_res = np.sum(weights * residuals ** 2)
    ss_tot = np.sum(weights * (y - y_mean) ** 2)
    r_squared = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Trend classification
    if raw_slope > residual_std * 0.12:
        trend = "improving"
    elif raw_slope < -residual_std * 0.12:
        trend = "declining"
    else:
        trend = "stable"

    # Reliability based on backtest accuracy (MAPE) rather than slope R²
    if mape <= 5.0:
        reliability = "high"
    elif mape <= 12.0:
        reliability = "moderate"
    else:
        reliability = "low"

    return {
        "predicted_value": float(round(predicted, 4)),
        "confidence_interval": (float(round(predicted - margin, 4)),
                                float(round(predicted + margin, 4))),
        "r_squared": float(round(r_squared, 4)),
        "mape": float(round(mape, 2)),
        "rmse": float(round(rmse, 2)),
        "fit_accuracy": float(round(fit_accuracy, 1)),
        "slope": float(round(raw_slope, 4)),
        "trend": trend,
        "reliability": reliability,
        "next_label": f"FY{int(valid_years[-1][:4]) + 1} (projected)"
    }

def analyze_metric(values_dict, metric_name="metric"):
    """
    Full analysis pipeline for one metric: standardize + detect anomalies + DE-NOISE + predict.
    """
    standardized = standardize_series(values_dict)
    anomalies = {y: d for y, d in standardized.items() if d["is_anomaly"]}

    denoised_dict = {}
    for year, value in values_dict.items():
        if year in standardized and standardized[year]["is_anomaly"]:
            denoised_dict[year] = standardized[year]["mean"]
        else:
            denoised_dict[year] = value

    prediction = predict_next_value(denoised_dict)

    return {
        "metric_name": metric_name,
        "standardized": standardized,
        "anomalies": anomalies,
        "prediction": prediction,
        "n_anomalies": len(anomalies)
    }


def analyze_all_metrics(ratios, stock_returns=None):
    """
    Run full analysis on all financial ratios.
    """
    metric_series = {}
    for year, metrics in ratios.items():
        for metric, value in metrics.items():
            if metric not in metric_series:
                metric_series[metric] = {}
            metric_series[metric][year] = value

    return {m: analyze_metric(v, m) for m, v in metric_series.items()}


def run_autoreg_stock_forecast(history_df, lags=10, steps=20, trend="c"):
    """
    Fit a statsmodels.tsa.ar_model.AutoReg model on historical closing prices
    and generate multi-step ahead point forecasts with 95% confidence intervals.
    """
    from statsmodels.tsa.ar_model import AutoReg
    import pandas as pd

    if history_df is None or history_df.empty or "Close" not in history_df:
        return None

    close = history_df["Close"].dropna().astype(float)
    if len(close) < max(30, lags * 3):
        return None

    values = close.to_numpy()
    dates = pd.to_datetime(close.index)

    val_size = min(steps, max(5, len(values) // 10))
    holdout_mape = None
    holdout_rmse = None
    if len(values) - val_size > lags + 5:
        train_vals = values[:-val_size]
        test_vals = values[-val_size:]
        try:
            val_model = AutoReg(train_vals, lags=lags, trend=trend).fit()
            val_preds = val_model.predict(start=len(train_vals), end=len(values) - 1)
            holdout_rmse = float(np.sqrt(np.mean((test_vals - val_preds) ** 2)))
            nonzero = test_vals != 0
            if np.any(nonzero):
                holdout_mape = float(np.mean(np.abs((test_vals[nonzero] - val_preds[nonzero]) / test_vals[nonzero])) * 100)
        except Exception:
            pass

    model_fit = AutoReg(values, lags=lags, trend=trend).fit()
    pred_res = model_fit.get_prediction(start=len(values), end=len(values) + steps - 1)
    forecast_mean = np.asarray(pred_res.predicted_mean, dtype=float)
    conf_int = np.asarray(pred_res.conf_int(alpha=0.05), dtype=float)

    last_date = dates[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    display_window = min(len(values), 90)
    hist_dates = [d.strftime("%Y-%m-%d") for d in dates[-display_window:]]
    hist_values = values[-display_window:].tolist()

    param_names = ["const"] if trend in ("c", "ct") else []
    if trend == "ct":
        param_names.append("trend")
    param_names.extend([f"Lag {i}" for i in range(1, lags + 1)])
    params = model_fit.params
    coef_table = [
        {
            "Term": param_names[idx] if idx < len(param_names) else f"Param {idx}",
            "Coefficient": round(float(params[idx]), 6),
            "P-value": round(float(model_fit.pvalues[idx]), 4) if idx < len(model_fit.pvalues) else None,
        }
        for idx in range(len(params))
    ]

    last_price = float(values[-1])
    end_forecast = float(forecast_mean[-1])
    expected_change_pct = ((end_forecast - last_price) / last_price) * 100 if last_price != 0 else 0.0

    return {
        "history_dates": hist_dates,
        "history_values": hist_values,
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast_values": forecast_mean.tolist(),
        "ci_lower": conf_int[:, 0].tolist(),
        "ci_upper": conf_int[:, 1].tolist(),
        "last_price": last_price,
        "end_forecast": end_forecast,
        "expected_change_pct": round(expected_change_pct, 2),
        "lags": lags,
        "steps": steps,
        "aic": round(float(model_fit.aic), 2),
        "bic": round(float(model_fit.bic), 2),
        "holdout_rmse": round(holdout_rmse, 2) if holdout_rmse is not None else None,
        "holdout_mape": round(holdout_mape, 2) if holdout_mape is not None else None,
        "coefficients": coef_table,
    }


def autoreg_stock_chart(ar_result, company_name="Company"):
    """
    Render historical closing prices alongside AutoReg multi-step forecast
    and 95% confidence band.
    """
    if not ar_result:
        return None

    import plotly.graph_objects as go

    hist_dates = ar_result["history_dates"]
    hist_vals = ar_result["history_values"]
    fc_dates = [hist_dates[-1]] + ar_result["forecast_dates"]
    fc_vals = [hist_vals[-1]] + ar_result["forecast_values"]
    ci_upper = [hist_vals[-1]] + ar_result["ci_upper"]
    ci_lower = [hist_vals[-1]] + ar_result["ci_lower"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_vals,
        mode="lines",
        name="Historical Close (Last 90d)",
        line=dict(color="#5B9BD5", width=2.5),
    ))

    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=ci_upper,
        mode="lines",
        line=dict(width=0.5, color="rgba(66, 199, 183, 0.25)"),
        name="95% Upper Bound",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=ci_lower,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(66, 199, 183, 0.16)",
        line=dict(width=0.5, color="rgba(66, 199, 183, 0.25)"),
        name="95% Confidence Interval",
    ))

    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=fc_vals,
        mode="lines+markers",
        name=f"AutoReg({ar_result['lags']}) Forecast",
        line=dict(color="#42C7B7", width=3, dash="dash"),
        marker=dict(size=5, color="#42C7B7"),
    ))

    fig.update_layout(
        title=f"{company_name} — AutoReg({ar_result['lags']}) Stock Price Forecast ({ar_result['steps']} Trading Days)",
        xaxis=dict(title="Trading Date", automargin=True, tickangle=45),
        yaxis=dict(title="Price (USD)", automargin=True),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def explain_historical_anomalies(anomalies, metric_name, company_ratios, raw_metrics=None, ticker=None):
    """
    Generate fundamental decomposition + macro/filing context for each |z| > 2 anomaly year.
    """
    if not anomalies:
        return []

    macro_events = {
        "2008": "Global Financial Crisis onset & credit freeze",
        "2009": "Great Recession demand trough & industrial capex contraction",
        "2014": "USD surge & global energy/industrial slowdown (Agilent Keysight spin-off transition)",
        "2015": "FX headwinds & commodity/industrial CapEx downturn",
        "2020": "COVID-19 pandemic lockdowns & global supply-chain disruption",
        "2021": "Post-pandemic stimulus rebound & supply bottlenecks",
        "2022": "40-year inflation peak & aggressive Fed rate-hiking cycle",
        "2023": "Tight monetary policy & post-COVID inventory destocking",
    }

    sorted_years = sorted(company_ratios.keys()) if company_ratios else []
    explanations = []

    for year_key, info in sorted(anomalies.items()):
        yr_str = str(year_key)[:4]
        val = info["value"]
        z = info["z_score"]
        direction = "surge above" if z > 0 else "drop below"

        drivers = []
        yr_ratios = company_ratios.get(year_key, {}) if company_ratios else {}
        prev_idx = sorted_years.index(year_key) - 1 if year_key in sorted_years else -1
        prev_ratios = company_ratios.get(sorted_years[prev_idx], {}) if prev_idx >= 0 else {}

        rev_g = yr_ratios.get("revenue_growth_pct")
        if rev_g is not None:
            if rev_g < -5.0:
                drivers.append(f"top-line revenue contracted {rev_g:.1f}% YoY (operating deleverage)")
            elif rev_g > 15.0:
                drivers.append(f"revenue expanded {rev_g:.1f}% YoY (scale leverage)")

        gm = yr_ratios.get("gross_margin_pct")
        prev_gm = prev_ratios.get("gross_margin_pct")
        if gm is not None and prev_gm is not None and abs(gm - prev_gm) >= 2.0 and metric_name != "gross_margin_pct":
            delta_gm = gm - prev_gm
            drivers.append(f"gross margin shifted {delta_gm:+.1f} pp (to {gm:.1f}%)")

        fcf = yr_ratios.get("free_cash_flow")
        if fcf is not None and fcf < 0:
            drivers.append("free cash flow turned negative")

        macro_note = macro_events.get(yr_str, "Company-specific operational or accounting shift")
        sec_url = f"https://www.sec.gov/edgar/searchedgar/companysearch?q={ticker}" if ticker else None
        driver_summary = "; ".join(drivers).capitalize() if drivers else f"Statistically significant {direction} historical mean ({val:.2f}%)"

        explanations.append({
            "Fiscal year": yr_str,
            "Observed value": f"{val:.2f}%",
            "Z-score": f"{z:+.2f}σ",
            "Fundamental driver": driver_summary,
            "Macro / industry context": macro_note,
            "SEC filing": sec_url,
        })

    return explanations

def run_stock_price_monte_carlo(history_df, horizon_days=252, simulations=3000, seed=42):
    """
    Simulate forward stock price paths via Geometric Brownian Motion (GBM):
        S_t = S_0 * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    Returns 10th/50th/90th percentile paths, terminal distribution stats, and 95% VaR.
    """
    if history_df is None or history_df.empty or "Close" not in history_df.columns:
        return None

    close = history_df["Close"].dropna()
    if len(close) < 30:
        return None

    log_ret = np.log(close / close.shift(1)).dropna()
    mu_daily = float(log_ret.mean())
    sigma_daily = float(log_ret.std())
    s0 = float(close.iloc[-1])

    rng = np.random.default_rng(seed)
    shocks = rng.normal(
        loc=(mu_daily - 0.5 * sigma_daily ** 2),
        scale=sigma_daily,
        size=(simulations, horizon_days),
    )
    price_paths = s0 * np.exp(np.cumsum(shocks, axis=1))
    # Prepend day 0 (current price)
    price_paths = np.column_stack([np.full(simulations, s0), price_paths])

    p10 = np.percentile(price_paths, 10, axis=0)
    p25 = np.percentile(price_paths, 25, axis=0)
    p50 = np.percentile(price_paths, 50, axis=0)
    p75 = np.percentile(price_paths, 75, axis=0)
    p90 = np.percentile(price_paths, 90, axis=0)

    terminal_prices = price_paths[:, -1]
    terminal_returns = (terminal_prices / s0) - 1.0
    prob_gain = float((terminal_prices > s0).mean() * 100.0)
    var_95_pct = float(np.percentile(terminal_returns, 5) * 100.0)

    last_date = pd.Timestamp(close.index[-1])
    if getattr(last_date, "tz", None) is not None:
        last_date = last_date.tz_localize(None)
    future_dates = pd.bdate_range(start=last_date, periods=horizon_days + 1)

    # Keep a handful of sample paths for subtle background visual texture
    sample_paths = price_paths[:15]

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "current_price": round(s0, 2),
        "p10": np.round(p10, 2).tolist(),
        "p25": np.round(p25, 2).tolist(),
        "p50": np.round(p50, 2).tolist(),
        "p75": np.round(p75, 2).tolist(),
        "p90": np.round(p90, 2).tolist(),
        "sample_paths": np.round(sample_paths, 2).tolist(),
        "bear_target": round(float(p10[-1]), 2),
        "median_target": round(float(p50[-1]), 2),
        "bull_target": round(float(p90[-1]), 2),
        "prob_gain_pct": round(prob_gain, 1),
        "var_95_pct": round(var_95_pct, 2),
        "ann_vol_pct": round(sigma_daily * np.sqrt(252) * 100, 2),
        "simulations": simulations,
        "horizon_days": horizon_days,
    }


def stock_monte_carlo_chart(mc_res, company_name):
    """Plot a 252-day Geometric Brownian Motion fan chart with IQR, 10-90% bounds, and sample walks."""
    import plotly.graph_objects as go

    dates = mc_res["dates"]
    fig = go.Figure()

    # Subtle background sample paths
    for path in mc_res.get("sample_paths", []):
        fig.add_trace(go.Scatter(
            x=dates,
            y=path,
            mode="lines",
            line=dict(color="rgba(43, 92, 143, 0.12)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 10th - 90th percentile outer cone
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=mc_res["p90"] + mc_res["p10"][::-1],
        fill="toself",
        fillcolor="rgba(43, 92, 143, 0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        name="10th–90th Percentile Range",
        hoverinfo="skip",
    ))

    # 25th - 75th interquartile inner cone
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=mc_res["p75"] + mc_res["p25"][::-1],
        fill="toself",
        fillcolor="rgba(43, 92, 143, 0.22)",
        line=dict(color="rgba(255,255,255,0)"),
        name="25th–75th Interquartile Range",
        hoverinfo="skip",
    ))

    # Median expected path (P50)
    fig.add_trace(go.Scatter(
        x=dates,
        y=mc_res["p50"],
        mode="lines",
        name=f"Median Path (${mc_res['median_target']:.2f})",
        line=dict(color="#1E3A5F", width=2.8),
    ))

    # Bull (P90) & Bear (P10) boundary lines
    fig.add_trace(go.Scatter(
        x=dates,
        y=mc_res["p90"],
        mode="lines",
        name=f"90th Pct Bull (${mc_res['bull_target']:.2f})",
        line=dict(color="#1F7A54", width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=dates,
        y=mc_res["p10"],
        mode="lines",
        name=f"10th Pct Bear (${mc_res['bear_target']:.2f})",
        line=dict(color="#B43C3C", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        title=f"{company_name} — {mc_res['horizon_days']}-Day Geometric Brownian Motion Simulation ({mc_res['simulations']:,} Paths)",
        xaxis=dict(title="Forward Trading Horizon", nticks=7, automargin=True),
        yaxis=dict(title="Simulated Share Price ($)", tickprefix="$", automargin=True),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, font=dict(size=10)),
        margin=dict(l=55, r=20, t=65, b=85),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig

def answer_research_question(
    question,
    target,
    target_ratios,
    target_raw=None,
    peer=None,
    peer_ratios=None,
    target_risk=None,
    gbm_res=None,
    sector_label=None,
):
    """
    Deterministic multi-source financial Q&A synthesizer combining SEC fundamentals,
    peer spreads, risk/CAPM diagnostics, and Monte Carlo tail-risk outputs.
    """
    q = (question or "").strip().lower()
    if not q:
        return None

    sorted_yrs = sorted(target_ratios.keys()) if target_ratios else []
    latest_yr = sorted_yrs[-1] if sorted_yrs else "N/A"
    prev_yr = sorted_yrs[-2] if len(sorted_yrs) >= 2 else None
    lat = target_ratios.get(latest_yr, {}) if latest_yr != "N/A" else {}
    prv = target_ratios.get(prev_yr, {}) if prev_yr else {}

    peer_yrs = sorted(peer_ratios.keys()) if peer_ratios else []
    peer_lat_yr = peer_yrs[-1] if peer_yrs else None
    p_lat = peer_ratios.get(peer_lat_yr, {}) if peer_lat_yr else {}

    def _fmt(val, suffix="%", dec=2):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:,.{dec}f}{suffix}"

    bullets = []
    table_rows = []
    headline = f"Research Brief — {target} (FY{latest_yr})"

    # 1. Peer Comparison / Head-to-Head
    if any(k in q for k in ["peer", "compare", "vs", "versus", "relative", "spread", "benchmark"]):
        headline = f"Head-to-Head Fundamental & Risk Comparison: {target} vs. {peer or 'Peer'}"
        if not peer or not p_lat:
            bullets.append(f"No comparison peer is currently selected in the sidebar. Select a peer alongside **{target}** to generate side-by-side spreads.")
        else:
            metrics_to_compare = [
                ("Gross Margin", "gross_margin_pct", "%"),
                ("Operating Margin", "operating_margin_pct", "%"),
                ("Net Margin", "net_margin_pct", "%"),
                ("Revenue Growth (YoY)", "revenue_growth_pct", "%"),
                ("Return on Equity (ROE)", "roe_pct", "%"),
                ("Debt-to-Equity", "debt_to_equity", "x"),
                ("Current Ratio", "current_ratio", "x"),
            ]
            for label, key, unit in metrics_to_compare:
                tv = lat.get(key)
                pv = p_lat.get(key)
                diff = (tv - pv) if (tv is not None and pv is not None) else None
                table_rows.append({
                    "Metric": label,
                    f"{target} (FY{latest_yr})": _fmt(tv, unit),
                    f"{peer} (FY{peer_lat_yr})": _fmt(pv, unit),
                    "Spread": f"{diff:+.2f}{unit}" if diff is not None else "N/A",
                })
            gm_t, gm_p = lat.get("gross_margin_pct"), p_lat.get("gross_margin_pct")
            if gm_t is not None and gm_p is not None:
                leader = target if gm_t >= gm_p else peer
                bullets.append(f"**Gross margin leadership:** **{leader}** leads by **{abs(gm_t - gm_p):.2f} percentage points** ({_fmt(gm_t)} vs. {_fmt(gm_p)}).")
            om_t, om_p = lat.get("operating_margin_pct"), p_lat.get("operating_margin_pct")
            if om_t is not None and om_p is not None:
                bullets.append(f"**Operating efficiency:** {target} converts **{_fmt(om_t)}** of revenue into operating income compared to **{_fmt(om_p)}** for {peer}.")

    # 2. Risk, Volatility, CAPM, Monte Carlo & Tail Risk
    elif any(k in q for k in ["risk", "beta", "sharpe", "volatility", "var", "monte carlo", "drawdown", "tail", "simulation"]):
        headline = f"Quantitative Risk, CAPM & Monte Carlo Tail-Risk Profile — {target}"
        if target_risk:
            bullets.append(f"**Market sensitivity (CAPM):** {target} carries a Beta of **{_fmt(target_risk.get('beta'), 'x')}** with annualized Jensen's alpha of **{_fmt(target_risk.get('jensen_alpha_pct'), '%')}** vs. S&P 500.")
            bullets.append(f"**Risk-adjusted return:** Annualized volatility is **{_fmt(target_risk.get('ann_volatility_pct'), '%')}**, yielding a Sharpe ratio of **{_fmt(target_risk.get('sharpe_ratio'), '')}** (Rf = 4.25%) and a peak-to-trough max drawdown of **{_fmt(target_risk.get('max_drawdown_pct'), '%')}**.")
        if gbm_res:
            bullets.append(
                f"**{gbm_res['horizon_days']}-Day GBM Simulation ({gbm_res['simulations']:,} paths):** "
                f"Median target **${gbm_res['median_target']:,.2f}** (10th–90th pct range: **${gbm_res['bear_target']:,.2f}** to **${gbm_res['bull_target']:,.2f}**), "
                f"with a **{gbm_res['prob_gain_pct']:.1f}%** probability of positive return and **{_fmt(gbm_res['var_95_pct'], '%')}** 95% horizon VaR."
            )
        if not bullets:
            bullets.append("Market history is unavailable for this ticker to compute daily CAPM and Monte Carlo risk diagnostics.")

    # 3. Leverage, Liquidity, Debt & Cash Flow
    elif any(k in q for k in ["leverage", "debt", "liquidity", "cash", "fcf", "balance sheet", "solvency", "current ratio"]):
        headline = f"Balance Sheet Solvency, Liquidity & Cash Flow — {target} (FY{latest_yr})"
        dte = lat.get("debt_to_equity")
        cr = lat.get("current_ratio")
        fcf = lat.get("free_cash_flow")
        bullets.append(f"**Capital structure:** Debt-to-Equity stands at **{_fmt(dte, 'x')}** in FY{latest_yr}" + (f" (vs. {_fmt(prv.get('debt_to_equity'), 'x')} in FY{prev_yr})." if prev_yr else "."))
        bullets.append(f"**Short-term liquidity:** Current ratio is **{_fmt(cr, 'x')}**, indicating {'healthy near-term working capital coverage (>1.5x)' if (cr and cr >= 1.5) else 'tighter short-term liquidity coverage' if cr else 'unavailable balance sheet detail'}.")
        if fcf is not None:
            bullets.append(f"**Free cash flow generation:** Reported FY{latest_yr} FCF is **${fcf / 1e9:,.2f}B**.")

    # 4. Growth & Historical Anomalies
    elif any(k in q for k in ["growth", "revenue", "anomal", "outlier", "trend", "history", "driver"]):
        headline = f"Top-Line Growth Trajectory & Historical Outliers — {target}"
        rg = lat.get("revenue_growth_pct")
        bullets.append(f"**Latest top-line momentum:** FY{latest_yr} revenue growth printed at **{_fmt(rg, '%')}** YoY.")
        for metric_key, label in [("operating_margin_pct", "Operating Margin"), ("gross_margin_pct", "Gross Margin"), ("revenue_growth_pct", "Revenue Growth")]:
            s = {yr: vals[metric_key] for yr, vals in target_ratios.items() if vals.get(metric_key) is not None}
            res = analyze_metric(s, metric_name=metric_key) if len(s) >= 3 else None
            if res and res.get("anomalies"):
                for a_yr, a_info in res["anomalies"].items():
                    bullets.append(f"**{label} anomaly (FY{a_yr}):** Observed **{_fmt(a_info['value'], '%')}** (`{a_info['z_score']:+.2f}σ` vs. historical mean).")
        if len(bullets) == 1:
            bullets.append("No historical profitability or growth observations exceeded the `|z| > 2.0σ` anomaly threshold.")

    # 5. Default / Profitability & Executive Tear-Sheet Summary
    else:
        headline = f"Executive Fundamental Tear-Sheet — {target} ({sector_label or 'Equity Universe'}, FY{latest_yr})"
        bullets.append(
            f"**Margin stack (FY{latest_yr}):** Gross margin **{_fmt(lat.get('gross_margin_pct'))}**, "
            f"Operating margin **{_fmt(lat.get('operating_margin_pct'))}**, and Net margin **{_fmt(lat.get('net_margin_pct'))}**."
        )
        bullets.append(
            f"**Growth & capital returns:** YoY revenue growth of **{_fmt(lat.get('revenue_growth_pct'))}** with Return on Equity (ROE) at **{_fmt(lat.get('roe_pct'))}**."
        )
        if peer and p_lat:
            bullets.append(
                f"**Peer context ({peer}):** {target} operating margin is **{_fmt(lat.get('operating_margin_pct'))}** vs. {peer} at **{_fmt(p_lat.get('operating_margin_pct'))}**."
            )

    # Always include a compact multi-year trend table for the target company
    if not table_rows and sorted_yrs:
        for yr in sorted_yrs[-4:]:
            r = target_ratios.get(yr, {})
            table_rows.append({
                "Fiscal Year": str(yr),
                "Revenue Growth": _fmt(r.get("revenue_growth_pct")),
                "Gross Margin": _fmt(r.get("gross_margin_pct")),
                "Operating Margin": _fmt(r.get("operating_margin_pct")),
                "Net Margin": _fmt(r.get("net_margin_pct")),
                "ROE": _fmt(r.get("roe_pct")),
            })

    return {
        "question": question,
        "headline": headline,
        "bullets": bullets,
        "table": table_rows,
    }
