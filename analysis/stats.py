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
    Fit a weighted linear trend to historical values and predict n_ahead steps forward.
    """
    years = sorted(values_dict.keys())
    values = [values_dict[y] for y in years if values_dict[y] is not None]
    valid_years = [y for y in years if values_dict[y] is not None]

    if len(values) < 3:
        return None

    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)

    weights = np.exp(np.linspace(0, 1, len(values)))
    weights = weights / weights.sum()

    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)

    slope = np.sum(weights * (x - x_mean) * (y - y_mean)) / np.sum(weights * (x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean

    next_x = len(values) - 1 + n_ahead
    predicted = slope * next_x + intercept

    y_pred = slope * x + intercept
    residuals = y - y_pred
    ss_res = np.sum(weights * residuals ** 2)
    ss_tot = np.sum(weights * (y - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    residual_std = np.sqrt(np.sum(weights * residuals ** 2))
    margin = 1.96 * residual_std * np.sqrt(len(values))

    if slope > residual_std * 0.1:
        trend = "improving"
    elif slope < -residual_std * 0.1:
        trend = "declining"
    else:
        trend = "stable"

    if r_squared >= 0.5:
        reliability = "high"
    elif r_squared >= 0.2:
        reliability = "moderate"
    else:
        reliability = "low"

    return {
        "predicted_value": float(round(predicted, 4)),
        "confidence_interval": (float(round(predicted - margin, 4)),
                                float(round(predicted + margin, 4))),
        "r_squared": float(round(r_squared, 4)),
        "slope": float(round(slope, 4)),
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
