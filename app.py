"""FinSight — Institutional Financial Research & Quantitative Analytics.

Run from the project root:
    streamlit run app.py

The UI orchestrates the user's existing SEC/XBRL, ratio, market-data,
statistical forecasting, and Ridge Regression modules. Forecasts are
research outputs, not investment recommendations.
"""
from __future__ import annotations

import io
import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from analysis.metrics import (
    COMPANIES, get_company_metrics, calculate_ratios,
    get_quarterly_metrics, get_industry_sector, get_metric_explanation,
)
from analysis.stock_data import get_all_stock_data
from analysis.stats import (
    analyze_metric, run_monte_carlo_forecast, monte_carlo_forecast_chart,
)
from analysis.ridge_model import run_per_company_models, reliability_label
from analysis.charts import (
    yoy_line_chart, yoy_comparison_chart, quarterly_bar_chart,
    margin_breakdown_stacked_chart, revenue_vs_profit_chart,
    price_history_chart, returns_comparison_chart, prediction_chart,
)
from analysis.theme import (
    apply_plotly_theme, NAVY, PANEL, PANEL_ALT, BORDER,
    TEXT, MUTED, BLUE, TEAL, GREEN, RED, AMBER,
)

from analysis.metrics import (
    COMPANIES, COMPANY_LABELS, get_company_metrics, calculate_ratios,
    get_quarterly_metrics, get_industry_sector, get_metric_explanation,
)

st.set_page_config(
    page_title="FinSight | Institutional Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    :root {
      --fs-bg: #f8f9fb;
      --fs-panel: #ffffff;
      --fs-panel2: #f1f4f9;
      --fs-border: #e2e8f0;
      --fs-text: #0f172a;
      --fs-muted: #475569;
      --fs-orange: #ea580c;
      --fs-orange-dark: #c2410c;
      --fs-orange-soft: #fff7ed;
      --fs-green: #15803d;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--fs-text); }
    .stApp {
      background: radial-gradient(ellipse at 80% -10%, #ffedd5 0%, #f8f9fb 42%);
      color: var(--fs-text);
    }
    [data-testid="stHeader"] { background: rgba(248, 249, 251, 0.88); }
    [data-testid="stSidebar"] {
      background: #ffffff;
      border-right: 1px solid var(--fs-border);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.25rem; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1600px; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; color: var(--fs-text) !important; }
    h1 { font-size: clamp(2rem, 3vw, 2.8rem) !important; font-weight: 800 !important; }
    h2 { font-size: 1.55rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.1rem !important; font-weight: 700 !important; }
    p, label, .stMarkdown { color: var(--fs-text); }

    .fs-eyebrow { color: var(--fs-orange); font: 600 11px 'DM Mono', monospace; letter-spacing: .16em; text-transform: uppercase; }
    .fs-subtitle { color: var(--fs-muted); font-size: .98rem; margin-top: -.55rem; }
    .fs-brand { font: 800 1.35rem 'Manrope', sans-serif; letter-spacing: -.04em; color: #0f172a; }
    .fs-brand span { color: var(--fs-orange); }
    .fs-sidebar-note { color: var(--fs-muted); font-size: .76rem; line-height: 1.5; }
    .fs-divider { height: 2px; background: linear-gradient(90deg, var(--fs-orange), rgba(234, 88, 12, 0.1)); margin: 1.2rem 0 1.4rem; }

    /* KPI & Content Cards */
    .fs-card {
      background: #ffffff;
      border: 1px solid var(--fs-border);
      border-top: 3px solid var(--fs-orange);
      border-radius: 12px;
      padding: 17px 19px;
      min-height: 114px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .fs-card-label { color: var(--fs-muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 700; }
    .fs-card-value { color: var(--fs-text); font: 800 1.55rem 'Manrope', sans-serif; margin-top: 8px; letter-spacing: -.04em; }
    .fs-card-foot { color: var(--fs-muted); font-size: .75rem; margin-top: 4px; }

    .fs-section-label { color: var(--fs-orange); font: 600 10px 'DM Mono', monospace; letter-spacing: .16em; text-transform: uppercase; margin-bottom: .35rem; }
    .fs-panel {
      background: #ffffff;
      border: 1px solid var(--fs-border);
      border-radius: 14px;
      padding: 18px 20px;
      box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
    }
    .fs-pill {
      display: inline-block; padding: 4px 10px;
      border: 1px solid #fed7aa; background: var(--fs-orange-soft);
      border-radius: 999px; color: var(--fs-orange-dark);
      font: 600 10px 'DM Mono', monospace; letter-spacing: .08em;
    }
    .fs-status { color: var(--fs-green); font: 600 10px 'DM Mono', monospace; letter-spacing: .1em; }
    .fs-muted { color: var(--fs-muted); }

    /* Streamlit native metrics & inputs */
    div[data-testid="stMetric"] {
      background: #ffffff;
      border: 1px solid var(--fs-border);
      border-left: 3px solid var(--fs-orange);
      padding: 14px 16px;
      border-radius: 12px;
      box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
    }
    div[data-testid="stMetricLabel"] { color: var(--fs-muted) !important; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: var(--fs-text) !important; font-family: 'Manrope', sans-serif; font-weight: 800; }

    /* Primary Action Buttons (Financial Orange) */
    .stButton > button, .stDownloadButton > button {
      border: 1px solid var(--fs-orange);
      background: linear-gradient(135deg, #f97316, #ea580c);
      color: #ffffff !important;
      border-radius: 9px;
      font-weight: 700;
      letter-spacing: .03em;
      box-shadow: 0 2px 6px rgba(234, 88, 12, 0.25);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
      background: linear-gradient(135deg, #ea580c, #c2410c);
      border-color: var(--fs-orange-dark);
      color: #ffffff !important;
    }

    /* Dropdowns & Inputs */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
      background: #ffffff;
      border-color: #cbd5e1;
      color: var(--fs-text);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid var(--fs-border); }
    .stTabs [data-baseweb="tab"] { color: var(--fs-muted); font-weight: 600; padding: 12px 15px; }
    .stTabs [aria-selected="true"] { color: var(--fs-orange) !important; border-bottom: 2px solid var(--fs-orange) !important; }

    /* Alert / Info boxes (Fixes white-on-white text!) */
    div[data-testid="stAlert"] {
      background-color: var(--fs-orange-soft) !important;
      border: 1px solid #fdba74 !important;
      color: #7c2d12 !important;
      border-radius: 10px;
    }
    div[data-testid="stAlert"] p { color: #7c2d12 !important; font-weight: 500; }

    div[data-testid="stExpander"] { border: 1px solid var(--fs-border); border-radius: 12px; background: #ffffff; }
    .fs-footer { color: #64748b; font: 500 10px 'DM Mono', monospace; letter-spacing: .08em; border-top: 1px solid var(--fs-border); padding-top: 16px; margin-top: 32px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# AutoReg (statsmodels.tsa.ar_model.AutoReg) Stock Forecasting
# ---------------------------------------------------------------------
def run_autoreg_stock_forecast(history_df, lags=10, steps=20, trend="c"):
    """Fit statsmodels.tsa.ar_model.AutoReg on historical closing prices."""
    try:
        from statsmodels.tsa.ar_model import AutoReg
    except ImportError as exc:
        return {"error": f"statsmodels is not installed yet ({exc}). Please reboot the app after updating requirements.txt."}

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
    if not ar_result or "error" in ar_result:
        return None

    hist_dates = ar_result["history_dates"]
    hist_vals = ar_result["history_values"]
    fc_dates = [hist_dates[-1]] + ar_result["forecast_dates"]
    fc_vals = [hist_vals[-1]] + ar_result["forecast_values"]
    ci_upper = [hist_vals[-1]] + ar_result["ci_upper"]
    ci_lower = [hist_vals[-1]] + ar_result["ci_lower"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_vals, mode="lines",
        name="Historical Close (Last 90d)", line=dict(color="#5B9BD5", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=ci_upper, mode="lines",
        line=dict(width=0.5, color="rgba(66, 199, 183, 0.25)"),
        name="95% Upper Bound", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=ci_lower, mode="lines", fill="tonexty",
        fillcolor="rgba(66, 199, 183, 0.16)",
        line=dict(width=0.5, color="rgba(66, 199, 183, 0.25)"),
        name="95% Confidence Interval",
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_vals, mode="lines+markers",
        name=f"AutoReg({ar_result['lags']}) Forecast",
        line=dict(color="#42C7B7", width=3, dash="dash"),
        marker=dict(size=5, color="#42C7B7"),
    ))
    fig.update_layout(
        title=f"{company_name} — AutoReg({ar_result['lags']}) Stock Price Forecast ({ar_result['steps']} Trading Days)",
        xaxis=dict(title="Trading Date", automargin=True, tickangle=45),
        yaxis=dict(title="Price (USD)", automargin=True),
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_fundamentals(ticker: str):
    raw = get_company_metrics(ticker)
    return raw, calculate_ratios(raw)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(tickers: tuple[str, ...], period: str):
    return get_all_stock_data(list(tickers), period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_quarterly_fundamentals(ticker: str):
    return get_quarterly_metrics(ticker)


def fmt_money(value, compact=False):
    if value is None:
        return "—"
    try:
        value = float(value)
        if not math.isfinite(value):
            return "—"
        if compact:
            absolute = abs(value)
            if absolute >= 1e12: return f"${value / 1e12:,.2f}T"
            if absolute >= 1e9: return f"${value / 1e9:,.2f}B"
            if absolute >= 1e6: return f"${value / 1e6:,.2f}M"
        return f"${value:,.2f}"
    except (TypeError, ValueError):
        return "—"


def fmt_number(value, suffix="", decimals=2):
    if value is None:
        return "—"
    try:
        value = float(value)
        return f"{value:,.{decimals}f}{suffix}" if math.isfinite(value) else "—"
    except (TypeError, ValueError):
        return "—"


def latest_year(ratios):
    return max(ratios.keys()) if ratios else None


def latest_value(ratios, key):
    year = latest_year(ratios)
    return ratios.get(year, {}).get(key) if year else None


def metric_series(ratios, metric):
    return {
        str(year): values.get(metric)
        for year, values in sorted(ratios.items())
        if values.get(metric) is not None
    }


def styled_chart(fig, height=390):
    if fig is not None:
        apply_plotly_theme(fig, height=height)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})
    return fig


def card(label, value, foot=""):
    st.markdown(
        f"""<div class="fs-card">
        <div class="fs-card-label">{label}</div>
        <div class="fs-card-value">{value}</div>
        <div class="fs-card-foot">{foot}</div></div>""",
        unsafe_allow_html=True,
    )


def section_header(eyebrow, title, description=None):
    st.markdown(f'<div class="fs-section-label">{eyebrow}</div>', unsafe_allow_html=True)
    st.subheader(title)
    if description:
        st.caption(description)


def safe_ratio_frame(ratios_by_ticker):
    rows = []
    for ticker, years in ratios_by_ticker.items():
        year = latest_year(years)
        if not year:
            continue
        row = {"Ticker": ticker, "Fiscal year": year}
        row.update(years[year])
        rows.append(row)
    return pd.DataFrame(rows)


def make_export_html(figures, title, ticker, peer):
    chunks = []
    for heading, fig in figures:
        if fig is not None:
            chunks.append(
                f"<section><h2>{heading}</h2>{pio.to_html(fig, full_html=False, include_plotlyjs='cdn')}</section>"
            )
    peer_label = f" vs {peer}" if peer else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
    <title>{title}</title><style>
    body{{background:#0b1220;color:#e7edf5;font:14px Arial,sans-serif;margin:0;padding:36px}}
    header{{border-bottom:1px solid #26364d;padding-bottom:22px;margin-bottom:26px}}
    h1{{font-size:28px;margin:0 0 8px}} h2{{font-size:17px;color:#9fc5e7;margin:0 0 10px}}
    p{{color:#91a1b7}} section{{background:#111c2e;border:1px solid #26364d;border-radius:14px;padding:18px;margin:18px 0}}
    </style></head><body><header><p>FINSIGHT / INSTITUTIONAL RESEARCH</p>
    <h1>{title}</h1><p>{ticker}{peer_label} · Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
    <p>Analytical research output. Forecasts are model-based estimates, not investment advice.</p></header>
    {''.join(chunks)}</body></html>"""


# ---------------------------------------------------------------------
# Sidebar / run configuration
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="fs-brand">◈ FIN<span>SIGHT</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="fs-sidebar-note">INSTITUTIONAL RESEARCH TERMINAL<br>FUNDAMENTALS · MARKETS · QUANTITATIVE MODELS</div>', unsafe_allow_html=True)
    st.markdown('<div class="fs-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="fs-section-label">01 / Universe</div>', unsafe_allow_html=True)
    available_tickers = sorted(COMPANIES.keys())
    if not available_tickers:
        st.error("No SEC company identifiers are configured in analysis/metrics.py.")
        st.stop()

    target = st.selectbox(
    "Target company",
    available_tickers,
    index=0,
    format_func=lambda t: COMPANY_LABELS.get(t, t),
    )
    peer_options = ["None"] + [ticker for ticker in available_tickers if ticker != target]
    peer_choice = st.selectbox(
        "Comparison peer",
        peer_options,
        index=1 if len(peer_options) > 1 else 0,
        format_func=lambda t: "None" if t == "None" else COMPANY_LABELS.get(t, t),
    )
    peer = None if peer_choice == "None" else peer_choice

    st.markdown('<div class="fs-section-label">02 / Market data</div>', unsafe_allow_html=True)
    period_label = st.selectbox("Price history window", ["1 year", "2 years", "5 years", "Maximum"], index=1)
    period_map = {"1 year": "1y", "2 years": "2y", "5 years": "5y", "Maximum": "max"}

    st.markdown('<div class="fs-divider"></div>', unsafe_allow_html=True)
    run_clicked = st.button("↗  RUN RESEARCH", use_container_width=True, type="primary")
    st.markdown(
        '<div class="fs-sidebar-note">SEC EDGAR / XBRL · Yahoo Finance<br>Run a fresh analysis to refresh the selected universe.</div>',
        unsafe_allow_html=True,
    )

selected_tickers = tuple([target] + ([peer] if peer else []))
if run_clicked:
    raw_data, ratio_data, errors = {}, {}, []
    with st.spinner("Retrieving SEC company facts and market history…"):
        for ticker in selected_tickers:
            try:
                raw_data[ticker], ratio_data[ticker] = fetch_company_fundamentals(ticker)
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")
        market_data = fetch_market_data(selected_tickers, period_map[period_label])

    st.session_state["raw_data"] = raw_data
    st.session_state["ratio_data"] = ratio_data
    st.session_state["market_data"] = market_data
    st.session_state["active_tickers"] = selected_tickers
    st.session_state["run_timestamp"] = datetime.now(timezone.utc).isoformat()
    st.session_state["load_errors"] = errors

has_data = (
    st.session_state.get("active_tickers") == selected_tickers
    and bool(st.session_state.get("ratio_data"))
)
if not has_data:
    st.markdown('<div class="fs-eyebrow">Quantitative research workspace</div>', unsafe_allow_html=True)
    st.title("FinSight")
    st.markdown('<div class="fs-subtitle">A structured view of public-company fundamentals, market behavior and quantitative expectations.</div>', unsafe_allow_html=True)
    st.markdown('<div class="fs-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown("### Research, not noise.")
        st.write(
            "FinSight combines SEC-reported financial statements with market history, "
            "derived operating ratios, statistical trend analysis and Ridge Regression."
        )
        st.markdown(
            '<div class="fs-panel"><div class="fs-section-label">Research workflow</div>'
            '<p><b>01</b> Select a company and optional peer.</p>'
            '<p><b>02</b> Retrieve annual SEC/XBRL fundamentals and market history.</p>'
            '<p><b>03</b> Review financial performance, market context and model outputs.</p>'
            '<p><b>04</b> Export an interactive HTML research pack.</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("### Analytical stack")
        for name, detail in [
            ("SEC / XBRL", "Reported financial statement facts"),
            ("Ratio engine", "Profitability, growth and balance-sheet metrics"),
            ("Statistical layer", "Z-score anomaly flags, weighted trend and Monte Carlo"),
            ("Ridge Regression", "Regularized multi-factor, one-year-ahead estimates"),
        ]:
            st.markdown(
                f'<div class="fs-panel" style="padding:12px 15px;margin-bottom:9px">'
                f'<b>{name}</b><br><span class="fs-muted" style="font-size:12px">{detail}</span></div>',
                unsafe_allow_html=True,
            )
    st.info("Choose your universe in the left panel, then select **RUN RESEARCH**.")
    st.stop()

raw = st.session_state["raw_data"]
ratios = st.session_state["ratio_data"]
stocks = st.session_state.get("market_data", {})
errors = st.session_state.get("load_errors", [])
company_names = {ticker: ticker for ticker in selected_tickers}
target_ratios = ratios.get(target, {})
target_raw = raw.get(target, {})
target_stock = stocks.get(target, {})
target_year = latest_year(target_ratios)
peer_ratios = ratios.get(peer, {}) if peer else {}

# Header
st.markdown('<div class="fs-eyebrow">Institutional research / company intelligence</div>', unsafe_allow_html=True)
st.markdown(f"<h1>{target}  <span style='color:#42c7b7'>/</span> Research Overview</h1>", unsafe_allow_html=True)
st.markdown(
    f'<div class="fs-subtitle">Fundamental performance, market context and quantitative outlook'
    f'{" · Peer: " + peer if peer else ""}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<span class="fs-pill">SEC / XBRL</span>   <span class="fs-pill">QUANTITATIVE MODELS</span> '
    f'  <span class="fs-pill">PUBLIC MARKETS</span>   '
    f'<span class="fs-status">● DATA LOADED</span>',
    unsafe_allow_html=True,
)
if errors:
    for error in errors:
        st.warning(f"Fundamental data issue — {error}")
if target not in raw or not target_ratios:
    st.error(f"No usable annual SEC fundamentals were returned for {target}. Check the ticker/CIK registry and SEC connectivity.")
    st.stop()

st.markdown('<div class="fs-divider"></div>', unsafe_allow_html=True)

# Executive metrics row
revenue = latest_value(target_ratios, "revenue")
revenue_values = target_raw.get("revenue", {}).get("values", {})
if target_year:
    rev_for_year = next((v for k, v in revenue_values.items() if str(k)[:4] == str(target_year)[:4]), None)
    if rev_for_year is not None:
        revenue = rev_for_year
latest_price = target_stock.get("stats", {}).get("current_price")
market_cap = target_stock.get("stats", {}).get("market_cap")
gross_margin = latest_value(target_ratios, "gross_margin_pct")
operating_margin = latest_value(target_ratios, "operating_margin_pct")
net_margin = latest_value(target_ratios, "net_margin_pct")
revenue_growth = latest_value(target_ratios, "revenue_growth_pct")

kpis = st.columns(5)
with kpis[0]: card("Latest reported revenue", fmt_money(revenue, True), f"FY {target_year or '—'} · SEC reported")
with kpis[1]: card("Gross margin", fmt_number(gross_margin, "%"), "Latest available fiscal year")
with kpis[2]: card("Operating margin", fmt_number(operating_margin, "%"), "Latest available fiscal year")
with kpis[3]: card("Share price", fmt_money(latest_price), "Yahoo Finance snapshot")
with kpis[4]: card("Market capitalization", fmt_money(market_cap, True), "Yahoo Finance snapshot")

tabs = st.tabs([
    "01  Executive overview",
    "02  Fundamental analysis",
    "03  Market intelligence",
    "04  Quantitative forecasts",
    "05  Research Q&A",
    "06  Data & export",
])

# ---------------------------------------------------------------------
# 01 Executive overview
# ---------------------------------------------------------------------
with tabs[0]:
    section_header("01 / Executive brief", "Operating performance at a glance",
                   "The latest available annual SEC filing is used for fundamental indicators.")
    left, right = st.columns([1.35, 1])
    with left:
        fig = revenue_vs_profit_chart(target_raw, company_name=target)
        styled_chart(fig, 410)
    with right:
        st.markdown('<div class="fs-panel">', unsafe_allow_html=True)
        st.markdown("#### Latest reported indicators")
        for label, value in [
            ("Net margin", fmt_number(net_margin, "%")),
            ("Revenue growth", fmt_number(revenue_growth, "%")),
            ("Inventory turnover", fmt_number(latest_value(target_ratios, "inventory_turnover"), "x")),
            ("Debt / equity", fmt_number(latest_value(target_ratios, "debt_to_equity"), "x")),
            ("Current ratio", fmt_number(latest_value(target_ratios, "current_ratio"), "x")),
            ("Free cash flow", fmt_money(latest_value(target_ratios, "free_cash_flow"), True)),
        ]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;border-bottom:1px solid #26364d;padding:9px 0">'
                f'<span class="fs-muted">{label}</span><b>{value}</b></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    section_header("02 / Peer lens", "Company comparison",
                   "Comparison uses the selected peer only; industry-wide distributions are not inferred from a two-company sample.")
    if peer and peer_ratios:
        peer_year = latest_year(peer_ratios)
        compare_cols = st.columns(4)
        compare_metrics = [
            ("Gross margin", "gross_margin_pct", "%"),
            ("Operating margin", "operating_margin_pct", "%"),
            ("Revenue growth", "revenue_growth_pct", "%"),
            ("Debt / equity", "debt_to_equity", "x"),
        ]
        for col, (label, key, suffix) in zip(compare_cols, compare_metrics):
            with col:
                card(f"{label} · {target}", fmt_number(latest_value(target_ratios, key), suffix), f"FY {target_year}")
                card(f"{label} · {peer}", fmt_number(latest_value(peer_ratios, key), suffix), f"FY {peer_year}")
        compare_fig = yoy_comparison_chart(
            {target: target_ratios, peer: peer_ratios},
            "gross_margin_pct", company_names, target_company=target,
        )
        styled_chart(compare_fig, 360)
    else:
        st.info("Select a comparison peer in the sidebar and rerun the analysis to activate peer benchmarking.")

# ---------------------------------------------------------------------
# 02 Fundamental analysis
# ---------------------------------------------------------------------
with tabs[1]:
    section_header("03 / Fundamental analysis", "Financial statement trends",
                   "Annual metrics are extracted from SEC EDGAR company facts and transformed into operating ratios.")
    metric_options = {
        "Gross margin": "gross_margin_pct",
        "Operating margin": "operating_margin_pct",
        "Net margin": "net_margin_pct",
        "Revenue growth": "revenue_growth_pct",
        "Gross profit growth": "gross_profit_growth_pct",
        "Inventory turnover": "inventory_turnover",
        "Days inventory outstanding": "days_inventory_outstanding",
        "Debt to equity": "debt_to_equity",
        "Current ratio": "current_ratio",
    }
    chosen_label = st.selectbox("Explore financial metric", list(metric_options.keys()), key="fund_metric")
    chosen_metric = metric_options[chosen_label]
    explain = get_metric_explanation(chosen_metric)
    st.caption(f"{explain.get('category', 'Financial metric')} · {explain.get('explanation', '')}")

    c1, c2 = st.columns(2)
    with c1:
        fig = yoy_line_chart(target_ratios, chosen_metric, target)
        styled_chart(fig, 380)
    with c2:
        if peer and peer_ratios:
            fig = yoy_comparison_chart(
                {target: target_ratios, peer: peer_ratios},
                chosen_metric, company_names, target_company=target,
            )
            styled_chart(fig, 380)
        else:
            fig = margin_breakdown_stacked_chart(target_raw, target)
            if fig is not None:
                styled_chart(fig, 380)
            else:
                st.info("No margin breakdown chart is available for this company.")

    c3, c4 = st.columns(2)
    with c3:
        fig = revenue_vs_profit_chart(target_raw, target)
        styled_chart(fig, 380)
    with c4:
        if peer and peer_ratios:
            peer_bar = {
                target: target_ratios,
                peer: peer_ratios,
            }
            fig = yoy_comparison_chart(peer_bar, "operating_margin_pct", company_names, target_company=target)
            styled_chart(fig, 380)
        else:
            st.markdown('<div class="fs-panel">', unsafe_allow_html=True)
            st.markdown("#### Ratio interpretation")
            for key in ["gross_margin_pct", "operating_margin_pct", "net_margin_pct", "current_ratio"]:
                item = get_metric_explanation(key)
                st.markdown(f"**{item['label']}**")
                st.caption(item["explanation"])
            st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Quarterly statement detail · optional SEC retrieval"):
        st.caption("Quarterly values are retrieved on demand so the initial dashboard run stays lean.")
        if st.button("Load quarterly breakdown", key="load_quarterly"):
            with st.spinner("Retrieving quarterly SEC filing facts…"):
                try:
                    st.session_state["quarterly_data"] = fetch_quarterly_fundamentals(target)
                    st.session_state["quarterly_error"] = None
                except Exception as exc:
                    st.session_state["quarterly_error"] = str(exc)
        if st.session_state.get("quarterly_error"):
            st.warning(f"Quarterly data unavailable: {st.session_state['quarterly_error']}")
        quarterly = st.session_state.get("quarterly_data", {})
        if quarterly:
            for metric_name, annual_data in quarterly.items():
                matching_years = [year for year in annual_data if str(year)[:4] == str(target_year)[:4]]
                if matching_years:
                    selected_fy = sorted(matching_years)[-1]
                    fig = quarterly_bar_chart(annual_data[selected_fy], metric_name, selected_fy, target)
                    styled_chart(fig, 320)
        elif not st.session_state.get("quarterly_error"):
            st.caption("Select “Load quarterly breakdown” to retrieve quarterly values for the latest reported fiscal year.")

# ---------------------------------------------------------------------
# 03 Market intelligence
# ---------------------------------------------------------------------
with tabs[2]:
    section_header("04 / Market intelligence", "Price behavior & return profile",
                   "Market data is retrieved separately from SEC fundamentals and may be delayed or incomplete.")
    if not stocks or target not in stocks:
        st.warning("Market history is unavailable for the selected ticker. Fundamental analysis remains available.")
    else:
        stats = target_stock.get("stats", {})
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Last close", fmt_money(float(target_stock["history"]["Close"].dropna().iloc[-1])) if not target_stock["history"]["Close"].dropna().empty else "—")
        with m2: st.metric("52-week high", fmt_money(stats.get("52_week_high")))
        with m3: st.metric("52-week low", fmt_money(stats.get("52_week_low")))
        with m4: st.metric("Trailing P/E", fmt_number(stats.get("pe_ratio"), "x"))
        price_fig = price_history_chart(stocks, company_names, target_company=target)
        styled_chart(price_fig, 430)
        if peer and peer in stocks:
            returns_fig = returns_comparison_chart(stocks, company_names, target_company=target)
            styled_chart(returns_fig, 360)
        else:
            returns = target_stock.get("returns", {})
            cols = st.columns(4)
            for col, (label, key) in zip(cols, [
                ("1-month return", "1_month_return_pct"),
                ("3-month return", "3_month_return_pct"),
                ("6-month return", "6_month_return_pct"),
                ("1-year return", "1_year_return_pct"),
            ]):
                with col:
                    st.metric(label, fmt_number(returns.get(key), "%"))

        st.markdown('<div class="fs-divider"></div>', unsafe_allow_html=True)
        section_header(
            "04B / Autoregressive price model",
            "Stock Price Forecast (statsmodels.tsa.AutoReg)",
            "Fits an AR(p) time-series model on daily closing prices. Forecasts are statistical extrapolations, not investment advice.",
        )
        ar_c1, ar_c2, ar_c3 = st.columns(3)
        with ar_c1:
            ar_lags = st.slider("Autoregressive lag order (trading days)", min_value=2, max_value=30, value=10, step=1, key="ar_lags")
        with ar_c2:
            ar_steps = st.slider("Forecast horizon (trading days)", min_value=5, max_value=60, value=20, step=5, key="ar_steps")
        with ar_c3:
            ar_trend = st.selectbox("Deterministic trend", ["c", "ct", "n"], index=0, format_func=lambda x: {"c": "Constant (c)", "ct": "Constant + Linear Trend (ct)", "n": "No Trend (n)"}[x], key="ar_trend")

        ar_result = run_autoreg_stock_forecast(target_stock.get("history"), lags=ar_lags, steps=ar_steps, trend=ar_trend)
        if ar_result and "error" in ar_result:
            st.warning(ar_result["error"])
        elif ar_result:
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                card("Last close", fmt_money(ar_result["last_price"]), "Latest trading session")
            with k2:
                card(
                    f"Projected ({ar_steps}d)",
                    fmt_money(ar_result["end_forecast"]),
                    f"{ar_result['expected_change_pct']:+.2f}% implied move",
                )
            with k3:
                card(
                    "Holdout RMSE / MAPE",
                    fmt_money(ar_result["holdout_rmse"]) if ar_result["holdout_rmse"] is not None else "—",
                    f"MAPE: {fmt_number(ar_result['holdout_mape'], '%')}" if ar_result["holdout_mape"] is not None else "Backtest window",
                )
            with k4:
                card(
                    "Information criteria",
                    f"AIC {ar_result['aic']}",
                    f"BIC {ar_result['bic']} · AR({ar_lags})",
                )
            ar_fig = autoreg_stock_chart(ar_result, company_name=target)
            styled_chart(ar_fig, 420)
            with st.expander(f"AutoReg({ar_lags}) lag parameters & p-values"):
                st.dataframe(pd.DataFrame(ar_result["coefficients"]), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough historical trading days to fit the AutoReg model with the selected lag order.")

# ---------------------------------------------------------------------
# 04 Quantitative forecasts
# ---------------------------------------------------------------------
with tabs[3]:
    section_header("05 / Quantitative research", "Forecasts, distributions & model diagnostics",
                   "Historical patterns are extrapolated for research. They are not price targets or guarantees.")
    forecast_choice = st.selectbox(
        "Statistical forecast target",
        ["Gross margin", "Operating margin", "Net margin", "Revenue growth"],
        key="forecast_metric",
    )
    forecast_map = {
        "Gross margin": "gross_margin_pct",
        "Operating margin": "operating_margin_pct",
        "Net margin": "net_margin_pct",
        "Revenue growth": "revenue_growth_pct",
    }
    forecast_metric = forecast_map[forecast_choice]
    series = metric_series(target_ratios, forecast_metric)
    analysis_result = analyze_metric(series, metric_name=forecast_metric) if series else None
    if analysis_result:
        prediction = analysis_result.get("prediction")
        if prediction:
            p1, p2, p3, p4 = st.columns(4)
            with p1: card("Trend direction", prediction["trend"].upper(), "Weighted linear trend")
            with p2: card("Projected next value", fmt_number(prediction["predicted_value"], "%" if "pct" in forecast_metric else ""), prediction["next_label"])
            with p3: card("Model R²", fmt_number(prediction["r_squared"]), "In-sample trend fit")
            with p4: card("Reliability label", prediction["reliability"].upper(), "Heuristic based on R²")
        pred_fig = prediction_chart(analysis_result, target)
        dist_fig = None
        try:
            from analysis.charts import normal_distribution_chart
            dist_fig = normal_distribution_chart(analysis_result, target)
        except Exception:
            dist_fig = None
        f1, f2 = st.columns(2)
        with f1:
            if pred_fig is not None: styled_chart(pred_fig, 390)
            else: st.info("At least three annual observations are required for a trend forecast.")
        with f2:
            if dist_fig is not None: styled_chart(dist_fig, 390)
            else: st.info("Distribution view requires at least three valid annual observations.")
        anomalies = analysis_result.get("anomalies", {})
        if anomalies:
            st.markdown("#### Historical anomaly flags")
            anomaly_df = pd.DataFrame([
                {"Fiscal year": str(year), "Observed value": info["value"], "Z-score": info["z_score"]}
                for year, info in anomalies.items()
            ])
            st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No historical observations crossed the current |z-score| > 2 anomaly threshold.")
        mc_data = run_monte_carlo_forecast(series, steps=3, simulations=3000)
        if mc_data:
            st.markdown("#### Monte Carlo scenario range")
            mc_fig = monte_carlo_forecast_chart(mc_data, forecast_metric, target)
            styled_chart(mc_fig, 410)
            st.caption("Shaded range shows the simulated 10th–90th percentile path using historical log changes. This is a scenario distribution, not a calibrated forecast interval.")
    else:
        st.info("Not enough annual data to construct a statistical forecast for this metric.")

    st.markdown('<div class="fs-divider"></div>', unsafe_allow_html=True)
    section_header("06 / Multivariate model", "Ridge Regression",
                   "The original feature set is retained: inventory turnover, current ratio, leverage, margins, revenue growth and free cash flow.")
    with st.spinner("Fitting per-company Ridge models…"):
        ridge_results = run_per_company_models({target: target_ratios}, {target: target_raw})
    target_models = ridge_results.get(target, {})
    ridge_rows = []
    for metric, result in target_models.items():
        ridge_rows.append({
            "Target": metric.replace("_", " ").title(),
            "Prediction": result.get("predicted"),
            "LOO CV R²": result.get("r2"),
            "In-sample R²": result.get("in_sample_r2"),
            "Alpha": result.get("alpha"),
            "Samples": result.get("n_samples"),
            "Reliability": reliability_label(result.get("r2")),
            "Status": result.get("reason", "Available" if result.get("predicted") is not None else "Unavailable"),
        })
    ridge_df = pd.DataFrame(ridge_rows)
    st.dataframe(ridge_df, use_container_width=True, hide_index=True)
    for metric, result in target_models.items():
        if result.get("importances"):
            with st.expander(f"Feature coefficients · {metric.replace('_', ' ').title()}"):
                importance = pd.DataFrame(
                    [{"Feature": key.replace("_", " ").title(), "Coefficient": value}
                     for key, value in sorted(result["importances"].items(), key=lambda item: abs(item[1]), reverse=True)]
                )
                st.dataframe(importance, use_container_width=True, hide_index=True)
    st.caption("LOO CV R² is calculated from pooled leave-one-out predictions. Small annual samples can produce unstable estimates; inspect sample counts and diagnostics before drawing conclusions.")

# ---------------------------------------------------------------------
# 05 Research Q&A
# ---------------------------------------------------------------------
with tabs[4]:
    section_header("07 / Research desk", "Structured financial Q&A",
                   "Answers are grounded in the loaded SEC-derived annual metrics. This interface does not claim to retrieve filing text or provide a live RAG answer.")
    query = st.text_input("Ask about profitability, growth, leverage, liquidity or inventory", placeholder="e.g. What is the latest gross margin?")
    if query:
        q = query.lower()
        mapping = {
            "gross margin": "gross_margin_pct",
            "operating margin": "operating_margin_pct",
            "net margin": "net_margin_pct",
            "revenue growth": "revenue_growth_pct",
            "gross profit growth": "gross_profit_growth_pct",
            "inventory turnover": "inventory_turnover",
            "inventory days": "days_inventory_outstanding",
            "days inventory": "days_inventory_outstanding",
            "debt to equity": "debt_to_equity",
            "debt/equity": "debt_to_equity",
            "current ratio": "current_ratio",
            "free cash flow": "free_cash_flow",
            "operating cash flow": "operating_cash_flow",
        }
        selected = next((key for phrase, key in mapping.items() if phrase in q), None)
        if selected:
            value = latest_value(target_ratios, selected)
            explanation = get_metric_explanation(selected)
            st.markdown('<div class="fs-panel">', unsafe_allow_html=True)
            st.markdown(f"#### {explanation.get('label', selected)} · {target}")
            st.markdown(f"### {fmt_money(value) if selected in ['free_cash_flow', 'operating_cash_flow'] else fmt_number(value, '%' if selected.endswith('_pct') else 'x' if selected in ['inventory_turnover', 'debt_to_equity', 'current_ratio'] else '')}")
            st.write(explanation.get("explanation", ""))
            st.caption(f"Source: SEC EDGAR/XBRL-derived annual metrics · FY {target_year or '—'}")
            st.markdown("</div>", unsafe_allow_html=True)
        elif "revenue" in q:
            st.info(f"Latest reported revenue for {target}: {fmt_money(revenue, True)} (FY {target_year or '—'}).")
        else:
            st.warning("I couldn't map that question to a metric in the current structured dataset. Try a metric such as gross margin, revenue growth, inventory turnover, debt-to-equity or free cash flow.")

# ---------------------------------------------------------------------
# 06 Data and export
# ---------------------------------------------------------------------
with tabs[5]:
    section_header("08 / Data room", "Source data & export",
                   "Inspect the latest derived ratios and download the current research pack.")
    st.markdown("#### Latest annual ratio snapshot")
    snapshot = safe_ratio_frame({target: target_ratios, **({peer: peer_ratios} if peer and peer_ratios else {})})
    st.dataframe(snapshot, use_container_width=True, hide_index=True)
    csv = snapshot.to_csv(index=False).encode("utf-8")
    st.download_button("↓  Download ratio snapshot (CSV)", data=csv, file_name=f"finsight_{target.lower()}_ratios.csv", mime="text/csv")

    export_figures = [
        ("Revenue vs net income", revenue_vs_profit_chart(target_raw, target)),
        ("Selected fundamental trend", yoy_line_chart(target_ratios, chosen_metric if "chosen_metric" in globals() else "gross_margin_pct", target)),
        ("Stock price history", price_history_chart(stocks, company_names, target_company=target) if target in stocks else None),
    ]
    if peer and peer_ratios:
        export_figures.append(("Peer gross margin comparison", yoy_comparison_chart({target: target_ratios, peer: peer_ratios}, "gross_margin_pct", company_names, target_company=target)))
    export_html = make_export_html(export_figures, "FinSight Research Pack", target, peer)
    st.download_button(
        "↓  Export interactive HTML dashboard",
        data=export_html.encode("utf-8"),
        file_name=f"finsight_{target.lower()}_research.html",
        mime="text/html",
        type="primary",
    )
    st.caption("The HTML export includes the charts available in the current session. Interactive charts load Plotly from its CDN.")

st.markdown(
    f'<div class="fs-footer">FINSIGHT · INSTITUTIONAL RESEARCH TERMINAL   /   '
    f'UNIVERSE: {", ".join(selected_tickers)}   /   '
    f'LAST RUN: {st.session_state.get("run_timestamp", "—")}</div>',
    unsafe_allow_html=True,
)
