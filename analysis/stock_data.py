"""Market data retrieval for FinSight.

Returns one consistent structure per ticker:
{ticker, history, returns, stats}. Network failures are isolated per ticker.
"""
import yfinance as yf
import pandas as pd
import numpy as np


def get_price_history(ticker, period="2y"):
    return yf.Ticker(ticker).history(period=period)


def get_key_stats(ticker):
    """Fetch market snapshot and company background fields; missing Yahoo fields remain None."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    hq_parts = [info.get("city"), info.get("state"), info.get("country")]
    headquarters = ", ".join(p for p in hq_parts if p) or None

    return {
        "current_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "pe_ratio": info.get("trailingPE"),
        "average_volume": info.get("averageVolume"),
        "dividend_yield": info.get("dividendYield"),
        "currency": info.get("currency", "USD"),
        "short_name": info.get("shortName") or info.get("longName") or ticker,
        "long_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "headquarters": headquarters,
        "employees": info.get("fullTimeEmployees"),
        "website": info.get("website"),
        "summary": info.get("longBusinessSummary"),
    }


def calculate_returns(history):
    """Calendar-based trailing returns; unavailable periods return None."""
    if history is None or history.empty or "Close" not in history:
        return {}
    close = history["Close"].dropna()
    if close.empty:
        return {}

    latest_price = float(close.iloc[-1])
    latest_date = close.index[-1]

    def return_since(days_ago):
        cutoff = latest_date - pd.Timedelta(days=days_ago)
        prior = close.loc[close.index <= cutoff]
        if prior.empty or float(prior.iloc[-1]) == 0:
            return None
        return round((latest_price / float(prior.iloc[-1]) - 1) * 100, 2)

    return {
        "1_month_return_pct": return_since(30),
        "3_month_return_pct": return_since(90),
        "6_month_return_pct": return_since(180),
        "1_year_return_pct": return_since(365),
    }


def compute_risk_metrics(history_df, risk_free_rate=0.0425):
    """
    Compute annualized volatility, Sharpe ratio, max drawdown, and CAPM Beta
    vs. S&P 500 (^GSPC) over the active price history window.
    """
    if history_df is None or history_df.empty or "Close" not in history_df.columns:
        return None

    close = history_df["Close"].dropna()
    if len(close) < 20:
        return None

    daily_ret = close.pct_change().dropna()
    ann_vol = float(daily_ret.std() * np.sqrt(252))
    ann_ret = float((close.iloc[-1] / close.iloc[0]) ** (252 / len(daily_ret)) - 1)
    sharpe = float((ann_ret - risk_free_rate) / ann_vol) if ann_vol > 1e-8 else None

    rolling_max = close.cummax()
    max_drawdown = float(((close - rolling_max) / rolling_max).min())

    # CAPM Beta vs. S&P 500 (^GSPC)
    beta = None
    alpha = None
    try:
        start_dt = close.index.min().strftime("%Y-%m-%d")
        end_dt = (close.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        spy = yf.download("^GSPC", start=start_dt, end=end_dt, progress=False, auto_adjust=True)
        if not spy.empty and "Close" in spy.columns:
            mkt_close = spy["Close"].squeeze().dropna()
            if getattr(daily_ret.index, "tz", None) is not None:
                daily_ret.index = daily_ret.index.tz_localize(None)
            if getattr(mkt_close.index, "tz", None) is not None:
                mkt_close.index = mkt_close.index.tz_localize(None)

            mkt_ret = mkt_close.pct_change().dropna()
            aligned = pd.concat([daily_ret.rename("asset"), mkt_ret.rename("mkt")], axis=1).dropna()
            if len(aligned) >= 20:
                cov = aligned["asset"].cov(aligned["mkt"])
                var_m = aligned["mkt"].var()
                if var_m > 1e-10:
                    beta = float(cov / var_m)
                    mkt_ann_ret = float((mkt_close.iloc[-1] / mkt_close.iloc[0]) ** (252 / len(mkt_ret)) - 1)
                    alpha = float(ann_ret - (risk_free_rate + beta * (mkt_ann_ret - risk_free_rate)))
    except Exception:
        beta = None

    return {
        "ann_volatility_pct": round(ann_vol * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "sharpe_ratio": round(sharpe, 2) if sharpe is not None else None,
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "beta": round(beta, 2) if beta is not None else None,
        "jensen_alpha_pct": round(alpha * 100, 2) if alpha is not None else None,
    }
