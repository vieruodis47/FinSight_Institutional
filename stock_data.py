"""Market data retrieval for FinSight.

Returns one consistent structure per ticker:
{ticker, history, returns, stats}. Network failures are isolated per ticker.
"""
import yfinance as yf
import pandas as pd


def get_price_history(ticker, period="2y"):
    return yf.Ticker(ticker).history(period=period)


def get_key_stats(ticker):
    """Fetch market snapshot fields; missing Yahoo fields remain None."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    return {
        "current_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "pe_ratio": info.get("trailingPE"),
        "average_volume": info.get("averageVolume"),
        "dividend_yield": info.get("dividendYield"),
        "currency": info.get("currency", "USD"),
        "short_name": info.get("shortName", ticker),
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


def get_all_stock_data(tickers=None, period="2y"):
    """Download price history, trailing returns, and market snapshot for tickers."""
    if tickers is None:
        tickers = ["GAP", "PVH", "AEO"]

    results = {}
    for ticker in dict.fromkeys(str(t).upper().strip() for t in tickers if t):
        try:
            history = get_price_history(ticker, period=period)
            if history is None or history.empty:
                continue
            results[ticker] = {
                "ticker": ticker,
                "history": history,
                "returns": calculate_returns(history),
                "stats": get_key_stats(ticker),
            }
        except Exception as exc:
            print(f"[FinSight] Market data unavailable for {ticker}: {exc}")
    return results
