"""Command-line runner for FinSight's statistical forecasts.

Run from the project root with: python run_predictions.py
"""
from analysis.metrics import COMPANIES, get_company_metrics, calculate_ratios
from analysis.stats import analyze_metric

METRICS_TO_FORECAST = ["revenue", "gross_margin_pct", "operating_margin_pct"]

print("=" * 68)
print("FINSIGHT / STATISTICAL TREND FORECASTS")
print("=" * 68)

for ticker in COMPANIES:
    try:
        raw = get_company_metrics(ticker)
        ratios = calculate_ratios(raw)
        series_by_metric = {}
        for year, metric_values in ratios.items():
            for metric, value in metric_values.items():
                if value is not None:
                    series_by_metric.setdefault(metric, {})[str(year)] = value
        series_by_metric["revenue"] = {
            str(date)[:4]: value
            for date, value in raw.get("revenue", {}).get("values", {}).items()
        }

        print(f"\n{ticker}")
        for metric in METRICS_TO_FORECAST:
            series = series_by_metric.get(metric, {})
            result = analyze_metric(series, metric_name=metric) if series else None
            prediction = result.get("prediction") if result else None
            if not prediction:
                print(f"  {metric}: insufficient annual history")
                continue
            unit = "$" if metric == "revenue" else ""
            lower, upper = prediction["confidence_interval"]
            print(f"  {metric}: {unit}{prediction['predicted_value']:,.2f}")
            print(f"    95% interval: {unit}{lower:,.2f} to {unit}{upper:,.2f}")
            print(f"    Trend: {prediction['trend']} | R²: {prediction['r_squared']} | "
                  f"Reliability: {prediction['reliability']}")
            if result["anomalies"]:
                print(f"    Anomaly flags: {len(result['anomalies'])}")
    except Exception as exc:
        print(f"  {ticker}: analysis failed — {exc}")
