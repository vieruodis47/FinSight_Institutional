"""Ridge Regression framework used by FinSight.

Keeps the original feature set, one-year-ahead target construction,
multicollinearity screen, per-company model, and pooled cross-company model.
Cross-validation is calculated from pooled leave-one-out predictions so the
reported out-of-fold R² is meaningful (rather than averaging single-row R²).
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

FEATURE_METRICS = [
    "inventory_turnover", "current_ratio", "debt_to_equity",
    "gross_margin_pct", "operating_margin_pct", "revenue_growth_pct",
    "free_cash_flow",
]
TARGET_METRICS = ["revenue", "gross_margin_pct", "operating_margin_pct"]


def filter_multicollinear_features(X, feature_names, threshold=0.75):
    if X.shape[1] <= 1:
        return X, list(feature_names)
    corr = pd.DataFrame(X, columns=feature_names).corr().abs()
    to_drop = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            value = corr.iloc[i, j]
            if pd.notna(value) and value > threshold:
                to_drop.add(corr.columns[i])
    kept = [name for name in feature_names if name not in to_drop]
    if not kept:
        kept = list(feature_names)
    indices = [feature_names.index(name) for name in kept]
    return X[:, indices], kept


def build_xy(ratios, raw_metrics, target_metric, normalize_revenue=False):
    years = sorted(ratios.keys())
    revenue_values = raw_metrics.get("revenue", {}).get("values", {})
    rows_x, rows_y, row_years = [], [], []

    for i in range(len(years) - 1):
        current_year, next_year = years[i], years[i + 1]
        features = [ratios[current_year].get(feature) for feature in FEATURE_METRICS]
        if any(value is None or not np.isfinite(float(value)) for value in features):
            continue

        if target_metric == "revenue":
            next_revenue = revenue_values.get(next_year)
            if normalize_revenue:
                current_revenue = revenue_values.get(current_year)
                if current_revenue in (None, 0) or next_revenue is None:
                    continue
                target = (next_revenue / current_revenue - 1) * 100
            else:
                target = next_revenue
        else:
            target = ratios[next_year].get(target_metric)

        if target is None or not np.isfinite(float(target)):
            continue
        rows_x.append([float(v) for v in features])
        rows_y.append(float(target))
        row_years.append((current_year, next_year))

    if len(rows_x) < 3:
        return None, None, None
    return np.asarray(rows_x), np.asarray(rows_y), row_years


def train_ridge(X, y, feature_names):
    """Screen correlated inputs, select alpha by pooled LOO R², then refit."""
    X_filtered, surviving = filter_multicollinear_features(
        np.asarray(X, dtype=float), list(feature_names)
    )
    if len(y) < 3:
        raise ValueError("At least three observations are required for Ridge modeling.")

    loo = LeaveOneOut()
    best_alpha, best_score = 1.0, -np.inf
    for alpha in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        fold_predictions = []
        # Fit scaling inside each fold to avoid validation leakage.
        for train_idx, test_idx in loo.split(X_filtered):
            fold_scaler = StandardScaler()
            x_train = fold_scaler.fit_transform(X_filtered[train_idx])
            x_test = fold_scaler.transform(X_filtered[test_idx])
            candidate = Ridge(alpha=alpha)
            candidate.fit(x_train, y[train_idx])
            fold_predictions.extend(candidate.predict(x_test).tolist())
        score = r2_score(y, fold_predictions) if np.var(y) > 0 else 0.0
        if np.isfinite(score) and score > best_score:
            best_score, best_alpha = float(score), alpha

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    model = Ridge(alpha=best_alpha)
    model.fit(X_scaled, y)
    in_sample = r2_score(y, model.predict(X_scaled)) if np.var(y) > 0 else 0.0
    return model, scaler, round(best_score if np.isfinite(best_score) else 0.0, 4), best_alpha, round(float(in_sample), 4), surviving


def predict_next(model, scaler, latest_ratios, raw_metrics, target_metric, surviving_features):
    values = [latest_ratios.get(feature) for feature in surviving_features]
    if any(value is None for value in values):
        return None, None
    x_scaled = scaler.transform(np.asarray(values, dtype=float).reshape(1, -1))
    predicted = float(model.predict(x_scaled)[0])
    importances = {
        feature: round(float(coef), 4)
        for feature, coef in zip(surviving_features, model.coef_)
    }
    return round(predicted, 4), importances


def _empty_result(reason):
    return {
        "predicted": None, "r2": None, "in_sample_r2": None,
        "alpha": None, "importances": {}, "n_samples": 0,
        "mode": "per_company", "reason": reason,
    }


def run_per_company_models(all_ratios, all_raw_metrics):
    results = {}
    for ticker, ratios in all_ratios.items():
        results[ticker] = {}
        if not ratios:
            for target in TARGET_METRICS:
                results[ticker][target] = _empty_result("No annual ratio history.")
            continue
        raw = all_raw_metrics.get(ticker, {})
        latest_year = sorted(ratios.keys())[-1]
        latest = ratios[latest_year]
        for target in TARGET_METRICS:
            X, y, _ = build_xy(ratios, raw, target)
            if X is None:
                results[ticker][target] = _empty_result("Insufficient complete annual observations.")
                continue
            try:
                model, scaler, cv_r2, alpha, in_sample, kept = train_ridge(X, y, FEATURE_METRICS)
                predicted, importances = predict_next(model, scaler, latest, raw, target, kept)
                results[ticker][target] = {
                    "predicted": predicted, "r2": cv_r2,
                    "in_sample_r2": in_sample, "alpha": alpha,
                    "importances": importances or {}, "n_samples": len(y),
                    "mode": "per_company",
                }
            except Exception as exc:
                results[ticker][target] = _empty_result(str(exc))
    return results


def run_cross_company_models(all_ratios, all_raw_metrics):
    results = {ticker: {} for ticker in all_ratios}
    for target in TARGET_METRICS:
        all_x, all_y = [], []
        for ticker, ratios in all_ratios.items():
            X, y, _ = build_xy(
                ratios, all_raw_metrics.get(ticker, {}), target,
                normalize_revenue=(target == "revenue"),
            )
            if X is not None:
                all_x.append(X)
                all_y.append(y)
        if not all_x:
            continue
        X_combined, y_combined = np.vstack(all_x), np.concatenate(all_y)
        try:
            model, scaler, cv_r2, alpha, in_sample, kept = train_ridge(
                X_combined, y_combined, FEATURE_METRICS
            )
            for ticker, ratios in all_ratios.items():
                if not ratios:
                    continue
                year = sorted(ratios.keys())[-1]
                predicted, importances = predict_next(
                    model, scaler, ratios[year], all_raw_metrics.get(ticker, {}),
                    target, kept,
                )
                results[ticker][target] = {
                    "predicted": predicted, "r2": cv_r2,
                    "in_sample_r2": in_sample, "alpha": alpha,
                    "importances": importances or {}, "n_samples": len(y_combined),
                    "mode": "cross_company",
                }
        except Exception as exc:
            for ticker in results:
                results[ticker][target] = _empty_result(str(exc))
                results[ticker][target]["mode"] = "cross_company"
    return results


def reliability_label(r2):
    if r2 is None:
        return "UNAVAILABLE"
    if r2 >= 0.5:
        return "HIGH"
    if r2 >= 0.2:
        return "MODERATE"
    return "LOW"
