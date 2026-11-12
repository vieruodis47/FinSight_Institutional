
"""
analysis/metrics.py

Extracts financial metrics from SEC EDGAR XBRL data dynamically.
Supports a large SEC company registry through sec_companies.json.

Features:
- Dynamic company ticker and CIK mapping
- Robust SEC registry loading
- Annual financial metric extraction
- Quarterly financial metric extraction
- Financial ratio calculations
- Cross-industry XBRL tag support
"""

import os
import re
import json
import requests

from datetime import datetime


# ============================================================
# 1. CONFIGURATION
# ============================================================

HEADERS = {
    "User-Agent": "FinSight Financial Research contact@example.com"
}

# Core fallback companies in case the SEC registry is unavailable.
COMPANIES = {
    "GAP": "0000039911",
    "PVH": "0000078239",
    "AEO": "0000919012",
}

# Absolute path to sec_companies.json in the repository root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
JSON_PATH = os.path.join(PROJECT_ROOT, "sec_companies.json")

master_sec_data = {}
REGISTRY_LOADED = False
REGISTRY_RECORD_COUNT = 0


# ============================================================
# 2. LOAD SEC COMPANY REGISTRY
# ============================================================

def load_sec_registry():
    """
    Load company tickers and CIK identifiers from sec_companies.json.

    Expected JSON format:
    {
        "companies": [
            {
                "name": "APPLE INC",
                "ticker": "AAPL",
                "cik": "0000320193"
            }
        ]
    }

    Existing fallback companies are preserved if loading fails.
    """

    global master_sec_data
    global REGISTRY_LOADED
    global REGISTRY_RECORD_COUNT

    if not os.path.isfile(JSON_PATH):
        print(f"[WARNING] SEC registry not found: {JSON_PATH}")
        return

    try:
        with open(JSON_PATH, "r", encoding="utf-8") as file:
            loaded_data = json.load(file)

        if not isinstance(loaded_data, dict):
            raise ValueError("SEC registry must contain a JSON object.")

        company_list = loaded_data.get("companies", [])

        if not isinstance(company_list, list):
            raise ValueError("'companies' must be a list.")

        master_sec_data = loaded_data
        REGISTRY_RECORD_COUNT = len(company_list)

        loaded_count = 0

        for item in company_list:

            if not isinstance(item, dict):
                continue

            ticker = item.get("ticker")
            raw_cik = item.get("cik")

            if not ticker or not raw_cik:
                continue

            clean_ticker = str(ticker).strip().upper()

            if not clean_ticker:
                continue

            digit_match = re.findall(r"\d+", str(raw_cik))

            if not digit_match:
                continue

            clean_cik = "".join(digit_match).zfill(10)

            # Validate CIK format.
            if len(clean_cik) != 10:
                continue

            COMPANIES[clean_ticker] = clean_cik
            loaded_count += 1

        REGISTRY_LOADED = True

        print("\n========== SEC REGISTRY STATUS ==========")
        print(f"[SUCCESS] Registry loaded: {JSON_PATH}")
        print(f"[INFO] JSON records: {REGISTRY_RECORD_COUNT}")
        print(f"[INFO] Valid mappings processed: {loaded_count}")
        print(f"[INFO] Unique tickers available: {len(COMPANIES)}")
        print("=========================================\n")

    except Exception as error:
        print(f"[ERROR] SEC registry loading failed: {error}")
        print("[INFO] Continuing with fallback companies.")


# Load registry when this module is imported.
load_sec_registry()


# ============================================================
# 3. COMPANY LOOKUP UTILITIES
# ============================================================

def normalize_ticker(ticker):
    """Normalize user-provided stock ticker input."""

    if ticker is None:
        raise ValueError("A company ticker is required.")

    return str(ticker).strip().upper()


def get_company_cik(ticker):
    """Return the SEC CIK associated with a stock ticker."""

    clean_ticker = normalize_ticker(ticker)

    if clean_ticker not in COMPANIES:
        raise ValueError(
            f"Ticker '{clean_ticker}' is not in the SEC company registry."
        )

    return COMPANIES[clean_ticker]


def get_company_name(ticker):
    """Return the registered company name when available."""

    clean_ticker = normalize_ticker(ticker)

    for item in master_sec_data.get("companies", []):
        if str(item.get("ticker", "")).strip().upper() == clean_ticker:
            return item.get("name", clean_ticker)

    return clean_ticker


# ============================================================
# 4. INDUSTRY CLASSIFICATION
# ============================================================

def get_industry_sector(sic_code):
    """
    Groups official SEC SIC codes into readable industry sectors.
    """

    if sic_code is None or str(sic_code).strip().upper() in [
        "", "NONE", "NAN"
    ]:
        return "Diversified Conglomerates"

    try:
        clean_digits = re.sub(
            r"\D",
            "",
            str(sic_code).split(".")[0]
        )

        if not clean_digits:
            return "Diversified Conglomerates"

        sic = int(clean_digits)

    except (ValueError, TypeError):
        return "Diversified Conglomerates"

    if 100 <= sic <= 999:
        return "Agriculture, Forestry, & Fishing"

    elif 1000 <= sic <= 1499:
        return "Mining, Energy, & Drilling"

    elif 1500 <= sic <= 1799:
        return "Construction & Contractors"

    elif 2000 <= sic <= 3999:
        return "Manufacturing & Heavy Industrial"

    elif 4000 <= sic <= 4999:
        return "Technology, Infrastructure, & Utilities"

    elif 5000 <= sic <= 5199:
        return "Wholesale Trade"

    elif 5200 <= sic <= 5999:
        return "Retail Trade & Apparel"

    elif 6000 <= sic <= 6799:
        return "Finance, Banking, & Real Estate"

    elif 7000 <= sic <= 8999:
        return "Services, Software, & Healthcare"

    else:
        return "Other Operational Entities"


# ============================================================
# 5. XBRL METRIC DEFINITIONS
# ============================================================

METRIC_FIELDS = {

    "revenue": (
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ],
        False,
    ),

    "cogs": (
        [
            "CostOfGoodsAndServicesSold",
            "CostOfRevenue",
            "CostOfGoodsSold",
        ],
        False,
    ),

    "gross_profit": (
        ["GrossProfit"],
        False,
    ),

    "operating_income": (
        ["OperatingIncomeLoss"],
        False,
    ),

    "net_income": (
        ["NetIncomeLoss"],
        False,
    ),

    "operating_cash_flow": (
        ["NetCashProvidedByUsedInOperatingActivities"],
        False,
    ),

    "inventory": (
        [
            "InventoryNet",
            "InventoryFinishedGoodsNetOfReserves",
            "InventoryFinishedGoods",
            "InventoryRawMaterialsAndSupplies",
            "InventoryWorkInProcess",
        ],
        True,
    ),

    "total_assets": (
        ["Assets"],
        True,
    ),

    "total_debt": (
        [
            "LongTermDebtNoncurrent",
            "LongTermDebt",
            "DebtCurrent",
        ],
        True,
    ),

    "equity": (
        [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        True,
    ),

    "capex": (
        [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsForCapitalImprovements",
        ],
        False,
    ),

    "current_assets": (
        ["AssetsCurrent"],
        True,
    ),

    "current_liabilities": (
        ["LiabilitiesCurrent"],
        True,
    ),
}


QUARTERLY_METRIC_FIELDS = {

    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],

    "cogs": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
    ],

    "gross_profit": ["GrossProfit"],

    "operating_income": ["OperatingIncomeLoss"],

    "net_income": ["NetIncomeLoss"],
}


# ============================================================
# 6. METRIC EXPLANATIONS
# ============================================================

METRIC_EXPLANATIONS = {

    "gross_margin_pct": {
        "label": "Gross Margin %",
        "explanation": (
            "The percentage of revenue remaining after the cost of goods "
            "sold. Higher margins generally indicate stronger product "
            "profitability."
        ),
        "category": "Profitability",
    },

    "operating_margin_pct": {
        "label": "Operating Margin %",
        "explanation": (
            "Profit remaining after operating expenses. Measures how "
            "efficiently a company runs its core business."
        ),
        "category": "Profitability",
    },

    "net_margin_pct": {
        "label": "Net Margin %",
        "explanation": (
            "The percentage of revenue remaining after all expenses, "
            "interest, and taxes."
        ),
        "category": "Profitability",
    },

    "inventory_turnover": {
        "label": "Inventory Turnover",
        "explanation": (
            "How many times a company sells through its average inventory "
            "during a year."
        ),
        "category": "Fast Fashion Specific",
    },

    "days_inventory_outstanding": {
        "label": "Days Inventory Outstanding",
        "explanation": (
            "The average number of days inventory remains before being sold."
        ),
        "category": "Fast Fashion Specific",
    },

    "revenue_growth_pct": {
        "label": "Revenue Growth %",
        "explanation": (
            "The percentage change in revenue compared with the previous year."
        ),
        "category": "Growth",
    },

    "gross_profit_growth_pct": {
        "label": "Gross Profit Growth %",
        "explanation": (
            "The percentage change in gross profit compared with the previous year."
        ),
        "category": "Growth",
    },

    "debt_to_equity": {
        "label": "Debt-to-Equity",
        "explanation": (
            "Total debt divided by shareholder equity. Measures the use "
            "of debt relative to equity financing."
        ),
        "category": "Financial Health",
    },

    "operating_cash_flow": {
        "label": "Operating Cash Flow",
        "explanation": (
            "Cash generated from a company's core business operations."
        ),
        "category": "Financial Health",
    },

    "free_cash_flow": {
        "label": "Free Cash Flow",
        "explanation": (
            "Operating cash flow minus capital expenditures."
        ),
        "category": "Financial Health",
    },

    "current_ratio": {
        "label": "Current Ratio",
        "explanation": (
            "Current assets divided by current liabilities. "
            "Measures short-term liquidity."
        ),
        "category": "Financial Health",
    },
}


def get_metric_explanation(metric_name):
    """Return the label and explanation for a metric."""

    return METRIC_EXPLANATIONS.get(
        metric_name,
        {
            "label": metric_name.replace("_", " ").title(),
            "explanation": "No explanation available for this metric.",
            "category": "Other",
        },
    )


# ============================================================
# 7. SEC EDGAR DATA RETRIEVAL
# ============================================================

def get_company_facts(cik):
    """Fetch raw XBRL company facts JSON from SEC EDGAR."""

    clean_cik = str(cik).strip().zfill(10)

    url = (
        "https://data.sec.gov/api/xbrl/companyfacts/"
        f"CIK{clean_cik}.json"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# 8. ANNUAL XBRL EXTRACTION
# ============================================================

def _extract_annual_field(data, field_name):
    """Extract annual flow values from 10-K filings."""

    facts = data.get("facts", {}).get("us-gaap", {})

    if field_name not in facts:
        return {}

    units = facts[field_name].get("units", {})
    usd_data = units.get("USD", [])

    annual = []

    for entry in usd_data:

        if "10-K" not in str(entry.get("form", "")):
            continue

        if not entry.get("start") or not entry.get("end"):
            continue

        try:
            start = datetime.strptime(entry["start"], "%Y-%m-%d")
            end = datetime.strptime(entry["end"], "%Y-%m-%d")
        except ValueError:
            continue

        duration_days = (end - start).days

        if 360 <= duration_days <= 372:
            annual.append(entry)

    by_end = {}

    for entry in annual:

        key = entry["end"]

        if (
            key not in by_end
            or entry.get("filed", "") > by_end[key].get("filed", "")
        ):
            by_end[key] = entry

    return {
        key: value["val"]
        for key, value in sorted(by_end.items())
    }


def _extract_instant_field(data, field_name):
    """Extract annual instant values from 10-K filings."""

    facts = data.get("facts", {}).get("us-gaap", {})

    if field_name not in facts:
        return {}

    units = facts[field_name].get("units", {})
    usd_data = units.get("USD", [])

    annual = [
        entry
        for entry in usd_data
        if "10-K" in str(entry.get("form", ""))
        and entry.get("end")
    ]

    by_end = {}

    for entry in annual:

        key = entry["end"]

        if (
            key not in by_end
            or entry.get("filed", "") > by_end[key].get("filed", "")
        ):
            by_end[key] = entry

    return {
        key: value["val"]
        for key, value in sorted(by_end.items())
    }


def extract_metric(data, field_options, instant=False):
    """
    Combine compatible XBRL tags across different reporting periods.
    Earlier tags in the options list receive priority for duplicate dates.
    """

    combined_results = {}
    last_used_field = None

    for field_name in field_options:

        if instant:
            result = _extract_instant_field(data, field_name)
        else:
            result = _extract_annual_field(data, field_name)

        if result:

            last_used_field = field_name

            for year_date, value in result.items():

                if year_date not in combined_results:
                    combined_results[year_date] = value

    return combined_results, last_used_field


def extract_all_metrics(data):
    """Extract all configured annual financial metrics."""

    results = {}

    for metric_name, (field_options, is_instant) in METRIC_FIELDS.items():

        values, used_field = extract_metric(
            data,
            field_options,
            instant=is_instant,
        )

        results[metric_name] = {
            "values": values,
            "field_used": used_field,
        }

    return results


# ============================================================
# 9. COMPANY FINANCIAL METRICS
# ============================================================

def get_company_metrics(ticker):
    """Fetch and extract annual financial metrics for a ticker."""

    cik = get_company_cik(ticker)

    data = get_company_facts(cik)

    return extract_all_metrics(data)


# ============================================================
# 10. FINANCIAL RATIO CALCULATIONS
# ============================================================

def calculate_ratios(metrics):
    """Calculate annual financial ratios and growth rates."""

    ratios = {}

    def normalize_to_years(data_dict):

        normalized = {}

        for date_str, value in data_dict.items():

            year_key = str(date_str)[:4]

            normalized[year_key] = value

        return normalized

    revenue = normalize_to_years(metrics["revenue"]["values"])
    cogs = normalize_to_years(metrics["cogs"]["values"])
    gross_profit = normalize_to_years(metrics["gross_profit"]["values"])
    operating_income = normalize_to_years(metrics["operating_income"]["values"])
    net_income = normalize_to_years(metrics["net_income"]["values"])
    inventory = normalize_to_years(metrics["inventory"]["values"])
    total_debt = normalize_to_years(metrics["total_debt"]["values"])
    equity = normalize_to_years(metrics["equity"]["values"])
    operating_cash_flow = normalize_to_years(
        metrics["operating_cash_flow"]["values"]
    )
    capex = normalize_to_years(metrics["capex"]["values"])
    current_assets = normalize_to_years(metrics["current_assets"]["values"])
    current_liabilities = normalize_to_years(
        metrics["current_liabilities"]["values"]
    )

    all_years = (
        set(revenue.keys())
        .union(gross_profit.keys())
        .union(net_income.keys())
        .union(cogs.keys())
    )

    years = sorted(all_years)

    for year in years:

        r = {}

        rev = revenue.get(year)
        gp = gross_profit.get(year)
        oi = operating_income.get(year)
        ni = net_income.get(year)
        cg = cogs.get(year)
        inv = inventory.get(year)

        debt = total_debt.get(year, 0)
        cap = capex.get(year)
        ca = current_assets.get(year)
        cl = current_liabilities.get(year)
        eq = equity.get(year)
        ocf = operating_cash_flow.get(year)

        # Reconstruct gross profit when the XBRL tag is unavailable.
        if gp is None and rev is not None and cg is not None:
            gp = rev - cg

        # Profitability ratios.
        if rev is not None and rev != 0:

            if gp is not None:
                r["gross_margin_pct"] = round(gp / rev * 100, 2)

            if oi is not None:
                r["operating_margin_pct"] = round(oi / rev * 100, 2)

            if ni is not None:
                r["net_margin_pct"] = round(ni / rev * 100, 2)

        # Inventory turnover.
        if cg is not None and inv is not None:

            previous_year = str(int(year) - 1)
            previous_inventory = inventory.get(previous_year)

            average_inventory = (
                (inv + previous_inventory) / 2
                if previous_inventory is not None
                else inv
            )

            if average_inventory > 0:

                turnover = cg / average_inventory

                r["inventory_turnover"] = round(turnover, 2)
                r["days_inventory_outstanding"] = round(
                    365 / turnover, 1
                )

        # Revenue growth.
        previous_year = str(int(year) - 1)
        previous_revenue = revenue.get(previous_year)

        if (
            rev is not None
            and previous_revenue is not None
            and previous_revenue != 0
        ):

            r["revenue_growth_pct"] = round(
                (rev - previous_revenue) / abs(previous_revenue) * 100,
                2,
            )

        # Gross profit growth.
        previous_gp = gross_profit.get(previous_year)

        if (
            gp is not None
            and previous_gp is not None
            and previous_gp != 0
        ):

            r["gross_profit_growth_pct"] = round(
                (gp - previous_gp) / abs(previous_gp) * 100,
                2,
            )

        # Financial health.
        if eq is not None and eq != 0:
            r["debt_to_equity"] = round(debt / eq, 2)

        if ocf is not None:
            r["operating_cash_flow"] = ocf

        if ocf is not None and cap is not None:
            r["free_cash_flow"] = ocf - cap

        if ca is not None and cl is not None and cl != 0:
            r["current_ratio"] = round(ca / cl, 2)

        ratios[year] = r

    return ratios


# ============================================================
# 11. QUARTERLY XBRL EXTRACTION
# ============================================================

def extract_quarterly_field(data, field_name):
    """Extract quarterly values and reconstruct Q4 from annual filings."""

    facts = data.get("facts", {}).get("us-gaap", {})

    if field_name not in facts:
        return {}

    units = facts[field_name].get("units", {})
    usd_data = units.get("USD", [])

    quarterly_entries = []
    annual_entries = []

    for entry in usd_data:

        if not entry.get("start") or not entry.get("end"):
            continue

        try:
            start = datetime.strptime(entry["start"], "%Y-%m-%d")
            end = datetime.strptime(entry["end"], "%Y-%m-%d")
        except ValueError:
            continue

        duration_days = (end - start).days
        form = str(entry.get("form", ""))

        if "10-Q" in form and 80 <= duration_days <= 100:
            quarterly_entries.append(entry)

        elif "10-K" in form and 360 <= duration_days <= 372:
            annual_entries.append(entry)

    q_by_end = {}

    for entry in quarterly_entries:

        key = entry["end"]

        if (
            key not in q_by_end
            or entry.get("filed", "") > q_by_end[key].get("filed", "")
        ):
            q_by_end[key] = entry

    a_by_end = {}

    for entry in annual_entries:

        key = entry["end"]

        if (
            key not in a_by_end
            or entry.get("filed", "") > a_by_end[key].get("filed", "")
        ):
            a_by_end[key] = entry

    annual_end_dates = sorted(a_by_end.keys())

    result = {}

    for q_end, entry in q_by_end.items():

        q_end_date = datetime.strptime(q_end, "%Y-%m-%d")

        matching_fy_end = None

        for a_end in annual_end_dates:

            a_end_date = datetime.strptime(a_end, "%Y-%m-%d")

            if a_end_date >= q_end_date:
                matching_fy_end = a_end
                break

        if matching_fy_end is None:
            continue

        if matching_fy_end not in result:
            result[matching_fy_end] = {}

        result[matching_fy_end][q_end] = entry["val"]

    final_result = {}

    for fy_end, quarters_by_date in result.items():

        sorted_dates = sorted(quarters_by_date.keys())

        labeled = {}

        for i, date in enumerate(sorted_dates[:3]):
            labeled[f"Q{i + 1}"] = quarters_by_date[date]

        # Reconstruct Q4 from annual value minus Q1-Q3.
        if fy_end in a_by_end:

            annual_value = a_by_end[fy_end]["val"]

            if all(
                quarter in labeled
                for quarter in ["Q1", "Q2", "Q3"]
            ):

                labeled["Q4"] = (
                    annual_value
                    - labeled["Q1"]
                    - labeled["Q2"]
                    - labeled["Q3"]
                )

        final_result[fy_end] = labeled

    return final_result


# ============================================================
# 12. QUARTERLY COMPANY METRICS
# ============================================================

def get_quarterly_metrics(ticker, year=None):
    """Fetch quarterly metrics for a company."""

    cik = get_company_cik(ticker)

    data = get_company_facts(cik)

    results = {}

    for metric_name, field_options in QUARTERLY_METRIC_FIELDS.items():

        quarterly_data = {}

        for field_name in field_options:

            extracted = extract_quarterly_field(
                data,
                field_name,
            )

            if extracted:
                quarterly_data = extracted
                break

        if year is not None:

            quarterly_data = (
                {year: quarterly_data[year]}
                if year in quarterly_data
                else {}
            )

        results[metric_name] = quarterly_data

    return results


# ============================================================
# 13. AVAILABLE FISCAL YEARS
# ============================================================

def get_available_years(ticker):
    """Return fiscal years with all four quarterly revenue values."""

    quarterly = get_quarterly_metrics(ticker)

    revenue_years = quarterly.get("revenue", {})

    complete_years = [
        year
        for year, quarters in revenue_years.items()
        if all(
            quarter in quarters
            for quarter in ["Q1", "Q2", "Q3", "Q4"]
        )
    ]

    return sorted(complete_years)


# ============================================================
# 14. INTERNAL TEST
# ============================================================

if __name__ == "__main__":

    print("\n--- FinSight SEC Registry Integration Test ---")

    print(f"Registry path: {JSON_PATH}")
    print(f"Registry exists: {os.path.isfile(JSON_PATH)}")
    print(f"Registry loaded: {REGISTRY_LOADED}")
    print(f"JSON records: {REGISTRY_RECORD_COUNT}")
    print(f"Available tickers: {len(COMPANIES)}")

    print("\nSample tickers:")

    for ticker in list(COMPANIES.keys())[:20]:
        print(
            f"{ticker}: {COMPANIES[ticker]} "
            f"({get_company_name(ticker)})"
        )

    test_tickers = ["GAP", "PVH", "AEO"]

    print("\n--- Financial Data Integration Test ---")

    for ticker in test_tickers:

        print(f"\nTesting {ticker}...")

        try:

            raw_metrics = get_company_metrics(ticker)

            revenue_data = raw_metrics["revenue"]["values"]

            if revenue_data:

                print(f"[SUCCESS] Retrieved {ticker} revenue data.")

                for year in sorted(revenue_data.keys())[-2:]:
                    print(
                        f"Year {year}: "
                        f"${revenue_data[year]:,.2f}"
                    )

            else:
                print(
                    f"[WARNING] No annual revenue data found for {ticker}."
                )

        except Exception as error:

            print(
                f"[ERROR] Integration test failed for "
                f"{ticker}: {error}"
            )

    print("\n--- Test Complete ---")

