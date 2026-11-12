"""Optional SEC EDGAR connectivity smoke test.

Run from the project root with: python test_xbrl.py
Uses the same SEC extraction helpers and User-Agent configured in metrics.py.
"""
from analysis.metrics import COMPANIES, get_company_facts, HEADERS


def main():
    ticker = "GAP" if "GAP" in COMPANIES else next(iter(COMPANIES))
    cik = COMPANIES[ticker]
    print(f"Checking SEC company facts for {ticker} (CIK {cik})…")
    facts = get_company_facts(cik)
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    print(f"SEC response received. us-gaap concepts available: {len(us_gaap):,}")
    revenue_tags = [
        tag for tag in us_gaap
        if "revenue" in tag.lower() or "sales" in tag.lower()
    ]
    print("Example revenue/sales tags:", ", ".join(revenue_tags[:12]) or "None found")


if __name__ == "__main__":
    main()
