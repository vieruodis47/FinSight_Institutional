# FinSight — Institutional Research Terminal

## Project layout

```text
finsight_institutional/
├── app.py
├── requirements.txt
├── run_predictions.py       # Original command-line prediction runner
├── test_xbrl.py              # Original SEC/XBRL smoke-test script
├── .streamlit/config.toml
└── analysis/
    ├── __init__.py
    ├── metrics.py          # Original SEC EDGAR / XBRL extraction and ratio framework
    ├── charts.py           # Original Plotly analytical chart functions
    ├── stats.py            # Original z-score, anomaly, weighted-trend and Monte Carlo framework
    ├── stock_data.py       # Normalized Yahoo Finance response structure
    ├── ridge_model.py      # Ridge models with robust small-sample handling
    └── theme.py            # Shared institutional Plotly theme
```

## Run locally

1. Unzip the project.
2. Open a terminal in the extracted `finsight_institutional` folder.
3. Create/activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

## SEC company registry

The supplied `analysis/metrics.py` currently includes GAP, PVH and AEO as fallback tickers.
It can also extend the registry from a `sec_companies.json` file placed in the project root.
Only tickers present in that registry are selectable in the sidebar.

## Important implementation notes

- The SEC/XBRL extraction and financial-ratio calculations are retained from the supplied source.
- The existing chart generators and statistical analysis module are retained. A shared theme is applied at render time.
- The market-data wrapper now returns `history`, `returns`, `stats`, and `ticker` consistently.
- Ridge Regression retains the supplied predictors and targets. It now handles insufficient data without referencing uninitialized variables, and uses pooled leave-one-out predictions to calculate out-of-fold R².
- The Research Q&A tab is deliberately described as structured metric lookup. The previous app displayed a hard-coded generic "AI Insight"; this redesign removes that unsupported claim rather than pretending a RAG pipeline is connected.
- Forecasts are analytical scenarios, not investment advice or guarantees.
- SEC requests require a descriptive User-Agent. Update `HEADERS` in `analysis/metrics.py` to your own contact details before public deployment if needed.

## Optional Gemini narrative module

`analysis/narrative.py` is included from the supplied project. It is not imported by the Streamlit app, because the supplied app did not connect a working Gemini-backed RAG/Q&A workflow and the original narrative module requires `google-genai` plus `GEMINI_API_KEY`. The Q&A page therefore answers only from loaded structured metrics rather than displaying fabricated AI commentary.
