# Cybersecurity vs Broad Tech Resilience Tracker

Small `yfinance` + FRED pipeline for comparing leading cybersecurity companies
against large broad technology companies. The thesis is no longer “which cyber
company has the best moat?” It is now:

Cybersecurity companies may have more durable growth because customers treat
security as mission-critical infrastructure, while broad tech companies can be
more exposed to discretionary technology spending.

## Quick Start

```bash
python3 scripts/cyber_tech_yfinance_fred_pipeline.py
```

Outputs are written under
`data/yfinance_fred_cyber_vs_tech/<run_date>_<run_time>/`.

## Interactive Streamlit Dashboard

Run the dashboard with:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

The root URL is a project overview. The sidebar starts with Summary Metrics,
followed by Data Validation, Statistical Analysis, Cyber Vs Tech Resilience,
and Cyber ETF Vs QQQ Downturn. Each page includes plain-English descriptions.
The resilience pages use maximum drawdown as a fixed test while keeping tickers
and downturn dates adjustable.

The dashboard has five thesis views:

- `Summary Metrics` - gives a plain-English overview of revenue growth, free
  cash flow margin, Rule of 40, maximum drawdown, and downturn return.

- `Cyber vs Tech Resilience` - tests the case that cyber was not necessarily
  more market-resilient than broad tech. You can change cyber tickers, broad
  tech tickers and downturn dates. Maximum drawdown is the single resilience
  measure, and the evaluator gives a binary verdict for the current chart.
- `Statistical Analysis` - runs cohort comparisons, exact permutation tests,
  correlations, regressions, and regime splits for the cyber-vs-tech data. It
  ends with a plain-English verdict on whether the data supports the thesis.
  The verdict is scored by `thesis_verdict_scoring.py`, which keeps the model
  static and reusable.
- `Cyber ETF vs QQQ Downturn` - compares adjustable cybersecurity ETFs with an
  adjustable broad-tech benchmark. It uses median maximum drawdown to produce a
  binary supports/does-not-support verdict.
- `Data Validation` - compares Yahoo Finance fundamentals with SEC EDGAR values
  and shows which source is selected for analysis.

The chart views let you download the rendered chart as a PNG and the
corresponding report data as a CSV. The statistical-analysis view lets you
download its tables as a CSV.

Useful files:

- `summary_metrics.csv` - one row per company with cohort, revenue growth,
  margins, cash/debt posture, valuation, employee productivity, and downturn
  stock resilience metrics.
- `company_profiles.csv` - Yahoo profile fields such as sector, industry,
  employee count, market cap, beta, and exchange.
- `raw_statements/*.csv` - annual and quarterly balance sheets, income
  statements, and cash-flow statements.
- `prices/*.csv` - daily adjusted price history for the lookback window.
- `macro/fred_series.csv` - FRED time series for downturn and macro context.
- `macro/fred_latest.csv` - latest value for each selected FRED series.
- `run_metadata.json` - tickers, run date, macro series, warnings, and output
  locations.

## Default Company Universe

Cybersecurity leaders:

`CRWD`, `PANW`, `FTNT`, `ZS`, `OKTA`

Broad technology comparables:

`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`

The default benchmark for downturn-relative stock performance is `QQQ`.

## Downturn And Resilience Metrics

The default downturn window is calendar year 2022, a period of rising rates and
multiple compression for software and technology equities.

Company-level metrics include:

- revenue growth
- gross, operating, net, and free cash flow margins
- Rule of 40
- R&D, sales and marketing, and stock compensation as a percentage of revenue
- cash, debt, net cash, current ratio, and debt-to-equity
- deferred revenue growth and billings proxy
- price return, maximum drawdown, volatility, and QQQ-relative return during the
  downturn window

FRED macro context includes unemployment, fed funds, high-yield spreads,
financial conditions, and consumer sentiment.

## Examples

Run a custom basket:

```bash
python3 scripts/cyber_tech_yfinance_fred_pipeline.py \
  --cyber-tickers CRWD PANW FTNT ZS OKTA \
  --tech-tickers AAPL MSFT GOOGL AMZN META
```

Pull quarterly metrics and ten years of prices:

```bash
python3 scripts/cyber_tech_yfinance_fred_pipeline.py --period quarterly --price-years 10
```

Change the downturn window:

```bash
python3 scripts/cyber_tech_yfinance_fred_pipeline.py \
  --downturn-start 2022-01-01 \
  --downturn-end 2022-12-31
```

Write somewhere else:

```bash
python3 scripts/cyber_tech_yfinance_fred_pipeline.py --output-dir data/cyber_vs_tech_run
```

## ETF Downturn Plot

To compare cybersecurity ETFs against `QQQ` during a downturn, run:

```bash
python3 scripts/plot_cyber_etf_vs_qqq_downturn.py
```

This writes a new run folder under `data/cyber_etf_vs_qqq_downturn/` with:

- `cibr_hack_vs_qqq_downturn.png` - indexed price performance chart
- `downturn_metrics.csv` - return, max drawdown, volatility, and relative
  return versus `QQQ`
- `prices.csv` - daily close prices
- `run_metadata.json` - tickers, date window, output paths, and warnings

The default date window is `2022-01-01` to `2022-12-31`. `CIBR` and `HACK` do
not have history back to 2002, so a 2002 run will not produce ETF price data.
Use `--start` and `--end` to test other available downturn windows.

To keep the ETF downturn chart and add an operating-growth variable, run:

```bash
python3 scripts/plot_cyber_etf_downturn_with_revenue_growth.py
```

This writes a separate folder under
`data/cyber_etf_downturn_with_revenue_growth/`. The chart keeps the indexed ETF
price lines for `CIBR`, `HACK`, and `QQQ`, then adds a second panel with annual
revenue growth for the representative cyber vendor basket: `CRWD`, `PANW`,
`FTNT`, `ZS`, and `OKTA`. Revenue growth is not ETF revenue; it is included as
an operating proxy for the cybersecurity companies inside the theme.

## Cyber Growth Versus Security Budget Variation

To test whether public cyber vendors kept growing while security budget growth
varied, run:

```bash
python3 scripts/plot_cyber_growth_vs_budget_variation.py
```

This writes a new run folder under `data/cyber_growth_vs_budget_variation/`
with:

- `cyber_growth_vs_budget_variation.png` - security budget growth bars versus
  cyber vendor revenue growth lines
- `cyber_company_revenue_growth.csv` - company-level annual revenue growth
- `cyber_growth_vs_budget_summary.csv` - median and mean cyber revenue growth
  versus the budget-growth proxy
- `run_metadata.json` - tickers, source notes, output paths, and warnings

The budget-growth proxy uses IANS/Artico Security Budget Benchmark figures
reported by WSJ: 17% in 2022, 6% in 2023, and 8% in 2024. This is not a pure
technology-sector-only data set, so the chart labels it as security budget
growth rather than exact tech-sector cyber spend.

## Notes

`yfinance` depends on Yahoo Finance endpoints. Missing fields are common,
especially for newly listed companies, delisted companies, or statement lines
that are named differently by company. FRED data is pulled from public CSV
endpoints. The script keeps going when a ticker or macro series has missing data
and records warnings in `run_metadata.json`.
## SEC EDGAR data

Pull SEC company facts and latest 10-K cybersecurity disclosures for five
cybersecurity companies and five large technology companies:

```bash
.venv/bin/python scripts/sec_edgar_cyber_bigtech_pipeline.py
```

Outputs are written to `edgar_data/`:

- `company_universe.csv`: ticker, cohort, CIK, and SEC company name.
- `sec_company_facts_metrics.csv`: latest annual revenue, profitability, cash
  flow, R&D, liquidity, and balance-sheet metrics.
- `sec_cybersecurity_disclosures.csv`: latest 10-K cybersecurity mentions,
  nearby dollar amounts, filing links, and review snippets.

The disclosure scan does not estimate cybersecurity spending. It identifies
potential disclosures for review because cybersecurity spend is generally not
reported as a standardized SEC line item.

### Validate yFinance fundamentals with EDGAR

After running both data pipelines, compare overlapping fundamentals:

```bash
.venv/bin/python scripts/validate_yfinance_with_edgar.py
```

This writes `data/validation/financial_metric_validation.csv` and
`data/validation/validation_summary.csv`. Differences up to 5% are validated,
5%-15% require review, and differences above 15% fail validation. The selected
value uses SEC EDGAR whenever available and falls back to yFinance otherwise.
