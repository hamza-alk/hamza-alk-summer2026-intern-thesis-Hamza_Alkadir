#!/usr/bin/env python3
"""Compare cybersecurity leaders with broad tech leaders using yfinance and FRED.

The pipeline collects company fundamentals, stock history, downturn-period price
behavior, and FRED macro indicators. The goal is to test whether cybersecurity
companies show more durable growth and resilience than large technology
companies whose security exposure is only one part of a broader business.
"""

from __future__ import annotations

import argparse
import json
import ssl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

import pandas as pd
import yfinance as yf

try:
    import certifi
except ImportError:  # pragma: no cover - optional runtime hardening.
    certifi = None


DEFAULT_CYBER_TICKERS = ["CRWD", "PANW", "FTNT", "ZS", "OKTA"]
DEFAULT_TECH_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
BENCHMARK_TICKER = "QQQ"

FRED_SERIES = {
    "UNRATE": "Unemployment Rate",
    "FEDFUNDS": "Effective Federal Funds Rate",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Option-Adjusted Spread",
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "UMCSENT": "University of Michigan Consumer Sentiment",
}

PROFILE_FIELDS = [
    "shortName",
    "longName",
    "sector",
    "industry",
    "country",
    "exchange",
    "currency",
    "marketCap",
    "enterpriseValue",
    "totalRevenue",
    "grossMargins",
    "operatingMargins",
    "profitMargins",
    "revenueGrowth",
    "earningsGrowth",
    "freeCashflow",
    "operatingCashflow",
    "totalCash",
    "totalDebt",
    "currentRatio",
    "debtToEquity",
    "returnOnAssets",
    "returnOnEquity",
    "beta",
    "fullTimeEmployees",
    "website",
]

LINE_ALIASES = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Operating Income Loss"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "research_development": [
        "Research And Development",
        "Research Development",
        "Research And Development Expense",
    ],
    "selling_general_admin": [
        "Selling General And Administration",
        "Selling General Administrative",
        "Selling General And Administrative Expense",
    ],
    "sales_marketing": ["Selling And Marketing Expense", "Sales And Marketing Expense"],
    "stock_compensation": ["Stock Based Compensation"],
    "free_cash_flow": ["Free Cash Flow"],
    "operating_cash_flow": ["Operating Cash Flow", "Total Cash From Operating Activities"],
    "capital_expenditure": ["Capital Expenditure", "Capital Expenditures"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "total_debt": ["Total Debt", "Long Term Debt And Capital Lease Obligation"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "deferred_revenue": [
        "Current Deferred Revenue",
        "Deferred Revenue",
        "Current Deferred Liabilities",
    ],
    "total_assets": ["Total Assets"],
    "stockholders_equity": ["Stockholders Equity", "Total Stockholder Equity"],
}


@dataclass
class CompanySpec:
    ticker: str
    cohort: str


@dataclass
class CompanyData:
    spec: CompanySpec
    info: dict[str, Any]
    income: pd.DataFrame
    balance: pd.DataFrame
    cashflow: pd.DataFrame
    q_income: pd.DataFrame
    q_balance: pd.DataFrame
    q_cashflow: pd.DataFrame
    prices: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull yfinance and FRED data for cyber vs broad tech resilience analysis."
    )
    parser.add_argument("--cyber-tickers", nargs="+", default=DEFAULT_CYBER_TICKERS)
    parser.add_argument("--tech-tickers", nargs="+", default=DEFAULT_TECH_TICKERS)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to data/yfinance_fred_cyber_vs_tech/YYYY-MM-DD_HH-MM-SS.",
    )
    parser.add_argument(
        "--period",
        choices=["annual", "quarterly"],
        default="annual",
        help="Statement cadence used for derived operating metrics.",
    )
    parser.add_argument("--price-years", type=int, default=5)
    parser.add_argument("--downturn-start", default="2022-01-01")
    parser.add_argument("--downturn-end", default="2022-12-31")
    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="Skip daily stock price downloads.",
    )
    parser.add_argument(
        "--skip-fred",
        action="store_true",
        help="Skip FRED macro data downloads.",
    )
    return parser.parse_args()


def clean_tickers(tickers: Iterable[str]) -> list[str]:
    return sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})


def build_universe(cyber_tickers: Iterable[str], tech_tickers: Iterable[str]) -> list[CompanySpec]:
    specs = [CompanySpec(ticker, "cybersecurity") for ticker in clean_tickers(cyber_tickers)]
    specs.extend(CompanySpec(ticker, "broad_tech") for ticker in clean_tickers(tech_tickers))
    return specs


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def latest_value(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    return float(cleaned.iloc[0]) if not cleaned.empty else None


def previous_value(series: pd.Series) -> float | None:
    if series is None or len(series.dropna()) < 2:
        return None
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    return float(cleaned.iloc[1]) if len(cleaned) >= 2 else None


def pct_change_latest(series: pd.Series) -> float | None:
    current = latest_value(series)
    previous = previous_value(series)
    if current is None or previous in (None, 0):
        return None
    return (current - previous) / abs(previous)


def statement_to_rows(df: pd.DataFrame, ticker: str, cohort: str, statement_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index.name = "line_item"
    out.columns = pd.to_datetime(out.columns, errors="coerce")
    out = (
        out.stack()
        .rename("value")
        .reset_index()
        .rename(columns={"level_1": "period_end"})
    )
    out.insert(0, "statement", statement_name)
    out.insert(0, "cohort", cohort)
    out.insert(0, "ticker", ticker)
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce").dt.date
    return out


def line_series(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    for alias in aliases:
        if alias in df.index:
            series = pd.to_numeric(df.loc[alias], errors="coerce")
            series.index = pd.to_datetime(series.index, errors="coerce")
            return series.sort_index(ascending=False)
    return pd.Series(dtype="float64")


def get_line(statements: dict[str, pd.DataFrame], metric: str) -> pd.Series:
    aliases = LINE_ALIASES[metric]
    for statement_name in ("income", "cashflow", "balance"):
        series = line_series(statements.get(statement_name, pd.DataFrame()), aliases)
        if not series.empty:
            return series
    return pd.Series(dtype="float64")


def collect_company(
    spec: CompanySpec, price_years: int, skip_prices: bool, warnings: list[str]
) -> CompanyData:
    stock = yf.Ticker(spec.ticker)
    try:
        info = stock.get_info()
    except Exception as exc:  # noqa: BLE001 - keep batch runs alive.
        warnings.append(f"{spec.ticker}: failed to fetch info: {exc}")
        info = {}

    def fetch_frame(attr_name: str) -> pd.DataFrame:
        try:
            frame = getattr(stock, attr_name)
            return frame if frame is not None else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{spec.ticker}: failed to fetch {attr_name}: {exc}")
            return pd.DataFrame()

    if skip_prices:
        prices = pd.DataFrame()
    else:
        try:
            prices = stock.history(period=f"{price_years}y", auto_adjust=False)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{spec.ticker}: failed to fetch prices: {exc}")
            prices = pd.DataFrame()

    company = CompanyData(
        spec=spec,
        info=info,
        income=fetch_frame("income_stmt"),
        balance=fetch_frame("balance_sheet"),
        cashflow=fetch_frame("cashflow"),
        q_income=fetch_frame("quarterly_income_stmt"),
        q_balance=fetch_frame("quarterly_balance_sheet"),
        q_cashflow=fetch_frame("quarterly_cashflow"),
        prices=prices,
    )
    has_statements = any(
        not frame.empty
        for frame in (
            company.income,
            company.balance,
            company.cashflow,
            company.q_income,
            company.q_balance,
            company.q_cashflow,
        )
    )
    if not info and not has_statements and (prices is None or prices.empty):
        warnings.append(f"{spec.ticker}: no yfinance profile, statement, or price data returned")
    return company


def price_metrics(
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    downturn_start: str,
    downturn_end: str,
) -> dict[str, float | None]:
    if prices is None or prices.empty or "Close" not in prices.columns:
        return {
            "price_return_downturn": None,
            "price_max_drawdown_downturn": None,
            "price_volatility_downturn": None,
            "benchmark_relative_return_downturn": None,
        }

    close = pd.to_numeric(prices["Close"], errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    window = close.loc[pd.to_datetime(downturn_start) : pd.to_datetime(downturn_end)]
    if window.empty:
        return {
            "price_return_downturn": None,
            "price_max_drawdown_downturn": None,
            "price_volatility_downturn": None,
            "benchmark_relative_return_downturn": None,
        }

    returns = window.pct_change().dropna()
    running_peak = window.cummax()
    max_drawdown = ((window / running_peak) - 1).min()
    price_return = (window.iloc[-1] / window.iloc[0]) - 1 if window.iloc[0] else None
    volatility = returns.std() * (252**0.5) if not returns.empty else None
    benchmark_return = benchmark_period_return(benchmark_prices, downturn_start, downturn_end)

    return {
        "price_return_downturn": float(price_return) if price_return is not None else None,
        "price_max_drawdown_downturn": float(max_drawdown),
        "price_volatility_downturn": float(volatility) if volatility is not None else None,
        "benchmark_relative_return_downturn": (
            float(price_return - benchmark_return)
            if price_return is not None and benchmark_return is not None
            else None
        ),
    }


def benchmark_period_return(
    prices: pd.DataFrame, downturn_start: str, downturn_end: str
) -> float | None:
    if prices is None or prices.empty or "Close" not in prices.columns:
        return None
    close = pd.to_numeric(prices["Close"], errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    window = close.loc[pd.to_datetime(downturn_start) : pd.to_datetime(downturn_end)]
    if window.empty or window.iloc[0] == 0:
        return None
    return float((window.iloc[-1] / window.iloc[0]) - 1)


def profile_row(company: CompanyData) -> dict[str, Any]:
    row = {"ticker": company.spec.ticker, "cohort": company.spec.cohort}
    for field in PROFILE_FIELDS:
        row[field] = company.info.get(field)
    return row


def metrics_row(
    company: CompanyData,
    period: str,
    benchmark_prices: pd.DataFrame,
    downturn_start: str,
    downturn_end: str,
) -> dict[str, Any]:
    statements = (
        {
            "income": company.q_income,
            "balance": company.q_balance,
            "cashflow": company.q_cashflow,
        }
        if period == "quarterly"
        else {
            "income": company.income,
            "balance": company.balance,
            "cashflow": company.cashflow,
        }
    )

    revenue = get_line(statements, "revenue")
    gross_profit = get_line(statements, "gross_profit")
    operating_income = get_line(statements, "operating_income")
    net_income = get_line(statements, "net_income")
    rd = get_line(statements, "research_development")
    sales_marketing = get_line(statements, "sales_marketing")
    sbc = get_line(statements, "stock_compensation")
    fcf = get_line(statements, "free_cash_flow")
    operating_cash_flow = get_line(statements, "operating_cash_flow")
    capex = get_line(statements, "capital_expenditure")
    cash = get_line(statements, "cash")
    debt = get_line(statements, "total_debt")
    current_assets = get_line(statements, "current_assets")
    current_liabilities = get_line(statements, "current_liabilities")
    deferred_revenue = get_line(statements, "deferred_revenue")
    equity = get_line(statements, "stockholders_equity")

    latest_revenue = latest_value(revenue)
    revenue_growth = pct_change_latest(revenue)
    latest_fcf = latest_value(fcf)
    if latest_fcf is None and latest_value(operating_cash_flow) is not None:
        # Yahoo may omit FCF; CFO plus capex approximates FCF because capex is usually negative.
        latest_fcf = latest_value(operating_cash_flow) + (latest_value(capex) or 0)

    latest_cash = latest_value(cash)
    latest_debt = latest_value(debt)
    fcf_margin = safe_div(latest_fcf, latest_revenue)
    operating_margin = safe_div(latest_value(operating_income), latest_revenue)
    net_cash = (
        latest_cash - latest_debt
        if latest_cash is not None and latest_debt is not None
        else None
    )

    previous_deferred_revenue = previous_value(deferred_revenue)
    latest_deferred_revenue = latest_value(deferred_revenue)
    deferred_revenue_change = (
        latest_deferred_revenue - previous_deferred_revenue
        if latest_deferred_revenue is not None and previous_deferred_revenue is not None
        else None
    )

    row = {
        "ticker": company.spec.ticker,
        "cohort": company.spec.cohort,
        "company": company.info.get("shortName") or company.info.get("longName"),
        "period": period,
        "latest_revenue": latest_revenue,
        "revenue_growth": revenue_growth,
        "gross_margin": safe_div(latest_value(gross_profit), latest_revenue),
        "operating_margin": operating_margin,
        "net_margin": safe_div(latest_value(net_income), latest_revenue),
        "free_cash_flow": latest_fcf,
        "free_cash_flow_margin": fcf_margin,
        "rule_of_40": (
            revenue_growth + fcf_margin
            if revenue_growth is not None and fcf_margin is not None
            else None
        ),
        "rd_percent_revenue": safe_div(latest_value(rd), latest_revenue),
        "sales_marketing_percent_revenue": safe_div(
            latest_value(sales_marketing), latest_revenue
        ),
        "stock_comp_percent_revenue": safe_div(latest_value(sbc), latest_revenue),
        "cash": latest_cash,
        "total_debt": latest_debt,
        "net_cash": net_cash,
        "net_cash_percent_revenue": safe_div(net_cash, latest_revenue),
        "current_ratio": safe_div(
            latest_value(current_assets), latest_value(current_liabilities)
        ),
        "debt_to_equity": safe_div(latest_debt, latest_value(equity)),
        "deferred_revenue": latest_deferred_revenue,
        "deferred_revenue_growth": pct_change_latest(deferred_revenue),
        "billings_proxy": (
            latest_revenue + deferred_revenue_change
            if latest_revenue is not None and deferred_revenue_change is not None
            else None
        ),
        "market_cap": company.info.get("marketCap"),
        "enterprise_value": company.info.get("enterpriseValue"),
        "ev_to_revenue": safe_div(company.info.get("enterpriseValue"), latest_revenue),
        "beta": company.info.get("beta"),
        "employees": company.info.get("fullTimeEmployees"),
        "revenue_per_employee": safe_div(
            latest_revenue, company.info.get("fullTimeEmployees")
        ),
    }
    row.update(price_metrics(company.prices, benchmark_prices, downturn_start, downturn_end))
    return row


def fetch_fred_series(
    series_ids: dict[str, str], observation_start: str, warnings: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    latest_rows = []
    ssl_context = (
        ssl.create_default_context(cafile=certifi.where())
        if certifi is not None
        else ssl.create_default_context()
    )
    for series_id, label in series_ids.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            with urlopen(url, context=ssl_context, timeout=30) as response:
                frame = pd.read_csv(response)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"FRED {series_id}: failed to fetch series: {exc}")
            continue

        value_col = series_id
        frame = frame.rename(columns={"observation_date": "date", value_col: "value"})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["date"])
        frame = frame[frame["date"] >= pd.to_datetime(observation_start)].copy()
        frame.insert(0, "series_name", label)
        frame.insert(0, "series_id", series_id)
        frames.append(frame)

        latest = frame.dropna(subset=["value"]).tail(1)
        if not latest.empty:
            latest_rows.append(
                {
                    "series_id": series_id,
                    "series_name": label,
                    "latest_date": latest["date"].iloc[0].date(),
                    "latest_value": latest["value"].iloc[0],
                }
            )

    all_series = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    latest_values = pd.DataFrame(latest_rows)
    return all_series, latest_values


def write_statement(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        pd.DataFrame().to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def write_outputs(
    companies: list[CompanyData],
    output_dir: Path,
    period: str,
    benchmark_prices: pd.DataFrame,
    downturn_start: str,
    downturn_end: str,
) -> None:
    raw_dir = output_dir / "raw_statements"
    price_dir = output_dir / "prices"
    raw_dir.mkdir(parents=True, exist_ok=True)
    price_dir.mkdir(parents=True, exist_ok=True)

    profiles = []
    metrics = []
    for company in companies:
        ticker = company.spec.ticker
        cohort = company.spec.cohort
        profiles.append(profile_row(company))
        metrics.append(metrics_row(company, period, benchmark_prices, downturn_start, downturn_end))

        statement_exports = {
            "annual_income": company.income,
            "annual_balance": company.balance,
            "annual_cashflow": company.cashflow,
            "quarterly_income": company.q_income,
            "quarterly_balance": company.q_balance,
            "quarterly_cashflow": company.q_cashflow,
        }
        for statement_name, frame in statement_exports.items():
            rows = statement_to_rows(frame, ticker, cohort, statement_name)
            write_statement(rows, raw_dir / f"{ticker}_{statement_name}.csv")

        if company.prices is not None and not company.prices.empty:
            prices = company.prices.copy()
            prices.index.name = "date"
            prices.insert(0, "cohort", cohort)
            prices.insert(0, "ticker", ticker)
            prices.reset_index().to_csv(price_dir / f"{ticker}_prices.csv", index=False)

    pd.DataFrame(profiles).sort_values(["cohort", "ticker"]).to_csv(
        output_dir / "company_profiles.csv", index=False
    )
    pd.DataFrame(metrics).sort_values(["cohort", "ticker"]).to_csv(
        output_dir / "summary_metrics.csv", index=False
    )


def main() -> int:
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("data")
        / "yfinance_fred_cyber_vs_tech"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    universe = build_universe(args.cyber_tickers, args.tech_tickers)
    companies = [
        collect_company(spec, args.price_years, args.skip_prices, warnings)
        for spec in universe
    ]

    benchmark_prices = pd.DataFrame()
    if not args.skip_prices:
        benchmark = collect_company(
            CompanySpec(BENCHMARK_TICKER, "benchmark"),
            args.price_years,
            False,
            warnings,
        )
        benchmark_prices = benchmark.prices

    write_outputs(
        companies,
        output_dir,
        args.period,
        benchmark_prices,
        args.downturn_start,
        args.downturn_end,
    )

    if not args.skip_fred:
        fred_series, fred_latest = fetch_fred_series(
            FRED_SERIES, args.downturn_start, warnings
        )
        macro_dir = output_dir / "macro"
        macro_dir.mkdir(parents=True, exist_ok=True)
        fred_series.to_csv(macro_dir / "fred_series.csv", index=False)
        fred_latest.to_csv(macro_dir / "fred_latest.csv", index=False)

    metadata = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "cyber_tickers": clean_tickers(args.cyber_tickers),
        "tech_tickers": clean_tickers(args.tech_tickers),
        "benchmark_ticker": BENCHMARK_TICKER,
        "period": args.period,
        "price_years": args.price_years,
        "downturn_start": args.downturn_start,
        "downturn_end": args.downturn_end,
        "skip_prices": args.skip_prices,
        "skip_fred": args.skip_fred,
        "fred_series": FRED_SERIES,
        "output_dir": str(output_dir),
        "warnings": warnings,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"Wrote {len(companies)} companies to {output_dir}")
    if warnings:
        print(f"Completed with {len(warnings)} warnings. See run_metadata.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
