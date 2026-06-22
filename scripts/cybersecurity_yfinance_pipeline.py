#!/usr/bin/env python3
"""Collect cybersecurity company financials from yfinance.

The script exports raw Yahoo Finance statements and creates a compact metrics
table aimed at spotting industry patterns: growth, margins, spending intensity,
cash/debt posture, billings proxy, and valuation multiples.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = [
    "PANW",
    "CRWD",
    "FTNT",
    "ZS",
    "OKTA",
    "S",
    "CHKP",
    "GEN",
    "TENB",
    "QLYS",
    "RPD",
    "VRNS",
    "NET",
    "AKAM",
    "DDOG",
]

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
class CompanyData:
    ticker: str
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
        description="Pull cybersecurity company statements and metrics from yfinance."
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to data/yfinance_cybersecurity/YYYY-MM-DD_HH-MM-SS.",
    )
    parser.add_argument(
        "--period",
        choices=["annual", "quarterly"],
        default="annual",
        help="Statement cadence used for derived metrics.",
    )
    parser.add_argument("--price-years", type=int, default=3)
    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="Skip daily price history downloads.",
    )
    return parser.parse_args()


def clean_tickers(tickers: Iterable[str]) -> list[str]:
    return sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})


def statement_to_rows(df: pd.DataFrame, ticker: str, statement_name: str) -> pd.DataFrame:
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
    out.insert(0, "ticker", ticker)
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce").dt.date
    return out


def write_statement(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        pd.DataFrame().to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def line_series(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    for alias in aliases:
        if alias in df.index:
            series = pd.to_numeric(df.loc[alias], errors="coerce")
            series.index = pd.to_datetime(series.index, errors="coerce")
            return series.sort_index(ascending=False)
    return pd.Series(dtype="float64")


def latest_value(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return None
    return float(cleaned.iloc[0])


def previous_value(series: pd.Series) -> float | None:
    if series is None or len(series.dropna()) < 2:
        return None
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    return float(cleaned.iloc[1]) if len(cleaned) >= 2 else None


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def pct_change_latest(series: pd.Series) -> float | None:
    current = latest_value(series)
    previous = previous_value(series)
    if current is None or previous in (None, 0):
        return None
    return (current - previous) / abs(previous)


def get_line(statements: dict[str, pd.DataFrame], metric: str) -> pd.Series:
    aliases = LINE_ALIASES[metric]
    for statement_name in ("income", "cashflow", "balance"):
        series = line_series(statements.get(statement_name, pd.DataFrame()), aliases)
        if not series.empty:
            return series
    return pd.Series(dtype="float64")


def collect_company(
    ticker: str, price_years: int, skip_prices: bool, warnings: list[str]
) -> CompanyData:
    stock = yf.Ticker(ticker)
    try:
        info = stock.get_info()
    except Exception as exc:  # noqa: BLE001 - keep batch runs alive.
        warnings.append(f"{ticker}: failed to fetch info: {exc}")
        info = {}

    def fetch_frame(attr_name: str) -> pd.DataFrame:
        try:
            frame = getattr(stock, attr_name)
            return frame if frame is not None else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{ticker}: failed to fetch {attr_name}: {exc}")
            return pd.DataFrame()

    if skip_prices:
        prices = pd.DataFrame()
    else:
        try:
            prices = stock.history(period=f"{price_years}y", auto_adjust=False)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{ticker}: failed to fetch prices: {exc}")
            prices = pd.DataFrame()

    company = CompanyData(
        ticker=ticker,
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
        warnings.append(f"{ticker}: no yfinance profile, statement, or price data returned")
    return company


def profile_row(company: CompanyData) -> dict[str, Any]:
    row = {"ticker": company.ticker}
    for field in PROFILE_FIELDS:
        row[field] = company.info.get(field)
    return row


def metrics_row(company: CompanyData, period: str) -> dict[str, Any]:
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
    sga = get_line(statements, "selling_general_admin")
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
    total_assets = get_line(statements, "total_assets")
    equity = get_line(statements, "stockholders_equity")

    latest_revenue = latest_value(revenue)
    previous_deferred_revenue = previous_value(deferred_revenue)
    latest_deferred_revenue = latest_value(deferred_revenue)
    deferred_revenue_change = (
        latest_deferred_revenue - previous_deferred_revenue
        if latest_deferred_revenue is not None and previous_deferred_revenue is not None
        else None
    )
    billings_proxy = (
        latest_revenue + deferred_revenue_change
        if latest_revenue is not None and deferred_revenue_change is not None
        else None
    )
    revenue_growth = pct_change_latest(revenue)
    fcf_margin = safe_div(latest_value(fcf), latest_revenue)

    if latest_value(fcf) is None and latest_value(operating_cash_flow) is not None:
        # If Yahoo omits FCF, approximate it from CFO minus capex. Capex is often negative.
        approx_fcf = latest_value(operating_cash_flow) + (latest_value(capex) or 0)
        fcf_margin = safe_div(approx_fcf, latest_revenue)
    else:
        approx_fcf = latest_value(fcf)

    market_cap = company.info.get("marketCap")
    enterprise_value = company.info.get("enterpriseValue")

    return {
        "ticker": company.ticker,
        "company": company.info.get("shortName") or company.info.get("longName"),
        "period": period,
        "latest_revenue": latest_revenue,
        "revenue_growth": revenue_growth,
        "gross_margin": safe_div(latest_value(gross_profit), latest_revenue),
        "operating_margin": safe_div(latest_value(operating_income), latest_revenue),
        "net_margin": safe_div(latest_value(net_income), latest_revenue),
        "free_cash_flow": approx_fcf,
        "free_cash_flow_margin": fcf_margin,
        "rule_of_40": (
            revenue_growth + fcf_margin
            if revenue_growth is not None and fcf_margin is not None
            else None
        ),
        "rd_spend": latest_value(rd),
        "rd_percent_revenue": safe_div(latest_value(rd), latest_revenue),
        "sga_spend": latest_value(sga),
        "sga_percent_revenue": safe_div(latest_value(sga), latest_revenue),
        "sales_marketing_spend": latest_value(sales_marketing),
        "sales_marketing_percent_revenue": safe_div(
            latest_value(sales_marketing), latest_revenue
        ),
        "stock_comp_percent_revenue": safe_div(latest_value(sbc), latest_revenue),
        "cash": latest_value(cash),
        "total_debt": latest_value(debt),
        "net_cash": (
            latest_value(cash) - latest_value(debt)
            if latest_value(cash) is not None and latest_value(debt) is not None
            else None
        ),
        "current_ratio": safe_div(
            latest_value(current_assets), latest_value(current_liabilities)
        ),
        "total_assets": latest_value(total_assets),
        "stockholders_equity": latest_value(equity),
        "deferred_revenue": latest_deferred_revenue,
        "deferred_revenue_growth": pct_change_latest(deferred_revenue),
        "billings_proxy_revenue_plus_deferred_change": billings_proxy,
        "billings_proxy_growth": (
            safe_div(billings_proxy - latest_revenue, latest_revenue)
            if billings_proxy is not None and latest_revenue not in (None, 0)
            else None
        ),
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "ev_to_revenue": safe_div(enterprise_value, latest_revenue),
        "market_cap_to_revenue": safe_div(market_cap, latest_revenue),
        "employees": company.info.get("fullTimeEmployees"),
        "revenue_per_employee": safe_div(
            latest_revenue, company.info.get("fullTimeEmployees")
        ),
    }


def write_outputs(companies: list[CompanyData], output_dir: Path, period: str) -> None:
    raw_dir = output_dir / "raw_statements"
    price_dir = output_dir / "prices"
    raw_dir.mkdir(parents=True, exist_ok=True)
    price_dir.mkdir(parents=True, exist_ok=True)

    profiles = []
    metrics = []
    for company in companies:
        profiles.append(profile_row(company))
        metrics.append(metrics_row(company, period))

        statement_exports = {
            "annual_income": company.income,
            "annual_balance": company.balance,
            "annual_cashflow": company.cashflow,
            "quarterly_income": company.q_income,
            "quarterly_balance": company.q_balance,
            "quarterly_cashflow": company.q_cashflow,
        }
        for statement_name, frame in statement_exports.items():
            rows = statement_to_rows(frame, company.ticker, statement_name)
            write_statement(rows, raw_dir / f"{company.ticker}_{statement_name}.csv")

        if company.prices is not None and not company.prices.empty:
            prices = company.prices.copy()
            prices.index.name = "date"
            prices.reset_index().to_csv(price_dir / f"{company.ticker}_prices.csv", index=False)

    pd.DataFrame(profiles).sort_values("ticker").to_csv(
        output_dir / "company_profiles.csv", index=False
    )
    pd.DataFrame(metrics).sort_values("ticker").to_csv(
        output_dir / "summary_metrics.csv", index=False
    )


def main() -> int:
    args = parse_args()
    tickers = clean_tickers(args.tickers)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("data")
        / "yfinance_cybersecurity"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    companies = [
        collect_company(ticker, args.price_years, args.skip_prices, warnings)
        for ticker in tickers
    ]
    write_outputs(companies, output_dir, args.period)

    metadata = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "tickers": tickers,
        "period": args.period,
        "price_years": args.price_years,
        "skip_prices": args.skip_prices,
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
