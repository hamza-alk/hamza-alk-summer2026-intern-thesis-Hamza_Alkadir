#!/usr/bin/env python3
"""Pull SEC EDGAR data for cyber and big tech companies.

Outputs CSVs under edgar_data/. The script uses SEC companyfacts for structured
financial data and scans the latest 10-K text for cybersecurity spending
references. Most companies do not disclose a clean cybersecurity-spend line
item, so the disclosure CSV makes that absence explicit instead of estimating.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover - optional runtime hardening.
    certifi = None


DEFAULT_CYBER_TICKERS = ["CRWD", "PANW", "FTNT", "ZS", "OKTA"]
DEFAULT_BIG_TECH_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
DEFAULT_OUTPUT_DIR = Path("edgar_data")
SEC_BASE = "https://www.sec.gov"
DATA_BASE = "https://data.sec.gov"

FINANCIAL_CONCEPTS = {
    "revenue": [
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "selling_general_admin": ["SellingGeneralAndAdministrativeExpense"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalentsAndShortTermInvestments",
    ],
    "total_debt": [
        "LongTermDebtAndFinanceLeaseObligationsCurrentAndNoncurrent",
        "LongTermDebtAndFinanceLeaseObligations",
        "LongTermDebt",
    ],
    "stockholders_equity": ["StockholdersEquity"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
}

CYBER_KEYWORDS = [
    "cybersecurity",
    "cyber security",
    "information security",
    "data security",
    "network security",
    "security program",
    "security incident",
    "cyberattack",
    "cyber attack",
]


@dataclass(frozen=True)
class Company:
    ticker: str
    cohort: str
    cik: str
    title: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull SEC EDGAR data for five cyber and five big tech companies."
    )
    parser.add_argument("--cyber-tickers", nargs="+", default=DEFAULT_CYBER_TICKERS)
    parser.add_argument("--big-tech-tickers", nargs="+", default=DEFAULT_BIG_TECH_TICKERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--user-agent",
        default="ThesisResearch hamza@example.com",
        help="SEC requires a descriptive User-Agent with contact information.",
    )
    parser.add_argument("--sleep", type=float, default=0.12, help="Delay between SEC requests.")
    return parser.parse_args()


def clean_tickers(tickers: list[str]) -> list[str]:
    return sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})


def ssl_context() -> ssl.SSLContext:
    return (
        ssl.create_default_context(cafile=certifi.where())
        if certifi is not None
        else ssl.create_default_context()
    )


def sec_get_json(url: str, user_agent: str, sleep_seconds: float) -> dict[str, Any]:
    time.sleep(sleep_seconds)
    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    with urlopen(request, timeout=60, context=ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


def sec_get_text(url: str, user_agent: str, sleep_seconds: float) -> str:
    time.sleep(sleep_seconds)
    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    with urlopen(request, timeout=60, context=ssl_context()) as response:
        return response.read().decode("utf-8", errors="replace")


def load_company_tickers(user_agent: str, sleep_seconds: float) -> dict[str, dict[str, Any]]:
    data = sec_get_json(f"{SEC_BASE}/files/company_tickers.json", user_agent, sleep_seconds)
    return {row["ticker"].upper(): row for row in data.values()}


def build_universe(
    cyber_tickers: list[str],
    big_tech_tickers: list[str],
    ticker_map: dict[str, dict[str, Any]],
) -> list[Company]:
    companies = []
    for cohort, tickers in (
        ("cybersecurity", clean_tickers(cyber_tickers)),
        ("big_tech", clean_tickers(big_tech_tickers)),
    ):
        for ticker in tickers:
            if ticker not in ticker_map:
                raise ValueError(f"{ticker} not found in SEC company_tickers.json")
            row = ticker_map[ticker]
            companies.append(
                Company(
                    ticker=ticker,
                    cohort=cohort,
                    cik=str(row["cik_str"]).zfill(10),
                    title=row["title"],
                )
            )
    return companies


def fact_units(companyfacts: dict[str, Any], concept: str) -> list[dict[str, Any]]:
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    concept_data = facts.get(concept)
    if not concept_data:
        return []
    units = concept_data.get("units", {})
    for preferred_unit in ("USD", "shares", "pure"):
        if preferred_unit in units:
            return units[preferred_unit]
    for values in units.values():
        return values
    return []


def latest_annual_fact(companyfacts: dict[str, Any], aliases: list[str]) -> tuple[float | None, str | None, str | None, str | None]:
    candidates = []
    for concept in aliases:
        for item in fact_units(companyfacts, concept):
            if item.get("form") not in {"10-K", "20-F", "40-F"}:
                continue
            if item.get("fp") not in {"FY", None}:
                continue
            value = item.get("val")
            end = item.get("end")
            filed = item.get("filed")
            fiscal_year = item.get("fy")
            if value is None or end is None:
                continue
            candidates.append(
                {
                    "value": float(value),
                    "concept": concept,
                    "period_end": end,
                    "filed": filed,
                    "fiscal_year": fiscal_year,
                }
            )
    if not candidates:
        return None, None, None, None
    candidates.sort(key=lambda row: (row["period_end"], row.get("filed") or ""), reverse=True)
    best = candidates[0]
    return best["value"], best["concept"], best["period_end"], str(best["fiscal_year"])


def company_metrics(company: Company, companyfacts: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ticker": company.ticker,
        "cohort": company.cohort,
        "cik": company.cik,
        "company": company.title,
    }
    concepts_used = {}
    period_ends = {}
    fiscal_years = {}
    for metric, aliases in FINANCIAL_CONCEPTS.items():
        value, concept, period_end, fiscal_year = latest_annual_fact(companyfacts, aliases)
        row[metric] = value
        concepts_used[f"{metric}_concept"] = concept
        period_ends[f"{metric}_period_end"] = period_end
        fiscal_years[f"{metric}_fiscal_year"] = fiscal_year

    row["free_cash_flow_proxy"] = (
        row["operating_cash_flow"] - row["capex"]
        if row.get("operating_cash_flow") is not None and row.get("capex") is not None
        else None
    )
    row["fcf_margin_proxy"] = (
        row["free_cash_flow_proxy"] / row["revenue"]
        if row.get("free_cash_flow_proxy") is not None and row.get("revenue") not in (None, 0)
        else None
    )
    row["rd_percent_revenue"] = (
        row["research_and_development"] / row["revenue"]
        if row.get("research_and_development") is not None and row.get("revenue") not in (None, 0)
        else None
    )
    row["sga_percent_revenue"] = (
        row["selling_general_admin"] / row["revenue"]
        if row.get("selling_general_admin") is not None and row.get("revenue") not in (None, 0)
        else None
    )
    row["net_cash_proxy"] = (
        row["cash"] - row["total_debt"]
        if row.get("cash") is not None and row.get("total_debt") is not None
        else None
    )
    row["current_ratio"] = (
        row["current_assets"] / row["current_liabilities"]
        if row.get("current_assets") is not None and row.get("current_liabilities") not in (None, 0)
        else None
    )
    row.update(concepts_used)
    row.update(period_ends)
    row.update(fiscal_years)
    return row


def latest_10k_submission(submissions: dict[str, Any]) -> dict[str, Any] | None:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])
    for form, accession, doc, filing_date in zip(forms, accessions, docs, dates):
        if form == "10-K":
            return {
                "form": form,
                "accession": accession,
                "primary_document": doc,
                "filing_date": filing_date,
            }
    return None


def filing_url(company: Company, filing: dict[str, Any]) -> str:
    accession_no_dashes = filing["accession"].replace("-", "")
    cik_no_padding = str(int(company.cik))
    return f"{SEC_BASE}/Archives/edgar/data/{cik_no_padding}/{accession_no_dashes}/{filing['primary_document']}"


def html_to_text(html: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cybersecurity_disclosure_row(company: Company, filing: dict[str, Any] | None, text: str) -> dict[str, Any]:
    if filing is None:
        return {
            "ticker": company.ticker,
            "cohort": company.cohort,
            "cik": company.cik,
            "company": company.title,
            "latest_10k_filing_date": None,
            "latest_10k_url": None,
            "cybersecurity_mentions": None,
            "explicit_cybersecurity_spending_found": False,
            "cybersecurity_dollar_amounts_nearby": "",
            "spending_search_note": "No 10-K found in recent SEC submissions.",
            "representative_snippets": "",
        }

    lowered = text.lower()
    mention_count = sum(lowered.count(keyword) for keyword in CYBER_KEYWORDS)
    snippets = []
    dollar_amounts = []
    explicit_spend_found = False
    spend_patterns = re.compile(
        r"(spend|spent|spending|expense|expenses|expenditure|expenditures|invest|investment|investments|cost|costs)",
        re.IGNORECASE,
    )
    money_pattern = re.compile(
        r"\$\s?\d+(?:\.\d+)?\s?(?:million|billion|thousand)?|\d+(?:\.\d+)?\s?(?:million|billion)\s+dollars",
        re.IGNORECASE,
    )

    for keyword in CYBER_KEYWORDS:
        for match in re.finditer(re.escape(keyword), lowered):
            start = max(0, match.start() - 320)
            end = min(len(text), match.end() + 320)
            snippet = text[start:end].strip()
            nearby_money = money_pattern.findall(snippet)
            if nearby_money:
                dollar_amounts.extend(nearby_money)
                if spend_patterns.search(snippet):
                    explicit_spend_found = True
            if len(snippets) < 5:
                snippets.append(snippet)

    unique_amounts = []
    for amount in dollar_amounts:
        cleaned = re.sub(r"\s+", " ", amount).strip()
        if cleaned not in unique_amounts:
            unique_amounts.append(cleaned)

    note = (
        "Potential cybersecurity spend language found near a dollar amount. Review snippets manually."
        if explicit_spend_found
        else "No explicit cybersecurity spending line item found in latest 10-K text scan."
    )
    return {
        "ticker": company.ticker,
        "cohort": company.cohort,
        "cik": company.cik,
        "company": company.title,
        "latest_10k_filing_date": filing["filing_date"],
        "latest_10k_url": filing.get("url"),
        "cybersecurity_mentions": mention_count,
        "explicit_cybersecurity_spending_found": explicit_spend_found,
        "cybersecurity_dollar_amounts_nearby": "; ".join(unique_amounts[:20]),
        "spending_search_note": note,
        "representative_snippets": " || ".join(snippets),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ticker_map = load_company_tickers(args.user_agent, args.sleep)
    companies = build_universe(args.cyber_tickers, args.big_tech_tickers, ticker_map)

    universe_rows = [company.__dict__ for company in companies]
    metrics_rows = []
    disclosure_rows = []
    raw_fact_dir = args.output_dir / "raw_companyfacts"
    raw_filing_dir = args.output_dir / "raw_latest_10k_text"
    raw_fact_dir.mkdir(parents=True, exist_ok=True)
    raw_filing_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    for company in companies:
        try:
            companyfacts = sec_get_json(
                f"{DATA_BASE}/api/xbrl/companyfacts/CIK{company.cik}.json",
                args.user_agent,
                args.sleep,
            )
            (raw_fact_dir / f"{company.ticker}_companyfacts.json").write_text(
                json.dumps(companyfacts, indent=2),
                encoding="utf-8",
            )
            metrics_rows.append(company_metrics(company, companyfacts))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            warnings.append(f"{company.ticker}: failed companyfacts request: {exc}")

        try:
            submissions = sec_get_json(
                f"{DATA_BASE}/submissions/CIK{company.cik}.json",
                args.user_agent,
                args.sleep,
            )
            filing = latest_10k_submission(submissions)
            filing_text = ""
            if filing is not None:
                url = filing_url(company, filing)
                filing["url"] = url
                filing_html = sec_get_text(url, args.user_agent, args.sleep)
                filing_text = html_to_text(filing_html)
                (raw_filing_dir / f"{company.ticker}_latest_10k.txt").write_text(
                    filing_text,
                    encoding="utf-8",
                )
            disclosure_rows.append(cybersecurity_disclosure_row(company, filing, filing_text))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            warnings.append(f"{company.ticker}: failed 10-K disclosure request: {exc}")
            disclosure_rows.append(cybersecurity_disclosure_row(company, None, ""))

    write_csv(args.output_dir / "company_universe.csv", universe_rows)
    write_csv(args.output_dir / "sec_company_facts_metrics.csv", metrics_rows)
    write_csv(args.output_dir / "sec_cybersecurity_disclosures.csv", disclosure_rows)

    metadata = {
        "cyber_tickers": clean_tickers(args.cyber_tickers),
        "big_tech_tickers": clean_tickers(args.big_tech_tickers),
        "output_dir": str(args.output_dir),
        "sec_user_agent": args.user_agent,
        "notes": [
            "Structured metrics come from SEC companyfacts us-gaap tags.",
            "Free cash flow is a proxy: operating cash flow minus capex.",
            "Cybersecurity spending is not usually disclosed as a standardized SEC line item.",
            "The disclosure CSV scans latest 10-K text for cybersecurity mentions and dollar amounts nearby.",
        ],
        "warnings": warnings,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote EDGAR CSVs to {args.output_dir}")
    if warnings:
        print(f"Completed with {len(warnings)} warnings. See run_metadata.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
