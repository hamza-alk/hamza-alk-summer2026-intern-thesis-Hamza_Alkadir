#!/usr/bin/env python3
"""Validate yFinance fundamentals against SEC EDGAR company facts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_YFINANCE_ROOT = Path("data/yfinance_fred_cyber_vs_tech")
DEFAULT_EDGAR_FILE = Path("edgar_data/sec_company_facts_metrics.csv")
DEFAULT_OUTPUT_DIR = Path("data/validation")

METRIC_MAP = {
    "revenue": ("latest_revenue", "revenue"),
    "free_cash_flow": ("free_cash_flow", "free_cash_flow_proxy"),
    "free_cash_flow_margin": ("free_cash_flow_margin", "fcf_margin_proxy"),
    "research_and_development_percent_revenue": ("rd_percent_revenue", "rd_percent_revenue"),
    "cash": ("cash", "cash"),
    "total_debt": ("total_debt", "total_debt"),
    "current_ratio": ("current_ratio", "current_ratio"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare yFinance summary metrics with audited SEC company facts."
    )
    parser.add_argument(
        "--yfinance-file",
        type=Path,
        help="Path to a yFinance summary_metrics.csv. Defaults to the newest pipeline run.",
    )
    parser.add_argument("--yfinance-root", type=Path, default=DEFAULT_YFINANCE_ROOT)
    parser.add_argument("--edgar-file", type=Path, default=DEFAULT_EDGAR_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validated-threshold", type=float, default=0.05)
    parser.add_argument("--review-threshold", type=float, default=0.15)
    return parser.parse_args()


def newest_summary(root: Path) -> Path:
    candidates = list(root.glob("*/summary_metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No summary_metrics.csv files found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def status_for_difference(
    difference_pct: float | None,
    validated_threshold: float,
    review_threshold: float,
) -> str:
    if difference_pct is None or pd.isna(difference_pct):
        return "Unavailable"
    absolute_difference = abs(difference_pct)
    if absolute_difference <= validated_threshold:
        return "Validated"
    if absolute_difference <= review_threshold:
        return "Review"
    return "Failed"


def validate_metrics(
    yfinance: pd.DataFrame,
    edgar: pd.DataFrame,
    validated_threshold: float = 0.05,
    review_threshold: float = 0.15,
) -> pd.DataFrame:
    yfinance_prefixed = yfinance.rename(
        columns={column: f"yf__{column}" for column in yfinance.columns if column != "ticker"}
    )
    edgar_prefixed = edgar.rename(
        columns={column: f"sec__{column}" for column in edgar.columns if column != "ticker"}
    )
    joined = yfinance_prefixed.merge(
        edgar_prefixed,
        on="ticker",
        how="outer",
        indicator=True,
    )
    rows = []
    for _, company in joined.iterrows():
        ticker = company["ticker"]
        cohort = company.get("sec__cohort")
        if pd.isna(cohort):
            cohort = company.get("yf__cohort")
        for metric, (yf_column, sec_column) in METRIC_MAP.items():
            yf_value = company.get(f"yf__{yf_column}")
            sec_value = company.get(f"sec__{sec_column}")
            yf_value = float(yf_value) if pd.notna(yf_value) else None
            sec_value = float(sec_value) if pd.notna(sec_value) else None
            difference = yf_value - sec_value if yf_value is not None and sec_value is not None else None
            difference_pct = (
                difference / abs(sec_value)
                if difference is not None and sec_value not in (None, 0)
                else None
            )
            status = status_for_difference(
                difference_pct,
                validated_threshold,
                review_threshold,
            )
            rows.append(
                {
                    "ticker": ticker,
                    "cohort": cohort,
                    "metric": metric,
                    "yfinance_value": yf_value,
                    "sec_value": sec_value,
                    "difference": difference,
                    "difference_pct": difference_pct,
                    "status": status,
                    "selected_value": sec_value if sec_value is not None else yf_value,
                    "selected_source": "SEC EDGAR" if sec_value is not None else "yFinance",
                    "sec_fiscal_year": company.get(f"sec__{sec_column}_fiscal_year"),
                    "sec_period_end": company.get(f"sec__{sec_column}_period_end"),
                    "sec_concept": company.get(f"sec__{sec_column}_concept"),
                }
            )
    return pd.DataFrame(rows).sort_values(["ticker", "metric"]).reset_index(drop=True)


def validation_summary(validation: pd.DataFrame) -> pd.DataFrame:
    summary = (
        validation.groupby(["ticker", "cohort", "status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for status in ("Validated", "Review", "Failed", "Unavailable"):
        if status not in summary:
            summary[status] = 0
    summary["metrics_checked"] = summary[
        ["Validated", "Review", "Failed", "Unavailable"]
    ].sum(axis=1)
    summary["validation_rate"] = summary["Validated"] / summary["metrics_checked"]
    return summary[
        [
            "ticker",
            "cohort",
            "metrics_checked",
            "Validated",
            "Review",
            "Failed",
            "Unavailable",
            "validation_rate",
        ]
    ].sort_values("ticker")


def main() -> int:
    args = parse_args()
    yfinance_file = args.yfinance_file or newest_summary(args.yfinance_root)
    if args.validated_threshold < 0 or args.review_threshold < args.validated_threshold:
        raise ValueError("Thresholds must satisfy 0 <= validated <= review.")

    yfinance = pd.read_csv(yfinance_file)
    edgar = pd.read_csv(args.edgar_file, dtype={"cik": str})
    validation = validate_metrics(
        yfinance,
        edgar,
        args.validated_threshold,
        args.review_threshold,
    )
    summary = validation_summary(validation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = args.output_dir / "financial_metric_validation.csv"
    summary_path = args.output_dir / "validation_summary.csv"
    validation.to_csv(validation_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"yFinance source: {yfinance_file}")
    print(f"EDGAR source: {args.edgar_file}")
    print(f"Wrote {validation_path}")
    print(f"Wrote {summary_path}")
    print(validation["status"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
