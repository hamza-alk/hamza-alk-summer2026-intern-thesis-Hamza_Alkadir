#!/usr/bin/env python3
"""Plot cyber company revenue growth against varied security budget growth.

The external budget series is a public CISO budget-growth proxy from IANS and
Artico as reported by WSJ: security budgets grew 17% in 2022, 6% in 2023, and
8% in 2024. The plot compares each budget year with vendor revenue growth in
the following fiscal year, since customer budgets are recognized as vendor
revenue over time. It is not technology-sector-only spending, so the output
labels it as security budget growth rather than exact tech-sector cyber spend.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont


DEFAULT_TICKERS = ["CRWD", "PANW", "FTNT", "ZS", "OKTA"]
DEFAULT_OUTPUT_ROOT = Path("data/cyber_growth_vs_budget_variation")
SECURITY_BUDGET_GROWTH = {
    2022: 0.17,
    2023: 0.06,
    2024: 0.08,
}
BUDGET_SOURCE = (
    "IANS/Artico Security Budget Benchmark, reported by WSJ on 2024-09-05: "
    "security budget growth 17% in 2022, 6% in 2023, 8% in 2024; "
    "security share of tech budgets rose from 8.6% in 2020 to 13.2% in 2024."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cyber vendor revenue growth versus security budget growth."
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data/cyber_growth_vs_budget_variation/YYYY-MM-DD_HH-MM-SS.",
    )
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def scale(value: float, source_min: float, source_max: float, target_min: int, target_max: int) -> float:
    if source_max == source_min:
        return (target_min + target_max) / 2
    return target_min + ((value - source_min) / (source_max - source_min)) * (target_max - target_min)


def quarterly_ttm_revenue_series(ticker: str, warnings: list[str]) -> pd.DataFrame:
    try:
        income = yf.Ticker(ticker).quarterly_income_stmt
    except Exception as exc:  # noqa: BLE001 - keep batch runs alive.
        warnings.append(f"{ticker}: failed to fetch quarterly income statement: {exc}")
        return pd.DataFrame()
    if income is None or income.empty:
        return pd.DataFrame()

    revenue_line = None
    for alias in ("Total Revenue", "Operating Revenue"):
        if alias in income.index:
            revenue_line = income.loc[alias]
            break
    if revenue_line is None:
        return pd.DataFrame()

    frame = (
        pd.to_numeric(revenue_line, errors="coerce")
        .dropna()
        .rename("quarterly_revenue")
        .reset_index()
        .rename(columns={"index": "period_end"})
    )
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame = frame.dropna(subset=["period_end"]).sort_values("period_end")
    frame["ttm_revenue"] = frame["quarterly_revenue"].rolling(4).sum()
    frame = frame.dropna(subset=["ttm_revenue"]).copy()
    frame["year"] = frame["period_end"].dt.year
    annual = frame.loc[frame.groupby("year")["period_end"].idxmax()].copy()
    annual["ticker"] = ticker
    annual["revenue"] = annual["ttm_revenue"]
    annual["revenue_growth"] = annual["revenue"].pct_change()
    annual["basis"] = "quarterly_ttm"
    return annual[["ticker", "period_end", "year", "revenue", "revenue_growth", "basis"]]


def annual_revenue_series(ticker: str, warnings: list[str]) -> pd.DataFrame:
    try:
        income = yf.Ticker(ticker).income_stmt
    except Exception as exc:  # noqa: BLE001 - keep batch runs alive.
        warnings.append(f"{ticker}: failed to fetch income statement: {exc}")
        return pd.DataFrame()
    if income is None or income.empty:
        warnings.append(f"{ticker}: no income statement returned")
        return pd.DataFrame()

    revenue_line = None
    for alias in ("Total Revenue", "Operating Revenue"):
        if alias in income.index:
            revenue_line = income.loc[alias]
            break
    if revenue_line is None:
        warnings.append(f"{ticker}: no revenue line found")
        return pd.DataFrame()

    frame = (
        pd.to_numeric(revenue_line, errors="coerce")
        .dropna()
        .rename("revenue")
        .reset_index()
        .rename(columns={"index": "period_end"})
    )
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame = frame.dropna(subset=["period_end"]).sort_values("period_end")
    frame["year"] = frame["period_end"].dt.year
    frame["ticker"] = ticker
    frame["revenue_growth"] = frame["revenue"].pct_change()
    frame["basis"] = "annual_statement"
    return frame[["ticker", "period_end", "year", "revenue", "revenue_growth", "basis"]]


def revenue_series(ticker: str, warnings: list[str]) -> pd.DataFrame:
    quarterly = quarterly_ttm_revenue_series(ticker, warnings)
    if not quarterly.empty and quarterly["year"].isin(SECURITY_BUDGET_GROWTH).any():
        return quarterly
    return annual_revenue_series(ticker, warnings)


def collect_revenue_growth(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    frames = []
    for ticker in tickers:
        frame = revenue_series(ticker, warnings)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), warnings
    return pd.concat(frames, ignore_index=True), warnings


def aggregate_growth(revenue_growth: pd.DataFrame) -> pd.DataFrame:
    budget = pd.DataFrame(
        [
            {"year": year, "security_budget_growth": growth}
            for year, growth in SECURITY_BUDGET_GROWTH.items()
        ]
    )
    annual = revenue_growth.copy()
    annual["budget_year"] = annual["year"] - 1
    annual = annual[annual["budget_year"].isin(SECURITY_BUDGET_GROWTH)].copy()
    cohort = (
        annual.groupby("budget_year")["revenue_growth"]
        .agg(cyber_revenue_growth_median="median", cyber_revenue_growth_mean="mean", company_count="count")
        .reset_index()
        .rename(columns={"budget_year": "year"})
    )
    return budget.merge(cohort, on="year", how="left").sort_values("year")


def draw_plot(revenue_growth: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    width, height = 1300, 820
    left, right, top, bottom = 110, 150, 100, 135
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom

    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    subtitle_font = load_font(16)
    axis_font = load_font(17, bold=True)
    tick_font = load_font(13)
    label_font = load_font(15, bold=True)
    note_font = load_font(14)

    years = summary["year"].tolist()
    x_positions = {
        year: scale(idx, 0, max(len(years) - 1, 1), plot_left + 95, plot_right - 95)
        for idx, year in enumerate(years)
    }

    all_growth = pd.concat(
        [
            revenue_growth["revenue_growth"].dropna(),
            summary["security_budget_growth"].dropna(),
            summary["cyber_revenue_growth_median"].dropna(),
        ],
        ignore_index=True,
    )
    y_min = min(0, float(all_growth.min()) - 0.05)
    y_max = max(0.45, float(all_growth.max()) + 0.08)

    draw.text(
        (left, 28),
        "Cyber Vendor Revenue Growth vs Security Budget Variation",
        fill="#111827",
        font=title_font,
    )
    draw.text(
        (left, 66),
        "Budget year is compared with following fiscal-year vendor revenue growth.",
        fill="#374151",
        font=subtitle_font,
    )

    for tick_pct in range(int(y_min * 100) - 5, int(y_max * 100) + 1, 10):
        tick = tick_pct / 100
        y = scale(tick, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        draw.text((50, y - 8), f"{tick_pct}%", fill="#374151", font=tick_font)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#111827", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#111827", width=2)

    bar_width = 54
    bar_gap = 12
    for _, row in summary.iterrows():
        x = x_positions[row["year"]]
        budget_y = scale(row["security_budget_growth"], y_min, y_max, plot_bottom, plot_top)
        budget_left = x - bar_width - bar_gap / 2
        budget_right = x - bar_gap / 2
        draw.rectangle(
            (budget_left, budget_y, budget_right, plot_bottom),
            fill="#d1d5db",
            outline="#6b7280",
        )
        draw.text((x - 20, plot_bottom + 14), str(int(row["year"])), fill="#111827", font=label_font)
        draw.text(
            (budget_left + 8, budget_y - 24),
            f"{row['security_budget_growth']:.0%}",
            fill="#374151",
            font=tick_font,
        )
        if pd.notna(row["cyber_revenue_growth_median"]):
            median_y = scale(row["cyber_revenue_growth_median"], y_min, y_max, plot_bottom, plot_top)
            median_left = x + bar_gap / 2
            median_right = x + bar_width + bar_gap / 2
            draw.rectangle(
                (median_left, median_y, median_right, plot_bottom),
                fill="#2563eb",
                outline="#111827",
            )
            draw.text(
                (median_left + 8, median_y - 24),
                f"{row['cyber_revenue_growth_median']:.0%}",
                fill="#111827",
                font=tick_font,
            )

    plotted_growth = revenue_growth.dropna(subset=["revenue_growth"]).copy()
    plotted_growth["budget_year"] = plotted_growth["year"] - 1
    for ticker, group in plotted_growth.groupby("ticker"):
        group = group[group["budget_year"].isin(years)].sort_values("budget_year")
        points = [
            (
                x_positions[row["budget_year"]],
                scale(row["revenue_growth"], y_min, y_max, plot_bottom, plot_top),
            )
            for _, row in group.iterrows()
        ]
        if len(points) >= 2:
            draw.line(points, fill="#93c5fd", width=2)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#93c5fd")

    legend_x = plot_right - 330
    legend_y = plot_top + 18
    draw.rectangle((legend_x - 16, legend_y - 14, legend_x + 300, legend_y + 98), fill="#ffffff", outline="#e5e7eb")
    draw.rectangle((legend_x, legend_y + 4, legend_x + 28, legend_y + 22), fill="#d1d5db", outline="#6b7280")
    draw.text((legend_x + 42, legend_y + 1), "Security budget growth", fill="#111827", font=note_font)
    draw.rectangle((legend_x, legend_y + 44, legend_x + 28, legend_y + 62), fill="#2563eb", outline="#111827")
    draw.text((legend_x + 42, legend_y + 43), "Median cyber revenue growth", fill="#111827", font=note_font)
    draw.line((legend_x, legend_y + 82, legend_x + 28, legend_y + 82), fill="#93c5fd", width=2)
    draw.text((legend_x + 42, legend_y + 73), "Individual cyber companies", fill="#111827", font=note_font)

    draw.text((plot_left + 370, height - 82), "Security Budget Year", fill="#111827", font=axis_font)
    draw.text((20, 385), "YoY Growth", fill="#111827", font=axis_font)
    draw.text(
        (plot_left, height - 42),
        "Budget-growth proxy source: IANS/Artico as reported by WSJ. Revenue growth uses the following fiscal year from Yahoo Finance.",
        fill="#4b5563",
        font=note_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    args = parse_args()
    tickers = sorted({ticker.strip().upper() for ticker in args.tickers if ticker.strip()})
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    revenue_growth, warnings = collect_revenue_growth(tickers)
    if revenue_growth.empty:
        warnings.append("No revenue growth data available; plot was not created.")
        summary = aggregate_growth(pd.DataFrame(columns=["year", "revenue_growth"]))
    else:
        summary = aggregate_growth(revenue_growth)
        revenue_growth.to_csv(output_dir / "cyber_company_revenue_growth.csv", index=False)
        summary.to_csv(output_dir / "cyber_growth_vs_budget_summary.csv", index=False)
        draw_plot(
            revenue_growth,
            summary,
            output_dir / "cyber_growth_vs_budget_variation.png",
        )

    metadata = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "tickers": tickers,
        "budget_growth_series": SECURITY_BUDGET_GROWTH,
        "budget_source_note": BUDGET_SOURCE,
        "output_dir": str(output_dir),
        "warnings": warnings,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"Wrote cyber growth vs budget outputs to {output_dir}")
    if warnings:
        print(f"Completed with {len(warnings)} warnings. See run_metadata.json.")
    return 0 if revenue_growth is not None and not revenue_growth.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
