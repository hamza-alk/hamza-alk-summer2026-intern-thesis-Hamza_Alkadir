#!/usr/bin/env python3
"""Plot CIBR/HACK vs QQQ downturn performance with cyber vendor revenue growth.

The ETFs do not have revenue, so the revenue-growth variable is added from a
representative public cyber vendor basket: CRWD, PANW, FTNT, ZS, and OKTA.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ETF_TICKERS = ["CIBR", "HACK", "QQQ"]
DEFAULT_REVENUE_TICKERS = ["CRWD", "PANW", "FTNT", "ZS", "OKTA"]
DEFAULT_OUTPUT_ROOT = Path("data/cyber_etf_downturn_with_revenue_growth")
ETF_COLORS = {
    "CIBR": "#2563eb",
    "HACK": "#7c3aed",
    "QQQ": "#16a34a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CIBR/HACK/QQQ downturn performance with cyber revenue growth."
    )
    parser.add_argument("--etf-tickers", nargs="+", default=DEFAULT_ETF_TICKERS)
    parser.add_argument("--revenue-tickers", nargs="+", default=DEFAULT_REVENUE_TICKERS)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument(
        "--revenue-year",
        type=int,
        default=None,
        help="Fiscal year for revenue growth. Defaults to the year after --start.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data/cyber_etf_downturn_with_revenue_growth/YYYY-MM-DD_HH-MM-SS.",
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


def fetch_prices(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    frames = []
    for ticker in tickers:
        try:
            history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        except Exception as exc:  # noqa: BLE001 - keep batch runs alive.
            warnings.append(f"{ticker}: failed to fetch prices: {exc}")
            continue
        if history.empty or "Close" not in history.columns:
            warnings.append(f"{ticker}: no close prices returned for {start} to {end}")
            continue
        frame = history[["Close"]].copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame = frame.rename(columns={"Close": "close"})
        frame.insert(0, "ticker", ticker)
        frames.append(frame.reset_index().rename(columns={"Date": "date"}))
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close"]), warnings
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices.sort_values(["ticker", "date"]), warnings


def calculate_metrics(prices: pd.DataFrame, benchmark_ticker: str = "QQQ") -> pd.DataFrame:
    rows = []
    benchmark_return = None
    grouped = {ticker: group.sort_values("date") for ticker, group in prices.groupby("ticker")}
    if benchmark_ticker in grouped:
        benchmark = grouped[benchmark_ticker]["close"]
        if not benchmark.empty and benchmark.iloc[0] != 0:
            benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1

    for ticker, group in grouped.items():
        close = pd.to_numeric(group["close"], errors="coerce").dropna()
        if close.empty or close.iloc[0] == 0:
            continue
        daily_returns = close.pct_change().dropna()
        period_return = (close.iloc[-1] / close.iloc[0]) - 1
        max_drawdown = ((close / close.cummax()) - 1).min()
        volatility = daily_returns.std() * (252**0.5) if not daily_returns.empty else None
        rows.append(
            {
                "ticker": ticker,
                "start_date": group["date"].iloc[0].date(),
                "end_date": group["date"].iloc[-1].date(),
                "start_close": close.iloc[0],
                "end_close": close.iloc[-1],
                "period_return": period_return,
                "max_drawdown": max_drawdown,
                "annualized_volatility": volatility,
                "relative_return_vs_qqq": (
                    period_return - benchmark_return
                    if benchmark_return is not None and ticker != benchmark_ticker
                    else None
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("ticker")


def normalized_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date").copy()
        first_close = group["close"].iloc[0]
        if first_close == 0:
            continue
        group["indexed_price"] = (group["close"] / first_close) * 100
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def revenue_growth_rows(tickers: list[str], target_year: int) -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    rows = []
    for ticker in tickers:
        try:
            income = yf.Ticker(ticker).income_stmt
        except Exception as exc:  # noqa: BLE001 - keep batch runs alive.
            warnings.append(f"{ticker}: failed to fetch income statement: {exc}")
            continue
        if income is None or income.empty:
            warnings.append(f"{ticker}: no income statement returned")
            continue

        revenue_line = None
        for alias in ("Total Revenue", "Operating Revenue"):
            if alias in income.index:
                revenue_line = income.loc[alias]
                break
        if revenue_line is None:
            warnings.append(f"{ticker}: no revenue line found")
            continue

        revenue = (
            pd.to_numeric(revenue_line, errors="coerce")
            .dropna()
            .rename("revenue")
            .reset_index()
            .rename(columns={"index": "period_end"})
        )
        revenue["period_end"] = pd.to_datetime(revenue["period_end"], errors="coerce")
        revenue = revenue.dropna(subset=["period_end"]).sort_values("period_end")
        revenue["fiscal_year"] = revenue["period_end"].dt.year
        revenue["revenue_growth"] = revenue["revenue"].pct_change()

        match = revenue[revenue["fiscal_year"] == target_year]
        if match.empty:
            warnings.append(f"{ticker}: no revenue growth found for fiscal year {target_year}")
            continue
        latest = match.iloc[-1]
        rows.append(
            {
                "ticker": ticker,
                "fiscal_year": int(latest["fiscal_year"]),
                "period_end": latest["period_end"].date(),
                "revenue": latest["revenue"],
                "revenue_growth": latest["revenue_growth"],
            }
        )
    return pd.DataFrame(rows).sort_values("ticker"), warnings


def draw_plot(
    indexed: pd.DataFrame,
    etf_metrics: pd.DataFrame,
    revenue_growth: pd.DataFrame,
    output_path: Path,
    start: str,
    end: str,
    revenue_year: int,
) -> None:
    width, height = 1400, 900
    left, right, top, bottom = 110, 310, 104, 120
    price_top, price_bottom = top, 560
    revenue_top, revenue_bottom = 645, height - bottom
    plot_left, plot_right = left, width - right

    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    subtitle_font = load_font(16)
    axis_font = load_font(17, bold=True)
    tick_font = load_font(13)
    label_font = load_font(15, bold=True)
    note_font = load_font(14)

    dates = pd.to_datetime(indexed["date"])
    x_min = dates.min()
    x_max = dates.max()
    y_min = max(0, int(indexed["indexed_price"].min() - 5))
    y_max = int(indexed["indexed_price"].max() + 5)

    draw.text(
        (left, 28),
        f"CIBR/HACK vs QQQ Downturn + Cyber Revenue Growth ({start} to {end})",
        fill="#111827",
        font=title_font,
    )
    draw.text(
        (left, 66),
        f"ETF prices show market stress; revenue bars show FY{revenue_year} operating growth for cyber vendors.",
        fill="#374151",
        font=subtitle_font,
    )

    for tick in range(y_min - (y_min % 10), y_max + 1, 10):
        y = scale(tick, y_min, y_max, price_bottom, price_top)
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        draw.text((48, y - 8), f"{tick}", fill="#374151", font=tick_font)

    month_ticks = pd.date_range(x_min, x_max, periods=6)
    for date in month_ticks:
        x = scale(date.timestamp(), x_min.timestamp(), x_max.timestamp(), plot_left, plot_right)
        draw.line((x, price_top, x, price_bottom), fill="#f3f4f6", width=1)
        draw.text((x - 35, price_bottom + 12), date.strftime("%b %Y"), fill="#374151", font=tick_font)

    draw.line((plot_left, price_bottom, plot_right, price_bottom), fill="#111827", width=2)
    draw.line((plot_left, price_top, plot_left, price_bottom), fill="#111827", width=2)
    draw.text((plot_left, price_top - 26), "ETF Indexed to 100", fill="#111827", font=axis_font)

    for ticker, group in indexed.groupby("ticker"):
        group = group.sort_values("date")
        points = [
            (
                scale(pd.Timestamp(row["date"]).timestamp(), x_min.timestamp(), x_max.timestamp(), plot_left, plot_right),
                scale(row["indexed_price"], y_min, y_max, price_bottom, price_top),
            )
            for _, row in group.iterrows()
        ]
        color = ETF_COLORS.get(ticker, "#6b7280")
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        if points:
            draw.text((points[-1][0] + 8, points[-1][1] - 8), ticker, fill=color, font=label_font)

    plotted_revenue_growth = revenue_growth.dropna(subset=["revenue_growth"]).copy()
    if not plotted_revenue_growth.empty:
        growth_max = max(0.4, float(plotted_revenue_growth["revenue_growth"].max()) + 0.08)
        growth_min = min(0, float(plotted_revenue_growth["revenue_growth"].min()) - 0.04)
        for tick_pct in range(int(growth_min * 100), int(growth_max * 100) + 1, 10):
            tick = tick_pct / 100
            y = scale(tick, growth_min, growth_max, revenue_bottom, revenue_top)
            draw.line((plot_left, y, plot_right, y), fill="#f3f4f6", width=1)
            draw.text((50, y - 8), f"{tick_pct}%", fill="#374151", font=tick_font)

        bar_gap = 34
        bar_width = min(
            90,
            (plot_right - plot_left - bar_gap * (len(plotted_revenue_growth) + 1))
            / len(plotted_revenue_growth),
        )
        x = plot_left + bar_gap
        for _, row in plotted_revenue_growth.iterrows():
            bar_left = x
            bar_right = x + bar_width
            y = scale(row["revenue_growth"], growth_min, growth_max, revenue_bottom, revenue_top)
            draw.rectangle((bar_left, y, bar_right, revenue_bottom), fill="#f59e0b", outline="#111827")
            draw.text((bar_left + 4, y - 22), f"{row['revenue_growth']:.0%}", fill="#111827", font=label_font)
            draw.text((bar_left + 2, revenue_bottom + 12), row["ticker"], fill="#111827", font=label_font)
            x += bar_width + bar_gap

        median_growth = plotted_revenue_growth["revenue_growth"].median()
        median_y = scale(median_growth, growth_min, growth_max, revenue_bottom, revenue_top)
        draw.line((plot_left, median_y, plot_right, median_y), fill="#b45309", width=3)
        draw.text(
            (plot_right - 170, median_y - 24),
            f"Median: {median_growth:.0%}",
            fill="#92400e",
            font=label_font,
        )

    draw.line((plot_left, revenue_bottom, plot_right, revenue_bottom), fill="#111827", width=2)
    draw.line((plot_left, revenue_top, plot_left, revenue_bottom), fill="#111827", width=2)
    draw.text((plot_left, revenue_top - 28), f"FY{revenue_year} Revenue Growth", fill="#111827", font=axis_font)

    legend_x = plot_right + 34
    draw.text((legend_x, price_top), "ETF Downturn Metrics", fill="#111827", font=axis_font)
    metric_y = price_top + 36
    for _, row in etf_metrics.iterrows():
        color = ETF_COLORS.get(row["ticker"], "#6b7280")
        draw.ellipse((legend_x, metric_y + 4, legend_x + 13, metric_y + 17), fill=color, outline="#111827")
        draw.text((legend_x + 24, metric_y), row["ticker"], fill="#111827", font=label_font)
        draw.text((legend_x + 24, metric_y + 22), f"Return: {row['period_return']:.1%}", fill="#374151", font=note_font)
        draw.text((legend_x + 24, metric_y + 42), f"Max drawdown: {row['max_drawdown']:.1%}", fill="#374151", font=note_font)
        metric_y += 86

    draw.text((legend_x, revenue_top), f"FY{revenue_year} Revenue", fill="#111827", font=axis_font)
    draw.rectangle((legend_x, revenue_top + 36, legend_x + 30, revenue_top + 56), fill="#f59e0b", outline="#111827")
    draw.text((legend_x + 42, revenue_top + 34), "Cyber vendor growth", fill="#111827", font=note_font)
    draw.line((legend_x, revenue_top + 82, legend_x + 30, revenue_top + 82), fill="#b45309", width=3)
    draw.text((legend_x + 42, revenue_top + 73), "Median growth", fill="#111827", font=note_font)

    draw.text(
        (plot_left, height - 40),
        "ETF prices and company revenue data pulled from Yahoo Finance. Revenue growth is not ETF revenue; it is the underlying cyber vendor basket.",
        fill="#4b5563",
        font=note_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    args = parse_args()
    etf_tickers = sorted({ticker.strip().upper() for ticker in args.etf_tickers if ticker.strip()})
    revenue_tickers = sorted({ticker.strip().upper() for ticker in args.revenue_tickers if ticker.strip()})
    revenue_year = args.revenue_year or (pd.to_datetime(args.start).year + 1)
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    prices, price_warnings = fetch_prices(etf_tickers, args.start, args.end)
    revenue_growth, revenue_warnings = revenue_growth_rows(revenue_tickers, revenue_year)
    warnings = price_warnings + revenue_warnings

    if prices.empty:
        metadata = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "etf_tickers": etf_tickers,
            "revenue_tickers": revenue_tickers,
            "start": args.start,
            "end": args.end,
            "revenue_year": revenue_year,
            "output_dir": str(output_dir),
            "warnings": warnings,
        }
        with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print(f"No ETF price data found. Wrote metadata to {output_dir / 'run_metadata.json'}")
        return 1

    etf_metrics = calculate_metrics(prices)
    indexed = normalized_prices(prices)
    prices.to_csv(output_dir / "etf_prices.csv", index=False)
    etf_metrics.to_csv(output_dir / "etf_downturn_metrics.csv", index=False)
    revenue_growth.to_csv(output_dir / "cyber_vendor_revenue_growth.csv", index=False)

    plot_path = output_dir / "cyber_etf_downturn_with_revenue_growth.png"
    draw_plot(indexed, etf_metrics, revenue_growth, plot_path, args.start, args.end, revenue_year)

    metadata = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "etf_tickers": etf_tickers,
        "revenue_tickers": revenue_tickers,
        "start": args.start,
        "end": args.end,
        "revenue_year": revenue_year,
        "output_dir": str(output_dir),
        "plot_path": str(plot_path),
        "warnings": warnings,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"Wrote ETF downturn + revenue growth outputs to {output_dir}")
    if warnings:
        print(f"Completed with {len(warnings)} warnings. See run_metadata.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
