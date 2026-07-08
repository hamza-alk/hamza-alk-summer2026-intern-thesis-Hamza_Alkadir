#!/usr/bin/env python3
"""Plot cybersecurity ETFs against QQQ during a downturn window.

CIBR and HACK do not have price history back to 2002. The default window uses
calendar year 2022, but --start and --end can be changed for any available
history.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont


DEFAULT_TICKERS = ["CIBR", "HACK", "QQQ"]
DEFAULT_OUTPUT_ROOT = Path("data/cyber_etf_vs_qqq_downturn")
COLORS = {
    "CIBR": "#2563eb",
    "HACK": "#7c3aed",
    "QQQ": "#16a34a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull CIBR/HACK/QQQ prices and plot downturn performance."
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data/cyber_etf_vs_qqq_downturn/YYYY-MM-DD_HH-MM-SS.",
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
        running_peak = close.cummax()
        max_drawdown = ((close / running_peak) - 1).min()
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


def draw_plot(indexed: pd.DataFrame, metrics: pd.DataFrame, output_path: Path, start: str, end: str) -> None:
    width, height = 1300, 820
    left, right, top, bottom = 110, 280, 100, 110
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom

    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    axis_font = load_font(17, bold=True)
    tick_font = load_font(13)
    label_font = load_font(15, bold=True)
    note_font = load_font(15)

    y_min = max(0, int(indexed["indexed_price"].min() - 5))
    y_max = int(indexed["indexed_price"].max() + 5)
    dates = pd.to_datetime(indexed["date"])
    x_min = dates.min()
    x_max = dates.max()

    draw.text(
        (left, 28),
        f"CIBR/HACK vs QQQ Downturn Performance ({start} to {end})",
        fill="#111827",
        font=title_font,
    )

    for tick in range(y_min - (y_min % 10), y_max + 1, 10):
        y = scale(tick, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        draw.text((48, y - 8), f"{tick}", fill="#374151", font=tick_font)

    month_ticks = pd.date_range(x_min, x_max, periods=6)
    for date in month_ticks:
        x = scale(date.timestamp(), x_min.timestamp(), x_max.timestamp(), plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill="#f3f4f6", width=1)
        draw.text((x - 35, plot_bottom + 14), date.strftime("%b %Y"), fill="#374151", font=tick_font)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#111827", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#111827", width=2)
    draw.text((plot_left + 320, height - 74), "Date", fill="#111827", font=axis_font)
    draw.text((22, 385), "Indexed to 100", fill="#111827", font=axis_font)

    for ticker, group in indexed.groupby("ticker"):
        group = group.sort_values("date")
        points = [
            (
                scale(pd.Timestamp(row["date"]).timestamp(), x_min.timestamp(), x_max.timestamp(), plot_left, plot_right),
                scale(row["indexed_price"], y_min, y_max, plot_bottom, plot_top),
            )
            for _, row in group.iterrows()
        ]
        color = COLORS.get(ticker, "#6b7280")
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        if points:
            draw.text((points[-1][0] + 8, points[-1][1] - 8), ticker, fill=color, font=label_font)

    legend_x = plot_right + 35
    draw.text((legend_x, plot_top), "Downturn Metrics", fill="#111827", font=axis_font)
    metric_y = plot_top + 36
    for _, row in metrics.iterrows():
        color = COLORS.get(row["ticker"], "#6b7280")
        draw.ellipse((legend_x, metric_y + 4, legend_x + 13, metric_y + 17), fill=color, outline="#111827")
        draw.text((legend_x + 24, metric_y), row["ticker"], fill="#111827", font=label_font)
        draw.text(
            (legend_x + 24, metric_y + 22),
            f"Return: {row['period_return']:.1%}",
            fill="#374151",
            font=note_font,
        )
        draw.text(
            (legend_x + 24, metric_y + 42),
            f"Max drawdown: {row['max_drawdown']:.1%}",
            fill="#374151",
            font=note_font,
        )
        metric_y += 86

    draw.text(
        (plot_left, plot_bottom + 78),
        "Higher line = better preservation of capital. Metrics use daily close prices from Yahoo Finance.",
        fill="#374151",
        font=note_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    args = parse_args()
    tickers = sorted({ticker.strip().upper() for ticker in args.tickers if ticker.strip()})
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    prices, warnings = fetch_prices(tickers, args.start, args.end)
    if prices.empty:
        metadata = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "tickers": tickers,
            "start": args.start,
            "end": args.end,
            "output_dir": str(output_dir),
            "warnings": warnings,
        }
        with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print(f"No price data found. Wrote metadata to {output_dir / 'run_metadata.json'}")
        return 1

    metrics = calculate_metrics(prices)
    indexed = normalized_prices(prices)
    prices.to_csv(output_dir / "prices.csv", index=False)
    metrics.to_csv(output_dir / "downturn_metrics.csv", index=False)
    plot_path = output_dir / "cibr_hack_vs_qqq_downturn.png"
    draw_plot(indexed, metrics, plot_path, args.start, args.end)

    metadata = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "output_dir": str(output_dir),
        "plot_path": str(plot_path),
        "warnings": warnings,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"Wrote ETF downturn outputs to {output_dir}")
    if warnings:
        print(f"Completed with {len(warnings)} warnings. See run_metadata.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
