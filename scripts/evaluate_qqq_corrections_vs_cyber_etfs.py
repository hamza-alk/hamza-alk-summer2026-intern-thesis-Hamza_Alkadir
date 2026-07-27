#!/usr/bin/env python3
"""Find QQQ corrections and compare cyber ETFs over the same windows.

Definition used here:
    A QQQ correction is triggered when QQQ falls at least 10% from a prior
    closing peak. The peak date is not part of the correction. The correction
    starts on the next trading day after the peak date, ends on the last trading
    day before QQQ first closes back at or above the peak close, and recovers on
    that first close back at or above the peak close.

The ETF comparison window is the QQQ correction period, excluding the peak and
recovery dates. This keeps the event definition objective and applies the exact
same dates to each ETF.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ["CIBR", "HACK", "QQQ"]
DEFAULT_BENCHMARK = "QQQ"
DEFAULT_OUTPUT_ROOT = Path("data/cyber_etf_qqq_corrections")
EVENT_METRIC_COLUMNS = [
    "event_id",
    "ticker",
    "available",
    "period_start",
    "period_end",
    "period_return",
    "max_drawdown",
    "annualized_volatility",
    "relative_return_vs_qqq",
]


@dataclass
class CorrectionEvent:
    event_id: int
    peak_date: pd.Timestamp
    correction_start_date: pd.Timestamp
    trigger_date: pd.Timestamp
    trough_date: pd.Timestamp
    correction_end_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    peak_close: float
    correction_start_close: float
    trigger_close: float
    trough_close: float
    correction_end_close: float
    recovery_close: float | None
    is_recovered: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find all QQQ corrections and compare cyber ETFs on those same dates."
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--start",
        default="2015-07-01",
        help="Default starts near CIBR's launch, so both CIBR and HACK can be compared.",
    )
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Correction threshold as a decimal. 0.10 means a 10%% QQQ drop.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data/cyber_etf_qqq_corrections/YYYY-MM-DD_HH-MM-SS.",
    )
    return parser.parse_args()


def fetch_prices(tickers: list[str], start: str, end: str | None) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    frames = []
    for ticker in tickers:
        try:
            history = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=False,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch runs alive.
            warnings.append(f"{ticker}: failed to fetch prices: {exc}")
            continue
        if history.empty or "Close" not in history.columns:
            warnings.append(f"{ticker}: no close prices returned for {start} to {end or 'latest'}")
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


def find_corrections(
    benchmark_prices: pd.DataFrame,
    threshold: float,
) -> list[CorrectionEvent]:
    close = benchmark_prices.sort_values("date")[["date", "close"]].dropna()
    if close.empty:
        return []

    events: list[CorrectionEvent] = []
    event_id = 1
    rows = list(close.itertuples(index=False))
    peak_idx = 0
    peak_date = rows[peak_idx].date
    peak_close = float(rows[peak_idx].close)
    in_correction = False
    trigger_date: pd.Timestamp | None = None
    trigger_close: float | None = None
    correction_start_idx: int | None = None

    for idx, row in enumerate(rows[1:], start=1):
        date = row.date
        price = float(row.close)

        if not in_correction:
            if price >= peak_close:
                peak_idx = idx
                peak_date = date
                peak_close = price
                continue

            drawdown = (price / peak_close) - 1
            if drawdown <= -threshold:
                in_correction = True
                correction_start_idx = peak_idx + 1
                trigger_date = date
                trigger_close = price
            continue

        if price >= peak_close:
            correction_end_idx = idx - 1
            if correction_start_idx is None or correction_start_idx > correction_end_idx:
                correction_start_idx = peak_idx + 1
            correction_rows = rows[correction_start_idx : correction_end_idx + 1]
            trough_row = min(correction_rows, key=lambda item: float(item.close))
            correction_start_row = rows[correction_start_idx]
            correction_end_row = rows[correction_end_idx]
            events.append(
                CorrectionEvent(
                    event_id=event_id,
                    peak_date=peak_date,
                    correction_start_date=correction_start_row.date,
                    trigger_date=trigger_date or date,
                    trough_date=trough_row.date,
                    correction_end_date=correction_end_row.date,
                    recovery_date=date,
                    peak_close=peak_close,
                    correction_start_close=float(correction_start_row.close),
                    trigger_close=trigger_close or price,
                    trough_close=float(trough_row.close),
                    correction_end_close=float(correction_end_row.close),
                    recovery_close=price,
                    is_recovered=True,
                )
            )
            event_id += 1
            in_correction = False
            peak_idx = idx
            peak_date = date
            peak_close = price
            trigger_date = None
            trigger_close = None
            correction_start_idx = None

    if in_correction:
        correction_end_idx = len(rows) - 1
        if correction_start_idx is None or correction_start_idx > correction_end_idx:
            correction_start_idx = peak_idx + 1
        correction_rows = rows[correction_start_idx : correction_end_idx + 1]
        trough_row = min(correction_rows, key=lambda item: float(item.close))
        correction_start_row = rows[correction_start_idx]
        correction_end_row = rows[correction_end_idx]
        events.append(
            CorrectionEvent(
                event_id=event_id,
                peak_date=peak_date,
                correction_start_date=correction_start_row.date,
                trigger_date=trigger_date or correction_start_row.date,
                trough_date=trough_row.date,
                correction_end_date=correction_end_row.date,
                recovery_date=None,
                peak_close=peak_close,
                correction_start_close=float(correction_start_row.close),
                trigger_close=trigger_close or float(correction_start_row.close),
                trough_close=float(trough_row.close),
                correction_end_close=float(correction_end_row.close),
                recovery_close=None,
                is_recovered=False,
            )
        )

    return events


def events_to_frame(events: list[CorrectionEvent], threshold: float) -> pd.DataFrame:
    rows = []
    for event in events:
        rows.append(
            {
                "event_id": event.event_id,
                "definition": f"QQQ close fell at least {threshold:.0%} from prior closing peak",
                "peak_date": event.peak_date.date(),
                "correction_start_date": event.correction_start_date.date(),
                "trigger_date": event.trigger_date.date(),
                "trough_date": event.trough_date.date(),
                "correction_end_date": event.correction_end_date.date(),
                "is_recovered": event.is_recovered,
                "peak_close": event.peak_close,
                "correction_start_close": event.correction_start_close,
                "trigger_close": event.trigger_close,
                "trough_close": event.trough_close,
                "correction_end_close": event.correction_end_close,
                "qqq_drawdown_to_trigger": (event.trigger_close / event.peak_close) - 1,
                "qqq_peak_to_trough_return": (event.trough_close / event.peak_close) - 1,
                "qqq_correction_period_return": (
                    event.correction_end_close / event.correction_start_close
                )
                - 1,
                "peak_to_trough_days": (event.trough_date - event.peak_date).days,
                "trigger_to_trough_days": (event.trough_date - event.trigger_date).days,
                "correction_period_days": (
                    event.correction_end_date - event.correction_start_date
                ).days,
            }
        )
    return pd.DataFrame(rows)


def closest_window_prices(group: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    window = group[(group["date"] >= start) & (group["date"] <= end)].sort_values("date").copy()
    return window


def calculate_event_metrics(
    prices: pd.DataFrame,
    events: list[CorrectionEvent],
    benchmark: str,
) -> pd.DataFrame:
    rows = []
    grouped = {ticker: group.sort_values("date") for ticker, group in prices.groupby("ticker")}

    for event in events:
        benchmark_return = None
        if benchmark in grouped:
            benchmark_window = closest_window_prices(
                grouped[benchmark],
                event.correction_start_date,
                event.correction_end_date,
            )
            if not benchmark_window.empty and benchmark_window["close"].iloc[0] != 0:
                benchmark_return = (benchmark_window["close"].iloc[-1] / benchmark_window["close"].iloc[0]) - 1

        for ticker, group in grouped.items():
            window = closest_window_prices(group, event.correction_start_date, event.correction_end_date)
            if window.empty or len(window) < 2 or window["close"].iloc[0] == 0:
                rows.append(
                    {
                        "event_id": event.event_id,
                        "ticker": ticker,
                        "available": False,
                        "period_start": None,
                        "period_end": None,
                        "period_return": None,
                        "max_drawdown": None,
                        "annualized_volatility": None,
                        "relative_return_vs_qqq": None,
                    }
                )
                continue

            close = pd.to_numeric(window["close"], errors="coerce").dropna()
            daily_returns = close.pct_change().dropna()
            period_return = (close.iloc[-1] / close.iloc[0]) - 1
            running_peak = close.cummax()
            max_drawdown = ((close / running_peak) - 1).min()
            volatility = daily_returns.std() * (252**0.5) if not daily_returns.empty else None
            rows.append(
                {
                    "event_id": event.event_id,
                    "ticker": ticker,
                    "available": True,
                    "period_start": window["date"].iloc[0].date(),
                    "period_end": window["date"].iloc[-1].date(),
                    "period_return": period_return,
                    "max_drawdown": max_drawdown,
                    "annualized_volatility": volatility,
                    "relative_return_vs_qqq": (
                        period_return - benchmark_return
                        if benchmark_return is not None and ticker != benchmark
                        else None
                    ),
                }
            )

    return pd.DataFrame(rows, columns=EVENT_METRIC_COLUMNS)


def summarize_metrics(metrics: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if metrics.empty or "available" not in metrics.columns:
        return pd.DataFrame(
            columns=[
                "ticker",
                "events_available",
                "median_period_return",
                "median_max_drawdown",
                "median_annualized_volatility",
                "median_relative_return_vs_qqq",
                "outperformed_qqq_count",
                "underperformed_qqq_count",
            ]
        )
    available = metrics[metrics["available"]].copy()
    rows = []
    for ticker, group in available.groupby("ticker"):
        cyber_group = group[group["ticker"] != benchmark]
        rows.append(
            {
                "ticker": ticker,
                "events_available": int(group["event_id"].nunique()),
                "median_period_return": group["period_return"].median(),
                "median_max_drawdown": group["max_drawdown"].median(),
                "median_annualized_volatility": group["annualized_volatility"].median(),
                "median_relative_return_vs_qqq": cyber_group["relative_return_vs_qqq"].median()
                if not cyber_group.empty
                else None,
                "outperformed_qqq_count": int((cyber_group["relative_return_vs_qqq"] > 0).sum())
                if not cyber_group.empty
                else None,
                "underperformed_qqq_count": int((cyber_group["relative_return_vs_qqq"] < 0).sum())
                if not cyber_group.empty
                else None,
            }
        )
    return pd.DataFrame(rows).sort_values("ticker")


def write_verdict(
    output_dir: Path,
    event_metrics: pd.DataFrame,
    benchmark: str,
) -> None:
    cyber = event_metrics[(event_metrics["available"]) & (event_metrics["ticker"] != benchmark)].copy()
    benchmark_rows = event_metrics[(event_metrics["available"]) & (event_metrics["ticker"] == benchmark)].copy()
    if cyber.empty or benchmark_rows.empty:
        verdict = "Does not support the thesis."
        detail = "There was not enough overlapping ETF data to compare cyber ETFs with QQQ."
    else:
        cyber_event = cyber.groupby("event_id")["period_return"].median().rename("median_cyber_return")
        qqq_event = benchmark_rows.set_index("event_id")["period_return"].rename("qqq_return")
        comparison = pd.concat([cyber_event, qqq_event], axis=1).dropna()
        outperformance_rate = (comparison["median_cyber_return"] > comparison["qqq_return"]).mean()
        if outperformance_rate >= 0.60:
            verdict = "Supports the thesis."
        else:
            verdict = "Does not support the thesis."
        detail = (
            f"Median cyber ETF returns beat QQQ in {outperformance_rate:.0%} of available "
            "QQQ correction events. The support threshold is 60%."
        )

    text = (
        f"{verdict}\n\n"
        "Test: all QQQ corrections using a 10% drawdown from a prior closing peak.\n"
        "Comparison: CIBR/HACK median performance versus QQQ over each QQQ correction period.\n\n"
        f"{detail}\n"
    )
    filename = "verdict_supports_thesis.txt" if verdict.startswith("Supports") else "verdict_does_not_support_thesis.txt"
    (output_dir / filename).write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    tickers = sorted({ticker.strip().upper() for ticker in args.tickers if ticker.strip()})
    benchmark = args.benchmark.strip().upper()
    if benchmark not in tickers:
        tickers.append(benchmark)
        tickers = sorted(tickers)

    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    prices, warnings = fetch_prices(tickers, args.start, args.end)
    benchmark_prices = prices[prices["ticker"] == benchmark].copy()
    events = find_corrections(benchmark_prices, args.threshold)
    events_frame = events_to_frame(events, args.threshold)
    event_metrics = calculate_event_metrics(prices, events, benchmark)
    summary = summarize_metrics(event_metrics, benchmark)

    prices.to_csv(output_dir / "prices.csv", index=False)
    events_frame.to_csv(output_dir / "qqq_correction_events.csv", index=False)
    event_metrics.to_csv(output_dir / "etf_metrics_by_correction.csv", index=False)
    summary.to_csv(output_dir / "summary_by_ticker.csv", index=False)
    write_verdict(output_dir, event_metrics, benchmark)

    metadata = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "definition": f"{benchmark} close fell at least {args.threshold:.0%} from prior closing peak",
        "comparison_window": (
            f"{benchmark} correction start date through {benchmark} correction end date; "
            "peak and recovery dates are excluded"
        ),
        "tickers": tickers,
        "benchmark": benchmark,
        "start": args.start,
        "end": args.end,
        "threshold": args.threshold,
        "correction_count": len(events),
        "output_dir": str(output_dir),
        "warnings": warnings,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"Found {len(events)} {benchmark} corrections.")
    print(f"Wrote all-corrections ETF outputs to {output_dir}")
    if warnings:
        print(f"Completed with {len(warnings)} warnings. See run_metadata.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
