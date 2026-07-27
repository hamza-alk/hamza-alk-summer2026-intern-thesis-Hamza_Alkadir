from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt
from PIL import Image, ImageDraw, ImageFont

from scripts.plot_cyber_etf_vs_qqq_downturn import (
    COLORS as ETF_COLORS,
    calculate_metrics as calculate_etf_metrics,
    draw_plot as draw_etf_plot,
    fetch_prices as fetch_etf_prices,
    normalized_prices as normalize_etf_prices,
)
from thesis_verdict_scoring import score_thesis_verdict


DEFAULT_CYBER_TICKERS = "CRWD PANW FTNT ZS OKTA"
DEFAULT_TECH_TICKERS = "AAPL MSFT GOOGL AMZN META"
VALIDATION_FILE = Path("data/validation/financial_metric_validation.csv")
VALIDATION_SUMMARY_FILE = Path("data/validation/validation_summary.csv")
YFINANCE_RESULTS_ROOT = Path("data/yfinance_fred_cyber_vs_tech")
QQQ_CORRECTIONS_ROOT = Path("data/cyber_etf_qqq_corrections")
EDGAR_COMPANYFACTS_ROOT = Path("edgar_data/raw_companyfacts")
DEFAULT_BUDGET_GROWTH = pd.DataFrame(
    {
        "budget_year": [2022, 2023, 2024],
        "security_budget_growth_pct": [17.0, 6.0, 8.0],
    }
)

COHORT_COLORS = {
    "cybersecurity": "#2563eb",
    "broad_tech": "#16a34a",
}

REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)

THESIS_STATEMENT = (
    "Cybersecurity companies may have more durable growth than broad technology "
    "companies because customers treat security as mission-critical infrastructure, "
    "while broader technology spending can be more discretionary. If that durability "
    "creates a genuine moat, cybersecurity companies should show both stronger "
    "operating performance and better resilience during difficult market conditions."
)


def apply_dashboard_style() -> None:
    """Small usability improvements without turning the app into a design showcase."""
    st.markdown(
        """
        <style>
        .block-container {max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem;}
        h1, h2, h3 {letter-spacing: -0.02em;}
        div[data-testid="stMetric"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: #ffffff;
        }
        .cohort-key {
            display: flex;
            gap: 1.25rem;
            align-items: center;
            margin: .25rem 0 1.25rem;
            color: #475569;
            font-size: .9rem;
        }
        .cohort-dot {
            display: inline-block;
            width: .7rem;
            height: .7rem;
            border-radius: 50%;
            margin-right: .4rem;
        }
        .thesis-box {
            border-left: 4px solid #2563eb;
            background: #f8fafc;
            padding: 1rem 1.15rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 1.25rem;
        }
        .takeaway {
            border: 1px solid #e2e8f0;
            background: #f8fafc;
            border-radius: 8px;
            padding: .8rem 1rem;
            margin-bottom: .55rem;
            font-size: 1.08rem;
        }
        .major-finding {
            border-left: 5px solid #2563eb;
            background: #eff6ff;
            border-radius: 0 8px 8px 0;
            padding: 1rem 1.2rem;
            margin: .75rem 0 1.25rem;
            font-size: 1.25rem;
            line-height: 1.45;
            font-weight: 650;
            color: #172554;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_cohort_key() -> None:
    st.markdown(
        """
        <div class="cohort-key">
          <span><span class="cohort-dot" style="background:#2563eb"></span>Cybersecurity</span>
          <span><span class="cohort-dot" style="background:#16a34a"></span>Broad tech</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_major_finding(text: str) -> None:
    st.markdown(f"<div class='major-finding'>{text}</div>", unsafe_allow_html=True)


def toggle_data_table(state_key: str) -> None:
    st.session_state[state_key] = not st.session_state.get(state_key, False)


def render_data_table(data: pd.DataFrame, key: str, **kwargs) -> None:
    """Keep supporting tables available without letting them dominate a page."""
    state_key = f"show_data_{key}"
    is_visible = st.session_state.get(state_key, False)
    st.button(
        "Hide data" if is_visible else "Show data",
        key=f"{state_key}_button",
        on_click=toggle_data_table,
        args=(state_key,),
    )
    is_visible = st.session_state.get(state_key, False)
    if is_visible:
        st.dataframe(data, **kwargs)


@st.dialog("What is a downturn?")
def show_downturn_definition() -> None:
    st.write(
        "A downturn is a period when the market falls significantly from a "
        "recent high. This dataset treats a QQQ decline of at least 10% from "
        "its prior peak as a downturn."
    )


@dataclass
class ImageResult:
    image: Image.Image
    csv: pd.DataFrame
    filename_base: str


def parse_tickers(value: str) -> list[str]:
    return sorted({ticker.strip().upper() for ticker in value.replace(",", " ").split() if ticker.strip()})


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
    if pd.isna(value) or source_max == source_min:
        return (target_min + target_max) / 2
    return target_min + ((value - source_min) / (source_max - source_min)) * (target_max - target_min)


def png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def draw_qqq_correction_drawdown_plot(
    drawdown_data: pd.DataFrame,
    metrics: pd.DataFrame,
    start: str,
    end: str,
) -> Image.Image:
    width, height = 1300, 860
    left, right, top, bottom = 120, 290, 100, 120
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom

    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    axis_font = load_font(17, bold=True)
    tick_font = load_font(13)
    label_font = load_font(15, bold=True)
    note_font = load_font(15)

    chart_data = drawdown_data.copy()
    y_min = int(np.floor(chart_data["drawdown_pct"].min() / 5) * 5) - 5
    y_max = max(5, int(np.ceil(chart_data["drawdown_pct"].max() / 5) * 5) + 5)
    dates = pd.to_datetime(chart_data["date"])
    x_min = dates.min()
    x_max = dates.max()

    draw.text(
        (left, 28),
        f"ETF Drawdown From Pre-Correction Peak ({start} to {end})",
        fill="#111827",
        font=title_font,
    )

    for tick in range(y_min - (y_min % 5), y_max + 1, 5):
        y = scale(tick, y_min, y_max, plot_bottom, plot_top)
        line_color = "#cbd5e1" if tick == 0 else "#e5e7eb"
        line_width = 2 if tick == 0 else 1
        draw.line((plot_left, y, plot_right, y), fill=line_color, width=line_width)
        draw.text((52, y - 8), f"{tick}%", fill="#374151", font=tick_font)

    date_ticks = pd.date_range(x_min, x_max, periods=6)
    for tick_date in date_ticks:
        x = scale(tick_date.timestamp(), x_min.timestamp(), x_max.timestamp(), plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill="#f3f4f6", width=1)
        draw.text((x - 35, plot_bottom + 14), tick_date.strftime("%b %d"), fill="#374151", font=tick_font)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#111827", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#111827", width=2)
    draw.text((plot_left + 320, height - 80), "Date", fill="#111827", font=axis_font)
    draw.text((18, 395), "Drawdown (%)", fill="#111827", font=axis_font)

    for ticker, group in chart_data.groupby("ticker"):
        group = group.sort_values("date")
        points = [
            (
                scale(pd.Timestamp(row["date"]).timestamp(), x_min.timestamp(), x_max.timestamp(), plot_left, plot_right),
                scale(row["drawdown_pct"], y_min, y_max, plot_bottom, plot_top),
            )
            for _, row in group.iterrows()
        ]
        color = ETF_COLORS.get(ticker, "#6b7280")
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        peak_rows = group[group["is_peak_marker"]] if "is_peak_marker" in group else pd.DataFrame()
        if not peak_rows.empty:
            peak_row = peak_rows.iloc[0]
            peak_x = scale(pd.Timestamp(peak_row["date"]).timestamp(), x_min.timestamp(), x_max.timestamp(), plot_left, plot_right)
            peak_y = scale(peak_row["drawdown_pct"], y_min, y_max, plot_bottom, plot_top)
            draw.ellipse((peak_x - 7, peak_y - 7, peak_x + 7, peak_y + 7), fill=color, outline="#111827", width=2)
            draw.text((peak_x + 8, peak_y - 24), "Peak", fill=color, font=tick_font)
        if points:
            draw.text((points[-1][0] + 8, points[-1][1] - 8), ticker, fill=color, font=label_font)

    legend_x = plot_right + 35
    draw.text((legend_x, plot_top), "Downturn Metrics", fill="#111827", font=axis_font)
    metric_y = plot_top + 36
    for _, row in metrics.sort_values("ticker").iterrows():
        color = ETF_COLORS.get(row["ticker"], "#6b7280")
        draw.ellipse((legend_x, metric_y + 4, legend_x + 13, metric_y + 17), fill=color, outline="#111827")
        draw.text((legend_x + 24, metric_y), row["ticker"], fill="#111827", font=label_font)
        draw.text(
            (legend_x + 24, metric_y + 22),
            f"Max drawdown: {row['max_drawdown']:.1%}",
            fill="#374151",
            font=note_font,
        )
        metric_y += 66

    draw.text(
        (plot_left, plot_bottom + 78),
        "Dots mark each ETF's peak before the correction starts. Closer to 0% means a smaller loss.",
        fill="#374151",
        font=note_font,
    )
    return image


def correction_start_drawdown_data(
    prices: pd.DataFrame,
    event_metrics: pd.DataFrame,
    start_date: str,
    trough_date: str,
    peak_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    trough = pd.to_datetime(trough_date)
    frames = []
    for ticker in event_metrics["ticker"].dropna().unique():
        group = prices[
            (prices["ticker"] == ticker)
            & (prices["date"] >= start)
            & (prices["date"] <= trough)
        ].sort_values("date").copy()
        peak_close = None
        peak_date = None
        if peak_table is not None and not peak_table.empty:
            peak_match = peak_table[peak_table["ticker"].eq(ticker)]
            if not peak_match.empty:
                peak_close = peak_match["peak_close"].iloc[0]
                peak_date = peak_match["peak_date"].iloc[0]
        if pd.isna(peak_close) or peak_close in (None, 0):
            peak_close = group["close"].iloc[0] if not group.empty else None
        if group.empty or peak_close in (None, 0):
            continue
        if peak_date is not None and not pd.isna(peak_date):
            peak_row = pd.DataFrame(
                [
                    {
                        "date": pd.to_datetime(peak_date),
                        "ticker": ticker,
                        "close": peak_close,
                        "drawdown_pct": 0.0,
                        "is_peak_marker": True,
                    }
                ]
            )
        else:
            peak_row = pd.DataFrame()
        group["drawdown_pct"] = ((group["close"] / peak_close) - 1) * 100
        group["is_peak_marker"] = False
        if not peak_row.empty:
            group = pd.concat([peak_row, group], ignore_index=True)
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def etf_peak_before_correction(
    prices: pd.DataFrame,
    event_metrics: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    rows = []
    for ticker in event_metrics["ticker"].dropna().unique():
        group = prices[
            (prices["ticker"] == ticker)
            & (prices["date"] < start)
        ].sort_values("date").copy()
        if group.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "peak_date": None,
                    "peak_close": None,
                    "peak_is_above_all_correction_closes": False,
                    "correction_start_close": None,
                    "drop_from_peak_to_start": None,
                }
            )
            continue

        correction_group = prices[
            (prices["ticker"] == ticker)
            & (prices["date"] >= start)
            & (prices["date"] <= end)
        ].sort_values("date")
        correction_max = correction_group["close"].max() if not correction_group.empty else None
        valid_group = (
            group[group["close"] > correction_max]
            if correction_max is not None and not pd.isna(correction_max)
            else group
        )
        peak_is_valid = not valid_group.empty
        peak_row = (
            valid_group.loc[valid_group["close"].idxmax()]
            if peak_is_valid
            else group.loc[group["close"].idxmax()]
        )
        start_group = prices[
            (prices["ticker"] == ticker)
            & (prices["date"] >= start)
        ].sort_values("date")
        start_close = start_group["close"].iloc[0] if not start_group.empty else None
        rows.append(
            {
                "ticker": ticker,
                "peak_date": peak_row["date"].date(),
                "peak_close": peak_row["close"],
                "peak_is_above_all_correction_closes": peak_is_valid,
                "correction_start_close": start_close,
                "drop_from_peak_to_start": (
                    (start_close / peak_row["close"]) - 1
                    if start_close is not None and peak_row["close"] != 0
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def revenue_growth_for_ticker(ticker: str) -> float | None:
    try:
        income = yf.Ticker(ticker).income_stmt
    except Exception:
        return None
    if income is None or income.empty:
        return None
    for alias in ("Total Revenue", "Operating Revenue"):
        if alias in income.index:
            revenue = pd.to_numeric(income.loc[alias], errors="coerce").dropna()
            revenue.index = pd.to_datetime(revenue.index, errors="coerce")
            revenue = revenue.sort_index(ascending=False)
            if len(revenue) >= 2 and revenue.iloc[1] != 0:
                return float((revenue.iloc[0] - revenue.iloc[1]) / abs(revenue.iloc[1]))
    return None


def fcf_margin_for_ticker(ticker: str) -> float | None:
    try:
        stock = yf.Ticker(ticker)
        income = stock.income_stmt
        cashflow = stock.cashflow
    except Exception:
        return None
    if income is None or income.empty or cashflow is None or cashflow.empty:
        return None
    revenue = None
    for alias in ("Total Revenue", "Operating Revenue"):
        if alias in income.index:
            revenue = pd.to_numeric(income.loc[alias], errors="coerce").dropna()
            revenue.index = pd.to_datetime(revenue.index, errors="coerce")
            revenue = revenue.sort_index(ascending=False)
            break
    if revenue is None or revenue.empty or revenue.iloc[0] == 0:
        return None

    fcf = None
    if "Free Cash Flow" in cashflow.index:
        fcf = pd.to_numeric(cashflow.loc["Free Cash Flow"], errors="coerce").dropna()
    elif "Operating Cash Flow" in cashflow.index and "Capital Expenditure" in cashflow.index:
        fcf = (
            pd.to_numeric(cashflow.loc["Operating Cash Flow"], errors="coerce")
            + pd.to_numeric(cashflow.loc["Capital Expenditure"], errors="coerce")
        ).dropna()
    if fcf is None or fcf.empty:
        return None
    fcf.index = pd.to_datetime(fcf.index, errors="coerce")
    fcf = fcf.sort_index(ascending=False)
    return float(fcf.iloc[0] / revenue.iloc[0])


def validated_fundamental(ticker: str, metric: str, fallback: float | None) -> tuple[float | None, str]:
    if not VALIDATION_FILE.exists():
        return fallback, "yFinance"
    validation = pd.read_csv(VALIDATION_FILE)
    match = validation[
        validation["ticker"].eq(ticker) & validation["metric"].eq(metric)
    ]
    if match.empty or pd.isna(match.iloc[0]["selected_value"]):
        return fallback, "yFinance"
    return float(match.iloc[0]["selected_value"]), str(match.iloc[0]["selected_source"])


@st.cache_data(show_spinner=False, ttl=60 * 30)
def ticker_price_metrics(ticker: str, start: date, end: date) -> dict[str, float | str | None]:
    history = yf.Ticker(ticker).history(start=str(start), end=str(end), auto_adjust=False)
    if history is None or history.empty or "Close" not in history.columns:
        return {
            "ticker": ticker,
            "period_return": None,
            "max_drawdown": None,
            "annualized_volatility": None,
        }
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if close.empty or close.iloc[0] == 0:
        return {
            "ticker": ticker,
            "period_return": None,
            "max_drawdown": None,
            "annualized_volatility": None,
        }
    daily_returns = close.pct_change().dropna()
    return {
        "ticker": ticker,
        "period_return": float((close.iloc[-1] / close.iloc[0]) - 1),
        "max_drawdown": float((close / close.cummax() - 1).min()),
        "annualized_volatility": float(daily_returns.std() * (252**0.5)) if not daily_returns.empty else None,
    }


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cyber_vs_tech_data(cyber_tickers: tuple[str, ...], tech_tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    rows = []
    for cohort, tickers in (("cybersecurity", cyber_tickers), ("broad_tech", tech_tickers)):
        for ticker in tickers:
            row = ticker_price_metrics(ticker, start, end)
            row["cohort"] = cohort
            row["revenue_growth"] = revenue_growth_for_ticker(ticker)
            yf_fcf_margin = fcf_margin_for_ticker(ticker)
            row["free_cash_flow_margin"], row["fcf_margin_source"] = validated_fundamental(
                ticker,
                "free_cash_flow_margin",
                yf_fcf_margin,
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    df["rule_of_40"] = df["revenue_growth"] + df["free_cash_flow_margin"]
    return df


def metric_label(metric: str) -> str:
    return {
        "max_drawdown": "Max Drawdown",
        "period_return": "Period Return",
        "annualized_volatility": "Annualized Volatility",
        "revenue_growth": "Revenue Growth",
        "free_cash_flow_margin": "FCF Margin",
        "rule_of_40": "Rule of 40",
    }[metric]


def metric_explanation(metric: str) -> str:
    return {
        "max_drawdown": (
            "Max drawdown measures the worst peak-to-trough stock decline during the selected period. "
            "Less negative is better because it means the stock lost less value at its worst point."
        ),
        "period_return": (
            "Period return measures total stock performance across the selected date window. "
            "Higher is better because it means investors lost less money or earned more."
        ),
        "annualized_volatility": (
            "Annualized volatility measures how unstable the stock price was during the selected period. "
            "Lower is better because resilient companies should move less violently in a downturn."
        ),
        "revenue_growth": (
            "Revenue growth measures how quickly company sales are expanding. "
            "Higher is better because durable demand should show up as continued growth."
        ),
        "free_cash_flow_margin": (
            "Free cash flow margin measures how much cash the company generates from each dollar of revenue. "
            "Higher is better because it shows growth is converting into cash."
        ),
        "rule_of_40": (
            "Rule of 40 combines revenue growth and free cash flow margin. "
            "Higher is better because it rewards companies that balance growth with cash generation."
        ),
    }[metric]


def metric_supports_thesis(metric: str, cyber_value: float, broad_value: float) -> bool:
    if pd.isna(cyber_value) or pd.isna(broad_value):
        return False
    if metric == "annualized_volatility":
        return cyber_value < broad_value
    return cyber_value > broad_value


def chart_evaluator(df: pd.DataFrame, metric: str, start: date, end: date) -> dict[str, str | float | bool]:
    plot_df = df.dropna(subset=[metric]).copy()
    medians = plot_df.groupby("cohort")[metric].median()
    cyber_median = medians.get("cybersecurity")
    broad_median = medians.get("broad_tech")
    supports = metric_supports_thesis(metric, cyber_median, broad_median)
    direction = "lower" if metric == "annualized_volatility" else "higher"
    verdict = "Supports thesis" if supports else "Does not support thesis"
    comparison = (
        f"Cyber median: {cyber_median:.1%}. Broad tech median: {broad_median:.1%}."
        if pd.notna(cyber_median) and pd.notna(broad_median)
        else "There is not enough data to compare the two cohorts."
    )
    if supports:
        thesis_text = (
            f"This supports the thesis for this chart because cyber has the better {metric_label(metric).lower()} "
            f"by the selected rule: {direction} is better."
        )
    else:
        thesis_text = (
            f"This goes against the thesis for this chart because cyber does not have the better "
            f"{metric_label(metric).lower()} by the selected rule: {direction} is better."
        )
    return {
        "verdict": verdict,
        "supports": supports,
        "cyber_median": cyber_median,
        "broad_median": broad_median,
        "explanation": metric_explanation(metric),
        "comparison": comparison,
        "thesis_text": thesis_text,
        "period": f"{start} to {end}",
    }


def draw_cyber_vs_tech_chart(df: pd.DataFrame, metric: str, start: date, end: date) -> ImageResult:
    plot_df = df.dropna(subset=[metric]).copy()
    plot_df = plot_df.sort_values(metric)
    width, height = 1350, 820
    left, right, top, bottom = 115, 250, 105, 150
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom

    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    subtitle_font = load_font(16)
    axis_font = load_font(17, bold=True)
    tick_font = load_font(13)
    label_font = load_font(14, bold=True)
    note_font = load_font(14)

    values = plot_df[metric].astype(float)
    y_min = min(0, float(values.min()) - 0.08)
    y_max = max(0.1, float(values.max()) + 0.08)
    if metric in ("max_drawdown", "period_return"):
        y_max = max(0, y_max)

    draw.text((left, 28), "Cyber Vs Broad Tech: Resilience Test", fill="#111827", font=title_font)
    draw.text(
        (left, 66),
        f"{metric_label(metric)} from {start} to {end}. If cyber bars are worse than broad tech, the market-resilience thesis weakens.",
        fill="#374151",
        font=subtitle_font,
    )

    for tick_pct in range(int(y_min * 100) - 5, int(y_max * 100) + 1, 10):
        tick = tick_pct / 100
        y = scale(tick, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        draw.text((45, y - 8), f"{tick_pct}%", fill="#374151", font=tick_font)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#111827", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#111827", width=2)

    bar_gap = 18
    bar_width = max(32, min(70, (plot_right - plot_left - bar_gap * (len(plot_df) + 1)) / max(len(plot_df), 1)))
    x = plot_left + bar_gap
    for _, row in plot_df.iterrows():
        value = float(row[metric])
        y = scale(value, y_min, y_max, plot_bottom, plot_top)
        zero_y = scale(0, y_min, y_max, plot_bottom, plot_top)
        color = COHORT_COLORS.get(row["cohort"], "#6b7280")
        draw.rectangle((x, min(y, zero_y), x + bar_width, max(y, zero_y)), fill=color, outline="#111827")
        draw.text((x + 2, min(y, zero_y) - 22), f"{value:.0%}", fill="#111827", font=tick_font)
        draw.text((x + 2, plot_bottom + 12), row["ticker"], fill="#111827", font=label_font)
        x += bar_width + bar_gap

    medians = plot_df.groupby("cohort")[metric].median()
    legend_x = plot_right + 30
    draw.text((legend_x, plot_top), "Cohorts", fill="#111827", font=axis_font)
    y = plot_top + 36
    for cohort, color in COHORT_COLORS.items():
        draw.rectangle((legend_x, y, legend_x + 24, y + 18), fill=color, outline="#111827")
        label = cohort.replace("_", " ").title()
        draw.text((legend_x + 36, y - 1), label, fill="#111827", font=note_font)
        if cohort in medians:
            draw.text((legend_x + 36, y + 22), f"Median: {medians[cohort]:.1%}", fill="#374151", font=note_font)
        y += 70

    draw.text((plot_left + 385, height - 70), "Ticker", fill="#111827", font=axis_font)
    draw.text((20, 365), metric_label(metric), fill="#111827", font=axis_font)
    draw.text(
        (plot_left, height - 38),
        "Yahoo Finance data. This chart is designed to test whether cyber was actually more resilient than broad tech.",
        fill="#4b5563",
        font=note_font,
    )
    return ImageResult(image=image, csv=df, filename_base=f"cyber_vs_tech_{metric}")


def annual_revenue_series(ticker: str) -> pd.DataFrame:
    income = yf.Ticker(ticker).income_stmt
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
        .rename("revenue")
        .reset_index()
        .rename(columns={"index": "period_end"})
    )
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame = frame.dropna(subset=["period_end"]).sort_values("period_end")
    frame["fiscal_year"] = frame["period_end"].dt.year
    frame["ticker"] = ticker
    frame["revenue_growth"] = frame["revenue"].pct_change()
    return frame[["ticker", "period_end", "fiscal_year", "revenue", "revenue_growth"]]


@st.cache_data(show_spinner=False)
def sec_annual_revenue_series(ticker: str) -> pd.DataFrame:
    """Return a deduplicated annual revenue history from local SEC company facts."""
    path = EDGAR_COMPANYFACTS_ROOT / f"{ticker}_companyfacts.json"
    if not path.exists():
        return pd.DataFrame()
    companyfacts = json.loads(path.read_text(encoding="utf-8"))
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    rows = []
    for priority, concept in enumerate(REVENUE_CONCEPTS):
        for item in facts.get(concept, {}).get("units", {}).get("USD", []):
            if item.get("form") != "10-K" or item.get("fp") not in {"FY", None}:
                continue
            start = pd.to_datetime(item.get("start"), errors="coerce")
            period_end = pd.to_datetime(item.get("end"), errors="coerce")
            value = pd.to_numeric(item.get("val"), errors="coerce")
            if pd.isna(start) or pd.isna(period_end) or pd.isna(value):
                continue
            duration_days = (period_end - start).days
            if not 250 <= duration_days <= 450:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "period_end": period_end,
                    "revenue": float(value),
                    "concept": concept,
                    "concept_priority": priority,
                    "filed": pd.to_datetime(item.get("filed"), errors="coerce"),
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values(
        ["period_end", "concept_priority", "filed"],
        ascending=[True, True, False],
    )
    frame = frame.drop_duplicates("period_end", keep="first").sort_values("period_end")
    frame["fiscal_year"] = frame["period_end"].dt.year
    frame["revenue_growth"] = frame["revenue"].pct_change()
    return frame[
        ["ticker", "period_end", "fiscal_year", "revenue", "revenue_growth", "concept"]
    ]


@st.cache_data(show_spinner=False)
def revenue_growth_by_downturn(
    events: pd.DataFrame,
    cyber_tickers: tuple[str, ...],
    tech_tickers: tuple[str, ...],
) -> pd.DataFrame:
    histories = {}
    for ticker in (*cyber_tickers, *tech_tickers):
        history = sec_annual_revenue_series(ticker)
        if history.empty:
            history = annual_revenue_series(ticker)
            if not history.empty:
                history["concept"] = "Yahoo Finance fallback"
        histories[ticker] = history

    rows = []
    for _, event in events.iterrows():
        trough_date = pd.to_datetime(event["trough_date"])
        for cohort, tickers in (
            ("cybersecurity", cyber_tickers),
            ("broad_tech", tech_tickers),
        ):
            for ticker in tickers:
                history = histories[ticker]
                eligible = (
                    history[history["period_end"] <= trough_date].sort_values("period_end")
                    if not history.empty
                    else pd.DataFrame()
                )
                selected = eligible.iloc[-1] if not eligible.empty else None
                rows.append(
                    {
                        "event_id": int(event["event_id"]),
                        "trough_date": trough_date,
                        "ticker": ticker,
                        "cohort": cohort,
                        "fiscal_year": selected["fiscal_year"] if selected is not None else None,
                        "period_end": selected["period_end"] if selected is not None else None,
                        "revenue": selected["revenue"] if selected is not None else None,
                        "revenue_growth": (
                            selected["revenue_growth"] if selected is not None else None
                        ),
                        "source": selected["concept"] if selected is not None else None,
                    }
                )
    return pd.DataFrame(rows)


def render_revenue_growth_downturns(events: pd.DataFrame) -> None:
    st.markdown("### Revenue growth during QQQ downturns")
    st.caption("Compares the latest available annual growth at each downturn trough.")

    selector_cols = st.columns(2)
    cyber_tickers = tuple(
        selector_cols[0].multiselect(
            "Cybersecurity companies",
            parse_tickers(DEFAULT_CYBER_TICKERS),
            default=parse_tickers(DEFAULT_CYBER_TICKERS),
        )
    )
    tech_tickers = tuple(
        selector_cols[1].multiselect(
            "Broad-tech companies",
            parse_tickers(DEFAULT_TECH_TICKERS),
            default=parse_tickers(DEFAULT_TECH_TICKERS),
        )
    )
    if not cyber_tickers or not tech_tickers:
        st.warning("Choose at least one company in each group.")
        return

    with st.spinner("Matching SEC revenue history to downturns..."):
        revenue_data = revenue_growth_by_downturn(events, cyber_tickers, tech_tickers)

    available = revenue_data.dropna(subset=["revenue_growth"])
    if available.empty:
        st.warning("No historical revenue growth is available for these companies.")
        return

    summary = (
        available.groupby(["event_id", "trough_date", "cohort"])["revenue_growth"]
        .median()
        .unstack("cohort")
        .reset_index()
    )
    summary["cyber_minus_broad"] = (
        summary.get("cybersecurity") - summary.get("broad_tech")
    )
    comparable = summary.dropna(subset=["cybersecurity", "broad_tech"])
    cyber_beat_rate = (
        (comparable["cybersecurity"] > comparable["broad_tech"]).mean()
        if not comparable.empty
        else None
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Comparable downturns", len(comparable))
    metric_cols[1].metric("Cyber growth beat rate", percent_text(cyber_beat_rate))
    metric_cols[2].metric(
        "Median growth advantage",
        percent_text(comparable["cyber_minus_broad"].median())
        if not comparable.empty
        else "N/A",
    )

    event_tabs = st.tabs(
        [f"Downturn #{int(row['event_id'])}" for _, row in events.iterrows()]
    )
    for tab, (_, event) in zip(event_tabs, events.iterrows()):
        with tab:
            event_id = int(event["event_id"])
            event_data = revenue_data[revenue_data["event_id"].eq(event_id)].copy()
            chart_data = event_data.dropna(subset=["revenue_growth"])
            st.caption(f"QQQ trough: {event['trough_date']}")
            if chart_data.empty:
                st.warning("No revenue history was available at this downturn.")
                continue
            medians = chart_data.groupby("cohort")["revenue_growth"].median()
            cyber_median = medians.get("cybersecurity")
            tech_median = medians.get("broad_tech")
            if pd.notna(cyber_median) and pd.notna(tech_median):
                leader = "Cybersecurity" if cyber_median > tech_median else "Broad tech"
                render_major_finding(
                    f"{leader} had higher median revenue growth. "
                    f"Cyber: {cyber_median:.1%}; Broad tech: {tech_median:.1%}."
                )

            chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("ticker:N", title=None, sort=None),
                    y=alt.Y("revenue_growth:Q", title="Revenue growth", axis=alt.Axis(format="%")),
                    color=alt.Color(
                        "cohort:N",
                        title=None,
                        scale=alt.Scale(
                            domain=["cybersecurity", "broad_tech"],
                            range=[
                                COHORT_COLORS["cybersecurity"],
                                COHORT_COLORS["broad_tech"],
                            ],
                        ),
                        legend=alt.Legend(
                            labelExpr="datum.label == 'cybersecurity' ? 'Cybersecurity' : 'Broad tech'"
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("ticker:N", title="Company"),
                        alt.Tooltip("cohort:N", title="Group"),
                        alt.Tooltip("revenue_growth:Q", title="Revenue growth", format=".1%"),
                        alt.Tooltip("fiscal_year:Q", title="Fiscal year", format=".0f"),
                    ],
                )
                .properties(height=390)
            )
            st.altair_chart(chart, width="stretch")
            display = event_data.copy()
            display["revenue"] = display["revenue"].map(simple_money)
            display["revenue_growth"] = display["revenue_growth"].map(percent_text)
            display["period_end"] = pd.to_datetime(display["period_end"]).dt.date
            render_data_table(
                display,
                f"revenue_growth_event_{event_id}",
                width="stretch",
                hide_index=True,
            )

    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download Revenue Growth CSV",
        data=csv_bytes(revenue_data),
        file_name="revenue_growth_by_qqq_downturn.csv",
        mime="text/csv",
        width="stretch",
    )
    download_cols[1].download_button(
        "Download Group Summary CSV",
        data=csv_bytes(summary),
        file_name="revenue_growth_group_summary.csv",
        mime="text/csv",
        width="stretch",
    )


def revenue_growth_dataset_tab() -> None:
    apply_dashboard_style()
    st.subheader("Revenue Growth During QQQ Downturns")
    st.caption("Compares cybersecurity and broad-tech company growth.")
    if st.button("What is a downturn?", key="revenue_explain_downturn"):
        show_downturn_definition()
    run_dir = latest_qqq_corrections_dir()
    if run_dir is None:
        st.warning("No QQQ downturn dataset is available.")
        return
    events = pd.read_csv(run_dir / "qqq_correction_events.csv")
    render_revenue_growth_downturns(events)


def sign_test_p_value(wins: int, losses: int) -> float | None:
    total = wins + losses
    if total == 0:
        return None
    smaller = min(wins, losses)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / (2**total)
    return min(1.0, 2 * tail)


def score_combined_thesis(
    events: pd.DataFrame,
    etf_metrics: pd.DataFrame,
    revenue_data: pd.DataFrame,
) -> dict[str, object]:
    revenue_available = revenue_data.dropna(subset=["revenue_growth"]).copy()
    revenue_summary = (
        revenue_available.groupby(["event_id", "cohort"])["revenue_growth"]
        .median()
        .unstack("cohort")
        .dropna()
        .reset_index()
    )
    revenue_summary["difference"] = (
        revenue_summary["cybersecurity"] - revenue_summary["broad_tech"]
    )
    unique_growth_periods = revenue_summary.drop_duplicates(
        ["cybersecurity", "broad_tech"]
    )
    growth_wins = int((unique_growth_periods["difference"] > 0).sum())
    growth_losses = int((unique_growth_periods["difference"] < 0).sum())
    growth_p_value = sign_test_p_value(growth_wins, growth_losses)
    growth_beat_rate = (
        float((revenue_summary["difference"] > 0).mean())
        if not revenue_summary.empty
        else None
    )
    growth_advantage = (
        float(revenue_summary["difference"].median())
        if not revenue_summary.empty
        else None
    )
    operating_points = (
        (2 if growth_advantage is not None and growth_advantage > 0 else 0)
        + (2 if growth_beat_rate is not None and growth_beat_rate > 0.5 else 0)
        + (2 if growth_p_value is not None and growth_p_value <= 0.05 else 0)
    )

    available_metrics = etf_metrics[etf_metrics["available"].eq(True)].copy()
    cyber_metrics = available_metrics[available_metrics["ticker"].ne("QQQ")]
    qqq_metrics = available_metrics[available_metrics["ticker"].eq("QQQ")]
    event_market = (
        cyber_metrics.groupby("event_id")
        .agg(
            cyber_drawdown=("max_drawdown", "median"),
            cyber_return=("period_return", "median"),
            cyber_volatility=("annualized_volatility", "median"),
        )
        .join(
            qqq_metrics.set_index("event_id")[
                ["max_drawdown", "period_return", "annualized_volatility"]
            ].rename(
                columns={
                    "max_drawdown": "qqq_drawdown",
                    "period_return": "qqq_return",
                    "annualized_volatility": "qqq_volatility",
                }
            ),
            how="inner",
        )
        .dropna()
    )
    drawdown_advantage = (
        float((event_market["cyber_drawdown"] - event_market["qqq_drawdown"]).median())
        if not event_market.empty
        else None
    )
    market_beat_rate = (
        float((event_market["cyber_return"] > event_market["qqq_return"]).mean())
        if not event_market.empty
        else None
    )
    volatility_advantage = (
        float((event_market["qqq_volatility"] - event_market["cyber_volatility"]).median())
        if not event_market.empty
        else None
    )
    market_points = (
        (2 if drawdown_advantage is not None and drawdown_advantage > 0 else 0)
        + (1 if market_beat_rate is not None and market_beat_rate > 0.5 else 0)
        + (1 if volatility_advantage is not None and volatility_advantage > 0 else 0)
    )

    total_points = operating_points + market_points
    if total_points >= 7:
        verdict = "Supports thesis"
    elif total_points >= 4:
        verdict = "Mixed evidence"
    else:
        verdict = "Does not support thesis"

    return {
        "verdict": verdict,
        "total_points": total_points,
        "operating_points": operating_points,
        "market_points": market_points,
        "growth_advantage": growth_advantage,
        "growth_beat_rate": growth_beat_rate,
        "growth_p_value": growth_p_value,
        "unique_growth_periods": len(unique_growth_periods),
        "drawdown_advantage": drawdown_advantage,
        "market_beat_rate": market_beat_rate,
        "volatility_advantage": volatility_advantage,
        "market_events": len(event_market),
    }


def deterministic_verdict_explanation(result: dict[str, object]) -> str:
    verdict = result["verdict"]
    growth = percent_text(result["growth_advantage"])
    growth_rate = percent_text(result["growth_beat_rate"])
    market_rate = percent_text(result["market_beat_rate"])
    return (
        f"{verdict}. Cyber's median revenue-growth advantage was {growth}, and "
        f"cyber grew faster in {growth_rate} of measured downturns. Cyber ETFs "
        f"beat QQQ on return in {market_rate} of downturns. This means the "
        "operating and market evidence do not necessarily point in the same direction."
    )


def ai_verdict_explanation(result: dict[str, object]) -> str:
    from openai import OpenAI

    client = OpenAI()
    prompt = (
        "Explain this cybersecurity thesis verdict to a university audience in "
        "120 words or fewer. Use plain English. State the strongest supporting "
        "evidence, strongest contradictory evidence, and one limitation. The "
        "score and verdict are fixed: do not recalculate, override, or soften them.\n\n"
        f"Structured results:\n{json.dumps(result, default=str, indent=2)}"
    )
    response = client.responses.create(
        model="gpt-5.6",
        input=prompt,
    )
    return response.output_text


@st.cache_data(show_spinner=False, ttl=60 * 30)
def budget_variation_data(tickers: tuple[str, ...], budget_table: tuple[tuple[int, float], ...], revenue_lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for ticker in tickers:
        frame = annual_revenue_series(ticker)
        if not frame.empty:
            frames.append(frame)
    revenue = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    budget = pd.DataFrame(budget_table, columns=["budget_year", "security_budget_growth_pct"])
    budget["security_budget_growth"] = budget["security_budget_growth_pct"] / 100
    if revenue.empty:
        summary = budget.copy()
        summary["cyber_revenue_growth_median"] = None
        summary["cyber_revenue_growth_mean"] = None
        summary["company_count"] = 0
        return revenue, summary

    revenue["budget_year"] = revenue["fiscal_year"] - revenue_lag
    filtered = revenue[revenue["budget_year"].isin(budget["budget_year"])].copy()
    grouped = (
        filtered.groupby("budget_year")["revenue_growth"]
        .agg(cyber_revenue_growth_median="median", cyber_revenue_growth_mean="mean", company_count="count")
        .reset_index()
    )
    summary = budget.merge(grouped, on="budget_year", how="left").sort_values("budget_year")
    return revenue, summary


def draw_budget_variation_chart(revenue: pd.DataFrame, summary: pd.DataFrame, revenue_lag: int) -> ImageResult:
    width, height = 1300, 820
    left, right, top, bottom = 110, 160, 100, 135
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

    years = summary["budget_year"].tolist()
    x_positions = {
        year: scale(idx, 0, max(len(years) - 1, 1), plot_left + 95, plot_right - 95)
        for idx, year in enumerate(years)
    }
    growth_values = [
        summary["security_budget_growth"].dropna(),
        summary["cyber_revenue_growth_median"].dropna(),
    ]
    if not revenue.empty and "revenue_growth" in revenue:
        growth_values.append(revenue["revenue_growth"].dropna())
    all_growth = pd.concat(growth_values, ignore_index=True)
    y_min = min(0, float(all_growth.min()) - 0.05) if not all_growth.empty else 0
    y_max = max(0.45, float(all_growth.max()) + 0.08) if not all_growth.empty else 0.45

    draw.text((left, 28), "Cyber Growth Vs Security Budget Variation", fill="#111827", font=title_font)
    draw.text(
        (left, 66),
        f"Budget year is compared with fiscal-year revenue growth {revenue_lag} year(s) later.",
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
        x = x_positions[row["budget_year"]]
        budget_y = scale(row["security_budget_growth"], y_min, y_max, plot_bottom, plot_top)
        budget_left = x - bar_width - bar_gap / 2
        budget_right = x - bar_gap / 2
        draw.rectangle((budget_left, budget_y, budget_right, plot_bottom), fill="#d1d5db", outline="#6b7280")
        draw.text((budget_left + 8, budget_y - 24), f"{row['security_budget_growth']:.0%}", fill="#374151", font=tick_font)
        if pd.notna(row["cyber_revenue_growth_median"]):
            median_y = scale(row["cyber_revenue_growth_median"], y_min, y_max, plot_bottom, plot_top)
            median_left = x + bar_gap / 2
            median_right = x + bar_width + bar_gap / 2
            draw.rectangle((median_left, median_y, median_right, plot_bottom), fill="#2563eb", outline="#111827")
            draw.text((median_left + 8, median_y - 24), f"{row['cyber_revenue_growth_median']:.0%}", fill="#111827", font=tick_font)
        draw.text((x - 20, plot_bottom + 14), str(int(row["budget_year"])), fill="#111827", font=label_font)

    if not revenue.empty:
        plotted_growth = revenue.dropna(subset=["revenue_growth"]).copy()
        plotted_growth = plotted_growth[plotted_growth["budget_year"].isin(years)]
        for _, group in plotted_growth.groupby("ticker"):
            group = group.sort_values("budget_year")
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
    draw.rectangle((legend_x - 16, legend_y - 14, legend_x + 310, legend_y + 98), fill="#ffffff", outline="#e5e7eb")
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
        "Security budget values are user-adjustable. Company revenue data from Yahoo Finance.",
        fill="#4b5563",
        font=note_font,
    )
    csv = summary.copy()
    return ImageResult(image=image, csv=csv, filename_base="cyber_growth_vs_budget_variation")


def render_downloads(result: ImageResult) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download PNG",
            data=png_bytes(result.image),
            file_name=f"{result.filename_base}.png",
            mime="image/png",
            width="stretch",
        )
    with col2:
        st.download_button(
            "Download CSV",
            data=csv_bytes(result.csv),
            file_name=f"{result.filename_base}.csv",
            mime="text/csv",
            width="stretch",
        )


def exact_permutation_p_value(df: pd.DataFrame, metric: str) -> tuple[float | None, float | None]:
    data = df[["cohort", metric]].dropna().copy()
    if data.empty or data["cohort"].nunique() != 2:
        return None, None
    values = data[metric].to_numpy(dtype=float)
    cyber_mask = data["cohort"].eq("cybersecurity").to_numpy()
    cyber_count = int(cyber_mask.sum())
    if cyber_count == 0 or cyber_count == len(values):
        return None, None

    observed = values[cyber_mask].mean() - values[~cyber_mask].mean()
    if len(values) > 16:
        return observed, None

    diffs = []
    for cyber_idx in itertools.combinations(range(len(values)), cyber_count):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(cyber_idx)] = True
        diffs.append(values[mask].mean() - values[~mask].mean())
    diffs = np.array(diffs)
    p_value = float((np.abs(diffs) >= abs(observed) - 1e-12).mean())
    return float(observed), p_value


def ols_summary(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> tuple[float | None, dict[str, float]]:
    data = df[[y_col] + x_cols].dropna().copy()
    if len(data) <= len(x_cols) + 1:
        return None, {}
    y = data[y_col].to_numpy(dtype=float)
    x = data[x_cols].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    coefs = np.linalg.lstsq(x, y, rcond=None)[0]
    prediction = x @ coefs
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else None
    names = ["intercept", *x_cols]
    return r2, dict(zip(names, [float(coef) for coef in coefs]))


def statistical_analysis_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_df = df.copy()
    analysis_df["cyber_dummy"] = analysis_df["cohort"].eq("cybersecurity").astype(int)

    metrics = [
        "period_return",
        "max_drawdown",
        "annualized_volatility",
        "revenue_growth",
        "free_cash_flow_margin",
        "rule_of_40",
    ]
    cohort_rows = []
    for metric in metrics:
        grouped = analysis_df.groupby("cohort")[metric]
        cyber_mean = grouped.mean().get("cybersecurity")
        broad_mean = grouped.mean().get("broad_tech")
        cyber_median = grouped.median().get("cybersecurity")
        broad_median = grouped.median().get("broad_tech")
        diff, p_value = exact_permutation_p_value(analysis_df, metric)
        cohort_rows.append(
            {
                "metric": metric_label(metric),
                "cyber_mean": cyber_mean,
                "broad_tech_mean": broad_mean,
                "cyber_minus_broad_mean": diff,
                "permutation_p_value": p_value,
                "cyber_median": cyber_median,
                "broad_tech_median": broad_median,
                "cyber_minus_broad_median": (
                    cyber_median - broad_median
                    if pd.notna(cyber_median) and pd.notna(broad_median)
                    else None
                ),
            }
        )
    cohort_table = pd.DataFrame(cohort_rows)

    corr_cols = [
        "max_drawdown",
        "period_return",
        "annualized_volatility",
        "revenue_growth",
        "free_cash_flow_margin",
        "rule_of_40",
        "cyber_dummy",
    ]
    corr_matrix = analysis_df[corr_cols].corr(numeric_only=True)
    corr_rows = []
    for target in ["max_drawdown", "period_return", "annualized_volatility"]:
        for factor in corr_cols:
            if factor != target and target in corr_matrix and factor in corr_matrix:
                corr_rows.append(
                    {
                        "target": metric_label(target),
                        "factor": metric_label(factor) if factor != "cyber_dummy" else "Cyber label",
                        "correlation": corr_matrix.loc[factor, target],
                    }
                )
    corr_table = pd.DataFrame(corr_rows).sort_values("correlation", key=lambda s: s.abs(), ascending=False)

    regression_rows = []
    model_specs = [
        ("max_drawdown", ["cyber_dummy"]),
        ("max_drawdown", ["cyber_dummy", "revenue_growth", "free_cash_flow_margin"]),
        ("period_return", ["cyber_dummy"]),
        ("period_return", ["cyber_dummy", "revenue_growth", "free_cash_flow_margin"]),
        ("annualized_volatility", ["cyber_dummy"]),
        ("annualized_volatility", ["cyber_dummy", "revenue_growth", "free_cash_flow_margin"]),
    ]
    for target, features in model_specs:
        r2, coefs = ols_summary(analysis_df, target, features)
        regression_rows.append(
            {
                "target": metric_label(target),
                "model": " + ".join("Cyber label" if feature == "cyber_dummy" else metric_label(feature) for feature in features),
                "r_squared": r2,
                "cyber_label_coefficient": coefs.get("cyber_dummy"),
            }
        )
    regression_table = pd.DataFrame(regression_rows)

    drawdown_median = analysis_df["max_drawdown"].median()
    rule40_median = analysis_df["rule_of_40"].median()
    regime = analysis_df[
        ["ticker", "cohort", "max_drawdown", "period_return", "annualized_volatility", "rule_of_40"]
    ].copy()
    regime["resilient_stock"] = regime["max_drawdown"] >= drawdown_median
    regime["high_operating_quality"] = regime["rule_of_40"] >= rule40_median
    regime = regime.sort_values(["resilient_stock", "high_operating_quality"], ascending=[False, False])

    export = analysis_df.copy()
    return cohort_table, corr_table, regression_table, regime, export


def format_pct_table(df: pd.DataFrame, pct_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{value:.1%}")
    if "permutation_p_value" in out.columns:
        out["permutation_p_value"] = out["permutation_p_value"].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    if "correlation" in out.columns:
        out["correlation"] = out["correlation"].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    if "r_squared" in out.columns:
        out["r_squared"] = out["r_squared"].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    return out


def cyber_vs_tech_tab() -> None:
    apply_dashboard_style()
    st.subheader("Cyber Vs Broad Tech Resilience")
    st.caption("Did cyber stocks fall less than broad tech?")
    controls, output = st.columns([0.34, 0.66], gap="large")
    with controls:
        cyber_text = st.text_area("Cyber tickers", DEFAULT_CYBER_TICKERS, height=80)
        tech_text = st.text_area("Broad tech tickers", DEFAULT_TECH_TICKERS, height=80)
        start = st.date_input("Downturn start", value=date(2022, 1, 1), key="cvt_start")
        end = st.date_input("Downturn end", value=date(2022, 12, 31), key="cvt_end")
        run = st.button("Update chart", type="primary", width="stretch")

    metric = "max_drawdown"
    cyber_tickers = tuple(parse_tickers(cyber_text))
    tech_tickers = tuple(parse_tickers(tech_text))
    input_key = (cyber_tickers, tech_tickers, start, end)
    if (
        run
        or "cyber_vs_tech_result" not in st.session_state
        or st.session_state.get("cyber_vs_tech_input_key") != input_key
    ):
        with st.spinner("Pulling Yahoo Finance data..."):
            df = cyber_vs_tech_data(cyber_tickers, tech_tickers, start, end)
            st.session_state.cyber_vs_tech_result = draw_cyber_vs_tech_chart(df, metric, start, end)
            st.session_state.cyber_vs_tech_evaluator = chart_evaluator(df, metric, start, end)
            st.session_state.cyber_vs_tech_input_key = input_key

    with output:
        result = st.session_state.cyber_vs_tech_result
        st.image(png_bytes(result.image), width="stretch")
        evaluator = st.session_state.cyber_vs_tech_evaluator
        st.markdown("### Finding")
        render_major_finding(f"{evaluator['verdict']}. {evaluator['thesis_text']}")
        st.markdown("**Measure: Maximum drawdown**")
        st.write(evaluator["explanation"])
        st.write(evaluator["comparison"])
        st.caption(f"Period: {evaluator['period']}")
        render_downloads(result)
        render_data_table(result.csv, "cyber_vs_tech_chart", width="stretch")


def statistical_analysis_tab() -> None:
    apply_dashboard_style()
    st.subheader("Does Cyber Beat Broad Tech?")
    st.caption("Tests market and company results together.")
    controls, output = st.columns([0.32, 0.68], gap="large")
    with controls:
        cyber_text = st.text_area("Cyber tickers", DEFAULT_CYBER_TICKERS, height=80, key="stats_cyber")
        tech_text = st.text_area("Broad tech tickers", DEFAULT_TECH_TICKERS, height=80, key="stats_tech")
        start = st.date_input("Downturn start", value=date(2022, 1, 1), key="stats_start")
        end = st.date_input("Downturn end", value=date(2022, 12, 31), key="stats_end")
        run = st.button("Run analysis", type="primary", width="stretch")

    if run or "statistical_analysis_result" not in st.session_state:
        cyber_tickers = tuple(parse_tickers(cyber_text))
        tech_tickers = tuple(parse_tickers(tech_text))
        with st.spinner("Pulling Yahoo Finance data and running statistics..."):
            df = cyber_vs_tech_data(cyber_tickers, tech_tickers, start, end)
            st.session_state.statistical_analysis_result = statistical_analysis_tables(df)

    cohort_table, corr_table, regression_table, regime, export = st.session_state.statistical_analysis_result
    verdict = score_thesis_verdict(cohort_table)

    with output:
        cyber_row = cohort_table.set_index("metric")
        mean_drawdown_diff = cyber_row.loc["Max Drawdown", "cyber_minus_broad_mean"]
        mean_return_diff = cyber_row.loc["Period Return", "cyber_minus_broad_mean"]
        vol_diff = cyber_row.loc["Annualized Volatility", "cyber_minus_broad_mean"]
        fcf_diff = cyber_row.loc["FCF Margin", "cyber_minus_broad_mean"]

        st.markdown("### Key results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cyber vs broad drawdown", f"{mean_drawdown_diff:.1%}")
        c2.metric("Cyber vs broad return", f"{mean_return_diff:.1%}")
        c3.metric("Cyber vs broad volatility", f"{vol_diff:.1%}")
        c4.metric("Cyber vs broad FCF margin", f"{fcf_diff:.1%}")

        st.caption("Negative return gaps and higher volatility mean cyber was less resilient.")

        st.markdown("### Group results")
        st.caption("A p-value below 0.05 is strong evidence of a real difference.")
        st.dataframe(
            format_pct_table(
                cohort_table,
                [
                    "cyber_mean",
                    "broad_tech_mean",
                    "cyber_minus_broad_mean",
                    "cyber_median",
                    "broad_tech_median",
                    "cyber_minus_broad_median",
                ],
            ),
            width="stretch",
        )

        st.markdown("### Metric relationships")
        st.caption("Correlation shows which measures move together.")
        st.dataframe(
            format_pct_table(corr_table.head(12), []),
            width="stretch",
        )

        st.markdown("### Does the cyber label matter?")
        st.caption("Controls for revenue growth and cash margins.")
        st.dataframe(
            format_pct_table(regression_table, ["cyber_label_coefficient"]),
            width="stretch",
        )

        st.markdown("### Company groups")
        st.caption("Classifies resilience and operating quality.")
        st.dataframe(
            format_pct_table(
                regime,
                ["max_drawdown", "period_return", "annualized_volatility", "rule_of_40"],
            ),
            width="stretch",
        )

        st.markdown("### Verdict score")
        st.caption("Market resilience carries the most weight.")
        score_cols = st.columns(4)
        score_cols[0].metric("Market score", f"{verdict.market_score}/6")
        score_cols[1].metric("Statistical score", f"{verdict.statistical_score}/3")
        score_cols[2].metric("Operating score", f"{verdict.operating_score}/3")
        score_cols[3].metric("Total score", f"{verdict.total_score}/{verdict.max_score}")
        st.caption("A market score of 2 or less cannot support the thesis.")
        st.dataframe(
            format_pct_table(
                verdict.details,
                ["cyber_median", "broad_tech_median", "cyber_minus_broad"],
            ),
            width="stretch",
        )

        export_parts = []
        for name, frame in (
            ("cohort_tests", cohort_table),
            ("correlations", corr_table),
            ("regressions", regression_table),
            ("regime_split", regime),
            ("verdict_scoring", verdict.details),
            ("raw_data", export),
        ):
            temp = frame.copy()
            temp.insert(0, "section", name)
            export_parts.append(temp)
        export_csv = pd.concat(export_parts, ignore_index=True, sort=False)
        st.download_button(
            "Download Statistical Analysis CSV",
            data=csv_bytes(export_csv),
            file_name="cyber_vs_tech_statistical_analysis.csv",
            mime="text/csv",
            width="stretch",
        )

        st.markdown("### Final Verdict")
        render_major_finding(f"{verdict.verdict}. {verdict.plain_english}")


def budget_variation_tab() -> None:
    apply_dashboard_style()
    st.subheader("Cyber Growth Vs Security Budget Variation: Supports Operating-Durability Claim")
    controls, output = st.columns([0.34, 0.66], gap="large")
    with controls:
        ticker_text = st.text_area("Cyber revenue tickers", DEFAULT_CYBER_TICKERS, height=90)
        budget_df = st.data_editor(
            DEFAULT_BUDGET_GROWTH,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "budget_year": st.column_config.NumberColumn("Budget year", step=1, format="%d"),
                "security_budget_growth_pct": st.column_config.NumberColumn("Security budget growth %", step=0.5, format="%.1f"),
            },
        )
        revenue_lag = st.number_input("Revenue fiscal-year lag", min_value=0, max_value=5, value=1, step=1)
        run = st.button("Update Budget Plot", type="primary", width="stretch")

    if run or "budget_variation_result" not in st.session_state:
        tickers = tuple(parse_tickers(ticker_text))
        clean_budget = budget_df.dropna(subset=["budget_year", "security_budget_growth_pct"]).copy()
        clean_budget["budget_year"] = clean_budget["budget_year"].astype(int)
        budget_table = tuple(
            (int(row["budget_year"]), float(row["security_budget_growth_pct"]))
            for _, row in clean_budget.iterrows()
        )
        with st.spinner("Pulling Yahoo Finance revenue data..."):
            revenue, summary = budget_variation_data(tickers, budget_table, int(revenue_lag))
            result = draw_budget_variation_chart(revenue, summary, int(revenue_lag))
            result.csv = summary
            st.session_state.budget_variation_result = result

    with output:
        result = st.session_state.budget_variation_result
        st.image(png_bytes(result.image), width="stretch")
        render_downloads(result)
        render_data_table(result.csv, "budget_chart", width="stretch")


@st.cache_data(show_spinner=False, ttl=60 * 30)
def etf_downturn_data(
    cyber_tickers: tuple[str, ...],
    benchmark: str,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    tickers = list(cyber_tickers) + [benchmark]
    prices, warnings = fetch_etf_prices(tickers, str(start), str(end))
    if prices.empty:
        return prices, pd.DataFrame(), pd.DataFrame(), warnings
    metrics = calculate_etf_metrics(prices, benchmark_ticker=benchmark)
    indexed = normalize_etf_prices(prices)
    return prices, metrics, indexed, warnings


def cyber_etf_vs_qqq_tab() -> None:
    apply_dashboard_style()
    st.subheader("Cyber ETFs Vs QQQ During a Downturn")
    st.caption(
        "Tests whether cybersecurity ETFs preserved capital better than broad "
        "technology during the selected downturn."
    )
    controls, output = st.columns([0.32, 0.68], gap="large")
    with controls:
        cyber_text = st.text_area("Cyber ETF tickers", "CIBR HACK", height=80)
        benchmark = st.text_input("Broad-tech benchmark", "QQQ").strip().upper()
        start = st.date_input("Downturn start", value=date(2022, 1, 1), key="etf_start")
        end = st.date_input("Downturn end", value=date(2022, 12, 31), key="etf_end")
        run = st.button("Update ETF Downturn Plot", type="primary", width="stretch")

    cyber_tickers = tuple(
        ticker for ticker in parse_tickers(cyber_text) if ticker != benchmark
    )
    input_key = (cyber_tickers, benchmark, start, end)
    if (
        run
        or "etf_downturn_result" not in st.session_state
        or st.session_state.get("etf_downturn_input_key") != input_key
    ):
        with st.spinner("Pulling ETF prices from Yahoo Finance..."):
            prices, metrics, indexed, warnings = etf_downturn_data(
                cyber_tickers,
                benchmark,
                start,
                end,
            )
            if indexed.empty or metrics.empty:
                st.session_state.etf_downturn_result = None
            else:
                image = draw_etf_plot(
                    indexed,
                    metrics,
                    None,
                    str(start),
                    str(end),
                )
                st.session_state.etf_downturn_result = ImageResult(
                    image=image,
                    csv=metrics,
                    filename_base="cyber_etf_vs_qqq_downturn",
                )
            st.session_state.etf_downturn_warnings = warnings
            st.session_state.etf_downturn_input_key = input_key

    with output:
        result = st.session_state.etf_downturn_result
        if result is None:
            st.error("No usable ETF price data was returned for this selection.")
            return

        st.image(png_bytes(result.image), width="stretch")
        metrics = result.csv
        cyber_metrics = metrics[metrics["ticker"].isin(cyber_tickers)]
        benchmark_metrics = metrics[metrics["ticker"].eq(benchmark)]
        enough_data = not cyber_metrics.empty and not benchmark_metrics.empty
        supports = (
            enough_data
            and cyber_metrics["max_drawdown"].median()
            > benchmark_metrics["max_drawdown"].iloc[0]
        )
        if supports:
            st.success(
                "**Supports the thesis.** The median cyber ETF maximum drawdown "
                f"was smaller than {benchmark}, indicating better capital preservation "
                "during this downturn."
            )
        else:
            st.error(
                "**Does not support the thesis.** The cyber ETF group did not have "
                f"a smaller median maximum drawdown than {benchmark} in this window."
            )
        st.write(
            "Maximum drawdown measures the worst peak-to-trough loss. A value "
            "closer to zero indicates stronger downside resilience."
        )
        for warning in st.session_state.get("etf_downturn_warnings", []):
            st.warning(warning)
        render_downloads(result)
        render_data_table(metrics, "single_downturn_metrics", width="stretch")


def latest_summary_metrics_file() -> Path | None:
    candidates = list(YFINANCE_RESULTS_ROOT.glob("*/summary_metrics.csv"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def latest_qqq_corrections_dir() -> Path | None:
    candidates = [
        path
        for path in QQQ_CORRECTIONS_ROOT.glob("*")
        if path.is_dir() and (path / "qqq_correction_events.csv").exists()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def percent_text(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def read_verdict_text(run_dir: Path) -> str | None:
    verdict_files = sorted(run_dir.glob("verdict_*.txt"))
    if not verdict_files:
        return None
    return verdict_files[0].read_text(encoding="utf-8").strip()


def simple_money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def summary_metrics_tab() -> None:
    apply_dashboard_style()
    st.subheader("Summary Metrics")
    st.caption("Cybersecurity compared with broad technology.")
    render_cohort_key()
    summary_file = latest_summary_metrics_file()
    if summary_file is None:
        st.warning(
            "No pipeline summary is available. Run "
            "`python3 scripts/cyber_tech_yfinance_fred_pipeline.py` first."
        )
        return

    data = pd.read_csv(summary_file)
    if VALIDATION_FILE.exists():
        validation = pd.read_csv(VALIDATION_FILE)
        selected_fcf = validation[
            validation["metric"].eq("free_cash_flow_margin")
        ][["ticker", "selected_value", "selected_source"]].rename(
            columns={
                "selected_value": "validated_fcf_margin",
                "selected_source": "fcf_source",
            }
        )
        data = data.merge(selected_fcf, on="ticker", how="left")
        data["free_cash_flow_margin"] = data["validated_fcf_margin"].combine_first(
            data["free_cash_flow_margin"]
        )
        data["fcf_source"] = data["fcf_source"].fillna("yFinance")
    else:
        data["fcf_source"] = "yFinance"
    data["rule_of_40"] = data["revenue_growth"] + data["free_cash_flow_margin"]

    cohort = data.groupby("cohort")[
        [
            "revenue_growth",
            "free_cash_flow_margin",
            "rule_of_40",
            "price_max_drawdown_downturn",
            "price_return_downturn",
        ]
    ].median()
    cyber = cohort.loc["cybersecurity"] if "cybersecurity" in cohort.index else pd.Series()
    tech = cohort.loc["broad_tech"] if "broad_tech" in cohort.index else pd.Series()

    card_specs = [
        ("Revenue growth", "revenue_growth", "How quickly sales grew."),
        ("FCF margin", "free_cash_flow_margin", "Cash kept from each sales dollar."),
        ("Rule of 40", "rule_of_40", "Growth plus free-cash-flow margin."),
        ("Maximum drawdown", "price_max_drawdown_downturn", "Worst stock decline."),
        ("Downturn return", "price_return_downturn", "Stock return in the downturn."),
    ]

    view = st.radio(
        "Choose a view",
        ["Simple overview", "Advanced details"],
        horizontal=True,
        help="Simple explains the result. Advanced shows company data.",
    )

    st.markdown("### Key takeaways")
    takeaways = []
    for label, column, _ in card_specs:
        cyber_value, tech_value = cyber.get(column), tech.get(column)
        if pd.isna(cyber_value) or pd.isna(tech_value):
            continue
        better = cyber_value > tech_value
        if column == "price_max_drawdown_downturn":
            better = cyber_value > tech_value  # A less-negative drawdown is better.
        leader = "Cybersecurity" if better else "Broad tech"
        takeaways.append(
            f"<div class='takeaway'><strong>{leader} leads on {label.lower()}.</strong> "
            f"Cyber: {cyber_value:.1%} · Broad tech: {tech_value:.1%}</div>"
        )
    for takeaway in takeaways[:3]:
        st.markdown(takeaway, unsafe_allow_html=True)

    if view == "Simple overview":
        st.markdown("### Side-by-side comparison")
        st.caption("Group medians reduce the effect of unusual companies.")
        for label, column, help_text in card_specs:
            cyber_value, tech_value = cyber.get(column), tech.get(column)
            label_col, cyber_col, tech_col = st.columns([1.5, 1, 1])
            label_col.markdown(f"**{label}**")
            label_col.caption(help_text)
            cyber_col.metric(
                "Cybersecurity",
                "N/A" if pd.isna(cyber_value) else f"{cyber_value:.1%}",
            )
            tech_col.metric(
                "Broad tech",
                "N/A" if pd.isna(tech_value) else f"{tech_value:.1%}",
            )

        with st.expander("What do these metrics mean?"):
            st.markdown(
                "- **Revenue growth:** how quickly company sales are expanding.\n"
                "- **FCF margin:** how much free cash a company keeps from each sales dollar.\n"
                "- **Rule of 40:** revenue growth plus free-cash-flow margin.\n"
                "- **Maximum drawdown:** the worst fall from a previous peak; closer to zero is better.\n"
                "- **Downturn return:** the total stock return over the selected difficult period."
            )
        return

    st.markdown("### Advanced company details")
    st.caption("Company results and data sources.")
    display = data[
        [
            "ticker",
            "cohort",
            "latest_revenue",
            "revenue_growth",
            "free_cash_flow_margin",
            "rule_of_40",
            "price_max_drawdown_downturn",
            "price_return_downturn",
            "fcf_source",
        ]
    ].copy()
    display.columns = [
        "Ticker",
        "Group",
        "Revenue",
        "Revenue growth",
        "FCF margin",
        "Rule of 40",
        "Maximum drawdown",
        "Downturn return",
        "FCF source",
    ]
    display["Group"] = display["Group"].replace(
        {"cybersecurity": "Cybersecurity", "broad_tech": "Broad tech"}
    )
    display["Revenue"] = display["Revenue"].map(simple_money)
    for column in [
        "Revenue growth",
        "FCF margin",
        "Rule of 40",
        "Maximum drawdown",
        "Downturn return",
    ]:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.1%}"
        )
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button(
        "Download Summary Metrics CSV",
        data=csv_bytes(data),
        file_name="summary_metrics.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption(f"Pipeline source: {summary_file}")


def qqq_corrections_dataset_tab() -> None:
    apply_dashboard_style()
    st.subheader("QQQ Corrections ETF Dataset")
    st.caption("Compares cyber ETFs with QQQ across many market declines.")
    if st.button("What is a downturn?", key="explain_downturn"):
        show_downturn_definition()
    run_dir = latest_qqq_corrections_dir()
    if run_dir is None:
        st.warning(
            "No QQQ correction dataset is available. Run "
            "`.venv/bin/python scripts/evaluate_qqq_corrections_vs_cyber_etfs.py` first."
        )
        return

    events = pd.read_csv(run_dir / "qqq_correction_events.csv")
    metrics = pd.read_csv(run_dir / "etf_metrics_by_correction.csv")
    summary = pd.read_csv(run_dir / "summary_by_ticker.csv")
    prices = pd.read_csv(run_dir / "prices.csv")
    prices["date"] = pd.to_datetime(prices["date"])
    verdict = read_verdict_text(run_dir)

    event_count = int(len(events))
    recovered_count = int(events["is_recovered"].sum()) if "is_recovered" in events else 0
    median_qqq_drop = events["qqq_peak_to_trough_return"].median()
    cyber_metrics = metrics[(metrics["available"] == True) & (metrics["ticker"] != "QQQ")]  # noqa: E712
    qqq_metrics = metrics[(metrics["available"] == True) & (metrics["ticker"] == "QQQ")]  # noqa: E712
    if not cyber_metrics.empty and not qqq_metrics.empty:
        cyber_event = cyber_metrics.groupby("event_id")["period_return"].median()
        qqq_event = qqq_metrics.set_index("event_id")["period_return"]
        comparison = pd.concat(
            [cyber_event.rename("median_cyber_return"), qqq_event.rename("qqq_return")],
            axis=1,
        ).dropna()
        beat_rate = (comparison["median_cyber_return"] > comparison["qqq_return"]).mean()
    else:
        beat_rate = None

    cards = st.columns(4)
    cards[0].metric("QQQ Downturns", event_count)
    cards[1].metric("Recovered events", recovered_count)
    cards[2].metric("Median QQQ drop", percent_text(median_qqq_drop))
    cards[3].metric("Cyber beat rate", percent_text(beat_rate))

    if verdict:
        render_major_finding(verdict.splitlines()[0])
        detail = "\n".join(verdict.splitlines()[1:]).strip()
        if detail:
            st.caption(detail)

    st.markdown("### Downturns")
    st.caption("Select a downturn to compare each ETF with QQQ.")
    event_tabs = st.tabs([f"Downturn #{int(row['event_id'])}" for _, row in events.iterrows()])
    for tab, (_, event) in zip(event_tabs, events.iterrows()):
        with tab:
            event_id = int(event["event_id"])
            event_metrics = metrics[
                (metrics["event_id"] == event_id) & (metrics["available"].eq(True))
            ].copy()
            if event_metrics.empty:
                st.warning("No ETF data is available for this downturn.")
                continue
            correction_start = event.get("correction_start_date", event["peak_date"])
            correction_end = event.get("correction_end_date", event["trough_date"])

            st.write(
                f"**Peak:** {event['peak_date']}  |  "
                f"**Correction start:** {correction_start}  |  "
                f"**Trough:** {event['trough_date']}  |  "
                f"**Correction end:** {correction_end}  |  "
                f"**QQQ peak-to-trough drop:** {percent_text(event['qqq_peak_to_trough_return'])}"
            )
            st.caption("Peak dots mark the high before each correction.")

            comparison = event_metrics[["ticker", "period_return"]].dropna().copy()
            comparison = comparison.sort_values("period_return", ascending=False)
            if comparison.empty:
                st.warning("Drop percentage is not available for this downturn.")
                continue

            peak_table = etf_peak_before_correction(
                prices,
                event_metrics,
                str(correction_start),
                str(correction_end),
            )
            st.markdown("#### ETF peaks")
            display_peak_table = peak_table.copy()
            display_peak_table["peak_close"] = display_peak_table["peak_close"].map(
                lambda value: "" if pd.isna(value) else f"${value:.2f}"
            )
            display_peak_table["correction_start_close"] = display_peak_table[
                "correction_start_close"
            ].map(lambda value: "" if pd.isna(value) else f"${value:.2f}")
            display_peak_table["drop_from_peak_to_start"] = display_peak_table[
                "drop_from_peak_to_start"
            ].map(percent_text)
            render_data_table(
                display_peak_table,
                f"correction_peaks_{int(event['event_id'])}",
                width="stretch",
                hide_index=True,
            )

            drawdown_event_prices = correction_start_drawdown_data(
                prices,
                event_metrics,
                str(correction_start),
                str(correction_end),
                peak_table,
            )
            if not drawdown_event_prices.empty:
                event_chart = draw_qqq_correction_drawdown_plot(
                    drawdown_event_prices,
                    event_metrics,
                    str(correction_start),
                    str(correction_end),
                )
                st.image(png_bytes(event_chart), width="stretch")
            else:
                st.warning("No price path is available to draw this downturn chart.")

            winner = comparison.iloc[0]
            cyber_values = event_metrics[event_metrics["ticker"] != "QQQ"]["period_return"].dropna()
            qqq_values = event_metrics[event_metrics["ticker"] == "QQQ"]["period_return"].dropna()
            if not cyber_values.empty and not qqq_values.empty:
                cyber_score = cyber_values.median()
                qqq_score = qqq_values.iloc[0]
                cyber_better = cyber_score > qqq_score
                if cyber_better:
                    render_major_finding(
                        "Cyber ETFs were more resilient. "
                        f"Median cyber: {percent_text(cyber_score)}; "
                        f"QQQ: {percent_text(qqq_score)}."
                    )
                else:
                    render_major_finding(
                        "QQQ was more resilient. "
                        f"Median cyber: {percent_text(cyber_score)}; "
                        f"QQQ: {percent_text(qqq_score)}."
                    )
            st.info(
                "Best individual ticker by drop percentage: "
                f"{winner['ticker']} at {percent_text(winner['period_return'])}."
            )

            display_event_metrics = event_metrics.copy()
            for column in [
                "period_return",
                "max_drawdown",
                "annualized_volatility",
                "relative_return_vs_qqq",
            ]:
                if column in display_event_metrics:
                    display_event_metrics[column] = display_event_metrics[column].map(percent_text)
            render_data_table(
                display_event_metrics,
                f"correction_metrics_{int(event['event_id'])}",
                width="stretch",
                hide_index=True,
            )

    st.markdown("### ETF summary")
    display_summary = summary.copy()
    for column in [
        "median_period_return",
        "median_max_drawdown",
        "median_annualized_volatility",
        "median_relative_return_vs_qqq",
    ]:
        if column in display_summary:
            display_summary[column] = display_summary[column].map(percent_text)
    render_data_table(display_summary, "corrections_ticker_summary", width="stretch", hide_index=True)

    st.markdown("### QQQ corrections")
    display_events = events.copy()
    if "recovery_date" in display_events.columns:
        display_events = display_events.drop(columns=["recovery_date"])
    for column in ["qqq_drawdown_to_trigger", "qqq_peak_to_trough_return"]:
        if column in display_events:
            display_events[column] = display_events[column].map(percent_text)
    render_data_table(display_events, "corrections_events", width="stretch", hide_index=True)

    st.markdown("### ETF results by correction")
    display_metrics = metrics.copy()
    for column in [
        "period_return",
        "max_drawdown",
        "annualized_volatility",
        "relative_return_vs_qqq",
    ]:
        if column in display_metrics:
            display_metrics[column] = display_metrics[column].map(percent_text)
    render_data_table(display_metrics, "corrections_all_metrics", width="stretch", hide_index=True)

    downloads = st.columns(3)
    downloads[0].download_button(
        "Download Events CSV",
        data=csv_bytes(events),
        file_name="qqq_correction_events.csv",
        mime="text/csv",
        width="stretch",
    )
    downloads[1].download_button(
        "Download ETF Metrics CSV",
        data=csv_bytes(metrics),
        file_name="etf_metrics_by_correction.csv",
        mime="text/csv",
        width="stretch",
    )
    downloads[2].download_button(
        "Download Summary CSV",
        data=csv_bytes(summary),
        file_name="summary_by_ticker.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption(f"Dataset source: {run_dir}")


def data_validation_tab() -> None:
    apply_dashboard_style()
    st.subheader("Are the Fundamentals Reliable?")
    st.caption("Compares Yahoo Finance with audited SEC filings.")
    if not VALIDATION_FILE.exists() or not VALIDATION_SUMMARY_FILE.exists():
        st.warning(
            "Validation files are not available. Run "
            "`python3 scripts/validate_yfinance_with_edgar.py` first."
        )
        return

    validation = pd.read_csv(VALIDATION_FILE)
    summary = pd.read_csv(VALIDATION_SUMMARY_FILE)
    status_counts = validation["status"].value_counts()
    checked = int(len(validation))
    validated = int(status_counts.get("Validated", 0))
    failed = int(status_counts.get("Failed", 0))
    unavailable = int(status_counts.get("Unavailable", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Metrics checked", checked)
    c2.metric("Validated", validated)
    c3.metric("Failed", failed)
    c4.metric("Unavailable", unavailable)
    st.caption("Validated: within 5% · Review: 5–15% · Failed: over 15%")

    if failed:
        render_major_finding(
            "Some Yahoo values differ from SEC filings by over 15%. "
            "Use the selected SEC values."
        )
    else:
        render_major_finding(
            "No available Yahoo values differ from SEC filings by over 15%."
        )

    st.markdown("### Company results")
    display_summary = summary.copy()
    display_summary["validation_rate"] = display_summary["validation_rate"].map(
        lambda value: "" if pd.isna(value) else f"{value:.0%}"
    )
    st.dataframe(display_summary, width="stretch")

    st.markdown("### Metric audit")
    ticker_options = ["All", *sorted(validation["ticker"].dropna().unique())]
    status_options = ["All", "Validated", "Review", "Failed", "Unavailable"]
    filter_cols = st.columns(2)
    ticker_filter = filter_cols[0].selectbox("Ticker", ticker_options)
    status_filter = filter_cols[1].selectbox("Validation status", status_options)
    filtered = validation.copy()
    if ticker_filter != "All":
        filtered = filtered[filtered["ticker"] == ticker_filter]
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]

    display = filtered.copy()
    display["difference_pct"] = display["difference_pct"].map(
        lambda value: "" if pd.isna(value) else f"{value:.1%}"
    )
    st.dataframe(display, width="stretch")
    st.caption(
        "Validated: difference <= 5%. Review: 5%-15%. Failed: >15%. "
        "SEC EDGAR is selected whenever an SEC value is available."
    )
    st.info(
        "This layer feeds the selected free-cash-flow margin into the resilience "
        "and statistical tabs. Revenue growth remains from Yahoo because the "
        "current EDGAR extract contains only the latest annual revenue value."
    )

    downloads = st.columns(2)
    downloads[0].download_button(
        "Download Validation CSV",
        data=csv_bytes(validation),
        file_name="financial_metric_validation.csv",
        mime="text/csv",
        width="stretch",
    )
    downloads[1].download_button(
        "Download Summary CSV",
        data=csv_bytes(summary),
        file_name="validation_summary.csv",
        mime="text/csv",
        width="stretch",
    )


def thesis_verdict_tab() -> None:
    apply_dashboard_style()
    st.subheader("Thesis Verdict")
    st.caption("Fixed scoring combines operating durability and market resilience.")

    run_dir = latest_qqq_corrections_dir()
    if run_dir is None:
        st.warning("No QQQ downturn dataset is available.")
        return
    events = pd.read_csv(run_dir / "qqq_correction_events.csv")
    etf_metrics = pd.read_csv(run_dir / "etf_metrics_by_correction.csv")
    revenue_data = revenue_growth_by_downturn(
        events,
        tuple(parse_tickers(DEFAULT_CYBER_TICKERS)),
        tuple(parse_tickers(DEFAULT_TECH_TICKERS)),
    )
    result = score_combined_thesis(events, etf_metrics, revenue_data)

    render_major_finding(
        f"{result['verdict']} — {result['total_points']}/10 points"
    )
    score_cols = st.columns(3)
    score_cols[0].metric("Operating durability", f"{result['operating_points']}/6")
    score_cols[1].metric("Market durability", f"{result['market_points']}/4")
    score_cols[2].metric("Total", f"{result['total_points']}/10")

    st.markdown("### What drove the result")
    evidence_cols = st.columns(2)
    with evidence_cols[0]:
        st.markdown("**Operating evidence**")
        st.metric("Median growth advantage", percent_text(result["growth_advantage"]))
        st.metric("Cyber growth beat rate", percent_text(result["growth_beat_rate"]))
        st.caption(
            f"Sign-test p-value: {result['growth_p_value']:.3f} across "
            f"{result['unique_growth_periods']} unique growth comparisons."
            if result["growth_p_value"] is not None
            else "Not enough unique growth comparisons for a sign test."
        )
    with evidence_cols[1]:
        st.markdown("**Market evidence**")
        st.metric("Median drawdown advantage", percent_text(result["drawdown_advantage"]))
        st.metric("Cyber ETF beat rate", percent_text(result["market_beat_rate"]))
        st.metric(
            "Volatility advantage",
            percent_text(result["volatility_advantage"]),
            help="Positive means cyber ETFs were less volatile than QQQ.",
        )

    st.markdown("### Score rules")
    score_table = pd.DataFrame(
        [
            ["Operating", "Positive median revenue-growth advantage", 2],
            ["Operating", "Cyber growth beat rate above 50%", 2],
            ["Operating", "Growth sign-test p-value at or below 0.05", 2],
            ["Market", "Smaller median maximum drawdown than QQQ", 2],
            ["Market", "Cyber ETF return beat rate above 50%", 1],
            ["Market", "Lower median volatility than QQQ", 1],
        ],
        columns=["Category", "Rule", "Points"],
    )
    st.dataframe(score_table, width="stretch", hide_index=True)
    st.caption("7–10: Supports · 4–6: Mixed · 0–3: Does not support")

    st.markdown("### Explanation")
    st.write(deterministic_verdict_explanation(result))
    if os.environ.get("OPENAI_API_KEY"):
        if st.button("Generate AI explanation", type="primary"):
            try:
                with st.spinner("Generating explanation..."):
                    st.session_state.ai_thesis_explanation = ai_verdict_explanation(result)
            except Exception as exc:
                st.error(f"AI explanation could not be generated: {exc}")
        if st.session_state.get("ai_thesis_explanation"):
            st.info(st.session_state.ai_thesis_explanation)
            st.caption("AI explains the fixed result; it does not determine the score.")
    else:
        st.button("Generate AI explanation", disabled=True)
        st.caption(
            "Optional AI explanation is available when OPENAI_API_KEY is configured. "
            "The score and verdict work without AI."
        )

    result_download = pd.DataFrame([result])
    st.download_button(
        "Download Verdict CSV",
        data=csv_bytes(result_download),
        file_name="thesis_verdict.csv",
        mime="text/csv",
        width="stretch",
    )


def main() -> None:
    st.set_page_config(page_title="Cyber Moat Thesis Dashboard", layout="wide")
    apply_dashboard_style()
    st.title("Cybersecurity Moat Thesis")
    render_cohort_key()
    st.markdown("### Thesis")
    st.markdown(
        f"<div class='thesis-box'>{THESIS_STATEMENT}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Current verdict")
    verdict_files = sorted(YFINANCE_RESULTS_ROOT.glob("*/verdict_*.txt"))
    if verdict_files:
        verdict_text = verdict_files[-1].read_text().strip()
        if "may not support" in verdict_text.lower() or "does not support" in verdict_text.lower():
            render_major_finding(
                "The current evidence does not support the full thesis. "
                "Cybersecurity may show operating strength, but it did not demonstrate "
                "enough market resilience relative to broad tech in the analyzed downturn."
            )
        else:
            render_major_finding(f"The current evidence supports the thesis. {verdict_text}")
        st.caption(f"Verdict source: {verdict_files[-1].parent.name}")
    else:
        st.warning("No saved verdict is available yet. Run the analysis pipeline to generate one.")

    st.subheader("Start here")
    st.write("Open **Summary Metrics** first. Use **Statistical Analysis** for the full test.")
    st.info(
        "The thesis verdict is evidence-driven. Changing the companies or downturn "
        "window can change the result."
    )


if __name__ == "__main__":
    main()
