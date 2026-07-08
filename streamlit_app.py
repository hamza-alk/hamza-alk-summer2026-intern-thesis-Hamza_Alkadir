from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont

from scripts.plot_cyber_etf_vs_qqq_downturn import (
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
    st.subheader("Cyber Vs Broad Tech Resilience")
    st.caption(
        "Tests one question: did cybersecurity stocks suffer a smaller maximum "
        "drawdown than broad technology stocks during the selected downturn?"
    )
    controls, output = st.columns([0.34, 0.66], gap="large")
    with controls:
        cyber_text = st.text_area("Cyber tickers", DEFAULT_CYBER_TICKERS, height=80)
        tech_text = st.text_area("Broad tech tickers", DEFAULT_TECH_TICKERS, height=80)
        start = st.date_input("Downturn start", value=date(2022, 1, 1), key="cvt_start")
        end = st.date_input("Downturn end", value=date(2022, 12, 31), key="cvt_end")
        run = st.button("Update Cyber Vs Tech Plot", type="primary", width="stretch")

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
        st.markdown("### Chart Evaluator")
        if evaluator["supports"]:
            st.success(f"**{evaluator['verdict']}.** {evaluator['thesis_text']}")
        else:
            st.error(f"**{evaluator['verdict']}.** {evaluator['thesis_text']}")
        st.markdown("**Resilience measure: Maximum Drawdown**")
        st.write(evaluator["explanation"])
        st.write(evaluator["comparison"])
        st.caption(f"Period analyzed: {evaluator['period']}")
        render_downloads(result)
        st.dataframe(result.csv, width="stretch")


def statistical_analysis_tab() -> None:
    st.subheader("Statistical Analysis: Does Cyber Beat Broad Tech?")
    st.caption(
        "Combines market performance and company fundamentals to test whether "
        "the cybersecurity group consistently performed better than broad tech."
    )
    controls, output = st.columns([0.32, 0.68], gap="large")
    with controls:
        cyber_text = st.text_area("Cyber tickers", DEFAULT_CYBER_TICKERS, height=80, key="stats_cyber")
        tech_text = st.text_area("Broad tech tickers", DEFAULT_TECH_TICKERS, height=80, key="stats_tech")
        start = st.date_input("Downturn start", value=date(2022, 1, 1), key="stats_start")
        end = st.date_input("Downturn end", value=date(2022, 12, 31), key="stats_end")
        run = st.button("Run Statistical Analysis", type="primary", width="stretch")

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

        st.markdown("### Key Findings")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cyber vs broad drawdown", f"{mean_drawdown_diff:.1%}")
        c2.metric("Cyber vs broad return", f"{mean_return_diff:.1%}")
        c3.metric("Cyber vs broad volatility", f"{vol_diff:.1%}")
        c4.metric("Cyber vs broad FCF margin", f"{fcf_diff:.1%}")

        st.info(
            "Read the first three cards as market resilience. Negative drawdown/return gaps and higher volatility mean cyber did not behave as the safer group. "
            "FCF margin uses the SEC-selected value when EDGAR validation is available."
        )

        st.markdown("### Group Comparison")
        st.write(
            "This table compares the average and middle result for each group. "
            "The probability value asks whether the observed gap could easily "
            "occur by chance; below 0.05 is commonly treated as strong evidence."
        )
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

        st.markdown("### Relationships Between Metrics")
        st.write(
            "Correlation shows whether two measurements tend to move together. "
            "Values near 1 move together, values near -1 move in opposite "
            "directions, and values near 0 have little relationship."
        )
        st.dataframe(format_pct_table(corr_table.head(12), []), width="stretch")

        st.markdown("### Does Being a Cyber Company Matter?")
        st.write(
            "Regression estimates whether the cyber label still relates to stock "
            "performance after accounting for revenue growth and cash margins. "
            "A larger R-squared means the model explains more of the differences."
        )
        st.dataframe(
            format_pct_table(regression_table, ["cyber_label_coefficient"]),
            width="stretch",
        )

        st.markdown("### Company Classification")
        st.caption(
            "A resilient stock had a smaller-than-typical drawdown. A high-quality "
            "operator had an above-typical combination of growth and cash generation."
        )
        st.dataframe(
            format_pct_table(
                regime,
                ["max_drawdown", "period_return", "annualized_volatility", "rule_of_40"],
            ),
            width="stretch",
        )

        st.markdown("### Verdict Scoring Model")
        st.write(
            "The model awards points for better market resilience, stronger "
            "statistical evidence, and better operating performance. Market "
            "resilience receives the most weight because durability should appear "
            "when conditions are difficult."
        )
        score_cols = st.columns(4)
        score_cols[0].metric("Market score", f"{verdict.market_score}/6")
        score_cols[1].metric("Statistical score", f"{verdict.statistical_score}/3")
        score_cols[2].metric("Operating score", f"{verdict.operating_score}/3")
        score_cols[3].metric("Total score", f"{verdict.total_score}/{verdict.max_score}")
        st.caption(
            "Static rule: market resilience has veto power. If market score is 2 or lower, the verdict is does not support thesis even if operating metrics look strong."
        )
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
        if verdict.verdict == "Supports thesis":
            st.success(f"**{verdict.verdict}.** {verdict.plain_english}")
        else:
            st.error(f"**{verdict.verdict}.** {verdict.plain_english}")


def budget_variation_tab() -> None:
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
        st.dataframe(result.csv, width="stretch")


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
        st.dataframe(metrics, width="stretch")


def latest_summary_metrics_file() -> Path | None:
    candidates = list(YFINANCE_RESULTS_ROOT.glob("*/summary_metrics.csv"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


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
    st.subheader("Summary Metrics")
    st.caption(
        "A simple overview of growth, cash generation, and downturn performance "
        "for the current cybersecurity and broad-technology groups."
    )
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

    st.markdown("### Group Medians")
    st.write(
        "The median is the middle company in each group, which prevents one very "
        "large or unusual company from controlling the comparison."
    )
    cards = st.columns(5)
    card_specs = [
        ("Revenue growth", "revenue_growth", "How quickly sales grew."),
        ("FCF margin", "free_cash_flow_margin", "Cash kept from each sales dollar."),
        ("Rule of 40", "rule_of_40", "Growth plus free-cash-flow margin."),
        ("Maximum drawdown", "price_max_drawdown_downturn", "Worst stock decline."),
        ("Downturn return", "price_return_downturn", "Stock return in the downturn."),
    ]
    for card, (label, column, help_text) in zip(cards, card_specs):
        cyber_value = cyber.get(column)
        tech_value = tech.get(column)
        delta = (
            cyber_value - tech_value
            if pd.notna(cyber_value) and pd.notna(tech_value)
            else None
        )
        card.metric(
            label,
            "N/A" if pd.isna(cyber_value) else f"{cyber_value:.1%}",
            None if delta is None else f"{delta:+.1%} vs broad tech",
            help=help_text,
        )
    st.caption(
        "Each card shows the cybersecurity median. The small number underneath "
        "shows how far cyber is above or below broad tech."
    )

    st.markdown("### How to read these metrics")
    explanations = pd.DataFrame(
        [
            ["Revenue growth", "Are company sales still expanding?", "Higher is better."],
            ["FCF margin", "How much spendable cash comes from sales?", "Higher is better."],
            ["Rule of 40", "Does the company combine growth with cash generation?", "Higher is better."],
            ["Maximum drawdown", "What was the worst stock-price loss?", "Closer to zero is better."],
            ["Downturn return", "How did the stock perform over the full downturn?", "Higher is better."],
        ],
        columns=["Metric", "Simple question", "Better result"],
    )
    st.dataframe(explanations, width="stretch", hide_index=True)

    st.markdown("### Company Details")
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


def data_validation_tab() -> None:
    st.subheader("EDGAR Validation: Are the Fundamentals Reliable?")
    st.caption(
        "Checks Yahoo Finance numbers against audited SEC filings before those "
        "fundamentals are used in the thesis analysis."
    )
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
    st.write(
        "**Validated** means the sources are within 5%. **Review** means they "
        "differ by 5%-15%. **Failed** means they differ by more than 15%. "
        "**Unavailable** means one source did not report a comparable value."
    )

    if failed:
        st.error(
            "Some Yahoo values differ from SEC filings by more than 15%. "
            "The analysis should use the SEC-selected values shown below."
        )
    else:
        st.success(
            "No available Yahoo values differ from SEC filings by more than 15%. "
            "SEC values remain the preferred source for audited fundamentals."
        )

    st.markdown("### Company Summary")
    display_summary = summary.copy()
    display_summary["validation_rate"] = display_summary["validation_rate"].map(
        lambda value: "" if pd.isna(value) else f"{value:.0%}"
    )
    st.dataframe(display_summary, width="stretch")

    st.markdown("### Metric-Level Audit")
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


def main() -> None:
    st.set_page_config(page_title="Cyber Moat Thesis Dashboard", layout="wide")
    st.title("Cybersecurity Moat Thesis")
    st.write(
        "This project asks a simple question: do cybersecurity companies keep "
        "growing and hold up better than broad technology companies when economic "
        "conditions become difficult?"
    )
    st.subheader("How to use the project")
    st.write(
        "Choose a page from the sidebar. Start with Summary Metrics for the main "
        "numbers, then use Data Validation to check their reliability. Statistical "
        "Analysis combines the evidence, while the two downturn pages let you "
        "change companies, ETFs, and dates."
    )
    st.info(
        "The thesis verdict is evidence-driven. Changing the companies or downturn "
        "window can change the result."
    )


if __name__ == "__main__":
    main()
