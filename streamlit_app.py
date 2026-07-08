from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CYBER_TICKERS = "CRWD PANW FTNT ZS OKTA"
DEFAULT_TECH_TICKERS = "AAPL MSFT GOOGL AMZN META"
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
            row["free_cash_flow_margin"] = fcf_margin_for_ticker(ticker)
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
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download CSV",
            data=csv_bytes(result.csv),
            file_name=f"{result.filename_base}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def cyber_vs_tech_tab() -> None:
    st.subheader("Cyber Vs Broad Tech: Does Not Support Market-Resilience Claim")
    controls, output = st.columns([0.34, 0.66], gap="large")
    with controls:
        cyber_text = st.text_area("Cyber tickers", DEFAULT_CYBER_TICKERS, height=80)
        tech_text = st.text_area("Broad tech tickers", DEFAULT_TECH_TICKERS, height=80)
        start = st.date_input("Downturn start", value=date(2022, 1, 1), key="cvt_start")
        end = st.date_input("Downturn end", value=date(2022, 12, 31), key="cvt_end")
        metric = st.selectbox(
            "Chart variable",
            ["max_drawdown", "period_return", "annualized_volatility", "revenue_growth", "free_cash_flow_margin", "rule_of_40"],
            format_func=metric_label,
        )
        run = st.button("Update Cyber Vs Tech Plot", type="primary", use_container_width=True)

    if run or "cyber_vs_tech_result" not in st.session_state:
        cyber_tickers = tuple(parse_tickers(cyber_text))
        tech_tickers = tuple(parse_tickers(tech_text))
        with st.spinner("Pulling Yahoo Finance data..."):
            df = cyber_vs_tech_data(cyber_tickers, tech_tickers, start, end)
            st.session_state.cyber_vs_tech_result = draw_cyber_vs_tech_chart(df, metric, start, end)

    with output:
        result = st.session_state.cyber_vs_tech_result
        st.image(png_bytes(result.image), use_container_width=True)
        render_downloads(result)
        st.dataframe(result.csv, use_container_width=True)


def budget_variation_tab() -> None:
    st.subheader("Cyber Growth Vs Security Budget Variation: Supports Operating-Durability Claim")
    controls, output = st.columns([0.34, 0.66], gap="large")
    with controls:
        ticker_text = st.text_area("Cyber revenue tickers", DEFAULT_CYBER_TICKERS, height=90)
        budget_df = st.data_editor(
            DEFAULT_BUDGET_GROWTH,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "budget_year": st.column_config.NumberColumn("Budget year", step=1, format="%d"),
                "security_budget_growth_pct": st.column_config.NumberColumn("Security budget growth %", step=0.5, format="%.1f"),
            },
        )
        revenue_lag = st.number_input("Revenue fiscal-year lag", min_value=0, max_value=5, value=1, step=1)
        run = st.button("Update Budget Plot", type="primary", use_container_width=True)

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
        st.image(png_bytes(result.image), use_container_width=True)
        render_downloads(result)
        st.dataframe(result.csv, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Cyber Moat Thesis Dashboard", layout="wide")
    st.title("Cyber Moat Thesis Dashboard")
    st.caption("Interactive plots for the strongest supporting and non-supporting evidence.")

    tab1, tab2 = st.tabs(["Cyber vs Tech Resilience", "Cyber Growth vs Budget"])
    with tab1:
        cyber_vs_tech_tab()
    with tab2:
        budget_variation_tab()


if __name__ == "__main__":
    main()
