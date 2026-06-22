#!/usr/bin/env python3
"""Create a simple moat efficiency scatter plot.

Reads the generated yfinance cybersecurity summary file and plots:
- X axis: revenue growth
- Y axis: free cash flow margin
"""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SUMMARY_PATH = Path("data/yfinance_cybersecurity/2026-06-22_public/summary_metrics.csv")
OUTPUT_PATH = Path("data/yfinance_cybersecurity/2026-06-22_public/moat_efficiency_simple.png")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
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


def draw_vertical_text(
    image: Image.Image,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    text_box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label = Image.new("RGBA", (text_width + 10, text_height + 10), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((5, 5), text, fill=fill, font=font)
    image.paste(label.rotate(90, expand=True), xy, label.rotate(90, expand=True))


def main() -> int:
    df = pd.read_csv(SUMMARY_PATH)
    df = df.dropna(subset=["revenue_growth", "free_cash_flow_margin"]).copy()
    df["revenue_growth_pct"] = df["revenue_growth"] * 100
    df["free_cash_flow_margin_pct"] = df["free_cash_flow_margin"] * 100

    width, height = 1200, 800
    left, right, top, bottom = 120, 70, 90, 115
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom

    x_min = 0
    x_max = max(35, int(df["revenue_growth_pct"].max() + 6))
    y_min = 0
    y_max = max(50, int(df["free_cash_flow_margin_pct"].max() + 6))

    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    axis_font = load_font(18, bold=True)
    tick_font = load_font(14)
    label_font = load_font(15, bold=True)
    note_font = load_font(16)

    draw.text(
        (left, 28),
        "Cybersecurity Moat Efficiency: Growth vs Free Cash Flow",
        fill="#111827",
        font=title_font,
    )

    for tick in range(0, x_max + 1, 5):
        x = scale(tick, x_min, x_max, plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill="#e5e7eb", width=1)
        draw.text((x - 12, plot_bottom + 12), f"{tick}%", fill="#374151", font=tick_font)

    for tick in range(0, y_max + 1, 5):
        y = scale(tick, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        draw.text((48, y - 8), f"{tick}%", fill="#374151", font=tick_font)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#111827", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#111827", width=2)

    draw.text(
        ((plot_left + plot_right) / 2 - 75, height - 55),
        "Revenue Growth",
        fill="#111827",
        font=axis_font,
    )
    draw_vertical_text(image, "Free Cash Flow Margin", (18, 285), axis_font, "#111827")

    for _, row in df.iterrows():
        x = scale(row["revenue_growth_pct"], x_min, x_max, plot_left, plot_right)
        y = scale(row["free_cash_flow_margin_pct"], y_min, y_max, plot_bottom, plot_top)
        radius = 7
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#2563eb", outline="#111827", width=1)
        draw.text((x + 9, y - 9), row["ticker"], fill="#111827", font=label_font)

    draw.text(
        (plot_right - 410, plot_bottom - 28),
        "Top right = faster growth + stronger cash generation",
        fill="#374151",
        font=note_font,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
