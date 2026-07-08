#!/usr/bin/env python3
"""Plot cyber vs broad tech growth and resilience from pipeline outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COHORT_COLORS = {
    "cybersecurity": "#2563eb",
    "broad_tech": "#16a34a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a cohort scatter plot from summary_metrics.csv."
    )
    parser.add_argument("summary_path", type=Path)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Defaults to cyber_vs_tech_resilience.png next to the summary file.",
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
    rotated = label.rotate(90, expand=True)
    image.paste(rotated, xy, rotated)


def main() -> int:
    args = parse_args()
    output_path = args.output_path or args.summary_path.with_name("cyber_vs_tech_resilience.png")
    df = pd.read_csv(args.summary_path)
    df = df.dropna(subset=["revenue_growth", "price_max_drawdown_downturn"]).copy()
    df["revenue_growth_pct"] = df["revenue_growth"] * 100
    df["downturn_drawdown_pct"] = df["price_max_drawdown_downturn"] * 100

    width, height = 1200, 800
    left, right, top, bottom = 130, 80, 100, 120
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom

    x_min = min(-20, int(df["downturn_drawdown_pct"].min() - 5))
    x_max = 0
    y_min = min(-10, int(df["revenue_growth_pct"].min() - 5))
    y_max = max(40, int(df["revenue_growth_pct"].max() + 5))

    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    axis_font = load_font(18, bold=True)
    tick_font = load_font(14)
    label_font = load_font(15, bold=True)
    note_font = load_font(16)

    draw.text(
        (left, 30),
        "Cybersecurity vs Broad Tech: Growth and Downturn Resilience",
        fill="#111827",
        font=title_font,
    )

    for tick in range(x_min, x_max + 1, 10):
        x = scale(tick, x_min, x_max, plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill="#e5e7eb", width=1)
        draw.text((x - 18, plot_bottom + 12), f"{tick}%", fill="#374151", font=tick_font)

    for tick in range(y_min, y_max + 1, 10):
        y = scale(tick, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        draw.text((55, y - 8), f"{tick}%", fill="#374151", font=tick_font)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#111827", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#111827", width=2)
    draw.text(
        ((plot_left + plot_right) / 2 - 135, height - 58),
        "Max Drawdown During Downturn",
        fill="#111827",
        font=axis_font,
    )
    draw_vertical_text(image, "Revenue Growth", (22, 315), axis_font, "#111827")

    for _, row in df.iterrows():
        x = scale(row["downturn_drawdown_pct"], x_min, x_max, plot_left, plot_right)
        y = scale(row["revenue_growth_pct"], y_min, y_max, plot_bottom, plot_top)
        color = COHORT_COLORS.get(row["cohort"], "#6b7280")
        radius = 7
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="#111827", width=1)
        draw.text((x + 9, y - 9), row["ticker"], fill="#111827", font=label_font)

    legend_x = plot_right - 260
    for idx, (cohort, color) in enumerate(COHORT_COLORS.items()):
        y = plot_top + idx * 28
        draw.ellipse((legend_x, y, legend_x + 14, y + 14), fill=color, outline="#111827", width=1)
        draw.text((legend_x + 24, y - 3), cohort.replace("_", " ").title(), fill="#111827", font=note_font)

    draw.text(
        (plot_right - 420, plot_bottom - 30),
        "Top right = higher growth + smaller downturn drawdown",
        fill="#374151",
        font=note_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
