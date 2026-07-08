"""Static scoring model for the cyber-vs-broad-tech thesis verdict.

The final verdict is intentionally binary:
- "Supports thesis"
- "Does not support thesis"

The model gives operating durability credit, but market resilience has veto
power. This avoids declaring support when cyber companies have better margins
but worse drawdowns, returns, or volatility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


MARKET_METRICS = {
    "max_drawdown": {
        "label": "Median max drawdown",
        "better": "higher",
        "points": 2,
        "p_value_metric": "max_drawdown",
    },
    "period_return": {
        "label": "Median downturn return",
        "better": "higher",
        "points": 2,
        "p_value_metric": "period_return",
    },
    "annualized_volatility": {
        "label": "Median annualized volatility",
        "better": "lower",
        "points": 2,
        "p_value_metric": "annualized_volatility",
    },
}

OPERATING_METRICS = {
    "revenue_growth": {
        "label": "Median revenue growth",
        "better": "higher",
        "points": 1,
    },
    "free_cash_flow_margin": {
        "label": "Median FCF margin",
        "better": "higher",
        "points": 1,
    },
    "rule_of_40": {
        "label": "Median Rule of 40",
        "better": "higher",
        "points": 1,
    },
}

SUGGESTIVE_P_VALUE = 0.10
MARKET_SCORE_VETO_THRESHOLD = 2
SUPPORT_TOTAL_SCORE_THRESHOLD = 7


@dataclass(frozen=True)
class ThesisVerdict:
    verdict: str
    plain_english: str
    market_score: int
    statistical_score: int
    operating_score: int
    total_score: int
    max_score: int
    details: pd.DataFrame


def _supports_cyber(cyber_value: float | None, broad_value: float | None, better: str) -> bool:
    if pd.isna(cyber_value) or pd.isna(broad_value):
        return False
    if better == "higher":
        return cyber_value > broad_value
    if better == "lower":
        return cyber_value < broad_value
    raise ValueError(f"Unknown better direction: {better}")


def _metric_difference(cyber_value: float | None, broad_value: float | None) -> float | None:
    if pd.isna(cyber_value) or pd.isna(broad_value):
        return None
    return cyber_value - broad_value


def score_thesis_verdict(
    cohort_table: pd.DataFrame,
    p_value_threshold: float = SUGGESTIVE_P_VALUE,
) -> ThesisVerdict:
    """Score whether the statistical analysis supports the thesis.

    Expected cohort_table columns:
    - metric
    - cyber_median
    - broad_tech_median
    - cyber_mean
    - broad_tech_mean
    - cyber_minus_broad_mean
    - permutation_p_value

    The `metric` values should be display labels such as "Max Drawdown" and
    "FCF Margin", matching streamlit_app.metric_label().
    """

    by_metric: dict[str, dict[str, Any]] = {
        str(row["metric"]): row.to_dict() for _, row in cohort_table.iterrows()
    }

    rows = []
    market_score = 0
    statistical_score = 0
    operating_score = 0

    for metric_name, config in MARKET_METRICS.items():
        label = config["label"]
        display_metric = {
            "max_drawdown": "Max Drawdown",
            "period_return": "Period Return",
            "annualized_volatility": "Annualized Volatility",
        }[metric_name]
        row = by_metric.get(display_metric, {})
        cyber_median = row.get("cyber_median")
        broad_median = row.get("broad_tech_median")
        supports = _supports_cyber(cyber_median, broad_median, config["better"])
        earned = config["points"] if supports else 0
        market_score += earned

        p_value = row.get("permutation_p_value")
        statistically_supports = supports and pd.notna(p_value) and p_value <= p_value_threshold
        stat_points = 1 if statistically_supports else 0
        statistical_score += stat_points

        rows.append(
            {
                "bucket": "Market resilience",
                "metric": label,
                "cyber_median": cyber_median,
                "broad_tech_median": broad_median,
                "cyber_minus_broad": _metric_difference(cyber_median, broad_median),
                "supports_cyber": supports,
                "points_earned": earned,
                "points_possible": config["points"],
                "p_value": p_value,
                "stat_points_earned": stat_points,
            }
        )

    for metric_name, config in OPERATING_METRICS.items():
        label = config["label"]
        display_metric = {
            "revenue_growth": "Revenue Growth",
            "free_cash_flow_margin": "FCF Margin",
            "rule_of_40": "Rule of 40",
        }[metric_name]
        row = by_metric.get(display_metric, {})
        cyber_median = row.get("cyber_median")
        broad_median = row.get("broad_tech_median")
        supports = _supports_cyber(cyber_median, broad_median, config["better"])
        earned = config["points"] if supports else 0
        operating_score += earned
        rows.append(
            {
                "bucket": "Operating durability",
                "metric": label,
                "cyber_median": cyber_median,
                "broad_tech_median": broad_median,
                "cyber_minus_broad": _metric_difference(cyber_median, broad_median),
                "supports_cyber": supports,
                "points_earned": earned,
                "points_possible": config["points"],
                "p_value": row.get("permutation_p_value"),
                "stat_points_earned": 0,
            }
        )

    total_score = market_score + statistical_score + operating_score
    max_score = 12

    if market_score <= MARKET_SCORE_VETO_THRESHOLD:
        verdict = "Does not support thesis"
        plain_english = (
            "Cyber may look better on some operating metrics, but it does not "
            "show enough market resilience. The thesis fails because cyber does "
            "not clearly beat broad tech on drawdown, return, and volatility."
        )
    elif total_score >= SUPPORT_TOTAL_SCORE_THRESHOLD:
        verdict = "Supports thesis"
        plain_english = (
            "Cyber beats broad tech on enough market-resilience and operating "
            "durability evidence to support the thesis."
        )
    else:
        verdict = "Does not support thesis"
        plain_english = (
            "The evidence is not strong enough. Cyber needs both market "
            "resilience and operating durability to support the thesis."
        )

    return ThesisVerdict(
        verdict=verdict,
        plain_english=plain_english,
        market_score=market_score,
        statistical_score=statistical_score,
        operating_score=operating_score,
        total_score=total_score,
        max_score=max_score,
        details=pd.DataFrame(rows),
    )
