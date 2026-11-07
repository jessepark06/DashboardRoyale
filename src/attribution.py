"""Luck vs location vs leak cause attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import machine_scorecard


CAUSE_SPEND = {
    "luck": "A-HOLD",
    "location": "A-WAYFINDING",
    "leak": "A-PREVENTIVE",
    "mix": "A-HOLD",
    "performing": "A-HOLD",
}


def attribute_causes(scorecard: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rule-based causes. Most machines are fine; only outliers get a label.

    Hold gap is not a separate 'config' cause: observed hold − theo hold is luck.
    A high published RTP (Royal Draw at 95%) is product mix, not a broken paytable.
    """
    m = machine_scorecard() if scorecard is None else scorecard.copy()

    faults_per_day = m["faults_opened"] / m["days_on_floor"].clip(lower=1)
    fault_p90 = float(faults_per_day.quantile(0.90))
    m["faults_per_day"] = faults_per_day

    m["score_luck"] = _z_clip((-m["luck_residual_usd"]).clip(lower=0))
    m["score_location"] = _z_clip(m["fleet_theo_gap"].clip(lower=0))
    m["score_leak"] = _z_clip(faults_per_day)
    m["score_mix"] = _z_clip((m["configured_rtp"] - 0.90).clip(lower=0))

    def _primary(row: pd.Series) -> str:
        theo = abs(float(row["theo_usd"])) + 1.0
        luck_share = -float(row["luck_residual_usd"]) / theo
        near_zone = float(row["theo_per_avail_hr"]) >= 0.85 * float(row["zone_theo_per_hr_med"] or 0)
        high_faults = fault_p90 > 0 and float(row["faults_per_day"]) >= fault_p90
        if float(row["availability_ratio"]) < 0.99 or high_faults:
            return "leak"
        if luck_share > 0.15 and near_zone:
            return "luck"
        if float(row["configured_rtp"]) >= 0.945 and float(row["coin_in_per_avail_hr"]) >= 0.7 * float(
            row["zone_coin_per_hr_med"] or 0
        ):
            return "mix"
        if float(row["fleet_theo_gap"]) > 4 and float(row["availability_ratio"]) >= 0.99:
            return "location"
        return "performing"

    def _alt(row: pd.Series, primary: str) -> str:
        order = ["luck", "location", "leak", "mix"]
        scores = {
            "luck": row["score_luck"],
            "location": row["score_location"],
            "leak": row["score_leak"],
            "mix": row["score_mix"],
        }
        for name in sorted(order, key=lambda k: scores[k], reverse=True):
            if name != primary:
                return name
        return "performing"

    m["primary_cause"] = m.apply(_primary, axis=1)
    m["alternative_cause"] = m.apply(lambda r: _alt(r, r["primary_cause"]), axis=1)
    m["preferred_action"] = m["primary_cause"].map(CAUSE_SPEND)

    m["is_unlucky"] = m["luck_residual_usd"] <= m["luck_residual_usd"].quantile(0.20)
    m["is_low_avail"] = m["availability_ratio"] < 0.99
    m["is_location_starved"] = m["fleet_theo_gap"] > 4
    m["is_available_but_empty"] = (m["occupancy_ratio"] < 0.25) & (m["availability_ratio"] > 0.99)
    m["is_back_forty"] = m["zone_id"] == "Z-B40"
    m["cause_note"] = m.apply(_note, axis=1)
    return m


def _z_clip(s: pd.Series) -> pd.Series:
    s = s.astype(float).fillna(0)
    std = s.std(ddof=0)
    if std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / std).clip(-3, 3)


def _note(row: pd.Series) -> str:
    cause = row["primary_cause"]
    alt = row["alternative_cause"]
    if cause == "luck":
        return (
            f"Primary: luck (residual ${row['luck_residual_usd']:,.0f}). "
            f"Theo/hr ${row['theo_per_avail_hr']:.2f} near peers — do not replace. Alt tested: {alt}."
        )
    if cause == "location":
        return (
            f"Primary: location. Theo/hr ${row['theo_per_avail_hr']:.2f} vs fleet "
            f"${row['fleet_theo_per_hr_med']:.2f}. Availability {row['availability_ratio']:.1%} — the box is up. "
            f"Alt tested: {alt}."
        )
    if cause == "leak":
        return (
            f"Primary: downtime ({int(row['faults_opened'])} faults, "
            f"avail {row['availability_ratio']:.1%}, lost theo ~${row['lost_theo_usd']:,.0f}). Alt tested: {alt}."
        )
    if cause == "mix":
        return (
            f"Primary: product mix. RTP {row['configured_rtp']:.1%} makes dollar win look low "
            f"even when coin-in is in line with the zone. Not a broken paytable. Alt tested: {alt}."
        )
    return (
        f"Performing. Theo/hr ${row['theo_per_avail_hr']:.2f}, avail {row['availability_ratio']:.1%}. "
        f"Alt tested: {alt}."
    )


def downtime_summary(scorecard: pd.DataFrame | None = None) -> dict:
    m = machine_scorecard() if scorecard is None else scorecard
    by_mfr = (
        m.groupby("manufacturer", as_index=False)
        .agg(
            machines=("machine_id", "nunique"),
            downtime_hours=("downtime_hours", "sum"),
            lost_theo_usd=("lost_theo_usd", "sum"),
            faults=("faults_opened", "sum"),
        )
        .sort_values("lost_theo_usd", ascending=False)
    )
    return {
        "total_downtime_hours": float(m["downtime_hours"].sum()),
        "total_lost_theo_usd": float(m["lost_theo_usd"].sum()),
        "by_manufacturer": by_mfr,
    }
