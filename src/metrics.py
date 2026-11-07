"""Fair machine scorecard metrics (ratio-of-sums, exposure, luck)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import calendar_day, on_floor


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce").astype(float)
    den = pd.to_numeric(den, errors="coerce").astype(float).replace(0, np.nan)
    return num / den


def machine_scorecard(
    daily: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """One row per machine with exposure-, luck-, and location-aware metrics."""
    df = on_floor(daily)
    if start:
        df = df[df["gaming_date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["gaming_date"] <= pd.Timestamp(end)]

    cal = calendar_day()[["gaming_date", "scheduled_seconds"]]
    df = df.merge(cal, on="gaming_date", how="left")
    df["scheduled_hours"] = df["scheduled_seconds"] / 3600.0

    # Latest known attributes per machine in window
    latest = (
        df.sort_values("gaming_date")
        .groupby("machine_id", as_index=False)
        .tail(1)[
            [
                "machine_id",
                "zone_id",
                "zone_name",
                "bank_id",
                "config_id",
                "theme",
                "configured_rtp",
                "volatility_class",
                "manufacturer",
                "lease_class",
            ]
        ]
    )

    agg = (
        df.groupby("machine_id", as_index=False)
        .agg(
            days_on_floor=("gaming_date", "nunique"),
            coin_in_usd=("coin_in_usd", "sum"),
            payout_usd=("payout_usd", "sum"),
            house_win_usd=("house_win_usd", "sum"),
            theo_usd=("theo_usd", "sum"),
            luck_residual_usd=("luck_residual_usd", "sum"),
            available_hours=("available_hours", "sum"),
            occupied_hours=("occupied_hours", "sum"),
            scheduled_hours=("scheduled_hours", "sum"),
            faults_opened=("faults_opened", "sum"),
            contribution_usd=("contribution_after_supplied_costs_usd", "sum"),
            paid_spins=("paid_spins", "sum"),
        )
    )

    m = agg.merge(latest, on="machine_id", how="left")
    for col in [
        "coin_in_usd",
        "payout_usd",
        "house_win_usd",
        "theo_usd",
        "luck_residual_usd",
        "available_hours",
        "occupied_hours",
        "scheduled_hours",
        "contribution_usd",
        "paid_spins",
        "faults_opened",
        "configured_rtp",
    ]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce").astype(float)
    m["observed_hold"] = _safe_div(m["house_win_usd"], m["coin_in_usd"])
    m["theo_hold"] = _safe_div(m["theo_usd"], m["coin_in_usd"])
    m["availability_ratio"] = _safe_div(m["available_hours"], m["scheduled_hours"])
    m["occupancy_ratio"] = _safe_div(m["occupied_hours"], m["available_hours"])
    m["theo_per_avail_hr"] = _safe_div(m["theo_usd"], m["available_hours"])
    m["house_win_per_avail_hr"] = _safe_div(m["house_win_usd"], m["available_hours"])
    m["coin_in_per_avail_hr"] = _safe_div(m["coin_in_usd"], m["available_hours"])
    m["luck_vs_theo"] = _safe_div(m["luck_residual_usd"], m["theo_usd"].abs())
    m["downtime_hours"] = (m["scheduled_hours"] - m["available_hours"]).clip(lower=0)
    m["lost_theo_usd"] = m["downtime_hours"] * m["theo_per_avail_hr"].fillna(0)

    # Zone peer medians (same zone)
    zone_peer = (
        m.groupby("zone_id")
        .agg(
            zone_theo_per_hr_med=("theo_per_avail_hr", "median"),
            zone_coin_per_hr_med=("coin_in_per_avail_hr", "median"),
            zone_avail_med=("availability_ratio", "median"),
        )
        .reset_index()
    )
    m = m.merge(zone_peer, on="zone_id", how="left")
    m["location_coin_gap"] = m["zone_coin_per_hr_med"] - m["coin_in_per_avail_hr"]
    m["location_theo_gap"] = m["zone_theo_per_hr_med"] - m["theo_per_avail_hr"]
    fleet_coin = float(m["coin_in_per_avail_hr"].median())
    fleet_theo = float(m["theo_per_avail_hr"].median())
    m["fleet_coin_per_hr_med"] = fleet_coin
    m["fleet_theo_per_hr_med"] = fleet_theo
    m["fleet_coin_gap"] = fleet_coin - m["coin_in_per_avail_hr"]
    m["fleet_theo_gap"] = fleet_theo - m["theo_per_avail_hr"]

    # Config peer medians
    cfg_peer = (
        m.groupby("config_id")
        .agg(cfg_theo_per_hr_med=("theo_per_avail_hr", "median"), cfg_hold_med=("observed_hold", "median"))
        .reset_index()
    )
    m = m.merge(cfg_peer, on="config_id", how="left")
    m["config_theo_gap"] = m["cfg_theo_per_hr_med"] - m["theo_per_avail_hr"]

    # Ranks (1 = worst)
    m["rank_house_win"] = m["house_win_usd"].rank(method="min", ascending=True)
    m["rank_contribution"] = m["contribution_usd"].rank(method="min", ascending=True)
    m["rank_theo_per_hr"] = m["theo_per_avail_hr"].rank(method="min", ascending=True)
    m["rank_availability"] = m["availability_ratio"].rank(method="min", ascending=True)

    # Fair underperformance score: high = more concerning after adjustments
    # Components: low theo/hr (not luck), low availability, large location gap
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    m["fair_weakness"] = (
        -z(m["theo_per_avail_hr"].fillna(m["theo_per_avail_hr"].median()))
        + z(m["downtime_hours"].fillna(0))
        + z(m["location_coin_gap"].fillna(0).clip(lower=0))
        - z(m["luck_residual_usd"].fillna(0))  # negative luck makes weakness look worse on raw; subtract so luck alone doesn't dominate replace decision
    )
    # Recompute: weakness should rise when theo/hr low, downtime high, location gap high,
    # but fall when negative luck explains low house win (machine is fine expected-value-wise)
    m["fair_weakness"] = (
        -z(m["theo_per_avail_hr"].fillna(m["theo_per_avail_hr"].median()))
        + z(m["downtime_hours"].fillna(0))
        + z(m["location_coin_gap"].fillna(0).clip(lower=0))
        + z((-m["luck_residual_usd"]).clip(lower=0)) * 0.15  # mild: very unlucky still noted
    )
    # Better formulation for "truly weak": low theo/hr + downtime + location, independent of luck
    m["structural_weakness"] = (
        -z(m["theo_per_avail_hr"].fillna(m["theo_per_avail_hr"].median()))
        + z(m["downtime_hours"].fillna(0))
        + z(m["location_coin_gap"].fillna(0).clip(lower=0))
    )
    m["rank_structural"] = m["structural_weakness"].rank(method="min", ascending=False)

    return m.sort_values("contribution_usd")


def naive_bottom(n: int = 10, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    sc = machine_scorecard(daily)
    return sc.nsmallest(n, "house_win_usd")


def fair_bottom(n: int = 10, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    sc = machine_scorecard(daily)
    return sc.nlargest(n, "structural_weakness")
