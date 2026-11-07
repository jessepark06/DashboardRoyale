"""Natural experiments: placement moves and repair downtime recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import on_floor, operations, placements


WINDOW_DAYS = 14


def move_experiment(window_days: int = WINDOW_DAYS) -> dict:
    """
    Pre/post comparison for machines that changed location_id.
    Returns per-move deltas and a shrunk average uplift in theo/hr and coin-in/hr.
    """
    daily = on_floor()
    place = placements().sort_values(["machine_id", "effective_from_utc"])

    # Detect location changes from placements
    place = place.copy()
    place["prev_location"] = place.groupby("machine_id")["location_id"].shift(1)
    moves = place[place["prev_location"].notna() & (place["location_id"] != place["prev_location"])].copy()

    # Also include operations move events as move dates
    ops = operations()
    ops_moves = ops[ops["event_type"] == "move"][["machine_id", "available_at_utc", "opened_at_utc"]].copy()

    results = []
    # Use placement effective_from as move instant
    for _, row in moves.iterrows():
        mid = row["machine_id"]
        move_ts = row["effective_from_utc"]
        if pd.isna(move_ts):
            continue
        move_date = pd.Timestamp(move_ts.tz_convert("America/Los_Angeles").date())
        # Gaming day boundary ~ 6am; approximate with calendar date of local
        sub = daily[daily["machine_id"] == mid].copy()
        if sub.empty:
            continue
        pre = sub[
            (sub["gaming_date"] >= move_date - pd.Timedelta(days=window_days))
            & (sub["gaming_date"] < move_date)
        ]
        post = sub[
            (sub["gaming_date"] > move_date)
            & (sub["gaming_date"] <= move_date + pd.Timedelta(days=window_days))
        ]
        if len(pre) < 5 or len(post) < 5:
            continue
        pre_coin = pre["coin_in_usd"].sum() / max(pre["available_hours"].sum(), 1e-6)
        post_coin = post["coin_in_usd"].sum() / max(post["available_hours"].sum(), 1e-6)
        pre_theo = pre["theo_usd"].sum() / max(pre["available_hours"].sum(), 1e-6)
        post_theo = post["theo_usd"].sum() / max(post["available_hours"].sum(), 1e-6)
        results.append(
            {
                "machine_id": mid,
                "move_date": move_date,
                "from_location": row["prev_location"],
                "to_location": row["location_id"],
                "pre_days": len(pre),
                "post_days": len(post),
                "delta_coin_per_hr": post_coin - pre_coin,
                "delta_theo_per_hr": post_theo - pre_theo,
                "pre_coin_per_hr": pre_coin,
                "post_coin_per_hr": post_coin,
            }
        )

    detail = pd.DataFrame(results)
    n = len(detail)
    if n == 0:
        # Fallback: no usable moves — return null EV with flag
        return {
            "n_moves": 0,
            "detail": detail,
            "mean_delta_theo_per_hr": 0.0,
            "mean_delta_coin_per_hr": 0.0,
            "shrunk_delta_theo_per_hr": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "usable": False,
            "note": "Too few placement moves with adequate pre/post windows; use peer-gap prior for moves.",
        }

    mean_theo = float(detail["delta_theo_per_hr"].mean())
    mean_coin = float(detail["delta_coin_per_hr"].mean())
    # James-Stein style shrink toward 0 when n small (prior: moves average to small effect)
    shrink = n / (n + 5)
    shrunk = mean_theo * shrink
    se = float(detail["delta_theo_per_hr"].std(ddof=1) / np.sqrt(n)) if n > 1 else abs(mean_theo)
    return {
        "n_moves": n,
        "detail": detail,
        "mean_delta_theo_per_hr": mean_theo,
        "mean_delta_coin_per_hr": mean_coin,
        "shrunk_delta_theo_per_hr": shrunk,
        "ci_low": shrunk - 1.96 * se,
        "ci_high": shrunk + 1.96 * se,
        "usable": n >= 2,
        "note": f"{n} placement changes with {window_days}d pre/post windows; shrunk EV={shrunk:.3f} theo$/hr.",
    }


def repair_downtime_experiment() -> dict:
    """
    Estimate theo lost during fault_repair outages and manufacturer aggregates.
    Recovery EV assumes a technician/PM recovers a fraction of historical lost theo rate.
    """
    daily = on_floor()
    ops = operations()
    faults = ops[ops["event_type"] == "fault_repair"].copy()

    # Duration hours
    faults["downtime_hours"] = (
        (faults["available_at_utc"] - faults["opened_at_utc"]).dt.total_seconds() / 3600.0
    ).clip(lower=0)
    faults = faults[faults["downtime_hours"].notna()]

    # Machine theo/hr over full period
    mach = (
        daily.groupby("machine_id", as_index=False)
        .agg(
            theo_usd=("theo_usd", "sum"),
            available_hours=("available_hours", "sum"),
            manufacturer=("manufacturer", "first"),
            zone_id=("zone_id", "first"),
        )
    )
    mach["theo_per_hr"] = mach["theo_usd"] / mach["available_hours"].replace(0, np.nan)

    faults = faults.merge(mach[["machine_id", "theo_per_hr", "manufacturer", "zone_id"]], on="machine_id", how="left")
    faults["lost_theo_usd"] = faults["downtime_hours"] * faults["theo_per_hr"].fillna(0)

    by_mfr = (
        faults.groupby("manufacturer", as_index=False)
        .agg(
            n_faults=("operation_id", "count"),
            downtime_hours=("downtime_hours", "sum"),
            lost_theo_usd=("lost_theo_usd", "sum"),
            median_hours=("downtime_hours", "median"),
        )
        .sort_values("lost_theo_usd", ascending=False)
    )

    by_machine = (
        faults.groupby("machine_id", as_index=False)
        .agg(
            n_faults=("operation_id", "count"),
            downtime_hours=("downtime_hours", "sum"),
            lost_theo_usd=("lost_theo_usd", "sum"),
            manufacturer=("manufacturer", "first"),
        )
        .sort_values("lost_theo_usd", ascending=False)
    )

    total_lost = float(faults["lost_theo_usd"].sum())
    # Daily lost-theo rate over 181 days
    daily_lost_rate = total_lost / 181.0

    return {
        "n_faults": int(len(faults)),
        "total_downtime_hours": float(faults["downtime_hours"].sum()),
        "total_lost_theo_usd": total_lost,
        "daily_lost_theo_rate": daily_lost_rate,
        "by_manufacturer": by_mfr,
        "by_machine": by_machine,
        "fault_detail": faults,
        "note": (
            f"{len(faults)} fault repairs; estimated lost theo ${total_lost:,.0f} "
            f"(~${daily_lost_rate:,.0f}/day). Tech/PM EV = recovery_rate × rate × operating days."
        ),
    }


def location_peer_gap_prior(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Machines with largest positive location coin/hr gaps — move candidates."""
    c = scorecard.copy()
    c = c[c["location_coin_gap"] > 0].sort_values("location_coin_gap", ascending=False)
    # Expected theo uplift if we close 40% of theo gap (conservative)
    c["move_ev_theo_per_hr"] = c["location_theo_gap"].clip(lower=0) * 0.40
    return c
