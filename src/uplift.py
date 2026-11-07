"""Map causal evidence to per-action expected July contribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import action_catalog
from .experiments import location_peer_gap_prior, move_experiment, repair_downtime_experiment
from .attribution import attribute_causes
from .metrics import machine_scorecard


HORIZON = 30
RECOVERY_RATE_TECH = 0.35  # fraction of historical lost-theo rate recovered by added tech
RECOVERY_RATE_PM = 0.25
MOVE_CAPTURE = 0.40  # fraction of location theo gap captured by a move
REPLACE_CAPTURE = 0.50  # fraction of gap to config median theo/hr


@dataclass
class ActionEV:
    action_type_id: str
    name: str
    unit_cost: float
    recurring_per_day: float
    lead_days: int
    quantity_limit: int
    incompatibility_group: str | None
    downtime_minutes: int
    # Candidate units with per-unit EV for operating days after lead
    candidates: pd.DataFrame  # columns: target_id, target_label, ev_usd, cost_usd, notes


def action_cost(unit_cost: float, recurring: float, lead_days: int, qty: int = 1) -> float:
    op_days = max(0, HORIZON - lead_days)
    return qty * (unit_cost + recurring * op_days)


def build_action_evs(
    recovery_tech: float = RECOVERY_RATE_TECH,
    recovery_pm: float = RECOVERY_RATE_PM,
    move_capture: float = MOVE_CAPTURE,
) -> dict[str, ActionEV]:
    catalog = action_catalog().copy()
    scorecard = attribute_causes(machine_scorecard())
    move_exp = move_experiment()
    repair_exp = repair_downtime_experiment()
    peer = location_peer_gap_prior(scorecard)

    # Shrink move experiment into peer-gap prior
    exp_theo_hr = move_exp["shrunk_delta_theo_per_hr"] if move_exp["usable"] else 0.0

    out: dict[str, ActionEV] = {}

    for _, row in catalog.iterrows():
        aid = row["action_type_id"]
        unit_cost = float(row["unit_cost_cents"]) / 100.0
        recurring = float(row["recurring_cost_cents_per_day"] or 0) / 100.0
        lead = int(row["lead_days"])
        limit = int(row["quantity_limit"])
        incompat = row.get("incompatibility_group")
        if pd.isna(incompat):
            incompat = None
        downtime_min = int(row["downtime_minutes"] or 0)
        op_days = max(0, HORIZON - lead)
        unit_budget_cost = action_cost(unit_cost, recurring, lead, 1)

        candidates = pd.DataFrame(columns=["target_id", "target_label", "ev_usd", "cost_usd", "notes"])

        if aid == "A-HOLD":
            candidates = pd.DataFrame(
                [
                    {
                        "target_id": "none",
                        "target_label": "Preserve budget",
                        "ev_usd": 0.0,
                        "cost_usd": 0.0,
                        "notes": "Valid if no positive EV after uncertainty.",
                    }
                ]
            )

        elif aid == "A-MOVE-MACHINE":
            # Top location-starved machines; EV = capture * gap * op_days * ~14 avail hrs/day proxy
            # Use theo gap * capture * op_days * typical available hours/day for that machine
            c = peer.head(40).copy()
            hours_per_day = (c["available_hours"] / c["days_on_floor"].clip(lower=1)).clip(1, 24)
            gap_hr = c["location_theo_gap"].clip(lower=0) * move_capture
            # Blend with experiment if positive
            if exp_theo_hr > 0:
                gap_hr = 0.5 * gap_hr + 0.5 * exp_theo_hr
            # Downtime cost of move
            move_down_hrs = downtime_min / 60.0
            down_cost = move_down_hrs * c["theo_per_avail_hr"].fillna(0)
            c["ev_usd"] = gap_hr * hours_per_day * op_days - down_cost
            c["cost_usd"] = unit_budget_cost
            c["target_id"] = c["machine_id"]
            c["target_label"] = c["machine_id"] + " @ " + c["zone_id"].astype(str)
            c["notes"] = (
                f"Close {move_capture:.0%} of location theo gap; "
                f"move experiment shrunk Δtheo/hr={exp_theo_hr:.3f} (n={move_exp['n_moves']})"
            )
            candidates = c[["target_id", "target_label", "ev_usd", "cost_usd", "notes"]].sort_values(
                "ev_usd", ascending=False
            )

        elif aid == "A-MOVE-BANK":
            bank = (
                scorecard.groupby(["bank_id", "zone_id"], as_index=False)
                .agg(
                    n=("machine_id", "count"),
                    mean_loc_gap=("location_theo_gap", "mean"),
                    mean_theo_hr=("theo_per_avail_hr", "mean"),
                    mean_avail_hr_day=("available_hours", "sum"),
                    days=("days_on_floor", "mean"),
                )
            )
            bank = bank[bank["n"] >= 3].copy()
            bank["hours_per_day"] = (bank["mean_avail_hr_day"] / bank["n"] / bank["days"].clip(lower=1)).clip(1, 24)
            bank["ev_usd"] = (
                bank["mean_loc_gap"].clip(lower=0) * move_capture * bank["n"] * bank["hours_per_day"] * op_days
                - (downtime_min / 60.0) * bank["mean_theo_hr"].fillna(0) * bank["n"]
            )
            bank["cost_usd"] = unit_budget_cost
            bank["target_id"] = bank["bank_id"]
            bank["target_label"] = bank["bank_id"].astype(str) + " (" + bank["zone_id"].astype(str) + ")"
            bank["notes"] = "Bank relocate; captures location theo gap across bank."
            candidates = bank[["target_id", "target_label", "ev_usd", "cost_usd", "notes"]].sort_values(
                "ev_usd", ascending=False
            )

        elif aid == "A-TECH-SHIFT":
            # Opportunity cost uses meter downtime (includes unrepaired slow periods), not only ticket spans.
            from .attribution import downtime_summary
            from .metrics import machine_scorecard as _ms

            dt = downtime_summary(_ms())
            daily_rate = max(repair_exp["daily_lost_theo_rate"], dt["total_lost_theo_usd"] / 181.0)
            # Tech also shortens queue: assume cuts mean repair duration ~25% on ~6.7 faults/day
            faults_per_day = repair_exp["n_faults"] / 181.0
            median_hrs = float(repair_exp["by_manufacturer"]["median_hours"].median())
            mttr_gain = faults_per_day * (0.25 * median_hrs) * (
                float(scorecard["theo_per_avail_hr"].median())
            )
            ev = recovery_tech * daily_rate * op_days + mttr_gain * op_days
            candidates = pd.DataFrame(
                [
                    {
                        "target_id": f"tech-{i}",
                        "target_label": f"Additional technician #{i}",
                        "ev_usd": ev * (1.0 if i == 1 else 0.65 if i == 2 else 0.4),
                        "cost_usd": unit_budget_cost,
                        "notes": (
                            f"Recover {recovery_tech:.0%} of ${daily_rate:,.0f}/day lost-theo "
                            f"+ MTTR cut on {faults_per_day:.1f} faults/day over {op_days}d. "
                            f"From {repair_exp['n_faults']} fault repairs."
                        ),
                    }
                    for i in range(1, limit + 1)
                ]
            )

        elif aid == "A-PREVENTIVE":
            by_mfr = repair_exp["by_manufacturer"].copy()
            # Share of lost theo by manufacturer; PM recovers recovery_pm of that mfr's rate
            total = by_mfr["lost_theo_usd"].sum() or 1.0
            rows = []
            for _, r in by_mfr.iterrows():
                share_daily = (r["lost_theo_usd"] / 181.0)
                # PM downtime cost: 90 min per machine
                n_mach = scorecard[scorecard["manufacturer"] == r["manufacturer"]].shape[0]
                down_cost = (downtime_min / 60.0) * scorecard.loc[
                    scorecard["manufacturer"] == r["manufacturer"], "theo_per_avail_hr"
                ].mean() * n_mach
                if np.isnan(down_cost):
                    down_cost = 0.0
                ev = recovery_pm * share_daily * op_days - down_cost
                rows.append(
                    {
                        "target_id": r["manufacturer"],
                        "target_label": f"PM: {r['manufacturer']}",
                        "ev_usd": ev,
                        "cost_usd": unit_budget_cost,
                        "notes": (
                            f"{int(r['n_faults'])} faults, lost theo ${r['lost_theo_usd']:,.0f}; "
                            f"recover {recovery_pm:.0%} of mfr daily rate over {op_days}d."
                        ),
                    }
                )
            candidates = pd.DataFrame(rows).sort_values("ev_usd", ascending=False)

        elif aid == "A-INSPECT":
            # Config-flagged machines; small expected recovery, high uncertainty
            cfg = scorecard[scorecard["primary_cause"] == "mix"].copy()
            if cfg.empty:
                cfg = scorecard.nlargest(15, "score_mix")
            cfg = cfg.head(limit)
            hold_gap = (cfg["theo_hold"] - cfg["observed_hold"]).clip(lower=0)
            # Assume 10% chance inspect finds fix worth 5% of remaining theo over op_days
            hours_per_day = (cfg["available_hours"] / cfg["days_on_floor"].clip(lower=1)).clip(1, 24)
            cfg["ev_usd"] = 0.10 * 0.05 * cfg["theo_per_avail_hr"].fillna(0) * hours_per_day * op_days
            cfg["cost_usd"] = unit_budget_cost
            cfg["target_id"] = cfg["machine_id"]
            cfg["target_label"] = cfg["machine_id"]
            cfg["notes"] = "Informational; EV assumes 10% chance of small hold fix."
            candidates = cfg[["target_id", "target_label", "ev_usd", "cost_usd", "notes"]].sort_values(
                "ev_usd", ascending=False
            )

        elif aid == "A-WAYFINDING":
            zone = (
                scorecard.groupby(["zone_id", "zone_name"], as_index=False)
                .agg(
                    n=("machine_id", "count"),
                    theo_hr=("theo_per_avail_hr", "median"),
                    hrs_day=("available_hours", "sum"),
                    days=("days_on_floor", "mean"),
                )
            )
            fleet_theo = float(scorecard["theo_per_avail_hr"].median())
            zone["hrs_per_day"] = zone["hrs_day"] / zone["n"] / zone["days"].clip(lower=1)
            # Close 8% of the zone's theo/hr gap vs the fleet. Signs do not create a new casino.
            capture = 0.08
            zone["gap"] = (fleet_theo - zone["theo_hr"]).clip(lower=0)
            zone["ev_usd"] = zone["gap"] * capture * zone["hrs_per_day"] * zone["n"] * op_days
            zone["cost_usd"] = unit_budget_cost
            zone["target_id"] = zone["zone_id"]
            zone["target_label"] = zone["zone_name"].astype(str)
            zone["notes"] = (
                f"Close {capture:.0%} of theo/hr gap vs fleet median ${fleet_theo:.2f} "
                f"over {op_days} days after lead. 5% capture is the kill line."
            )
            candidates = zone[["target_id", "target_label", "ev_usd", "cost_usd", "notes"]].sort_values(
                "ev_usd", ascending=False
            )

        elif aid in ("A-REPLACE-LEASE", "A-REPLACE-BUY"):
            # Truly structurally weak + not primarily luck
            weak = scorecard[
                (scorecard["primary_cause"].isin(["mix", "leak"]))
                | (scorecard["structural_weakness"] > scorecard["structural_weakness"].quantile(0.9))
            ].copy()
            weak = weak[~weak["is_unlucky"] | (weak["primary_cause"] != "luck")]
            weak = weak.nsmallest(30, "theo_per_avail_hr")
            hours_per_day = (weak["available_hours"] / weak["days_on_floor"].clip(lower=1)).clip(1, 24)
            # Uplift to config peer median theo/hr
            gap = (weak["cfg_theo_per_hr_med"] - weak["theo_per_avail_hr"]).clip(lower=0) * REPLACE_CAPTURE
            down_cost = (downtime_min / 60.0) * weak["theo_per_avail_hr"].fillna(0)
            lease_recurring_drag = recurring * op_days if aid == "A-REPLACE-LEASE" else 0.0
            weak["ev_usd"] = gap * hours_per_day * op_days - down_cost - lease_recurring_drag
            # For buy, capital is in cost; EV is gross contribution lift only
            weak["cost_usd"] = unit_budget_cost
            weak["target_id"] = weak["machine_id"]
            weak["target_label"] = weak["machine_id"]
            weak["notes"] = f"Replace toward config median theo/hr; {op_days} operating days after lead {lead}."
            candidates = weak[["target_id", "target_label", "ev_usd", "cost_usd", "notes"]].sort_values(
                "ev_usd", ascending=False
            )

        elif aid == "A-PROMO":
            # Incremental only: promo does NOT earn the existing weekend premium.
            # Assume +3–5% coin-in on two promo days above the weekday-model forecast, × ~9% hold.
            from .forecast import run_forecast

            fc = run_forecast()
            sat = float(fc["weekday_means"].get("Sat", fc["baseline_mean"]))
            hold = 0.09
            incr_pct = 0.05 if fc["favor_promo"] else 0.02
            # Two event days; second promo has diminishing return
            raw = sat * incr_pct * 2 * hold
            candidates = pd.DataFrame(
                [
                    {
                        "target_id": f"promo-{i}",
                        "target_label": f"Promotion weekend #{i}",
                        "ev_usd": raw * (1.0 if i == 1 else 0.5) - (0 if fc["favor_promo"] else raw * 0.5),
                        "cost_usd": unit_budget_cost,
                        "notes": (
                            f"Incremental +{incr_pct:.0%} coin-in × 2 days × {hold:.0%} hold "
                            f"on Sat-level volume (${sat:,.0f}). Not the weekend premium. "
                            + fc["decision"][:120]
                        ),
                    }
                    for i in range(1, limit + 1)
                ]
            )

        elif aid in ("A-COMP-PILOT", "A-COMP-CHANGE", "A-SURVEY"):
            # Default low EV — learning / uncertain; survey has no direct Jul cash
            ev_map = {"A-COMP-PILOT": 2000.0, "A-COMP-CHANGE": 500.0, "A-SURVEY": 0.0}
            candidates = pd.DataFrame(
                [
                    {
                        "target_id": aid,
                        "target_label": row["name"],
                        "ev_usd": ev_map[aid],
                        "cost_usd": unit_budget_cost,
                        "notes": "Low direct Jul cash EV; informational / policy risk.",
                    }
                ]
            )

        out[aid] = ActionEV(
            action_type_id=aid,
            name=str(row["name"]),
            unit_cost=unit_cost,
            recurring_per_day=recurring,
            lead_days=lead,
            quantity_limit=limit,
            incompatibility_group=incompat,
            downtime_minutes=downtime_min,
            candidates=candidates.reset_index(drop=True),
        )

    return out
