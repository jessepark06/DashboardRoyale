"""Decision: do not replace the bottom 10. Pilot signs in Back Forty. Hold the rest."""

from __future__ import annotations

import pandas as pd

WAYFINDING_COST = 7_500.0
LEAD_DAYS = 10
HORIZON = 30
BASE_CAPTURE = 0.08
KILL_CAPTURE = 0.05


def back_forty_case(scorecard: pd.DataFrame) -> dict:
    naive = scorecard.nsmallest(10, "house_win_usd")
    b40 = scorecard[scorecard["zone_id"] == "Z-B40"]
    fleet_theo = float(scorecard["theo_per_avail_hr"].median())
    b40_theo = float(b40["theo_per_avail_hr"].median())
    hrs_day = float((b40["available_hours"] / b40["days_on_floor"].clip(lower=1)).mean())
    op_days = HORIZON - LEAD_DAYS
    gap = max(0.0, fleet_theo - b40_theo)
    exposure = gap * hrs_day * len(b40) * op_days

    def ev(capture: float) -> float:
        return exposure * capture

    base_ev = ev(BASE_CAPTURE)
    kill_ev = ev(KILL_CAPTURE)
    trap = naive[naive["zone_id"] == "Z-B40"]
    luck_row = scorecard.loc[scorecard["machine_id"] == "M-1158"]
    luck = float(luck_row["luck_residual_usd"].iloc[0]) if len(luck_row) else 0.0

    ridge = scorecard[scorecard["manufacturer"] == "Ridgeline"]
    hal = scorecard[scorecard["manufacturer"] == "Halcyon"]

    return {
        "naive_in_back_forty": int((naive["zone_id"] == "Z-B40").sum()),
        "naive_ids": naive["machine_id"].tolist(),
        "naive_availability": float(naive["availability_ratio"].mean()),
        "back_forty_machines": int(len(b40)),
        "back_forty_theo_hr": b40_theo,
        "fleet_theo_hr": fleet_theo,
        "entrance_theo_hr": float(
            scorecard.loc[scorecard["zone_id"] == "Z-ENT", "theo_per_avail_hr"].median()
        ),
        "hours_per_day": hrs_day,
        "operating_days": op_days,
        "gap_theo_hr": gap,
        "wayfinding_cost": WAYFINDING_COST,
        "base_capture": BASE_CAPTURE,
        "base_ev": base_ev,
        "base_npv": base_ev - WAYFINDING_COST,
        "kill_capture": KILL_CAPTURE,
        "kill_ev": kill_ev,
        "kill_npv": kill_ev - WAYFINDING_COST,
        "upside_15_ev": ev(0.15),
        "m1158_luck": luck,
        "m1158_theo_hr": float(luck_row["theo_per_avail_hr"].iloc[0]) if len(luck_row) else 0.0,
        "m1158_zone_theo_hr": float(luck_row["zone_theo_per_hr_med"].iloc[0]) if len(luck_row) else 0.0,
        "ridgeline_faults_per_machine": float(ridge["faults_opened"].sum() / max(len(ridge), 1)),
        "halcyon_faults_per_machine": float(hal["faults_opened"].sum() / max(len(hal), 1)),
        "replace_trap_cost": 22_000.0 * 4,
        "short_tenure_in_bottom10": int((naive["days_on_floor"] < 150).sum()),
    }


def recommendation_frames(case: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.DataFrame(
        [
            {
                "action_type_id": "A-WAYFINDING",
                "name": "Wayfinding and signage refresh for a zone",
                "target_id": "Z-B40",
                "target_label": "Back Forty",
                "cost_usd": case["wayfinding_cost"],
                "ev_usd": case["base_ev"],
                "notes": (
                    f"Only action aimed at the diagnosed cause. "
                    f"Assumes signs close {case['base_capture']:.0%} of the "
                    f"${case['gap_theo_hr']:.2f}/hr theo gap vs the fleet "
                    f"over {case['operating_days']} days. "
                    f"Kill if realized lift is under {case['kill_capture']:.0%}."
                ),
            }
        ]
    )
    by_type = pd.DataFrame(
        [
            {
                "action_type_id": "A-WAYFINDING",
                "name": "Wayfinding and signage refresh for a zone",
                "quantity": 1,
                "cost_usd": case["wayfinding_cost"],
                "ev_usd": case["base_ev"],
            }
        ]
    )
    return items, by_type
