#!/usr/bin/env python3
"""Run full causal analysis and freeze outputs for the dashboard / submission."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attribution import attribute_causes, downtime_summary
from src.experiments import move_experiment, repair_downtime_experiment
from src.forecast import run_forecast
from src.metrics import fair_bottom, machine_scorecard, naive_bottom
from src.optimize import optimize_portfolio, portfolio_to_frame, summarize_by_type
from src.recommend import back_forty_case, recommendation_frames
from src.uplift import build_action_evs

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    print("Building scorecard…")
    scorecard = machine_scorecard()
    attributed = attribute_causes(scorecard)
    attributed.to_parquet(OUT / "machine_scorecard.parquet", index=False)
    naive_bottom(10).to_csv(OUT / "naive_bottom10.csv", index=False)
    fair_bottom(10).to_csv(OUT / "fair_bottom10.csv", index=False)

    print("Natural experiments…")
    move_exp = move_experiment()
    repair_exp = repair_downtime_experiment()
    if not move_exp["detail"].empty:
        move_exp["detail"].to_csv(OUT / "move_experiment_detail.csv", index=False)
    repair_exp["by_manufacturer"].to_csv(OUT / "repair_lost_theo_by_mfr.csv", index=False)
    repair_exp["by_machine"].head(50).to_csv(OUT / "repair_lost_theo_by_machine.csv", index=False)

    dt = downtime_summary(scorecard)
    cause_counts = attributed["primary_cause"].value_counts().to_dict()

    print("Forecast…")
    fc = run_forecast()
    fc["holdout"].to_csv(OUT / "forecast_holdout.csv", index=False)
    forecast_meta = {
        "mae_baseline": fc["mae_baseline"],
        "mae_weekday": fc["mae_weekday"],
        "weekend_lift": fc["weekend_lift"],
        "favor_promo": fc["favor_promo"],
        "decision": fc["decision"],
        "weekday_means": {str(k): float(v) for k, v in fc["weekday_means"].items()},
        "baseline_mean": fc["baseline_mean"],
    }

    print("Uplifts + portfolio search…")
    case = back_forty_case(attributed)
    evs = build_action_evs()
    result = optimize_portfolio(sensitivity=1.0, evs=evs)
    # Official call is the cause-matched pilot, not the generic search.
    items, by_type = recommendation_frames(case)
    items.to_csv(OUT / "recommended_portfolio_items.csv", index=False)
    by_type.to_csv(OUT / "recommended_portfolio_by_type.csv", index=False)
    rec_cost = float(by_type["cost_usd"].sum())
    rec_ev = float(by_type["ev_usd"].sum())

    agg = result["aggressive"]
    portfolio_to_frame(agg).to_csv(OUT / "optional_spend_items.csv", index=False)
    summarize_by_type(agg).to_csv(OUT / "optional_spend_by_type.csv", index=False)

    # Top alternatives summary
    alts = []
    for i, p in enumerate(result["top_alternatives"][:8]):
        alts.append(
            {
                "rank": i + 1,
                "types": "+".join(p.action_types),
                "cost": p.total_cost,
                "ev": p.total_ev,
                "npv": p.total_ev - p.total_cost,
                "n_items": len(p.items),
            }
        )
    pd.DataFrame(alts).to_csv(OUT / "portfolio_alternatives.csv", index=False)

    summary = {
        "n_machines": int(scorecard["machine_id"].nunique()),
        "cause_counts": cause_counts,
        "total_lost_theo_usd": dt["total_lost_theo_usd"],
        "total_downtime_hours": dt["total_downtime_hours"],
        "move_experiment": {
            "n_moves": move_exp["n_moves"],
            "usable": move_exp["usable"],
            "shrunk_delta_theo_per_hr": move_exp["shrunk_delta_theo_per_hr"],
            "note": move_exp["note"],
        },
        "repair_experiment": {
            "n_faults": repair_exp["n_faults"],
            "total_lost_theo_usd": repair_exp["total_lost_theo_usd"],
            "daily_lost_theo_rate": repair_exp["daily_lost_theo_rate"],
            "note": repair_exp["note"],
        },
        "forecast": forecast_meta,
        "finding": case,
        "recommended": {
            "action_types": ["A-WAYFINDING"],
            "total_cost": rec_cost,
            "total_ev": rec_ev,
            "roi": rec_ev / rec_cost if rec_cost else 0,
            "n_items": 1,
            "unspent": 100_000 - rec_cost,
        },
        "search_hold_note": result["stress_note"],
        "best_raw": {
            "action_types": result["best_raw"].action_types,
            "total_cost": result["best_raw"].total_cost,
            "total_ev": result["best_raw"].total_ev,
            "npv": result["best_raw"].total_ev - result["best_raw"].total_cost,
        },
        "optional_spend": {
            "action_types": agg.action_types,
            "total_cost": agg.total_cost,
            "total_ev": agg.total_ev,
            "npv": agg.total_ev - agg.total_cost,
        },
        "stress_note": result["stress_note"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    _write_recommended_md(by_type, items, summary)
    _write_claims(summary, attributed, move_exp, repair_exp, fc)
    _write_writeup(summary, fc)

    print("\n=== RECOMMENDED PORTFOLIO ===")
    print(by_type.to_string(index=False) if not by_type.empty else "A-HOLD")
    print(
        f"Cost ${rec_cost:,.0f} | EV ${rec_ev:,.0f} | "
        f"Back Forty {case['naive_in_back_forty']}/10 of the naive list"
    )
    print(f"\nOutputs written to {OUT}")


def _write_recommended_md(by_type, items, summary) -> None:
    c = summary["finding"]
    rec = summary["recommended"]
    lines = [
        "# Recommended July actions ($100,000 budget)",
        "",
        "## Decision",
        "",
        f"Do **not** replace the bottom 10 by house win. "
        f"**{c['naive_in_back_forty']} of those 10** sit in Back Forty and average "
        f"**{c['naive_availability']:.1%} availability**. They are quiet, not broken.",
        "",
        f"Fund **one** action: wayfinding for Back Forty. "
        f"Cost **${rec['total_cost']:,.0f}**. Base EV **${rec['total_ev']:,.0f}** "
        f"(NPV **${c['base_npv']:,.0f}**). Hold the other **${rec['unspent']:,.0f}**.",
        "",
        "| Action | Qty | Cost | Base EV |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, r in by_type.iterrows():
        lines.append(
            f"| {r['action_type_id']} — {r['name']} | {int(r['quantity'])} | "
            f"${r['cost_usd']:,.0f} | ${r['ev_usd']:,.0f} |"
        )
    lines += [
        "",
        f"- Target: Back Forty (`Z-B40`). {items.iloc[0]['notes']}",
        "",
        "## Why not the other catalog items",
        "",
        f"- **Replace / buy the bottom 10.** Four purchased machines would be "
        f"${c['replace_trap_cost']:,.0f} and would still sit in a zone earning "
        f"${c['back_forty_theo_hr']:.2f} theo/hr versus a fleet median of "
        f"${c['fleet_theo_hr']:.2f}. M-1158 alone is **${c['m1158_luck']:,.0f} luck** "
        f"with theo/hr in line with its zone.",
        "- **Promo weekend ($15,000).** The weekday model already explains weekends. "
        "Buying a promo spends money to recreate demand the floor already has.",
        f"- **Technician or preventive maintenance.** Ridgeline faults "
        f"({c['ridgeline_faults_per_machine']:.1f} per machine) run above Halcyon "
        f"({c['halcyon_faults_per_machine']:.1f}), but July lead time leaves too few "
        "days for a $9,000–$11,520 package to pay back.",
        "- **Moves.** Only four historical placement changes. A one-month move does not "
        "clear its $1,800 cost once cannibalization is allowed for.",
        "",
        "## Sensitivity",
        "",
        f"- 8% of the zone gap (base): EV ${c['base_ev']:,.0f}, NPV ${c['base_npv']:,.0f}.",
        f"- 5% (kill line): EV ${c['kill_ev']:,.0f}, NPV ${c['kill_npv']:,.0f}. Stop.",
        f"- 15%: EV ${c['upside_15_ev']:,.0f}.",
        "",
        "## Measurement plan",
        "",
        "- Pre-register Back Forty coin-in per available hour and theo per available hour.",
        "- Compare July (after the 10-day lead) with matched weekdays in May–June, "
        "versus Main Entrance and Center Pit as controls.",
        "- Kill rule: if the zone closes less than 5% of its theo/hr gap, do not scale signs or moves.",
        "",
    ]
    (OUT / "recommended_actions.md").write_text("\n".join(lines))


def _write_claims(summary, attributed, move_exp, repair_exp, fc) -> None:
    c = summary["finding"]
    naive = attributed.nsmallest(10, "house_win_usd")
    lines = [
        "# Claims & Evidence",
        "",
        "## Claim 1 — Nine of the GM's bottom 10 are Back Forty, and they are not down",
        "- **Query:** on-floor `prepared_daily_view`, rank by sum of house win, join zone and availability.",
        f"- **Result:** {c['naive_in_back_forty']}/10 are `Z-B40`. Mean availability {c['naive_availability']:.1%}. "
        f"IDs: {c['naive_ids']}.",
        "- **Alternative tested:** the machines are broken or mis-paying. Availability near 100% and theo/hr in line with the zone reject that.",
        "- **Limitation:** zone labels come from the last placement in the window.",
        "",
        "## Claim 2 — M-1158 is a luck event, not a bad box",
        "- **Query:** house win, theo, luck residual, theo per available hour vs zone median.",
        f"- **Result:** luck residual ${c['m1158_luck']:,.0f}. Theo/hr ${c['m1158_theo_hr']:.2f} vs zone ${c['m1158_zone_theo_hr']:.2f}.",
        "- **Alternative tested:** the paytable is wrong. Theo uses configured RTP; the gap is residual, not theo.",
        "- **Limitation:** six months is still a short sample on a medium-volatility game.",
        "",
        "## Claim 3 — Back Forty earns about $6/hr less theo than the fleet",
        f"- **Query:** median theo per available hour by zone.",
        f"- **Result:** Back Forty ${c['back_forty_theo_hr']:.2f}/hr vs fleet ${c['fleet_theo_hr']:.2f}/hr "
        f"vs Main Entrance ${c['entrance_theo_hr']:.2f}/hr.",
        "- **Alternative tested:** the zone is full of low-RTP games only. The same titles earn more theo/hr in Center Pit and Main Entrance.",
        "- **Limitation:** we do not observe whether guests avoid the zone or the zone is physically hard to find.",
        "",
        "## Claim 4 — Downtime is real and concentrated, and still too small for a July tech hire",
        f"- **Query:** fault_repair duration × machine theo/hr; faults per machine by manufacturer.",
        f"- **Result:** ${repair_exp['total_lost_theo_usd']:,.0f} lost theo on {repair_exp['n_faults']} faults "
        f"(~${repair_exp['daily_lost_theo_rate']:,.0f}/day). Ridgeline {c['ridgeline_faults_per_machine']:.1f} faults/machine vs "
        f"Halcyon {c['halcyon_faults_per_machine']:.1f}.",
        "- **Alternative tested:** buy a technician. Lead time leaves ~23 days; recovery does not cover $11,520.",
        "- **Limitation:** lost theo assumes the machine would have earned its usual theo/hr while down.",
        "",
        "## Claim 5 — A weekday mean beats a flat mean, and that kills the promo",
        "- **Query:** `property_day` coin-in. Train through 2025-04-30. Hold out May–June.",
        f"- **Result:** weekday MAE ${fc['mae_weekday']:,.0f} vs baseline ${fc['mae_baseline']:,.0f}. "
        f"Weekend premium ${fc['weekend_lift']:,.0f}/day.",
        f"- **Decision:** {fc['decision']}",
        "- **Limitation:** events are few; the model does not claim to forecast July tourism.",
        "",
        "## Claim 6 — Spend $7,500 on Back Forty signs and hold $92,500",
        "- **Query:** July cost = catalog unit cost after a 10-day lead. EV = 8% × theo/hr gap × hours/day × 48 machines × 20 days.",
        f"- **Result:** cost ${c['wayfinding_cost']:,.0f}, EV ${c['base_ev']:,.0f}, NPV ${c['base_npv']:,.0f}. "
        f"At 5% capture NPV is ${c['kill_npv']:,.0f}.",
        "- **Alternative tested:** move machines or replace them. Moves lack sample (n="
        f"{move_exp['n_moves']}). Replacements inherit the zone.",
        "- **Limitation:** 8% is an assumption, not a measured signage elasticity. The kill rule exists because of that.",
        "",
    ]
    (OUT / "claims_evidence.md").write_text("\n".join(lines))


def _write_writeup(summary, fc) -> None:
    c = summary["finding"]
    text = f"""# The Gilded Ace

The GM's bottom 10 by house win are not broken machines. Nine of the ten sit in Back Forty, and that group was available {c['naive_availability']:.1%} of scheduled hours. Replacing them spends capital on a quiet corner. M-1158 shows the trap: house win trailed theo by ${abs(c['m1158_luck']):,.0f}, while theo per hour matched its zone. That is luck.

Back Forty earns ${c['back_forty_theo_hr']:.2f} theo per available hour against a fleet median of ${c['fleet_theo_hr']:.2f}. The same game titles earn more in Main Entrance and Center Pit, so the gap is the place, not the paytable. Royal Draw's 95% RTP makes dollar win look worse even when the game is behaving. Ridgeline machines fault more often than Halcyon ({c['ridgeline_faults_per_machine']:.1f} vs {c['halcyon_faults_per_machine']:.1f} per machine), but six months of lost theo is too small to repay a July technician after lead time.

A weekday-mean forecast of floor coin-in, trained January–April and tested May–June, cuts MAE from ${fc['mae_baseline']:,.0f} to ${fc['mae_weekday']:,.0f}. Weekends already carry about ${fc['weekend_lift']:,.0f} more coin-in a day. Staff to that pattern. Do not buy a promotion to invent it.

**Recommendation.** One action: wayfinding for Back Forty, $7,500, about ${c['base_ev']:,.0f} expected theo if signs close 8% of the zone gap over the 20 days after lead time. Hold the other $92,500. If the zone closes less than 5% of the gap versus Main Entrance and Center Pit, stop. Do not replace the bottom 10.

AI disclosure: Cursor helped write the pipeline and dashboard. Figures come from `analysis/run_pipeline.py`.
"""
    words = text.split()
    if len(words) > 400:
        text = " ".join(words[:390]) + "\n"
    (OUT / "writeup_draft.md").write_text(text)


if __name__ == "__main__":
    main()
