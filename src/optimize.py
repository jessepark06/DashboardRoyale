"""Constrained portfolio search: ≤3 action types, $100k, conflicts."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .uplift import ActionEV, build_action_evs

BUDGET = 100_000.0
MAX_TYPES = 3
# Skip pure learning actions with ~0 Jul EV unless nothing else works
DEPRIORITIZE = {"A-SURVEY", "A-COMP-CHANGE"}


@dataclass
class PortfolioItem:
    action_type_id: str
    name: str
    target_id: str
    target_label: str
    cost_usd: float
    ev_usd: float
    notes: str


@dataclass
class Portfolio:
    items: list[PortfolioItem] = field(default_factory=list)
    total_cost: float = 0.0
    total_ev: float = 0.0
    action_types: list[str] = field(default_factory=list)

    @property
    def roi(self) -> float:
        return self.total_ev / self.total_cost if self.total_cost > 0 else 0.0


def _greedy_fill(
    action_ids: tuple[str, ...],
    evs: dict[str, ActionEV],
    budget: float = BUDGET,
    sensitivity: float = 1.0,
    require_positive_npv: bool = True,
) -> Portfolio:
    """Within a fixed set of action types, greedily add units."""
    # Comp policy mutual exclusion
    if "A-COMP-PILOT" in action_ids and "A-COMP-CHANGE" in action_ids:
        return Portfolio()

    # Avoid double-counting location: do not combine bank moves with machine moves
    if "A-MOVE-BANK" in action_ids and "A-MOVE-MACHINE" in action_ids:
        return Portfolio()

    pool: list[PortfolioItem] = []

    for aid in action_ids:
        ae = evs[aid]
        cands = ae.candidates.copy()
        if cands.empty:
            continue
        cands["adj_ev"] = cands["ev_usd"] * sensitivity
        cands = cands.sort_values("adj_ev", ascending=False)
        for _, r in cands.head(ae.quantity_limit).iterrows():
            pool.append(
                PortfolioItem(
                    action_type_id=aid,
                    name=ae.name,
                    target_id=str(r["target_id"]),
                    target_label=str(r["target_label"]),
                    cost_usd=float(r["cost_usd"]),
                    ev_usd=float(r["adj_ev"]),
                    notes=str(r["notes"]),
                )
            )

    def net(item: PortfolioItem) -> float:
        return item.ev_usd - item.cost_usd

    if require_positive_npv:
        pool.sort(key=net, reverse=True)
    else:
        # Rank by EV/cost for gross-EV spend portfolios
        pool.sort(key=lambda i: (i.ev_usd / i.cost_usd) if i.cost_usd > 0 else i.ev_usd, reverse=True)

    port = Portfolio(action_types=list(action_ids))
    type_counts: dict[str, int] = {a: 0 for a in action_ids}
    type_limits = {a: evs[a].quantity_limit for a in action_ids}
    same_machine_groups = {"A-MOVE-MACHINE", "A-MOVE-BANK", "A-REPLACE-LEASE", "A-REPLACE-BUY"}
    machine_claimed: set[str] = set()

    for item in pool:
        if item.action_type_id == "A-HOLD":
            continue
        if require_positive_npv and net(item) <= 0:
            continue
        if not require_positive_npv and item.ev_usd <= 0:
            continue
        if port.total_cost + item.cost_usd > budget + 1e-6:
            continue
        if type_counts[item.action_type_id] >= type_limits[item.action_type_id]:
            continue
        if item.action_type_id in same_machine_groups:
            tid = item.target_id
            if tid in machine_claimed:
                continue
            machine_claimed.add(tid)

        port.items.append(item)
        port.total_cost += item.cost_usd
        port.total_ev += item.ev_usd
        type_counts[item.action_type_id] += 1

    return port


def optimize_portfolio(
    sensitivity: float = 1.0,
    budget: float = BUDGET,
    evs: dict[str, ActionEV] | None = None,
) -> dict:
    """
    Exhaustive search over combinations of ≤3 action types (excluding pure A-HOLD combos
    except as benchmark), greedily filling quantities inside each combo.
    """
    if evs is None:
        evs = build_action_evs()

    actionable = [
        a
        for a in evs.keys()
        if a != "A-HOLD" and a not in DEPRIORITIZE
    ]
    # Also allow deprioritized alone in search but they'll usually lose
    actionable_full = [a for a in evs.keys() if a != "A-HOLD"]

    best = Portfolio()
    best.total_ev = -1e18
    all_ports: list[Portfolio] = []
    aggressive_ports: list[Portfolio] = []

    for k in range(1, MAX_TYPES + 1):
        for combo in itertools.combinations(actionable_full, k):
            if "A-COMP-PILOT" in combo and "A-COMP-CHANGE" in combo:
                continue
            port = _greedy_fill(
                combo, evs, budget=budget, sensitivity=sensitivity, require_positive_npv=True
            )
            if port.items:
                all_ports.append(port)
                best_net = best.total_ev - best.total_cost
                port_net = port.total_ev - port.total_cost
                if port_net > best_net:
                    best = port

            agr = _greedy_fill(
                combo, evs, budget=budget, sensitivity=sensitivity, require_positive_npv=False
            )
            if agr.items:
                aggressive_ports.append(agr)

    hold = Portfolio(
        items=[
            PortfolioItem(
                "A-HOLD",
                "Preserve budget",
                "none",
                "Preserve budget",
                0.0,
                0.0,
                "Keep $100k if no positive EV after uncertainty.",
            )
        ],
        total_cost=0.0,
        total_ev=0.0,
        action_types=["A-HOLD"],
    )

    if best.action_types and best.items:
        stress_port = _greedy_fill(
            tuple(best.action_types), evs, budget=budget, sensitivity=0.5, require_positive_npv=True
        )
    else:
        stress_port = hold

    best_net = best.total_ev - best.total_cost if best.items else -1.0
    stress_net = stress_port.total_ev - stress_port.total_cost if stress_port.items else -1.0

    if best_net <= 0 or not best.items:
        recommend = hold
        stress_note = (
            "No action clears positive July NPV (EV − cost) under catalog lead times; "
            "recommend A-HOLD. Optional spend portfolios below are for discussion only."
        )
    elif stress_net <= 0:
        recommend = hold
        stress_note = (
            f"Best base NPV is ${best_net:,.0f} on {best.action_types}, but at −50% EV "
            f"NPV ≤ 0 → recommend A-HOLD (valid under GUIDE luck/uncertainty rule)."
        )
    else:
        recommend = best
        stress_note = (
            f"At −50% EV, same type set NPV ${stress_net:,.0f} "
            f"(EV ${stress_port.total_ev:,.0f} on cost ${stress_port.total_cost:,.0f})."
        )

    aggressive = (
        max(aggressive_ports, key=lambda p: p.total_ev, default=hold) if aggressive_ports else hold
    )
    ranked = sorted(all_ports, key=lambda p: p.total_ev - p.total_cost, reverse=True)[:15]

    return {
        "recommended": recommend,
        "best_raw": best,
        "aggressive": aggressive,
        "hold": hold,
        "stress_port": stress_port,
        "stress_note": stress_note,
        "top_alternatives": ranked,
        "sensitivity": sensitivity,
        "evs": evs,
    }


def portfolio_to_frame(port: Portfolio) -> pd.DataFrame:
    if not port.items:
        return pd.DataFrame(
            columns=["action_type_id", "name", "target_id", "target_label", "cost_usd", "ev_usd", "notes"]
        )
    return pd.DataFrame(
        [
            {
                "action_type_id": i.action_type_id,
                "name": i.name,
                "target_id": i.target_id,
                "target_label": i.target_label,
                "cost_usd": i.cost_usd,
                "ev_usd": i.ev_usd,
                "notes": i.notes,
            }
            for i in port.items
        ]
    )


def summarize_by_type(port: Portfolio) -> pd.DataFrame:
    df = portfolio_to_frame(port)
    if df.empty:
        return df
    return (
        df.groupby(["action_type_id", "name"], as_index=False)
        .agg(quantity=("target_id", "count"), cost_usd=("cost_usd", "sum"), ev_usd=("ev_usd", "sum"))
        .sort_values("ev_usd", ascending=False)
    )
