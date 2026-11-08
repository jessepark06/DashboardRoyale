#!/usr/bin/env python3
"""The Gilded Ace — a short dashboard for the July decision."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

import src.attribution as attribution
import src.data as data_mod
import src.forecast as forecast
import src.metrics as metrics_mod

INK = "#1C211E"
MUTED = "#6B736E"
GOLD = "#9A7B4F"

PAGES = ["Decision", "Machines", "Checks"]
CAUSE_LABEL = {
    "performing": "Fine",
    "location": "Quiet spot",
    "leak": "Downtime",
    "mix": "High payout",
    "luck": "Luck",
}

st.set_page_config(page_title="Gilded Ace", page_icon="♠", layout="wide", initial_sidebar_state="expanded")


def _analysis_token() -> str:
    paths = [Path(data_mod.__file__), Path(metrics_mod.__file__), Path(attribution.__file__)]
    return "|".join(f"{path.name}:{path.stat().st_mtime_ns}" for path in paths)


def _file_mtime(path: Path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


def inject_css() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap" rel="stylesheet">
        <style>
          html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
          .stApp { background: #F7F6F3; }
          [data-testid="stSidebar"] { background: #F7F6F3; border-right: 1px solid #E4E3DE; }
          [data-testid="stSidebar"] * { color: #1C211E !important; }
          .block-container { padding: 2.2rem 2.4rem 4rem; max-width: 880px; }
          h1 {
            font-weight: 600 !important;
            letter-spacing: -0.03em;
            color: #1C211E !important;
            font-size: 1.85rem !important;
            margin-bottom: 0.25rem !important;
          }
          .page-sub { color: #6B736E; font-size: 1rem; line-height: 1.5; margin: 0 0 1.6rem; max-width: 38rem; }
          .brand { font-size: 1.15rem; font-weight: 600; letter-spacing: -0.02em; margin: 0.4rem 0 0.1rem; }
          .brand-sub { color: #6B736E; font-size: 0.8rem; margin-bottom: 1.4rem; }
          div[data-testid="stMetric"] {
            background: transparent;
            border: none;
            padding: 0 0 0.2rem;
          }
          div[data-testid="stMetricLabel"] { color: #6B736E !important; font-size: 0.78rem !important; }
          div[data-testid="stMetricValue"] {
            font-variant-numeric: tabular-nums;
            color: #1C211E !important;
            font-size: 1.55rem !important;
            font-weight: 600 !important;
          }
          .note {
            color: #1C211E;
            font-size: 0.98rem;
            line-height: 1.55;
            margin: 0.2rem 0 1.4rem;
            max-width: 40rem;
          }
          .quiet { color: #6B736E; font-size: 0.92rem; line-height: 1.5; margin: 0.3rem 0 0; }
          .proofs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem; margin: 0.2rem 0 1.6rem; }
          .proof-num {
            font-size: 2.4rem;
            font-weight: 600;
            letter-spacing: -0.03em;
            line-height: 1.05;
            color: #1C211E;
          }
          .proof-num em { font-style: normal; color: #9A7B4F; }
          .proof-label { color: #6B736E; font-size: 0.92rem; line-height: 1.35; margin-top: 0.35rem; max-width: 14rem; }
          #MainMenu, footer, header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart(fig: go.Figure, height: int = 280) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=INK, size=13),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=INK)),
    )
    return fig


def money(x: float) -> str:
    return f"${x:,.0f}"


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


@st.cache_data(show_spinner=False)
def load_scorecard(start: str, end: str, analysis_token: str) -> pd.DataFrame:
    del analysis_token
    importlib.reload(data_mod)
    importlib.reload(metrics_mod)
    importlib.reload(attribution)
    return attribution.attribute_causes(metrics_mod.machine_scorecard(start=start, end=end))


@st.cache_data(show_spinner=False)
def load_summary(mtime_ns: int) -> dict:
    del mtime_ns
    path = OUT / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_forecast() -> dict:
    importlib.reload(forecast)
    return forecast.run_forecast()


def main() -> None:
    inject_css()
    summary = load_summary(_file_mtime(OUT / "summary.json"))
    with st.sidebar:
        st.markdown("<div class='brand'>Gilded Ace</div>", unsafe_allow_html=True)
        st.markdown("<div class='brand-sub'>January–June 2025</div>", unsafe_allow_html=True)
        page = st.radio("Section", PAGES, label_visibility="collapsed")
        st.markdown("---")
        start = st.date_input("From", value=pd.Timestamp("2025-01-01"))
        end = st.date_input("To", value=pd.Timestamp("2025-06-30"))

    if start > end:
        st.error("Start must be on or before end.")
        return

    scorecard = load_scorecard(str(start), str(end), _analysis_token())
    if page == "Decision":
        page_decision(summary, scorecard)
    elif page == "Machines":
        page_machines(scorecard)
    else:
        page_checks(summary)


def page_decision(summary: dict, scorecard: pd.DataFrame) -> None:
    finding = summary.get("finding", {})
    rec = summary.get("recommended", {})
    st.markdown("<h1>Don't replace the bottom 10</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-sub'>Three numbers. The machines are on. The corner is quiet. One of them was unlucky.</p>",
        unsafe_allow_html=True,
    )

    if scorecard.empty:
        st.info("No machines in this date range.")
        return

    bottom = scorecard.nsmallest(min(10, len(scorecard)), "house_win_usd")
    n_b40 = int((bottom["zone_id"] == "Z-B40").sum())
    avail = float(bottom["availability_ratio"].mean())
    hit = scorecard[scorecard["machine_id"] == "M-1158"]
    luck = None if hit.empty else abs(float(hit.iloc[0]["luck_residual_usd"]))

    luck_num = "Outside this window" if luck is None else f"−{money(luck)}"
    luck_label = (
        "M-1158 is not on the floor in this window."
        if luck is None
        else "M-1158 missed theo. Its hourly rate still matches the corner. That is luck."
    )
    st.markdown(
        f"""
        <div class="proofs">
          <div>
            <div class="proof-num"><em>{n_b40}</em> of {len(bottom)}</div>
            <div class="proof-label">Lowest house-win machines sit in Back Forty.</div>
          </div>
          <div>
            <div class="proof-num"><em>{avail:.1%}</em></div>
            <div class="proof-label">Of the time those machines were actually on.</div>
          </div>
          <div>
            <div class="proof-num"><em>{luck_num}</em></div>
            <div class="proof-label">{luck_label}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    zones = ["Back Forty", "Bar Row", "Center Pit", "Main Entrance"]
    if not scorecard.empty and "zone_name" in scorecard.columns:
        med = (
            scorecard[scorecard["zone_name"].isin(zones)]
            .groupby("zone_name", as_index=False)["theo_per_avail_hr"]
            .median()
            .sort_values("theo_per_avail_hr")
        )
        colors = [GOLD if name == "Back Forty" else "#C9C8C2" for name in med["zone_name"]]
        fig = go.Figure(
            go.Bar(
                x=med["theo_per_avail_hr"],
                y=med["zone_name"],
                orientation="h",
                marker_color=colors,
                text=[f"${v:.0f}/hr" for v in med["theo_per_avail_hr"]],
                textposition="outside",
                textfont=dict(size=16, color=INK),
                cliponaxis=False,
            )
        )
        fig.update_layout(xaxis_title="", yaxis_title="", xaxis_range=[0, med["theo_per_avail_hr"].max() * 1.35])
        show(chart(fig, 240))
        st.markdown(
            "<p class='quiet'>Theo per hour. Same kind of play, much less of it in Back Forty.</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<p class='note'>So buy signs for Back Forty, {money(rec.get('total_cost', 7500))}, "
        f"and hold {money(rec.get('unspent', 92500))}. "
        f"If the signs recover 8% of that gap, they return about {money(finding.get('base_ev', rec.get('total_ev', 0)))}. "
        f"If they recover less than 5%, stop.</p>",
        unsafe_allow_html=True,
    )


def page_machines(scorecard: pd.DataFrame) -> None:
    st.markdown("<h1>Machines</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-sub'>House win is the list the general manager is using. Theo per hour and availability show whether the box itself is the problem.</p>",
        unsafe_allow_html=True,
    )
    if scorecard.empty:
        st.info("No machines in this date range.")
        return

    f1, f2 = st.columns(2)
    zones = ["All"] + sorted(scorecard["zone_id"].dropna().astype(str).unique().tolist())
    with f1:
        zone = st.selectbox("Zone", zones)
    view = scorecard if zone == "All" else scorecard[scorecard["zone_id"] == zone]
    banks = ["All"] + sorted(view["bank_id"].dropna().astype(str).unique().tolist())
    with f2:
        bank = st.selectbox("Bank", banks)
    if bank != "All":
        view = view[view["bank_id"] == bank]
    if view.empty:
        st.info("No machines in this zone and bank.")
        return

    bottom = view.nsmallest(10, "house_win_usd").copy()
    table = pd.DataFrame(
        {
            "Machine": bottom["machine_id"],
            "Zone": bottom["zone_id"],
            "House win": bottom["house_win_usd"].map(money),
            "Theo / hour": bottom["theo_per_avail_hr"].map(lambda v: f"${v:.2f}"),
            "Up": bottom["availability_ratio"].map(lambda v: f"{v:.0%}"),
            "Why": bottom["primary_cause"].map(lambda v: CAUSE_LABEL.get(v, v)),
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)

    ids = sorted(view["machine_id"].tolist())
    default = ids.index("M-1158") if "M-1158" in ids else 0
    mid = st.selectbox("One machine", ids, index=default)
    row = view[view["machine_id"] == mid].iloc[0]
    c1, c2 = st.columns(2)
    c1.metric("Luck", money(float(row["luck_residual_usd"])))
    c2.metric("Payout rate", f"{float(row['configured_rtp']):.0%}")
    st.markdown(f"<p class='note'>{row['cause_note']}</p>", unsafe_allow_html=True)


def page_checks(summary: dict) -> None:
    st.markdown("<h1>Checks</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-sub'>The forecast, the guests, and the reasons the rest of the budget stays put.</p>",
        unsafe_allow_html=True,
    )
    fc = load_forecast()
    st.markdown(
        f"""
        <div class="proofs">
          <div>
            <div class="proof-num"><em>{money(fc['mae_weekday'])}</em></div>
            <div class="proof-label">Error if you use the weekday pattern.</div>
          </div>
          <div>
            <div class="proof-num">{money(fc['mae_baseline'])}</div>
            <div class="proof-label">Error if you use one flat average.</div>
          </div>
          <div>
            <div class="proof-num">{money(fc['weekend_lift'])}</div>
            <div class="proof-label">Extra coin-in a weekend day already brings in. Do not buy a promo.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='quiet'>May–June, after training on January–April. The smaller error is the one that earns the decision.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<h1 style='font-size:1.25rem !important; margin-top:2rem;'>Guests</h1>", unsafe_allow_html=True)
    from src.data import player_period, property_day

    players = player_period().copy()
    for col in ("tracked_coin_in_usd", "tracked_net_usd", "comp_cost_usd"):
        players[col] = pd.to_numeric(players[col], errors="coerce")
    g1, g2 = st.columns(2)
    ages = ["All"] + sorted(players["age_band"].dropna().unique().tolist())
    regions = ["All"] + sorted(players["home_region"].dropna().unique().tolist())
    with g1:
        age = st.selectbox("Age", ages)
    with g2:
        region = st.selectbox("Home region", regions)
    view = players
    if age != "All":
        view = view[view["age_band"] == age]
    if region != "All":
        view = view[view["home_region"] == region]
    if view.empty:
        st.info("No carded play in this group.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Coin-in", money(float(view["tracked_coin_in_usd"].sum())))
        k2.metric("Net", money(float(view["tracked_net_usd"].sum())))
        k3.metric("Comps", money(float(view["comp_cost_usd"].sum())))
        by_player = view.groupby("player_id", as_index=False)["tracked_coin_in_usd"].sum().sort_values(
            "tracked_coin_in_usd", ascending=False
        )
        total = float(by_player["tracked_coin_in_usd"].sum()) or 1.0
        top1 = float(by_player.head(max(1, int(len(by_player) * 0.01)))["tracked_coin_in_usd"].sum()) / total
        top10 = float(by_player.head(max(1, int(len(by_player) * 0.10)))["tracked_coin_in_usd"].sum()) / total
        st.markdown(
            f"<p class='quiet'>Carded play only. The top 1% of these players are {top1:.0%} of the coin-in. The top 10% are {top10:.0%}.</p>",
            unsafe_allow_html=True,
        )

    prop = property_day().copy()
    prop["comp_cost_usd"] = pd.to_numeric(prop["comp_cost_usd"], errors="coerce")
    prop["theo_usd"] = pd.to_numeric(prop["theo_usd"], errors="coerce")
    comp = float(prop["comp_cost_usd"].sum())
    theo = float(prop["theo_usd"].sum()) or 1.0
    st.markdown(
        f"<p class='quiet'>On the whole floor, comps are about {comp / theo:.0%} of theo. That cost is already in the books once.</p>",
        unsafe_allow_html=True,
    )

    repair = summary.get("repair_experiment", {})
    move = summary.get("move_experiment", {})
    st.markdown("<h1 style='font-size:1.25rem !important; margin-top:2rem;'>Why not the rest</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p class='note'>Downtime cost about {money(repair.get('total_lost_theo_usd', 0))} over six months, "
        f"about {money(repair.get('daily_lost_theo_rate', 0))} a day. "
        f"Only {move.get('n_moves', 0)} machines were moved, and the gain was about "
        f"${move.get('shrunk_delta_theo_per_hr', 0):.0f} of theo per hour. "
        f"Neither pays back inside July.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
