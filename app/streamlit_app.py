#!/usr/bin/env python3
"""The Gilded Ace — causal profit dashboard."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import src.attribution as attribution
import src.data as data_mod
import src.experiments as experiments
import src.forecast as forecast
import src.metrics as metrics_mod

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"


def _analysis_token() -> str:
    """Cache key so a long-lived Streamlit process drops stale scorecards."""
    paths = [Path(data_mod.__file__), Path(metrics_mod.__file__), Path(attribution.__file__)]
    return "|".join(f"{path.name}:{path.stat().st_mtime_ns}" for path in paths)


def _file_mtime(path: Path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0

# Visual system
INK = "#1C211E"
MUTED = "#6B736E"
LINE = "#DDDDD6"
SURFACE = "#FFFFFF"
ACCENT = "#2F4A3C"
ACCENT_SOFT = "#E7EEE9"
GOLD = "#9A7B4F"
PLOTLY_COLORS = ["#2F4A3C", "#9A7B4F", "#5B7C6B", "#C4A574", "#1C211E", "#8A9490"]

PAGES = [
    ("Overview", "Overview"),
    ("Machines", "Machines"),
    ("Cause", "Cause"),
    ("Evidence", "Evidence"),
    ("Model", "Model"),
    ("Guests", "Guests"),
    ("Budget", "Budget"),
]

st.set_page_config(
    page_title="Gilded Ace",
    page_icon="♠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap" rel="stylesheet">
        <style>
          html, body, [class*="css"] {
            font-family: "DM Sans", sans-serif;
          }
          .stApp { background: #F6F5F2; }
          [data-testid="stSidebar"] {
            background: #EEEEE9;
            border-right: 1px solid #DDDDD6;
          }
          [data-testid="stSidebar"] * { color: #1C211E !important; }
          .block-container {
            padding: 1.5rem 2rem 3rem;
            max-width: 1080px;
          }
          h1, .page-title {
            font-family: "Fraunces", Georgia, serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
            color: #1C211E !important;
            margin-bottom: 0.2rem !important;
          }
          .page-sub {
            color: #6B736E;
            font-size: 0.95rem;
            margin: 0 0 1.4rem 0;
            line-height: 1.45;
            max-width: 42rem;
          }
          .brand {
            font-family: "Fraunces", Georgia, serif;
            font-size: 1.35rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin: 0.2rem 0 0.15rem;
          }
          .brand-sub {
            color: #6B736E;
            font-size: 0.78rem;
            margin-bottom: 1.2rem;
          }
          div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #DDDDD6;
            border-radius: 10px;
            padding: 0.85rem 1rem 0.7rem;
          }
          div[data-testid="stMetricLabel"] { color: #6B736E !important; }
          div[data-testid="stMetricValue"] {
            font-variant-numeric: tabular-nums;
            color: #1C211E !important;
            font-size: 1.45rem !important;
          }
          .callout {
            background: #E7EEE9;
            border-left: 3px solid #2F4A3C;
            border-radius: 0 8px 8px 0;
            padding: 0.85rem 1rem;
            color: #1C211E;
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0.4rem 0 1.2rem;
          }
          .section-label {
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #6B736E;
            font-weight: 600;
            margin: 1.4rem 0 0.55rem;
          }
          [data-testid="stHorizontalBlock"] > div { gap: 0.75rem; }
          hr { border: none; border-top: 1px solid #DDDDD6; margin: 1.2rem 0; }
          .stDataFrame { border: 1px solid #DDDDD6; border-radius: 10px; overflow: hidden; }
          #MainMenu { visibility: hidden; }
          footer { visibility: hidden; }
          header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=24, r=16, t=36, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=INK, size=12),
        title=dict(font=dict(size=14, color=INK, family="DM Sans, sans-serif")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        colorway=PLOTLY_COLORS,
        xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED)),
    )
    return fig


def header(title: str, subtitle: str) -> None:
    st.markdown(f"<h1 class='page-title'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='page-sub'>{subtitle}</p>", unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f"<div class='callout'>{text}</div>", unsafe_allow_html=True)


def money(x: float) -> str:
    return f"${x:,.0f}"


@st.cache_data(show_spinner=False)
def load_scorecard(start: str, end: str, analysis_token: str) -> pd.DataFrame:
    # The token is the cache key. Reload so this process runs the code on disk.
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
def load_csv(name: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    path = OUT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_forecast() -> dict:
    importlib.reload(forecast)
    return forecast.run_forecast()


@st.cache_data(show_spinner=False)
def load_moves() -> dict:
    importlib.reload(experiments)
    return experiments.move_experiment()


@st.cache_data(show_spinner=False)
def load_repairs() -> dict:
    importlib.reload(experiments)
    return experiments.repair_downtime_experiment()


def _dispatch(page: str, summary: dict, scorecard: pd.DataFrame) -> None:
    if page == "Overview":
        page_overview(summary, scorecard)
    elif page == "Machines":
        page_triage(scorecard)
    elif page == "Cause":
        page_cause(scorecard)
    elif page == "Evidence":
        page_evidence()
    elif page == "Model":
        page_model()
    elif page == "Guests":
        page_guests()
    else:
        page_budget(summary)


def main() -> None:
    inject_css()
    summary = load_summary(_file_mtime(OUT / "summary.json"))

    with st.sidebar:
        st.markdown("<div class='brand'>Gilded Ace</div>", unsafe_allow_html=True)
        st.markdown("<div class='brand-sub'>Floor analytics · 2025</div>", unsafe_allow_html=True)
        page = st.radio("Section", [p[0] for p in PAGES], label_visibility="collapsed")
        st.markdown("---")
        start = st.date_input("From", value=pd.Timestamp("2025-01-01"))
        end = st.date_input("To", value=pd.Timestamp("2025-06-30"))

    if start > end:
        st.error("Start must be before end.")
        return

    scorecard = load_scorecard(str(start), str(end), _analysis_token())
    _dispatch(page, summary, scorecard)


def page_overview(summary: dict, scorecard: pd.DataFrame) -> None:
    finding = summary.get("finding", {})
    n_b40 = finding.get("naive_in_back_forty", 9)
    header(
        "Don't replace the bottom 10",
        f"{n_b40} of the 10 lowest house-win machines sit in Back Forty. They are available almost all the time. The corner is quiet. The boxes are not broken.",
    )
    rec = summary.get("recommended", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Do this", "Signs in Back Forty")
    c2.metric("Cost", money(rec.get("total_cost", 7500)))
    c3.metric("Hold back", money(rec.get("unspent", 92500)))
    c4.metric("Base EV", money(rec.get("total_ev", 0)))

    if finding:
        callout(
            f"M-1158, on the GM's list, missed theo by {money(abs(finding.get('m1158_luck', 0)))}. "
            f"Its theo per hour (${finding.get('m1158_theo_hr', 0):.2f}) matches Back Forty "
            f"(${finding.get('m1158_zone_theo_hr', 0):.2f}). That is luck. "
            f"Signs are a ${finding.get('wayfinding_cost', 7500):,.0f} test: "
            f"NPV {money(finding.get('base_npv', 0))} if they close 8% of the zone gap, "
            f"and {money(finding.get('kill_npv', 0))} at 5%. Kill the idea if July misses 5%."
        )

    by_type = load_csv("recommended_portfolio_by_type.csv", _file_mtime(OUT / "recommended_portfolio_by_type.csv"))
    opt = load_csv("optional_spend_by_type.csv", _file_mtime(OUT / "optional_spend_by_type.csv"))

    left, right = st.columns((1.1, 1), gap="large")
    with left:
        section("Primary plan")
        if not by_type.empty:
            show = by_type.rename(
                columns={
                    "action_type_id": "Action",
                    "name": "Name",
                    "quantity": "Qty",
                    "cost_usd": "Cost",
                    "ev_usd": "EV",
                }
            )
            st.dataframe(show, use_container_width=True, hide_index=True)
        section("Rejected")
        st.markdown(
            "Replace, promo, technician, and moves stay unfunded. "
            "A new machine in Back Forty inherits the corner. "
            "Weekends are already busy. July lead time is too short for a tech hire."
        )

    section("Same game, different corner")
    st.caption("If the box were the problem, a title would earn the same theo per hour everywhere.")
    focus_zones = ["Back Forty", "Bar Row", "Center Pit", "Main Entrance"]
    themed = (
        scorecard[scorecard["zone_name"].isin(focus_zones)]
        .groupby(["theme", "zone_name"], as_index=False)["theo_per_avail_hr"]
        .median()
    )
    present = themed.groupby("theme")["zone_name"].nunique()
    keep = present[present >= 3].index
    themed = themed[themed["theme"].isin(keep)]
    fig = px.bar(
        themed,
        x="theme",
        y="theo_per_avail_hr",
        color="zone_name",
        barmode="group",
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"theo_per_avail_hr": "Theo / hour", "theme": "", "zone_name": ""},
    )
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)


def page_triage(scorecard: pd.DataFrame) -> None:
    header(
        "Machine triage",
        "Raw house win vs exposure-adjusted structural weakness.",
    )
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

    cols = [
        "machine_id",
        "zone_id",
        "theme",
        "house_win_usd",
        "theo_per_avail_hr",
        "luck_residual_usd",
        "availability_ratio",
        "primary_cause",
    ]
    a, b = st.columns(2, gap="large")
    with a:
        section("Naive bottom 10")
        st.dataframe(view.nsmallest(10, "house_win_usd")[cols], use_container_width=True, hide_index=True)
    with b:
        section("Structural bottom 10")
        st.dataframe(view.nlargest(10, "structural_weakness")[cols], use_container_width=True, hide_index=True)

    section("Theo per hour vs luck")
    fig = px.scatter(
        view,
        x="theo_per_avail_hr",
        y="luck_residual_usd",
        color="primary_cause",
        hover_data=["machine_id", "zone_id", "house_win_usd"],
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"theo_per_avail_hr": "Theo / available hour", "luck_residual_usd": "Luck residual ($)"},
    )
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)


def page_cause(scorecard: pd.DataFrame) -> None:
    header("Cause lab", "Luck, location, downtime, or a high-RTP mix. Most machines are none of these.")
    mid = st.selectbox("Machine", sorted(scorecard["machine_id"].tolist()))
    row = scorecard[scorecard["machine_id"] == mid].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Primary", str(row["primary_cause"]))
    c2.metric("Alternative", str(row["alternative_cause"]))
    c3.metric("Action", str(row["preferred_action"]))
    c4.metric("House win", money(float(row["house_win_usd"])))

    callout(str(row["cause_note"]))

    scores = pd.DataFrame(
        {
            "cause": ["luck", "location", "leak", "mix"],
            "score": [row["score_luck"], row["score_location"], row["score_leak"], row["score_mix"]],
        }
    )
    left, right = st.columns((1.1, 1), gap="large")
    with left:
        section("Evidence scores")
        fig = px.bar(scores, x="cause", y="score", color_discrete_sequence=[ACCENT])
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(style_fig(fig, 280), use_container_width=True)
    with right:
        section("Key metrics")
        metrics = pd.DataFrame(
            {
                "Metric": [
                    "Theo / avail hr",
                    "Zone peer theo/hr",
                    "Availability",
                    "Occupancy",
                    "Luck residual",
                    "Lost theo",
                    "Faults",
                    "Location coin gap",
                ],
                "Value": [
                    f"{row['theo_per_avail_hr']:.2f}",
                    f"{row['zone_theo_per_hr_med']:.2f}",
                    f"{row['availability_ratio']:.1%}",
                    f"{row['occupancy_ratio']:.1%}",
                    money(float(row["luck_residual_usd"])),
                    money(float(row["lost_theo_usd"])),
                    f"{int(row['faults_opened'])}",
                    f"{row['location_coin_gap']:.2f}",
                ],
            }
        )
        st.dataframe(metrics, use_container_width=True, hide_index=True)


def page_evidence() -> None:
    header(
        "Natural experiments",
        "Uplifts from placement moves and fault-repair downtime — not catalog promises.",
    )
    moves = load_moves()
    repairs = load_repairs()

    section("Placement moves")
    m1, m2, m3 = st.columns(3)
    m1.metric("Moves", moves["n_moves"])
    m2.metric("Shrunk Δ theo/hr", f"${moves['shrunk_delta_theo_per_hr']:.2f}")
    m3.metric("CI low / high", f"${moves['ci_low']:.2f} / ${moves['ci_high']:.2f}")
    st.caption(moves["note"])

    detail = moves["detail"]
    if detail is not None and not detail.empty:
        fig = px.bar(
            detail,
            x="machine_id",
            y="delta_theo_per_hr",
            color_discrete_sequence=[ACCENT],
            labels={"delta_theo_per_hr": "Δ theo $/hr", "machine_id": ""},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    section("Fault repairs")
    r1, r2, r3 = st.columns(3)
    r1.metric("Faults", f"{repairs['n_faults']:,}")
    r2.metric("Lost theo", money(repairs["total_lost_theo_usd"]))
    r3.metric("Per day", money(repairs["daily_lost_theo_rate"]))
    st.caption(repairs["note"])

    by_mfr = repairs["by_manufacturer"]
    fig = px.bar(
        by_mfr,
        x="manufacturer",
        y="lost_theo_usd",
        color_discrete_sequence=[GOLD],
        labels={"lost_theo_usd": "Lost theo ($)", "manufacturer": ""},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig, 300), use_container_width=True)


def page_model() -> None:
    header(
        "Demand model",
        "Weekday mean vs constant mean. Train Jan–Apr, holdout May–Jun.",
    )
    fc = load_forecast()
    a, b, c, d = st.columns(4)
    a.metric("Flat mean", money(fc["mae_baseline"]))
    b.metric("Weekday", money(fc["mae_weekday"]))
    c.metric("Weekday + events", money(fc.get("mae_event", fc["mae_weekday"])))
    d.metric("Weekend lift", money(fc["weekend_lift"]))
    callout(fc["decision"])

    hold = fc["holdout"].copy()
    hold["gaming_date"] = pd.to_datetime(hold["gaming_date"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hold["gaming_date"],
            y=hold["coin_in_usd"],
            name="Actual",
            mode="lines",
            line=dict(color=INK, width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hold["gaming_date"],
            y=hold["pred_weekday"],
            name="Weekday",
            mode="lines",
            line=dict(color=ACCENT, width=1.6),
        )
    )
    if "pred_event" in hold.columns:
        fig.add_trace(
            go.Scatter(
                x=hold["gaming_date"],
                y=hold["pred_event"],
                name="Weekday + events",
                mode="lines",
                line=dict(color=GOLD, width=1.2, dash="dot"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=hold["gaming_date"],
            y=hold["pred_baseline"],
            name="Flat mean",
            mode="lines",
            line=dict(color=MUTED, width=1.1, dash="dot"),
        )
    )
    fig.update_layout(yaxis_title="Coin-in ($)", xaxis_title="")
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    means = pd.DataFrame(
        {"weekday": list(fc["weekday_means"].keys()), "coin_in": list(fc["weekday_means"].values())}
    )
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    means["weekday"] = pd.Categorical(means["weekday"], categories=order, ordered=True)
    means = means.sort_values("weekday")
    section("Training weekday means")
    fig2 = px.bar(means, x="weekday", y="coin_in", color_discrete_sequence=[ACCENT])
    fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="Coin-in ($)")
    st.plotly_chart(style_fig(fig2, 280), use_container_width=True)


def page_guests() -> None:
    header("Guests & comps", "Carded play only — will not match machine meters.")
    from src.data import player_period, property_day

    pp = player_period().copy()
    for col in ("tracked_coin_in_usd", "tracked_net_usd", "comp_cost_usd", "theo_usd"):
        pp[col] = pd.to_numeric(pp[col], errors="coerce")
    g1, g2 = st.columns(2)
    ages = ["All"] + sorted(pp["age_band"].dropna().unique().tolist())
    regions = ["All"] + sorted(pp["home_region"].dropna().unique().tolist())
    with g1:
        age = st.selectbox("Age band", ages)
    with g2:
        region = st.selectbox("Home region", regions)
    view = pp
    if age != "All":
        view = view[view["age_band"] == age]
    if region != "All":
        view = view[view["home_region"] == region]
    k1, k2, k3 = st.columns(3)
    k1.metric("Carded coin-in", money(float(view["tracked_coin_in_usd"].sum())))
    k2.metric("Carded net", money(float(view["tracked_net_usd"].sum())))
    k3.metric("Comp cost", money(float(view["comp_cost_usd"].sum())))
    coin_col = "tracked_coin_in_usd"
    pp = view
    g = (
        pp.groupby("player_id", as_index=False)[coin_col]
        .sum()
        .sort_values(coin_col, ascending=False)
    )
    g["share"] = g[coin_col] / g[coin_col].sum()
    g["cum_share"] = g["share"].cumsum()
    top1 = float(g.head(max(1, int(len(g) * 0.01)))["share"].sum())
    top10 = float(g.head(max(1, int(len(g) * 0.10)))["share"].sum())

    c1, c2 = st.columns(2)
    c1.metric("Top 1% share", f"{top1:.1%}")
    c2.metric("Top 10% share", f"{top10:.1%}")

    section("Concentration")
    fig = px.line(
        g.reset_index(drop=True).reset_index(),
        x="index",
        y="cum_share",
        labels={"index": "Players ranked", "cum_share": "Cumulative share"},
    )
    fig.update_traces(line_color=ACCENT)
    st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    prop = property_day().copy()
    prop["comp_cost_usd"] = pd.to_numeric(prop["comp_cost_usd"], errors="coerce")
    prop["theo_usd"] = pd.to_numeric(prop["theo_usd"], errors="coerce")
    prop["gaming_date"] = pd.to_datetime(prop["gaming_date"])
    prop["month"] = prop["gaming_date"].dt.to_period("M").astype(str)
    monthly = prop.groupby("month", as_index=False).agg(
        comp_cost_usd=("comp_cost_usd", "sum"), theo_usd=("theo_usd", "sum")
    )
    monthly["comp_reinvestment"] = monthly["comp_cost_usd"] / monthly["theo_usd"].replace(0, pd.NA)

    section("Comp reinvestment")
    st.caption("Comp cost ÷ theo. Comp cost is already in POS cogs — subtract once.")
    fig2 = px.line(monthly, x="month", y="comp_reinvestment", markers=True)
    fig2.update_traces(line_color=GOLD, marker_color=GOLD)
    fig2.update_layout(xaxis_title="", yaxis_title="Comp / theo")
    st.plotly_chart(style_fig(fig2, 280), use_container_width=True)


def page_budget(summary: dict) -> None:
    header("Budget", "Catalog prices with July lead-time costs. Horizon: 30 days.")
    from src.data import action_catalog
    from src.uplift import action_cost

    cat = action_catalog().copy()
    cat["unit_cost_usd"] = cat["unit_cost_cents"] / 100.0
    cat["recurring_usd"] = cat["recurring_cost_cents_per_day"].fillna(0) / 100.0
    cat["july_unit_cost"] = cat.apply(
        lambda r: action_cost(r["unit_cost_usd"], r["recurring_usd"], int(r["lead_days"]), 1),
        axis=1,
    )
    section("Action catalog")
    st.dataframe(
        cat[
            [
                "action_type_id",
                "name",
                "july_unit_cost",
                "lead_days",
                "quantity_limit",
                "incompatibility_group",
            ]
        ].rename(
            columns={
                "action_type_id": "ID",
                "name": "Action",
                "july_unit_cost": "July cost",
                "lead_days": "Lead",
                "quantity_limit": "Limit",
                "incompatibility_group": "Conflict",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    section("Frozen recommendation")
    by_type = load_csv("recommended_portfolio_by_type.csv", _file_mtime(OUT / "recommended_portfolio_by_type.csv"))
    if by_type.empty:
        st.write("Run `python analysis/run_pipeline.py` first.")
        return
    st.dataframe(
        by_type.rename(
            columns={
                "action_type_id": "Action",
                "name": "Name",
                "quantity": "Qty",
                "cost_usd": "Cost",
                "ev_usd": "EV",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    remaining = 100_000 - float(summary.get("recommended", {}).get("total_cost", 0))
    st.metric("Unspent", money(remaining))


if __name__ == "__main__":
    main()
