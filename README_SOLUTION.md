# The Gilded Ace — Causal Profit Solution

Competition solution for **Dashboard Royale**: diagnose luck vs location vs leak, estimate action uplifts from natural experiments, then allocate ≤3 action types under $100k.

## Setup

```sh
cd DashboardRoyale
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce analysis

```sh
python analysis/run_pipeline.py
```

Writes frozen artifacts to `outputs/` (portfolio, forecast holdout, claims appendix, write-up draft).

## Launch dashboard

```sh
streamlit run app/streamlit_app.py
```

## Layout

| Path | Role |
| --- | --- |
| `src/metrics.py` | Fair machine scorecard |
| `src/attribution.py` | Luck / location / leak / config |
| `src/experiments.py` | Move + repair quasi-experiments |
| `src/forecast.py` | Weekday coin-in vs mean baseline |
| `src/uplift.py` | Evidence → action EVs |
| `src/optimize.py` | ≤3 types, $100k portfolio search |
| `app/streamlit_app.py` | Interactive dashboard |
| `outputs/` | Submission-ready tables + markdown |

## The decision

Nine of the ten machines with the lowest house win sit in **Back Forty** and were available about **99.7%** of the time. **M-1158** missed theo by about **$36,000** while earning a normal theo per hour for that zone. That is luck and location, not a broken box.

**Spend $7,500 on wayfinding for Back Forty. Hold $92,500.** Kill the pilot if July closes less than 5% of the zone's theo-per-hour gap versus Main Entrance and Center Pit.

## Judging walkthrough

1. **Overview** — the Back Forty finding and the $7,500 decision  
2. **Machines** — naive bottom 10 vs structural list; zone and bank filters  
3. **Cause** — one machine, competing explanations  
4. **Evidence** — moves, lost theo, Ridgeline vs Halcyon  
5. **Model** — May–June MAE vs a flat mean; do not buy a promo  
6. **Guests** — age and region drill, concentration, comps  
7. **Budget** — catalog lead times and the frozen plan  
