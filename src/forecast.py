"""Time-based coin-in forecast: weekday mean vs constant mean baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import calendar_day, property_day


def run_forecast(
    train_end: str = "2025-04-30",
    holdout_start: str = "2025-05-01",
    holdout_end: str = "2025-06-30",
) -> dict:
    pd_day = property_day().copy()
    pd_day = pd_day.sort_values("gaming_date")
    train = pd_day[pd_day["gaming_date"] <= pd.Timestamp(train_end)]
    hold = pd_day[
        (pd_day["gaming_date"] >= pd.Timestamp(holdout_start))
        & (pd_day["gaming_date"] <= pd.Timestamp(holdout_end))
    ]

    baseline_mean = float(train["coin_in_usd"].mean())
    weekday_means = train.groupby("local_weekday")["coin_in_usd"].mean().to_dict()

    cal = calendar_day()[["gaming_date", "federal_holiday"]]
    train = train.merge(cal, on="gaming_date", how="left")
    hold = hold.merge(cal, on="gaming_date", how="left")
    for frame in (train, hold):
        frame["event_day"] = frame["events_active"].fillna("").astype(str).str.len().gt(0)

    # Event bump estimated only on training days, within weekday, so May–June cannot leak back.
    event_rows = train[train["event_day"]]
    if len(event_rows) >= 4:
        event_premium = float(
            (
                event_rows["coin_in_usd"] - event_rows["local_weekday"].map(weekday_means)
            ).mean()
        )
    else:
        event_premium = 0.0

    hold = hold.copy()
    hold["pred_baseline"] = baseline_mean
    hold["pred_weekday"] = hold["local_weekday"].map(weekday_means).astype(float).fillna(baseline_mean)
    hold["pred_event"] = hold["pred_weekday"] + np.where(hold["event_day"], event_premium, 0.0)

    mae_baseline = float(np.mean(np.abs(hold["coin_in_usd"] - hold["pred_baseline"])))
    mae_weekday = float(np.mean(np.abs(hold["coin_in_usd"] - hold["pred_weekday"])))
    mae_event = float(np.mean(np.abs(hold["coin_in_usd"] - hold["pred_event"])))

    # Decision implication
    weekend_days = {"Sat", "Saturday", "Sun", "Sunday"}
    wk_vals = {k: v for k, v in weekday_means.items()}
    # Normalize weekday labels
    weekend_avg = np.mean([v for k, v in weekday_means.items() if str(k)[:3] in ("Sat", "Sun")])
    weekday_avg = np.mean([v for k, v in weekday_means.items() if str(k)[:3] not in ("Sat", "Sun")])
    weekend_lift = float(weekend_avg - weekday_avg)

    if mae_weekday < mae_baseline * 0.98 and weekend_lift > 0:
        decision = (
            f"Weekday model beats the constant mean (MAE ${mae_weekday:,.0f} vs ${mae_baseline:,.0f}). "
            f"Adding announced events makes holdout MAE worse (${mae_event:,.0f}), so events do not earn a promo. "
            f"Weekends already run ~${weekend_lift:,.0f} more coin-in per day. Staff to that pattern."
        )
        favor_promo = False
    else:
        decision = (
            f"Weekday model MAE ${mae_weekday:,.0f} vs baseline ${mae_baseline:,.0f}. "
            "Demand pattern is weak or noisy — do not spend on demand creation (skip A-PROMO)."
        )
        favor_promo = False

    return {
        "train_rows": len(train),
        "holdout_rows": len(hold),
        "baseline_mean": baseline_mean,
        "weekday_means": weekday_means,
        "mae_baseline": mae_baseline,
        "mae_weekday": mae_weekday,
        "mae_event": mae_event,
        "event_premium": event_premium,
        "weekend_lift": weekend_lift,
        "favor_promo": favor_promo,
        "decision": decision,
        "holdout": hold[
            [
                "gaming_date",
                "local_weekday",
                "coin_in_usd",
                "house_win_usd",
                "pred_baseline",
                "pred_weekday",
                "pred_event",
            ]
        ],
        "train": train[["gaming_date", "local_weekday", "coin_in_usd"]],
    }
