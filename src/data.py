"""Load Dashboard Royale parquet tables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _read(rel: str) -> pd.DataFrame:
    path = DATA / rel
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    # Parquet decimals → float for arithmetic
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or str(df[col].dtype).startswith("decimal"):
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass
        if hasattr(df[col].dtype, "name") and "decimal" in str(df[col].dtype).lower():
            df[col] = df[col].astype(float)
    # Explicit cast for common money/ratio columns
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(3)
        if len(sample) and all(hasattr(v, "as_tuple") for v in sample):
            df[col] = df[col].astype(float)
    return df


@lru_cache(maxsize=None)
def prepared_daily() -> pd.DataFrame:
    df = _read("1_prepared_views/prepared_daily_view.parquet")
    df["gaming_date"] = pd.to_datetime(df["gaming_date"]).dt.normalize()
    return df


@lru_cache(maxsize=None)
def property_day() -> pd.DataFrame:
    df = _read("1_prepared_views/property_day.parquet")
    df["gaming_date"] = pd.to_datetime(df["gaming_date"]).dt.normalize()
    return df


@lru_cache(maxsize=None)
def player_period() -> pd.DataFrame:
    return _read("1_prepared_views/player_period.parquet")


@lru_cache(maxsize=None)
def calendar_day() -> pd.DataFrame:
    df = _read("2_machine_decision/calendar_day.parquet")
    df["gaming_date"] = pd.to_datetime(df["gaming_date"]).dt.normalize()
    return df


@lru_cache(maxsize=None)
def action_catalog() -> pd.DataFrame:
    return _read("2_machine_decision/action_catalog.parquet")


@lru_cache(maxsize=None)
def placements() -> pd.DataFrame:
    df = _read("2_machine_decision/placements.parquet")
    for col in ("effective_from_utc", "effective_to_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


@lru_cache(maxsize=None)
def operations() -> pd.DataFrame:
    df = _read("2_machine_decision/operations.parquet")
    for col in (
        "opened_at_utc",
        "opened_at_local",
        "recorded_at_utc",
        "work_started_at_utc",
        "available_at_utc",
    ):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


@lru_cache(maxsize=None)
def machines() -> pd.DataFrame:
    return _read("2_machine_decision/machines.parquet")


@lru_cache(maxsize=None)
def game_math() -> pd.DataFrame:
    return _read("2_machine_decision/game_math.parquet")


@lru_cache(maxsize=None)
def events() -> pd.DataFrame:
    return _read("4_context/events.parquet")


def on_floor(daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Active on-floor machine-days (exclude pre-install / post-retire)."""
    df = prepared_daily() if daily is None else daily
    return df[df["location_id"].notna()].copy()
