import sys

import numpy as np
import pandas as pd

sys.path.append("..")

from src.config import cfg

CATEGORIES: dict[str, list[int]] = {
    "Hour": list(range(24)),
    "DayOfWeek": list(range(7)),
    "Month": list(range(1, 13)),
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[cfg.DATETIME_COL])
    out["Hour"] = dt.dt.hour.astype("int16")
    out["DayOfWeek"] = dt.dt.day_of_week.astype("int16")  # Monday = 0
    out["Month"] = dt.dt.month.astype("int16")
    out["Year"] = dt.dt.year.astype("int16")
    out["IsWeekend"] = (out["DayOfWeek"] >= 5).astype("int8")

    return out


def add_lag_features(df: pd.DataFrame, target: str = cfg.TARGET_COL) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(cfg.DATETIME_COL).reset_index(drop=True)
    for lag in cfg.LAGS:
        out[f"lag{lag}"] = out[target].shift(lag)
    out["roll_24"] = (
        out[target].shift(1).rolling(window=cfg.ROLLING_WINDOW, min_periods=cfg.ROLLING_WINDOW).mean()
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering step; drops the first 168 non-usable rows."""
    featured = add_lag_features(add_time_features(df))
    return featured.dropna(subset=[f"lag{lag}" for lag in cfg.LAGS] + ["roll_24"]).reset_index(drop=True)


def encode_features(featured: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return a numeric X (DatetimeIndex, fixed schema) and the aligned target y.

    Categories are declared explicitly so train and test always share exactly
    the same columns, even if a category were absent from one partition.
    """
    frame = featured.set_index(cfg.DATETIME_COL).copy()
    for column, categories in CATEGORIES.items():
        frame[column] = pd.Categorical(frame[column], categories=categories)
    X = pd.get_dummies(frame[list(cfg.MODEL_FEATURES)], columns=list(cfg.CATEGORICAL_FEATURES), dtype=float)
    X.index.name = cfg.DATETIME_COL
    y = frame[cfg.TARGET_COL].astype(float)
    return X, y


def get_feature_names() -> list[str]:
    """Names of the encoded columns (useful for the model card / documentation)."""
    frame = pd.DataFrame({column: CATEGORIES[column][:1] for column in cfg.CATEGORICAL_FEATURES})
    for column in cfg.MODEL_FEATURES:
        if column not in frame:
            frame[column] = [0.0]
    for column, categories in CATEGORIES.items():
        frame[column] = pd.Categorical(frame[column], categories=categories)
    return list(
        pd.get_dummies(frame[list(cfg.MODEL_FEATURES)], columns=list(cfg.CATEGORICAL_FEATURES), dtype=float).columns
    )
