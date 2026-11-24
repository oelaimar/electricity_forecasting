import pandas as pd
import logging

import sys
sys.path.append("..")

from src.config import cfg
from src.data_loader import load_raw

log = logging.getLogger(__name__)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    dt, y = cfg.DATETIME_COL, cfg.TARGET_COL
    df = df.copy()

    # Ensure datetime column has the correct type
    df[dt] = pd.to_datetime(df[dt], errors="coerce")

    # Ensure target column is numeric
    df[y] = pd.to_numeric(df[y], errors="coerce")

    # Remove rows with invalid dates or missing target
    df = df.dropna(subset=[dt, y])
    log.info("Dropped rows with invalid dates or missing target")

    # Remove exact duplicate rows
    df = df.drop_duplicates()
    log.info("Dropped exact duplicate rows")

    # Sort data chronologically
    df = df.sort_values(dt).reset_index(drop=True)
    log.info("Sorted data chronologically")

    # Remove duplicate timestamps (keep first observation)
    df = df.drop_duplicates(subset=[dt], keep="first")

    # Create a complete hourly datetime range
    full_range = pd.date_range(
        start=df[dt].min(),
        end=df[dt].max(),
        freq="h"
    )

    df = df.set_index(dt).reindex(full_range)
    df.index.name = dt

    log.info("Reindexed to complete hourly range")

    # Interpolate missing target values
    df[y] = df[y].interpolate(method="time")

    log.info("Interpolated missing target values")

    # Reset index
    df = df.reset_index()

    log.info(
        "Cleaning completed | Rows: %s | Datetime type: %s | Target type: %s",
        len(df), df[dt].dtype, df[y].dtype
    )

    return df


def save_clean(df: pd.DataFrame) -> None:
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.CLEAN_FILE, index=False)
    log.info("Cleaned data saved to: %s", cfg.CLEAN_FILE)