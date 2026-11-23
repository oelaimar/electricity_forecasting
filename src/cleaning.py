import pandas as pd
import logging

import sys
sys.path.append("..")

from src.config import cfg
from src.data_loader import load_raw

log = logging.getLogger(__name__)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    dt, y = cfg.DATA_RAW, cfg.TARGET_COL
    df = df.copy()
    
    df = df.dropna(subset=[dt, y])
    log.info("Drop rows with unparsable date or missing target")
    
    df = df.drop_duplicates()
    log.info("Drop exact duplicate rows")

    df = df.sort_values(dt).reset_index(drop=True)
    log.info("Sort chronologically")

    full_range = pd.date_range(df[dt].min(), df[dt].max(), freq="h")
    df = df.set_index(dt).reindex(full_range)
    df.index.name = dt
    log.info("Reindex to a complete hourly range (exposes DST gaps as NaN)")
       
    df[y] = df[y].interpolate(method="time")
    log.info("Fill missing hours by time-based linear interpolation")

    return df.reset_index()

def save_clean(df: pd.DataFrame) -> None:
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.CLEAN_FILE, index=False)
    log.info("cleand data saves to : %s", cfg.CLEAN_FILE)