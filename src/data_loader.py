import pandas as pd
import logging
from pathlib import Path

import sys
sys.path.append("..")

from src.config import cfg

log = logging.getLogger(__name__)

def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = cfg.RAW_FILE
        
    df = pd.read_csv(path)
    log.info("data loaded")
    return df