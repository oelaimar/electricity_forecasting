import pandas as pd
import logging

import sys
sys.path.append("..")

from src.config import cfg

log = logging.getLogger(__name__)

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(cfg.RAW_FILE)
    log.info("data loaded")
    return df