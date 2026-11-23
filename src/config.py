from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

class Config():
    """Singleton holding all project paths and settings."""
    
    _instance: "Config | None" = None
    
    def __new__(cls) -> Config:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self) -> None:
        # ---------- Root & env ----------
        self.ROOT = Path(
            os.getenv("PROJECT_ROOT") or Path(__file__).resolve().parents[1]
        ).resolve()
        load_dotenv(self.ROOT / ".env")  # optional, silently ignored if missing

        # ---------- Paths ----------
        self.DATA_RAW = self.ROOT / "data" / "raw"
        self.DATA_PROCESSED = self.ROOT / "data" / "processed"
        self.RAW_FILE = self.DATA_RAW / "dataset-electric.csv"
        self.CLEAN_FILE = self.DATA_PROCESSED / "clean.csv"
        self.FEATURES_FILE = self.DATA_PROCESSED / "features.csv"

        self.MODELS_DIR = self.ROOT / "models"
        self.MODEL_FILE = self.MODELS_DIR / "forecastingmodelv1.joblib"

        self.REPORTS_DIR = self.ROOT / "reports"
        self.FIGURES_DIR = self.REPORTS_DIR / "figures"

        # ---------- Dataset schema ----------
        self.DATETIME_COL = "Datetime"
        self.TARGET_COL = "PJME_MW"

        # ---------- Feature engineering ----------
        self.LAGS = (24, 168)          # lag_24, lag_168
        self.ROLLING_WINDOW = 24       # roll_24
        self.TIME_FEATURES = ["Hour", "DayOfWeek", "Month", "Year", "IsWeekend"]

        # ---------- Modeling ----------
        self.TRAIN_RATIO = 0.8         # chronological 80/20 split
        self.RANDOM_STATE = 42
        self.BASELINE_COL = "lag_24"

        # ---------- Azure (read from .env, never hardcode secrets) ----------
        self.AZURE_REGION = "northeurope"
        self.AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
        self.AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "")
        self.AZURE_ML_WORKSPACE = os.getenv("AZURE_ML_WORKSPACE", "")
        self.AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "")
        self.AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "energy-data")
        self.COMPUTE_NAME = "cpu-cluster"
        self.COMPUTE_SIZE = "Standard_DS11_v2"
        self.COMPUTE_MIN_NODES = 0
        self.COMPUTE_MAX_NODES = 1
        self.MODEL_NAME = "forecastingmodelv1"
        
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create output directories if they do not exist."""
        for d in (self.DATA_RAW, self.DATA_PROCESSED, self.MODELS_DIR,
                  self.REPORTS_DIR, self.FIGURES_DIR):
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()