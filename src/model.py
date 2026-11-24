import sys
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt 
import numpy as np  
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

sys.path.append("..")

from src.config import cfg


# Visualisations

def plot_time_series(df: pd.DataFrame, output_dir: Path = cfg.FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df[cfg.DATETIME_COL], df[cfg.TARGET_COL], linewidth=0.5, color="#1f77b4")
    ax.set(title="Hourly electricity consumption - full timeline", xlabel="Date", ylabel="MW")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = output_dir / "01_consumption_timeline.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_seasonality(df: pd.DataFrame, output_dir: Path = cfg.FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = pd.to_datetime(df[cfg.DATETIME_COL])
    hourly = df.groupby(dt.dt.hour)[cfg.TARGET_COL].mean()
    weekday = df.groupby(dt.dt.day_of_week)[cfg.TARGET_COL].mean()
    monthly = df.groupby(dt.dt.month)[cfg.TARGET_COL].mean()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].bar(hourly.index, hourly.values, color="#1f77b4")
    axes[0].set(title="By hour of day", xlabel="Hour", ylabel="Mean MW")
    axes[1].bar(weekday.index, weekday.values, color="#ff7f0e")
    axes[1].set(title="By day of week (0=Mon)", xlabel="Day", ylabel="Mean MW")
    axes[2].plot(monthly.index, monthly.values, marker="o", color="#2ca02c")
    axes[2].set(title="By month", xlabel="Month", ylabel="Mean MW")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Consumption patterns: daily, weekly and annual seasonality")
    fig.tight_layout()
    path = output_dir / "02_seasonality.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path