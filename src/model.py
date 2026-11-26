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


def chronological_split(X: pd.DataFrame, y: pd.Series, ratio: float = 0.80) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Timestamp]:
    if not X.index.is_monotonic_increasing:
        raise ValueError("Features must be sorted chronologically before splitting")
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be between 0 and 1")
    split_at = int(len(X) * ratio)
    split_timestamp = X.index[split_at]
    return X.iloc[:split_at], X.iloc[split_at:], y.iloc[:split_at], y.iloc[split_at:], split_timestamp


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R² and MAPE (%). MAPE uses a guarded denominator."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denominator = np.where(true == 0, np.finfo(float).eps, true)
    return {
        "MAE": float(mean_absolute_error(true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "R2": float(r2_score(true, pred)),
        "MAPE": float(np.mean(np.abs((true - pred) / denominator)) * 100),
    }


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    lag_column: str = "lag24",
) -> dict:
    """Compare the regression with the naive 'same hour yesterday' baseline."""
    predictions = pd.Series(model.predict(X_test), index=X_test.index, name="prediction")
    baseline = X_test[lag_column].astype(float)
    model_metrics = compute_metrics(y_test, predictions)
    baseline_metrics = compute_metrics(y_test, baseline)
    improvement = {
        metric: float((baseline_metrics[metric] - model_metrics[metric]) / abs(baseline_metrics[metric]) * 100)
        for metric in ("MAE", "RMSE", "MAPE")  # lower is better
    }
    return {
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "improvement_pct": improvement,
        "r2_gain_points": float(model_metrics["R2"] - baseline_metrics["R2"]),  # higher is better
        "predictions": predictions,
        "baseline": baseline,
    }


def residual_analysis(y_true: pd.Series, predictions: pd.Series, residuals: pd.Series | None = None) -> dict:
    """Error distribution, bias, autocorrelation and the worst forecast errors."""
    residuals = y_true - predictions if residuals is None else residuals
    rmse = float(np.sqrt(np.mean(residuals**2)))
    acf_1 = float(residuals.autocorr(lag=1)) if len(residuals) > 1 else float("nan")
    acf_24 = float(residuals.autocorr(lag=24)) if len(residuals) > 24 else float("nan")
    errors = (residuals / y_true).abs()
    worst = residuals.abs().nlargest(5)
    return {
        "mean_residual_MW": float(residuals.mean()),
        "std_residual_MW": float(residuals.std()),
        "median_abs_error_MW": float(residuals.abs().median()),
        "pct_within_5pct": float((errors <= 0.05).mean() * 100),
        "pct_within_10pct": float((errors <= 0.10).mean() * 100),
        "acf_lag_1": acf_1,
        "acf_lag_24": acf_24,
        "residual_std_by_hour": {
            str(hour): float(value)
            for hour, value in residuals.groupby(pd.to_datetime(residuals.index).hour).std().items()
        },
        "worst_errors": [
            {"timestamp": str(ts), "error_MW": float(value)} for ts, value in worst.items()
        ],
        "rmse_MW": rmse,
    }


def save_model(model: LinearRegression, path: Path | None = None) -> Path:
    path = path or cfg.MODELS_DIR / "forecastingmodelv1.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def save_metrics(payload: dict, path: Path | None = None) -> Path:
    path = path or REPORTS_DIR / METRICS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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


def plot_predictions(y_test: pd.Series, evaluation: dict, output_dir: Path = cfg.FIGURES_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = evaluation["predictions"]
    window = slice(None, 24 * 14)  # readable zoom on the first two test weeks
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_test.index[window], y_test.iloc[window].values, label="Actual", linewidth=1.2)
    ax.plot(predictions.index[window], predictions.iloc[window].values, label="Linear regression", linewidth=1.2)
    ax.plot(y_test.index[window], evaluation["baseline"].iloc[window].values, label="lag24 baseline", alpha=0.7)
    ax.set(title="Test set - actual vs predicted (first 14 days)", xlabel="Date", ylabel="MW")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    zoom_path = output_dir / "03_predictions_zoom.png"
    fig.savefig(zoom_path, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, predictions, s=4, alpha=0.25)
    limits = [float(y_test.min()), float(y_test.max())]
    ax.plot(limits, limits, color="red", linestyle="--", linewidth=1)
    ax.set(title="Predicted vs actual (whole test set)", xlabel="Actual MW", ylabel="Predicted MW")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    scatter_path = output_dir / "04_predicted_vs_actual.png"
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)
    return {"zoom": zoom_path, "scatter": scatter_path}


def plot_residuals(y_test: pd.Series, evaluation: dict, output_dir: Path = cfg.FIGURES_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_test - evaluation["predictions"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(residuals.index, residuals.values, linewidth=0.4, alpha=0.7)
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set(title="Residuals over time", xlabel="Date", ylabel="MW")
    axes[1].hist(residuals, bins=80, color="#9467bd", alpha=0.8)
    axes[1].set(title="Residual distribution", xlabel="MW", ylabel="Count")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    time_path = output_dir / "05_residuals.png"
    fig.savefig(time_path, dpi=120)
    plt.close(fig)

    by_hour = residuals.groupby(pd.to_datetime(residuals.index).hour).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_hour.index, by_hour.values, color="#8c564b")
    ax.axhline(0, color="red", linewidth=1)
    ax.set(title="Mean residual by hour (remaining bias)", xlabel="Hour", ylabel="MW")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    hour_path = output_dir / "06_residual_by_hour.png"
    fig.savefig(hour_path, dpi=120)
    plt.close(fig)
    return {"time": time_path, "by_hour": hour_path}
