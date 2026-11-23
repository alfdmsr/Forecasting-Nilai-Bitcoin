"""MinMaxScaler management: loads exact training scaler if available, or calibrated fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from deploy.indicators import MODEL_FEATURES

DEFAULT_SCALER_PATH = Path(__file__).resolve().parent / "artifacts" / "scaler.json"

# Calibrated fallback bounds matching Bitcoin3.csv's 70% training period
# (2017-09-21 to ~2021-12, BTC hourly data). Replaced automatically once
# `deploy/artifacts/scaler.json` is generated via `deploy/extract_scaler.py`.
FALLBACK_MIN = [
    3156.26,     # Close
    0.0,         # Volume USDT
    0.0,         # RSI
    -1200.0,     # MACD_Hist
    10.0,        # ATR
    -25.0,       # KAMAO
    3200.0,      # Close_Roll_Mean_24
    0.0,         # Close_Roll_Std_24
]
FALLBACK_MAX = [
    68789.63,        # Close
    1_500_000_000.0, # Volume USDT
    100.0,           # RSI
    1200.0,          # MACD_Hist
    3000.0,          # ATR
    25.0,            # KAMAO
    68000.0,         # Close_Roll_Mean_24
    3500.0,          # Close_Roll_Std_24
]


@dataclass
class ScalerBundle:
    scaler: MinMaxScaler
    mode: str  # "exact_training" | "estimated_fallback"
    source: str
    metadata: dict

    def transform_window(self, window_df: pd.DataFrame) -> np.ndarray:
        """Transform a (24, 8) feature DataFrame into a (1, 24, 8) float32 array."""
        ordered = window_df[MODEL_FEATURES].to_numpy(dtype=np.float64)
        scaled = self.scaler.transform(ordered)
        return np.expand_dims(scaled.astype(np.float32), axis=0)

    def inverse_transform_close(self, scaled_preds: np.ndarray) -> np.ndarray:
        """Inverse-scale predicted Close values (column index 0) back to USD."""
        arr = np.asarray(scaled_preds, dtype=np.float64)
        flat = arr.flatten()
        dummy = np.zeros((len(flat), self.scaler.n_features_in_), dtype=np.float64)
        dummy[:, 0] = flat
        inv = self.scaler.inverse_transform(dummy)[:, 0]
        return inv.reshape(arr.shape)


def _build_sklearn_scaler(data_min: list[float], data_max: list[float]) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    dummy_fit = np.vstack([np.asarray(data_min, dtype=np.float64), np.asarray(data_max, dtype=np.float64)])
    scaler.fit(dummy_fit)
    return scaler


def load_scaler(scaler_path: Path | None = None) -> ScalerBundle:
    path = scaler_path or DEFAULT_SCALER_PATH
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload.get("features", MODEL_FEATURES)
        if features != MODEL_FEATURES:
            raise ValueError(f"Urutan fitur di {path} tidak cocok: {features}")
        scaler = _build_sklearn_scaler(payload["data_min"], payload["data_max"])
        return ScalerBundle(
            scaler=scaler,
            mode="exact_training",
            source=str(path.name),
            metadata={
                "train_rows": payload.get("train_rows"),
                "date_start": payload.get("train_date_start"),
                "date_end": payload.get("train_date_end"),
                "close_min_usd": payload["data_min"][0],
                "close_max_usd": payload["data_max"][0],
            },
        )

    scaler = _build_sklearn_scaler(FALLBACK_MIN, FALLBACK_MAX)
    return ScalerBundle(
        scaler=scaler,
        mode="estimated_fallback",
        source="built-in BTC 2017-2021 calibration",
        metadata={
            "close_min_usd": FALLBACK_MIN[0],
            "close_max_usd": FALLBACK_MAX[0],
            "note": "Jalankan `python3 deploy/extract_scaler.py /path/to/Bitcoin3.csv` untuk kalibrasi 100% eksak.",
        },
    )
