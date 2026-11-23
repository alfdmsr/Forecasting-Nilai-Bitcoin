"""MinMaxScaler management (Pure NumPy): loads exact training scaler if available, or fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    data_min: np.ndarray
    data_max: np.ndarray
    mode: str  # "exact_training" | "estimated_fallback"
    source: str
    metadata: dict

    @property
    def data_range(self) -> np.ndarray:
        span = self.data_max - self.data_min
        return np.where(span == 0.0, 1.0, span)

    def transform_window(self, window_df: pd.DataFrame) -> np.ndarray:
        """Transform a (24, 8) feature DataFrame into a (1, 24, 8) float32 array."""
        ordered = window_df[MODEL_FEATURES].to_numpy(dtype=np.float64)
        scaled = (ordered - self.data_min) / self.data_range
        return np.expand_dims(scaled.astype(np.float32), axis=0)

    def inverse_transform_close(self, scaled_preds: np.ndarray) -> np.ndarray:
        """Inverse-scale predicted Close values (column index 0) back to USD."""
        arr = np.asarray(scaled_preds, dtype=np.float64)
        return arr * self.data_range[0] + self.data_min[0]


def load_scaler(scaler_path: Path | None = None) -> ScalerBundle:
    path = scaler_path or DEFAULT_SCALER_PATH
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload.get("features", MODEL_FEATURES)
        if features != MODEL_FEATURES:
            raise ValueError(f"Urutan fitur di {path} tidak cocok: {features}")
        dmin = np.asarray(payload["data_min"], dtype=np.float64)
        dmax = np.asarray(payload["data_max"], dtype=np.float64)
        return ScalerBundle(
            data_min=dmin,
            data_max=dmax,
            mode="exact_training",
            source=str(path.name),
            metadata={
                "train_rows": payload.get("train_rows"),
                "date_start": payload.get("train_date_start"),
                "date_end": payload.get("train_date_end"),
                "close_min_usd": float(dmin[0]),
                "close_max_usd": float(dmax[0]),
            },
        )

    dmin = np.asarray(FALLBACK_MIN, dtype=np.float64)
    dmax = np.asarray(FALLBACK_MAX, dtype=np.float64)
    return ScalerBundle(
        data_min=dmin,
        data_max=dmax,
        mode="estimated_fallback",
        source="built-in BTC 2017-2021 calibration",
        metadata={
            "close_min_usd": float(dmin[0]),
            "close_max_usd": float(dmax[0]),
            "note": "Jalankan `python3 deploy/extract_scaler.py /path/to/Bitcoin3.csv` untuk kalibrasi 100% eksak.",
        },
    )
