"""Model loading and multi-horizon forecasting inference service."""

from __future__ import annotations

import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Literal

# Default to CPU for deterministic, CUDA-free cloud/local inference unless USE_GPU=1
if os.environ.get("USE_GPU", "0") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd
import tensorflow as tf

from deploy.custom_layers import CUSTOM_OBJECTS, Seq2SeqTeacherForcing
from deploy.indicators import HORIZON, MODEL_FEATURES, WINDOW_SIZE
from deploy.scaler_utils import ScalerBundle, load_scaler

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_PATH = Path(
    os.environ.get("BASELINE_MODEL_PATH", str(REPO_ROOT / "model_baseline_LSTM.keras"))
)
DEFAULT_SEQ2SEQ_PATH = Path(
    os.environ.get("SEQ2SEQ_MODEL_PATH", str(REPO_ROOT / "model_seq2seq_LSTM.keras"))
)

ModelChoice = Literal["baseline", "seq2seq", "both"]


class ForecastingService:
    """Loads both trained Keras models once and serves 24-hour BTC forecasts."""

    def __init__(
        self,
        baseline_path: Path = DEFAULT_BASELINE_PATH,
        seq2seq_path: Path = DEFAULT_SEQ2SEQ_PATH,
    ):
        self.baseline_path = Path(baseline_path)
        self.seq2seq_path = Path(seq2seq_path)
        self.baseline_model: tf.keras.Model | None = None
        self.seq2seq_model: tf.keras.Model | None = None
        self.scaler_bundle: ScalerBundle = load_scaler()
        self.loaded: bool = False

    def load(self) -> None:
        if self.loaded:
            return

        if not self.baseline_path.exists():
            raise FileNotFoundError(f"Model baseline tidak ditemukan: {self.baseline_path}")
        if not self.seq2seq_path.exists():
            raise FileNotFoundError(f"Model seq2seq tidak ditemukan: {self.seq2seq_path}")

        self.baseline_model = tf.keras.models.load_model(
            self.baseline_path, custom_objects=CUSTOM_OBJECTS, compile=False
        )
        self.seq2seq_model = self._load_seq2seq_safe(self.seq2seq_path)
        self.scaler_bundle = load_scaler()
        self.loaded = True

    @staticmethod
    def _load_seq2seq_safe(path: Path) -> tf.keras.Model:
        try:
            return tf.keras.models.load_model(
                path, custom_objects=CUSTOM_OBJECTS, compile=False
            )
        except Exception:
            # Fallback: instantiate exact architecture and load model.weights.h5 from .keras archive
            model = Seq2SeqTeacherForcing(
                units=128, horizon=HORIZON, name="model_seq2seq_LSTM"
            )
            _ = model(
                tf.zeros((1, WINDOW_SIZE, len(MODEL_FEATURES))),
                targets=tf.zeros((1, HORIZON)),
                training=True,
            )
            with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as td:
                weights_path = Path(td) / "model.weights.h5"
                weights_path.write_bytes(zf.read("model.weights.h5"))
                model.load_weights(weights_path)
            return model

    def reload_scaler(self) -> ScalerBundle:
        self.scaler_bundle = load_scaler()
        return self.scaler_bundle

    def status(self) -> dict:
        return {
            "status": "ok" if self.loaded else "not_loaded",
            "models": {
                "baseline": {
                    "name": "model_baseline_LSTM",
                    "architecture": "LSTM(64) -> CustomMultiHeadAttention(2x32) -> Flatten -> CustomLeakyReLU -> CustomDense(24)",
                    "params": int(self.baseline_model.count_params()) if self.baseline_model else None,
                    "notebook_test_mae_scaled": 0.01149,
                    "loaded": self.baseline_model is not None,
                },
                "seq2seq": {
                    "name": "model_seq2seq_LSTM",
                    "architecture": "Encoder LSTM(128) -> Decoder LSTM(128) + Cross-Attention(2x32) -> CustomDense(1) x 24 autoregressive",
                    "params": int(self.seq2seq_model.count_params()) if self.seq2seq_model else None,
                    "notebook_test_mae_scaled": 0.02547,
                    "loaded": self.seq2seq_model is not None,
                },
            },
            "window_size": WINDOW_SIZE,
            "horizon": HORIZON,
            "features": MODEL_FEATURES,
            "scaler": {
                "mode": self.scaler_bundle.mode,
                "source": self.scaler_bundle.source,
                "metadata": self.scaler_bundle.metadata,
            },
        }

    def predict(
        self,
        window_df: pd.DataFrame,
        model_choice: ModelChoice = "both",
        actual_future_df: pd.DataFrame | None = None,
        source_info: dict | None = None,
    ) -> dict:
        """Run 24-hour forecast on a prepared 24-row x 8-feature DataFrame."""
        if not self.loaded:
            self.load()

        if len(window_df) != WINDOW_SIZE:
            raise ValueError(f"Window harus tepat {WINDOW_SIZE} baris, diterima {len(window_df)}.")

        x_input = self.scaler_bundle.transform_window(window_df)
        x_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)

        # Build historical and future timestamps
        if isinstance(window_df.index, pd.DatetimeIndex):
            hist_ts = [ts.isoformat() for ts in window_df.index]
            last_ts = window_df.index[-1]
            future_ts = [
                (last_ts + pd.Timedelta(hours=h)).isoformat() for h in range(1, HORIZON + 1)
            ]
        else:
            hist_ts = [f"T-{WINDOW_SIZE - i}h" for i in range(WINDOW_SIZE)]
            future_ts = [f"T+{h}h" for h in range(1, HORIZON + 1)]

        last_close = float(window_df["Close"].iloc[-1])
        hist_close = [round(float(v), 2) for v in window_df["Close"].tolist()]

        actual_future_usd: list[float] | None = None
        if actual_future_df is not None and len(actual_future_df) == HORIZON:
            actual_future_usd = [round(float(v), 2) for v in actual_future_df["Close"].tolist()]

        forecasts: dict[str, dict] = {}
        models_to_run = (
            ["baseline", "seq2seq"] if model_choice == "both" else [model_choice]
        )

        for name in models_to_run:
            t0 = time.perf_counter()
            if name == "baseline":
                assert self.baseline_model is not None
                scaled_out = self.baseline_model(x_tensor, training=False).numpy()[0]
            elif name == "seq2seq":
                assert self.seq2seq_model is not None
                scaled_out = self.seq2seq_model(x_tensor, targets=None, training=False).numpy()[0]
            else:
                raise ValueError(f"Model tidak dikenal: {name}")
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            usd_out = self.scaler_bundle.inverse_transform_close(scaled_out)
            usd_list = [round(float(v), 2) for v in usd_out.tolist()]
            scaled_list = [round(float(v), 6) for v in scaled_out.tolist()]

            end_price = usd_list[-1]
            delta_usd = round(end_price - last_close, 2)
            delta_pct = round((delta_usd / last_close) * 100.0, 2) if last_close else 0.0

            summary: dict = {
                "last_close_usd": round(last_close, 2),
                "predicted_24h_usd": end_price,
                "change_24h_usd": delta_usd,
                "change_24h_pct": delta_pct,
                "min_usd": round(min(usd_list), 2),
                "max_usd": round(max(usd_list), 2),
                "latency_ms": latency_ms,
            }

            if actual_future_usd is not None:
                actual_arr = np.asarray(actual_future_usd, dtype=np.float64)
                pred_arr = np.asarray(usd_list, dtype=np.float64)
                mae_usd = float(np.mean(np.abs(actual_arr - pred_arr)))
                mape_pct = float(np.mean(np.abs((actual_arr - pred_arr) / np.maximum(actual_arr, 1e-9))) * 100.0)
                summary["backtest_mae_usd"] = round(mae_usd, 2)
                summary["backtest_mape_pct"] = round(mape_pct, 2)

            forecasts[name] = {
                "usd": usd_list,
                "scaled": scaled_list,
                "summary": summary,
            }

        # Check if input close falls outside training scaler range (useful transparency warning)
        close_min = self.scaler_bundle.metadata.get("close_min_usd", 0.0)
        close_max = self.scaler_bundle.metadata.get("close_max_usd", float("inf"))
        out_of_range = last_close < close_min or last_close > close_max

        latest_indicators = {
            col: round(float(window_df[col].iloc[-1]), 4) for col in MODEL_FEATURES
        }

        return {
            "source": source_info or {},
            "scaler": {
                "mode": self.scaler_bundle.mode,
                "source": self.scaler_bundle.source,
                "close_min_usd": close_min,
                "close_max_usd": close_max,
                "input_out_of_training_range": out_of_range,
            },
            "history": {
                "timestamps": hist_ts,
                "close_usd": hist_close,
                "latest_indicators": latest_indicators,
            },
            "forecast": {
                "timestamps": future_ts,
                "actual_usd": actual_future_usd,
                "models": forecasts,
            },
        }


service = ForecastingService()
