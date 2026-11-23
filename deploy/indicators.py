"""Technical indicator & feature engineering pipeline matching model_forecasting.ipynb."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = ["Close", "Volume USDT", "RSI", "MACD_Hist", "ATR", "KAMAO"]
MODEL_FEATURES = [
    "Close",
    "Volume USDT",
    "RSI",
    "MACD_Hist",
    "ATR",
    "KAMAO",
    "Close_Roll_Mean_24",
    "Close_Roll_Std_24",
]
WINDOW_SIZE = 24
HORIZON = 24


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd_hist(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def compute_kamao(
    close: pd.Series, er_period: int = 10, fast: int = 2, slow: int = 30
) -> pd.Series:
    """Kaufman Adaptive Moving Average Oscillator: percentage spread from KAMA."""
    values = close.to_numpy(dtype=np.float64)
    n = len(values)
    kama = np.full(n, np.nan, dtype=np.float64)

    if n <= er_period:
        return pd.Series(np.zeros(n), index=close.index)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)

    kama[er_period - 1] = values[er_period - 1]
    for i in range(er_period, n):
        change = abs(values[i] - values[i - er_period])
        volatility = np.sum(np.abs(np.diff(values[i - er_period : i + 1])))
        er = change / volatility if volatility > 1e-12 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (values[i] - kama[i - 1])

    kama_series = pd.Series(kama, index=close.index)
    kamao = ((close - kama_series) / kama_series.replace(0.0, np.nan)) * 100.0
    return kamao


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common CSV column variations into canonical names."""
    out = df.copy()
    rename_map: dict[str, str] = {}
    lower_lookup = {c.strip().lower(): c for c in out.columns}

    aliases = {
        "Date": ["date", "timestamp", "time", "datetime", "open_time"],
        "Open": ["open"],
        "High": ["high"],
        "Low": ["low"],
        "Close": ["close", "close_price", "price"],
        "Volume USDT": ["volume usdt", "quote_volume", "volume_usdt", "vol_usdt"],
        "Volume": ["volume", "vol", "volume btc", "volume_btc"],
        "RSI": ["rsi", "rsi_14"],
        "MACD_Hist": ["macd_hist", "macd_histogram", "macd hist"],
        "ATR": ["atr", "atr_14"],
        "KAMAO": ["kamao", "kama_osc", "kama_oscillator"],
        "Close_Roll_Mean_24": ["close_roll_mean_24"],
        "Close_Roll_Std_24": ["close_roll_std_24"],
    }

    for canonical, candidates in aliases.items():
        if canonical in out.columns:
            continue
        for cand in candidates:
            if cand in lower_lookup:
                rename_map[lower_lookup[cand]] = canonical
                break

    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def prepare_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Build the 8-feature DataFrame required by the LSTM models.

    Supports two schemas:
    1. Native `Bitcoin3.csv` schema containing `BASE_FEATURES` directly.
    2. Generic OHLCV schema (`Close`, `Volume`/`Volume USDT`, optional `High`/`Low`),
       where missing technical indicators (`RSI`, `MACD_Hist`, `ATR`, `KAMAO`) are
       computed automatically.

    Returns:
        (prepared_df, schema_mode) where prepared_df has DateTimeIndex (or RangeIndex)
        and columns == MODEL_FEATURES with no NaNs.
    """
    df = _normalize_column_names(df_raw)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    if "Close" not in df.columns:
        raise ValueError(
            "Kolom 'Close' tidak ditemukan. Pastikan CSV memiliki kolom 'Close' (dan sebaiknya 'Date')."
        )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    has_native = all(col in df.columns for col in BASE_FEATURES)

    if has_native:
        schema_mode = "native_bitcoin3"
        feat_df = df[BASE_FEATURES].copy()
    else:
        schema_mode = "computed_from_ohlcv"
        feat_df = pd.DataFrame(index=df.index)
        feat_df["Close"] = df["Close"]

        if "Volume USDT" in df.columns:
            feat_df["Volume USDT"] = df["Volume USDT"]
        elif "Volume" in df.columns:
            feat_df["Volume USDT"] = df["Volume"] * df["Close"]
        else:
            feat_df["Volume USDT"] = 0.0

        high = df["High"] if "High" in df.columns else df["Close"] * 1.004
        low = df["Low"] if "Low" in df.columns else df["Close"] * 0.996

        feat_df["RSI"] = df["RSI"] if "RSI" in df.columns else compute_rsi(df["Close"])
        feat_df["MACD_Hist"] = (
            df["MACD_Hist"] if "MACD_Hist" in df.columns else compute_macd_hist(df["Close"])
        )
        feat_df["ATR"] = (
            df["ATR"] if "ATR" in df.columns else compute_atr(high, low, df["Close"])
        )
        feat_df["KAMAO"] = (
            df["KAMAO"] if "KAMAO" in df.columns else compute_kamao(df["Close"])
        )

    # Exact feature engineering from Cell 13 of model_forecasting.ipynb
    feat_df["Close_Roll_Mean_24"] = feat_df["Close"].rolling(window=24).mean()
    feat_df["Close_Roll_Std_24"] = feat_df["Close"].rolling(window=24).std()

    feat_df = feat_df[MODEL_FEATURES].dropna()

    if len(feat_df) < WINDOW_SIZE:
        raise ValueError(
            f"Data valid setelah feature engineering hanya {len(feat_df)} baris "
            f"(dibutuhkan minimal {WINDOW_SIZE} baris, jadi sediakan minimal 48 baris data per jam)."
        )

    return feat_df, schema_mode
