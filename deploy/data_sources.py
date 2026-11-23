"""Hybrid data sources: Live BTC/USDT API, Built-in Sample CSV, and User CSV upload."""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import pandas as pd

from deploy.indicators import HORIZON, WINDOW_SIZE, prepare_features

SAMPLE_CSV_PATH = Path(__file__).resolve().parent / "artifacts" / "sample_btc_hourly.csv"


async def fetch_live_btc_ohlcv(limit: int = 120) -> tuple[pd.DataFrame, str]:
    """Fetch recent hourly BTC/USDT candles using public APIs with automatic fallback.

    Order of providers (no API key required):
    1. Binance public market data mirror (`data-api.binance.vision`) — works globally including US IPs
    2. Binance main API (`api.binance.com`)
    3. Kraken public OHLC API (`api.kraken.com`, XBT/USD)
    """
    errors: list[str] = []
    timeout = httpx.Timeout(12.0, connect=6.0)

    async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "btc-lstm-forecast/1.0"}) as client:
        for provider_name, url in [
            (
                "Binance Data Vision (BTCUSDT 1h)",
                f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit={limit}",
            ),
            (
                "Binance API (BTCUSDT 1h)",
                f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit={limit}",
            ),
        ]:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                raw = resp.json()
                if not isinstance(raw, list) or len(raw) < 60:
                    raise ValueError("Data candle terlalu sedikit")

                rows = [
                    {
                        "Date": pd.to_datetime(int(k[0]), unit="ms", utc=True).tz_localize(None),
                        "Open": float(k[1]),
                        "High": float(k[2]),
                        "Low": float(k[3]),
                        "Close": float(k[4]),
                        "Volume": float(k[5]),
                        "Volume USDT": float(k[7]),
                    }
                    for k in raw
                ]
                return pd.DataFrame(rows), provider_name
            except Exception as exc:
                errors.append(f"{provider_name}: {exc}")

        # Fallback: Kraken Public OHLC (60-minute interval)
        try:
            url = "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=60"
            resp = await client.get(url)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("error"):
                raise ValueError(str(payload["error"]))
            result = payload["result"]
            pair_key = next(k for k in result.keys() if k != "last")
            candles = result[pair_key][-limit:]
            rows = [
                {
                    "Date": pd.to_datetime(int(k[0]), unit="s", utc=True).tz_localize(None),
                    "Open": float(k[1]),
                    "High": float(k[2]),
                    "Low": float(k[3]),
                    "Close": float(k[4]),
                    "Volume": float(k[6]),
                    "Volume USDT": float(k[6]) * float(k[4]),
                }
                for k in candles
            ]
            return pd.DataFrame(rows), "Kraken Public API (BTC/USD 1h)"
        except Exception as exc:
            errors.append(f"Kraken API: {exc}")

    raise RuntimeError(
        "Gagal mengambil data live BTC dari semua provider publik. Detail: " + " | ".join(errors)
    )


def load_sample_window(
    backtest: bool = True,
    offset_from_end: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Load a 24h input window (and optional 24h actual future) from built-in sample CSV."""
    if not SAMPLE_CSV_PATH.exists():
        raise FileNotFoundError(f"Sample dataset tidak ditemukan: {SAMPLE_CSV_PATH}")

    raw_df = pd.read_csv(SAMPLE_CSV_PATH)
    prepared_df, schema_mode = prepare_features(raw_df)
    total_valid = len(prepared_df)

    if backtest and total_valid >= WINDOW_SIZE + HORIZON:
        max_offset = total_valid - (WINDOW_SIZE + HORIZON)
        offset = max(0, min(offset_from_end, max_offset))
        end_future = total_valid - offset
        start_future = end_future - HORIZON
        start_window = start_future - WINDOW_SIZE
        window_df = prepared_df.iloc[start_window:start_future]
        actual_future_df = prepared_df.iloc[start_future:end_future]
    else:
        max_offset = max(0, total_valid - WINDOW_SIZE)
        offset = max(0, min(offset_from_end, max_offset))
        end_window = total_valid - offset
        start_window = end_window - WINDOW_SIZE
        window_df = prepared_df.iloc[start_window:end_window]
        actual_future_df = None

    info = {
        "type": "sample",
        "provider": "Built-in Sample BTC Hourly CSV",
        "schema_mode": schema_mode,
        "backtest_mode": actual_future_df is not None,
        "total_valid_rows": total_valid,
        "max_offset": max_offset,
        "offset_from_end": offset,
    }
    return window_df, actual_future_df, info


def parse_uploaded_csv(
    content: bytes,
    filename: str,
    backtest: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Parse user-uploaded CSV bytes into a 24h window (+ optional 24h backtest ground truth)."""
    try:
        raw_df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"File CSV tidak dapat dibaca: {exc}") from exc

    if raw_df.empty:
        raise ValueError("File CSV kosong.")

    prepared_df, schema_mode = prepare_features(raw_df)
    total_valid = len(prepared_df)

    actual_future_df: pd.DataFrame | None = None
    if backtest and total_valid >= WINDOW_SIZE + HORIZON:
        window_df = prepared_df.iloc[-(WINDOW_SIZE + HORIZON) : -HORIZON]
        actual_future_df = prepared_df.iloc[-HORIZON:]
    else:
        window_df = prepared_df.iloc[-WINDOW_SIZE:]

    info = {
        "type": "csv_upload",
        "provider": f"Uploaded CSV ({filename})",
        "schema_mode": schema_mode,
        "backtest_mode": actual_future_df is not None,
        "raw_rows": int(len(raw_df)),
        "valid_rows": total_valid,
    }
    return window_df, actual_future_df, info
