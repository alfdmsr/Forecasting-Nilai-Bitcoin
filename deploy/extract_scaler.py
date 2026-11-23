"""CLI tool to extract the exact MinMaxScaler parameters from Bitcoin3.csv.

Reproduces Cell 4, 6, 13, and 15 of `model_forecasting.ipynb` identically so the
deployed FastAPI app uses the exact training scaler (0% skew) without needing to
commit the full Bitcoin3.csv dataset into Git.

Usage:
    python3 deploy/extract_scaler.py /path/to/Bitcoin3.csv
    python3 -m deploy.extract_scaler /path/to/Bitcoin3.csv --also-export-sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.indicators import BASE_FEATURES, MODEL_FEATURES  # noqa: E402

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def extract_scaler_from_csv(csv_path: Path, also_export_sample: bool = False) -> Path:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    df_selected = df[BASE_FEATURES].copy()
    df_selected["Close_Roll_Mean_24"] = df_selected["Close"].rolling(window=24).mean()
    df_selected["Close_Roll_Std_24"] = df_selected["Close"].rolling(window=24).std()
    df_selected.dropna(inplace=True)

    n = len(df_selected)
    train_df = df_selected[0 : int(n * 0.7)]
    val_df = df_selected[int(n * 0.7) : int(n * 0.85)]
    test_df = df_selected[int(n * 0.85) :]

    scaler = MinMaxScaler()
    scaler.fit(train_df[MODEL_FEATURES])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_out = ARTIFACTS_DIR / "scaler.json"
    payload = {
        "features": MODEL_FEATURES,
        "data_min": [float(x) for x in scaler.data_min_],
        "data_max": [float(x) for x in scaler.data_max_],
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_date_start": str(train_df.index[0]),
        "train_date_end": str(train_df.index[-1]),
    }
    scaler_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Exact scaler saved to: {scaler_out}")
    print(f"     Train rows : {len(train_df)} ({train_df.index[0]} -> {train_df.index[-1]})")
    print(f"     Close range: ${scaler.data_min_[0]:,.2f} -> ${scaler.data_max_[0]:,.2f}")

    if also_export_sample:
        # Export a 168-hour (7-day) slice from the test split starting around sample #100
        start_idx = min(max(0, 76), max(0, len(test_df) - 168))
        sample_slice = df.loc[test_df.index[start_idx : start_idx + 168], BASE_FEATURES].copy()
        sample_out = ARTIFACTS_DIR / "sample_btc_hourly.csv"
        sample_slice.reset_index().to_csv(sample_out, index=False)
        print(f"[OK] Sample test slice ({len(sample_slice)} rows) saved to: {sample_out}")

    return scaler_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ekstrak parameter MinMaxScaler eksak dari Bitcoin3.csv untuk deployment."
    )
    parser.add_argument("csv_path", type=Path, help="Path menuju file Bitcoin3.csv")
    parser.add_argument(
        "--also-export-sample",
        action="store_true",
        help="Sekalian perbarui deploy/artifacts/sample_btc_hourly.csv dengan 168 jam dari data test asli.",
    )
    args = parser.parse_args()
    if not args.csv_path.exists():
        raise SystemExit(f"File tidak ditemukan: {args.csv_path}")
    extract_scaler_from_csv(args.csv_path, also_export_sample=args.also_export_sample)


if __name__ == "__main__":
    main()
