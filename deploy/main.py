"""FastAPI application serving Bitcoin 24-hour LSTM forecasts + interactive Web UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from deploy.data_sources import (
    SAMPLE_CSV_PATH,
    fetch_live_btc_ohlcv,
    load_sample_window,
    parse_uploaded_csv,
)
from deploy.indicators import WINDOW_SIZE, prepare_features
from deploy.model_service import ModelChoice, service

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="Bitcoin 24h Deep Learning Forecast API",
    description=(
        "REST API & Web UI untuk peramalan harga Close Bitcoin 24 jam ke depan "
        "menggunakan Baseline LSTM (Self-Attention) dan Seq2Seq LSTM (Cross-Attention)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SamplePredictRequest(BaseModel):
    model: ModelChoice = Field(default="both", description="Pilihan model: baseline, seq2seq, atau both")
    backtest: bool = Field(
        default=True,
        description="Jika True, menahan 24 jam terakhir sebagai ground truth aktual untuk menghitung MAE.",
    )
    offset_from_end: int = Field(
        default=0,
        ge=0,
        le=120,
        description="Pergeseran jendela sampel mundur dari akhir dataset (dalam jam).",
    )


class LivePredictRequest(BaseModel):
    model: ModelChoice = Field(default="both", description="Pilihan model: baseline, seq2seq, atau both")


@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health", tags=["System"])
async def health():
    if not service.loaded:
        service.load()
    return service.status()


@app.get("/api/sample/download", tags=["Sample"])
async def download_sample_csv():
    if not SAMPLE_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="File sampel tidak ditemukan.")
    return FileResponse(
        SAMPLE_CSV_PATH,
        media_type="text/csv",
        filename="sample_btc_hourly.csv",
    )


@app.post("/api/predict/sample", tags=["Forecast"])
async def predict_sample(req: SamplePredictRequest):
    try:
        window_df, actual_future_df, info = load_sample_window(
            backtest=req.backtest,
            offset_from_end=req.offset_from_end,
        )
        return service.predict(
            window_df=window_df,
            model_choice=req.model,
            actual_future_df=actual_future_df,
            source_info=info,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal memproses sampel: {exc}") from exc


@app.post("/api/predict/live", tags=["Forecast"])
async def predict_live(req: LivePredictRequest):
    try:
        raw_df, provider = await fetch_live_btc_ohlcv(limit=120)
        prepared_df, schema_mode = prepare_features(raw_df)
        window_df = prepared_df.iloc[-WINDOW_SIZE:]
        info = {
            "type": "live_api",
            "provider": provider,
            "schema_mode": schema_mode,
            "backtest_mode": False,
            "fetched_candles": int(len(raw_df)),
        }
        return service.predict(
            window_df=window_df,
            model_choice=req.model,
            actual_future_df=None,
            source_info=info,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal memproses data live: {exc}") from exc


@app.post("/api/predict/csv", tags=["Forecast"])
async def predict_csv(
    file: UploadFile = File(..., description="File CSV data historis BTC per jam (minimal 48 baris)"),
    model: ModelChoice = Form(default="both"),
    backtest: bool = Form(default=False),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File harus berformat .csv")

    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Ukuran file maksimal 15 MB.")

    try:
        window_df, actual_future_df, info = parse_uploaded_csv(
            content=content,
            filename=file.filename,
            backtest=backtest,
        )
        return service.predict(
            window_df=window_df,
            model_choice=model,
            actual_future_df=actual_future_df,
            source_info=info,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal memproses CSV: {exc}") from exc


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
