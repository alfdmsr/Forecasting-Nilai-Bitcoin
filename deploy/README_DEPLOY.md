# Panduan Deployment — Bitcoin 24h Deep Learning Forecast

Branch `deploy` ini menjalankan aplikasi **FastAPI + Material Design 3 Web UI** dengan **Pure NumPy Inference Engine** (`deploy/numpy_inference.py`) yang membaca langsung bobot dari file `.keras` aslimu (`model_baseline_LSTM.keras` & `model_seq2seq_LSTM.keras`) tanpa perlu meng-install TensorFlow di server produksi.

- **Konsumsi RAM**: ~141 MB *(turun dari ~726 MB saat memakai TensorFlow — sangat aman untuk **Render Free 512 MB tanpa kartu kredit**)*
- **Latensi Inferensi**: ~1 ms (Baseline) & ~3.4 ms (Seq2Seq)
- **Presisi**: Identik dengan TensorFlow hingga `~1e-7`
- **Branch `main`**: 100% utuh dan tidak berubah

---

## 1. Kalibrasi `MinMaxScaler` Asli dari `Bitcoin3.csv`

```bash
python3 deploy/extract_scaler.py Bitcoin3.csv --also-export-sample
```

Menghasilkan `deploy/artifacts/scaler.json` (< 1 KB) dan mengisi `deploy/artifacts/sample_btc_hourly.csv` dengan data test asli.

---

## 2. Menjalankan Secara Lokal

```bash
pip install -r deploy/requirements-deploy.txt
uvicorn deploy.main:app --host 0.0.0.0 --port 8000 --reload
```

Buka di browser:
- **Web UI (Material Design 3)**: `http://localhost:8000`
- **Swagger / OpenAPI Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/health`

---

## 3. Deploy Gratis Permanen di Render (Tanpa Kartu Kredit)

1. Push branch `deploy` ke GitHub (`git push origin deploy`).
2. Buka [Render Dashboard](https://dashboard.render.com) → login dengan GitHub (tidak perlu kartu kredit).
3. Klik **New +** → **Web Service** → pilih repo `Forecasting-Nilai-Bitcoin`.
4. Atur:
   - **Branch**: `deploy`
   - **Runtime**: `Docker`
   - **Instance Type**: `Free` (512 MB RAM — lega karena app ini hanya butuh ~141 MB!)
   - **Health Check Path**: `/api/health`
5. Klik **Deploy Web Service**. Build selesai dalam ±1–2 menit.

---

## 4. Menjalankan dengan Docker

```bash
docker build -t bitcoin-lstm-forecast .
docker run -p 8000:8000 bitcoin-lstm-forecast
```
