# Panduan Deployment — Bitcoin 24h Deep Learning Forecast

Branch `deploy` ini menambahkan aplikasi **FastAPI + Web UI interaktif** dan konfigurasi **Docker / Render / Railway** tanpa mengubah satu baris pun file asli di branch `main` (`model_forecasting.ipynb`, `README.md`, `requirements.txt`, maupun file `.keras`).

---

## 1. Kalibrasi `MinMaxScaler` Asli dari `Bitcoin3.csv` (Sangat Disarankan)

Secara default, aplikasi sudah bisa langsung dijalankan dengan **scaler estimasi (2017–2021)**. Untuk mendapatkan kalibrasi **100% identik** dengan notebook training, jalankan perintah berikut setelah kamu mengunduh `Bitcoin3.csv`:

```bash
python3 deploy/extract_scaler.py /path/to/Bitcoin3.csv --also-export-sample
```

Perintah tersebut akan:
1. Mereplikasi preprocessing & split 70% training dari `model_forecasting.ipynb`.
2. Menghasilkan `deploy/artifacts/scaler.json` (parameter `data_min` dan `data_max` eksak).
3. Memperbarui `deploy/artifacts/sample_btc_hourly.csv` dengan 168 jam data test asli (berkat flag `--also-export-sample`).

Setelah itu, commit & push `deploy/artifacts/scaler.json` ke branch `deploy` (ukurannya hanya < 1 KB). Badge di Web UI akan otomatis berubah menjadi **`Scaler: Exact Training`**.

---

## 2. Menjalankan Secara Lokal

```bash
# Install dependency deployment (atau gunakan environment Python 3.10 yang sudah punya TensorFlow 2.19)
pip install -r deploy/requirements-deploy.txt

# Jalankan server FastAPI
CUDA_VISIBLE_DEVICES=-1 uvicorn deploy.main:app --host 0.0.0.0 --port 8000 --reload
```

Buka di browser:
- **Web UI**: `http://localhost:8000`
- **Swagger / OpenAPI Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/health`

---

## 3. Menjalankan dengan Docker

```bash
docker build -t bitcoin-lstm-forecast .
docker run -p 8000:8000 bitcoin-lstm-forecast
```

---

## 4. Deploy ke Cloud (Render / Railway)

### Opsi A — Render (Gratis / Starter)
1. Push branch `deploy` ke GitHub:
   ```bash
   git push -u origin deploy
   ```
2. Buka [Render Dashboard](https://dashboard.render.com) → **New +** → **Web Service** (atau **Blueprint** untuk membaca `render.yaml` otomatis).
3. Pilih repository `Forecasting-Nilai-Bitcoin` dan pastikan **Branch = `deploy`** serta **Runtime = Docker**.
4. Klik **Deploy**. Render otomatis mendeteksi `Dockerfile` dan health check `/api/health`.

> **Catatan RAM**: TensorFlow CPU membutuhkan ~500–700 MB RAM saat memuat kedua model. Jika instance Free tier Render (512 MB RAM) mengalami OOM, gunakan Railway atau Render Starter (1 GB+ RAM).

### Opsi B — Railway
1. Push branch `deploy` ke GitHub.
2. Buka [Railway](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Pilih repo ini, lalu di **Settings → Source** ubah branch ke **`deploy`**.
4. Railway otomatis membangun dari `Dockerfile` dan menyuntikkan variabel `PORT`.

---

## 5. Daftar Endpoint REST API

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/` | Halaman Web UI interaktif |
| `GET` | `/api/health` | Status server, metadata model, & status kalibrasi scaler |
| `GET` | `/api/sample/download` | Unduh contoh file CSV (`sample_btc_hourly.csv`) |
| `POST` | `/api/predict/sample` | Prediksi 24 jam dari dataset sampel (+ opsional mode Backtest) |
| `POST` | `/api/predict/live` | Ambil 120 jam data BTC/USDT live (Binance Vision / Kraken) & prediksi 24 jam |
| `POST` | `/api/predict/csv` | Upload file `.csv` (`Bitcoin3.csv` atau OHLCV) & prediksi 24 jam |
