# Forecasting Nilai Bitcoin dengan Deep Learning (LSTM)

Peramalan (*forecasting*) harga **Bitcoin** untuk 24 jam ke depan menggunakan **LSTM**, dengan *custom layer* dan *custom training loop* yang dibangun dari nol menggunakan TensorFlow mencakup mekanisme **Multi-Head Attention** buatan sendiri hingga arsitektur **Seq2Seq** dengan *teacher forcing*.

## Tujuan

Memprediksi harga penutupan (*Close price*) Bitcoin untuk **24 jam ke depan**, berdasarkan data historis 24 jam sebelumnya (per jam), menggunakan pendekatan *deep learning* untuk data deret waktu (*time series*).

## Dataset

Dataset berupa data harga Bitcoin per jam (`Bitcoin3.csv`) yang mencakup fitur:

- `Close` — harga penutupan (target prediksi)
- `Volume USDT` — volume transaksi
- `RSI` — Relative Strength Index
- `MACD_Hist` — Moving Average Convergence Divergence (histogram)
- `ATR` — Average True Range
- `KAMAO` — indikator teknikal tambahan

Data displit menjadi:
- **70%** — Training
- **15%** — Validation
- **15%** — Test

## Alur Pengerjaan

### 1. Eksplorasi Data (EDA)
- Visualisasi **heatmap korelasi** antar fitur
- **Dekomposisi deret waktu** (trend, seasonal, residual) dengan periode mingguan (168 jam)
- Uji **ACF & PACF** untuk menentukan panjang *window* input yang optimal

### 2. Preprocessing & Data Pipeline
- **Feature engineering** — menghitung rolling mean & rolling std 24 jam untuk menangkap tren jangka pendek
- **Normalisasi** menggunakan `MinMaxScaler`
- Membentuk data menjadi *sequence* menggunakan `tf.data.Dataset`, dengan:
  - **Window size**: 24 jam (input)
  - **Horizon**: 24 jam (output/prediksi ke depan)

### 3. Arsitektur Model & Custom Layer

Beberapa komponen dibangun **dari nol (from scratch)** menggunakan TensorFlow `Layer` API, meliputi:
- `CustomDense` — layer Dense manual dengan `add_weight`
- `CustomMultiHeadAttention` — mekanisme *multi-head attention* buatan sendiri
- `CustomLeakyReLU` — fungsi aktivasi kustom

Dua arsitektur model dibangun dan dibandingkan:

| Model | Arsitektur |
|---|---|
| **Model 1 — Baseline LSTM** | LSTM → Custom Multi-Head Attention → Flatten → Custom LeakyReLU → Custom Dense |
| **Model 2 — Seq2Seq LSTM** | Encoder-Decoder LSTM dengan *teacher forcing* + Custom Multi-Head Attention (dibangun via *model subclassing*) |

### 4. Custom Training Loop & Evaluasi
- **Custom loss function** — *Weighted MAE*, memberi bobot lebih besar pada langkah prediksi yang lebih jauh ke depan
- **Custom training loop** menggunakan `tf.GradientTape` (bukan `model.fit()` bawaan Keras)
- Callback kustom untuk *early stopping* dan *reduce learning rate on plateau*
- Evaluasi akhir menggunakan **MAE (Mean Absolute Error)**

## Hasil Evaluasi

| Model | MAE (skala 0–1, ternormalisasi) |
|---|---|
| **Baseline LSTM** | **0.01149** |
| Seq2Seq LSTM | 0.02547 |

Model **Baseline LSTM** menghasilkan MAE yang lebih rendah dibandingkan Seq2Seq LSTM pada dataset ini.

## Struktur Project

```
Forecasting-Nilai-Bitcoin/
├── model_forecasting.ipynb   # Notebook utama: EDA, preprocessing, training, evaluasi
├── model_baseline_LSTM.keras                                 # Model Baseline LSTM hasil pelatihan
├── model_seq2seq_LSTM.keras                                  # Model Seq2Seq LSTM hasil pelatihan
├── best_model_seq2seq_LSTM.keras                              # Checkpoint model Seq2Seq LSTM terbaik
├── requirements.txt                                           # Daftar dependency environment
└── README.md
```

## Tech Stack

| Library | Versi |
|---|---|
| **Python** & **TensorFlow / Keras** | `2.19.0` — pembangunan model, custom layer, dan custom training loop |
| **Pandas** | `2.3.3` — manipulasi data time series |
| **NumPy** | `2.0.2` |
| **Statsmodels** | `0.14.6` — dekomposisi deret waktu, ACF/PACF |
| **Scikit-learn** | `1.6.1` — `MinMaxScaler`, `mean_absolute_error` |
| **Matplotlib** | `3.10.0` — visualisasi data & hasil prediksi |
| **Seaborn** | `0.13.2` — visualisasi heatmap korelasi |

## Cara Menjalankan

1. **Clone repository**
   ```sh
   git clone https://github.com/alfdmsr/Forecasting-Nilai-Bitcoin.git
   cd Forecasting-Nilai-Bitcoin
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Jalankan notebook**
   ```sh
   jupyter notebook model_forecasting.ipynb
   ```
   Jalankan seluruh cell secara berurutan. Perlu diperhatikan bahwa dataset dimuat dari path Kaggle (`/kaggle/input/...`), sesuaikan path tersebut jika menjalankan di luar lingkungan Kaggle.

4. **Menggunakan model yang sudah dilatih**

   Tidak perlu training ulang — model tersedia langsung dalam format `.keras`:
   ```python
   import tensorflow as tf

   model = tf.keras.models.load_model(
       "model_baseline_LSTM.keras",
       custom_objects={
           "CustomDense": CustomDense,
           "CustomMultiHeadAttention": CustomMultiHeadAttention,
           "CustomLeakyReLU": CustomLeakyReLU,
       }
   )
   ```
   > Karena model menggunakan *custom layer*, definisi kelasnya (ada di notebook) perlu di-import atau didefinisikan ulang saat memuat model.

## Catatan

Proyek ini dibuat untuk tujuan pembelajaran mendalam mengenai *time series forecasting* dengan pendekatan *deep learning*, mulai dari analisis time series klasik (dekomposisi, ACF/PACF), pembangunan *custom layer* dan mekanisme *attention* dari nol, hingga *custom training loop* tanpa mengandalkan `model.fit()` bawaan Keras.