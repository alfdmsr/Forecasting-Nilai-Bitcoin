const state = {
  activeTab: "sample",
  selectedModel: "both",
  chart: null,
  lastPayload: null,
};

const INDICATOR_DESC = {
  Close: "Harga penutupan terakhir (USD)",
  "Volume USDT": "Volume transaksi per jam",
  RSI: "Indeks momentum (0–100)",
  MACD_Hist: "Histogram tren jangka pendek",
  ATR: "Rentang volatilitas rata-rata",
  KAMAO: "Osilator tren adaptif (KAMA)",
  Close_Roll_Mean_24: "Rata-rata harga 24 jam",
  Close_Roll_Std_24: "Simpangan volatilitas 24 jam",
};

const els = {
  statusChip: document.getElementById("statusChip"),
  statusIcon: document.getElementById("statusIcon"),
  statusText: document.getElementById("statusText"),
  linearProgress: document.getElementById("linearProgress"),
  sourceTabs: document.querySelectorAll("[data-tab]"),
  modelBtns: document.querySelectorAll(".model-btn"),
  subSections: {
    sample: document.getElementById("sub-sample"),
    live: document.getElementById("sub-live"),
    csv: document.getElementById("sub-csv"),
  },
  sampleBacktest: document.getElementById("sampleBacktest"),
  sampleOffset: document.getElementById("sampleOffset"),
  offsetVal: document.getElementById("offsetVal"),
  dropzone: document.getElementById("dropzone"),
  csvFile: document.getElementById("csvFile"),
  csvFileName: document.getElementById("csvFileName"),
  csvBacktest: document.getElementById("csvBacktest"),
  runBtn: document.getElementById("runBtn"),
  runBtnText: document.getElementById("runBtnText"),
  errorBanner: document.getElementById("errorBanner"),
  errorText: document.getElementById("errorText"),
  infoBanner: document.getElementById("infoBanner"),
  infoText: document.getElementById("infoText"),
  summaryGrid: document.getElementById("summaryGrid"),
  sourceSubtitle: document.getElementById("sourceSubtitle"),
  exportCsvBtn: document.getElementById("exportCsvBtn"),
  detailTabs: document.querySelectorAll(".md-tab"),
  detailPanels: document.querySelectorAll(".detail-panel"),
  tableHead: document.querySelector("#predictionTable thead"),
  tableBody: document.querySelector("#predictionTable tbody"),
  indicatorGrid: document.getElementById("indicatorGrid"),
  aboutGrid: document.getElementById("aboutGrid"),
};

function fmtUSD(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  return "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtShortTs(iso) {
  if (!iso || iso.startsWith("T")) return iso;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"];
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  return `${dd} ${months[d.getMonth()]} ${hh}:00`;
}

function setLoading(loading) {
  els.runBtn.disabled = loading;
  els.linearProgress.classList.toggle("hidden", !loading);
  els.runBtnText.textContent = loading ? "Memproses..." : "Prediksi 24 Jam";
}

function showError(msg) {
  if (!msg) {
    els.errorBanner.classList.add("hidden");
    return;
  }
  els.errorText.textContent = msg;
  els.errorBanner.classList.remove("hidden");
}

function showInfo(msg) {
  if (!msg) {
    els.infoBanner.classList.add("hidden");
    return;
  }
  els.infoText.innerHTML = msg;
  els.infoBanner.classList.remove("hidden");
}

async function fetchHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error();
    els.statusChip.className = "md-assist-chip online";
    els.statusIcon.textContent = "check_circle";
    els.statusText.textContent = "Online";
  } catch {
    els.statusChip.className = "md-assist-chip offline";
    els.statusIcon.textContent = "cloud_off";
    els.statusText.textContent = "Offline";
  }
}

async function runForecast() {
  showError(null);
  setLoading(true);

  const model = state.selectedModel;

  try {
    let res;
    if (state.activeTab === "sample") {
      res = await fetch("/api/predict/sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          backtest: els.sampleBacktest.checked,
          offset_from_end: Number(els.sampleOffset.value),
        }),
      });
    } else if (state.activeTab === "live") {
      res = await fetch("/api/predict/live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
      });
    } else {
      const file = els.csvFile.files[0];
      if (!file) {
        throw new Error("Pilih file .csv terlebih dahulu untuk menjalankan prediksi.");
      }
      const fd = new FormData();
      fd.append("file", file);
      fd.append("model", model);
      fd.append("backtest", String(els.csvBacktest.checked));
      res = await fetch("/api/predict/csv", { method: "POST", body: fd });
    }

    const payload = await res.json();
    if (!res.ok) {
      throw new Error(payload.detail || `Permintaan gagal (HTTP ${res.status})`);
    }

    state.lastPayload = payload;
    renderAll(payload);
  } catch (err) {
    showError(err.message || "Terjadi kendala saat menghitung prediksi.");
  } finally {
    setLoading(false);
  }
}

function renderAll(data) {
  if (data.scaler.input_out_of_training_range) {
    showInfo(
      `Harga input berada di luar rentang data training (${fmtUSD(data.scaler.close_min_usd)} – ${fmtUSD(data.scaler.close_max_usd)}), sehingga hasil prediksi bersifat ekstrapolasi.`
    );
  } else {
    showInfo(null);
  }

  if (data.source?.max_offset !== undefined && state.activeTab === "sample") {
    els.sampleOffset.max = String(data.source.max_offset);
  }

  const modeText = data.source.backtest_mode ? "Mode Backtest (vs Aktual)" : "Ramalan 24 Jam ke Depan";
  els.sourceSubtitle.textContent = `${data.source.provider} • ${modeText}`;
  els.exportCsvBtn.classList.remove("hidden");

  renderSummary(data);
  renderChart(data);
  renderTable(data);
  renderIndicators(data);
  renderAbout(data);
}

function renderSummary(data) {
  const lastClose = data.history.close_usd[data.history.close_usd.length - 1];
  const lastTime = fmtShortTs(data.history.timestamps[data.history.timestamps.length - 1]);
  const models = data.forecast.models;

  const cards = [
    `
      <div class="md-metric-card">
        <div class="md-metric-head">
          <span class="md-label-medium">Harga Terakhir</span>
          <span class="md-badge md-badge-neutral">Input</span>
        </div>
        <div class="md-metric-value">${fmtUSD(lastClose)}</div>
        <div class="md-body-small">${lastTime}</div>
      </div>
    `,
  ];

  const meta = {
    baseline: { title: "Prediksi Baseline (+24j)", badge: "Rekomendasi", badgeCls: "md-badge-primary", highlight: true },
    seq2seq: { title: "Prediksi Seq2Seq (+24j)", badge: "Eksperimen", badgeCls: "md-badge-neutral", highlight: false },
  };

  for (const key of ["baseline", "seq2seq"]) {
    const m = models[key];
    if (!m) continue;
    const s = m.summary;
    const info = meta[key];
    const isUp = s.change_24h_usd >= 0;
    const icon = isUp ? "trending_up" : "trending_down";
    const sign = isUp ? "+" : "";

    const foot =
      s.backtest_mae_usd !== undefined
        ? `<div class="md-metric-foot">Rata-rata error (MAE): <strong>${fmtUSD(s.backtest_mae_usd)}</strong> (${s.backtest_mape_pct}%)</div>`
        : `<div class="md-metric-foot">Rentang 24 jam: <strong>${fmtUSD(s.min_usd)}</strong> – <strong>${fmtUSD(s.max_usd)}</strong></div>`;

    cards.push(`
      <div class="md-metric-card ${info.highlight ? "primary-highlight" : ""}">
        <div class="md-metric-head">
          <span class="md-label-medium">${info.title}</span>
          <span class="md-badge ${info.badgeCls}">${info.badge}</span>
        </div>
        <div class="md-metric-value">${fmtUSD(s.predicted_24h_usd)}</div>
        <div class="md-delta-row">
          <span class="md-delta-chip ${isUp ? "md-delta-up" : "md-delta-down"}">
            <span class="material-symbols-rounded" style="font-size:15px">${icon}</span>
            ${sign}${s.change_24h_pct}%
          </span>
          <span class="md-body-small">${sign}${fmtUSD(s.change_24h_usd)}</span>
        </div>
        ${foot}
      </div>
    `);
  }

  els.summaryGrid.innerHTML = cards.join("");
}

function renderChart(data) {
  const histLabels = data.history.timestamps.map(fmtShortTs);
  const futLabels = data.forecast.timestamps.map(fmtShortTs);
  const allLabels = [...histLabels, ...futLabels];

  const histLen = histLabels.length;
  const futLen = futLabels.length;
  const lastVal = data.history.close_usd[histLen - 1];

  const histSeries = [...data.history.close_usd, ...Array(futLen).fill(null)];
  const padForecast = (arr) => [...Array(histLen - 1).fill(null), lastVal, ...arr];

  const ctx = document.getElementById("forecastChart").getContext("2d");
  const grad = ctx.createLinearGradient(0, 0, 0, 340);
  grad.addColorStop(0, "rgba(26, 115, 232, 0.16)");
  grad.addColorStop(1, "rgba(26, 115, 232, 0.0)");

  const datasets = [
    {
      label: "Historis (24 Jam)",
      data: histSeries,
      borderColor: "#1a73e8",
      backgroundColor: grad,
      fill: true,
      borderWidth: 2.5,
      pointRadius: 0,
      pointHoverRadius: 5,
      tension: 0.25,
    },
  ];

  if (data.forecast.actual_usd) {
    datasets.push({
      label: "Aktual (+24 Jam)",
      data: padForecast(data.forecast.actual_usd),
      borderColor: "#9334e6",
      borderWidth: 2.5,
      pointRadius: 0,
      pointHoverRadius: 5,
      tension: 0.25,
    });
  }

  if (data.forecast.models.baseline) {
    datasets.push({
      label: "Baseline LSTM",
      data: padForecast(data.forecast.models.baseline.usd),
      borderColor: "#188038",
      borderDash: [6, 4],
      borderWidth: 2.5,
      pointRadius: 0,
      pointHoverRadius: 5,
      tension: 0.25,
    });
  }

  if (data.forecast.models.seq2seq) {
    datasets.push({
      label: "Seq2Seq LSTM",
      data: padForecast(data.forecast.models.seq2seq.usd),
      borderColor: "#e37400",
      borderDash: [4, 4],
      borderWidth: 2.2,
      pointRadius: 0,
      pointHoverRadius: 5,
      tension: 0.25,
    });
  }

  const dividerPlugin = {
    id: "mdDivider",
    afterDatasetsDraw(chart) {
      const { ctx: c, chartArea, scales } = chart;
      if (!scales.x) return;
      const x = scales.x.getPixelForValue(histLen - 1);
      c.save();
      c.beginPath();
      c.setLineDash([4, 4]);
      c.strokeStyle = "#c4c7c5";
      c.lineWidth = 1.2;
      c.moveTo(x, chartArea.top);
      c.lineTo(x, chartArea.bottom);
      c.stroke();
      c.fillStyle = "#444746";
      c.font = "500 11px Roboto, sans-serif";
      c.fillText("Prediksi 24 jam →", x + 8, chartArea.top + 14);
      c.restore();
    },
  };

  if (state.chart) {
    state.chart.destroy();
  }

  state.chart = new Chart(ctx, {
    type: "line",
    data: { labels: allLabels, datasets },
    plugins: [dividerPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          align: "start",
          labels: {
            color: "#444746",
            usePointStyle: true,
            pointStyle: "line",
            boxWidth: 22,
            padding: 18,
            font: { family: "Roboto, sans-serif", size: 12, weight: "500" },
          },
        },
        tooltip: {
          backgroundColor: "#303134",
          titleColor: "#f8f9fa",
          bodyColor: "#e8eaed",
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            label: (c) => ` ${c.dataset.label}: ${fmtUSD(c.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#5f6368", maxTicksLimit: 8, font: { family: "Roboto, sans-serif", size: 11 } },
          grid: { display: false },
        },
        y: {
          ticks: {
            color: "#5f6368",
            font: { family: "Roboto, sans-serif", size: 11 },
            callback: (v) => "$" + Number(v).toLocaleString("en-US"),
          },
          grid: { color: "#eef2f6" },
        },
      },
    },
  });
}

function renderTable(data) {
  const ts = data.forecast.timestamps;
  const actual = data.forecast.actual_usd;
  const base = data.forecast.models.baseline?.usd;
  const s2s = data.forecast.models.seq2seq?.usd;

  const headers = ["Horizon", "Waktu"];
  if (actual) headers.push("Aktual");
  if (base) headers.push("Baseline LSTM");
  if (actual && base) headers.push("Selisih Baseline");
  if (s2s) headers.push("Seq2Seq LSTM");
  if (actual && s2s) headers.push("Selisih Seq2Seq");

  els.tableHead.innerHTML = `<tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr>`;

  els.tableBody.innerHTML = ts
    .map((t, i) => {
      const cols = [`<td>+${i + 1} jam</td>`, `<td>${fmtShortTs(t)}</td>`];
      if (actual) cols.push(`<td>${fmtUSD(actual[i])}</td>`);
      if (base) cols.push(`<td>${fmtUSD(base[i])}</td>`);
      if (actual && base) cols.push(`<td>${fmtUSD(Math.abs(actual[i] - base[i]))}</td>`);
      if (s2s) cols.push(`<td>${fmtUSD(s2s[i])}</td>`);
      if (actual && s2s) cols.push(`<td>${fmtUSD(Math.abs(actual[i] - s2s[i]))}</td>`);
      return `<tr>${cols.join("")}</tr>`;
    })
    .join("");
}

function renderIndicators(data) {
  const ind = data.history.latest_indicators || {};
  els.indicatorGrid.innerHTML = Object.entries(ind)
    .map(([k, v]) => {
      const display =
        k === "Volume USDT"
          ? Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 })
          : Number(v).toLocaleString("en-US", { maximumFractionDigits: 3 });
      return `
        <div class="md-ind-tile">
          <div class="md-title-small">${k}</div>
          <div class="md-body-small">${INDICATOR_DESC[k] || ""}</div>
          <div class="md-ind-val">${display}</div>
        </div>
      `;
    })
    .join("");
}

function renderAbout(data) {
  const scalerMode =
    data.scaler.mode === "exact_training"
      ? `Menggunakan <strong>MinMaxScaler eksak</strong> dari data training asli (${fmtUSD(data.scaler.close_min_usd)} – ${fmtUSD(data.scaler.close_max_usd)}).`
      : `Menggunakan <strong>scaler estimasi (2017–2021)</strong>. Jalankan <code>python3 deploy/extract_scaler.py /path/to/Bitcoin3.csv</code> untuk kalibrasi 100% eksak.`;

  els.aboutGrid.innerHTML = `
    <div class="md-about-tile">
      <h3>Baseline LSTM (Rekomendasi)</h3>
      <p>Memprediksi langsung 24 jam sekaligus menggunakan LSTM(64) + Custom Multi-Head Self-Attention (72.216 parameter). Nilai MAE uji di notebook: <strong>0.01149</strong>.</p>
    </div>
    <div class="md-about-tile">
      <h3>Seq2Seq LSTM</h3>
      <p>Encoder–Decoder LSTM(128) + Cross-Attention (170.049 parameter) yang memprediksi secara berantai per jam. Nilai MAE uji: <strong>0.02547</strong>.</p>
    </div>
    <div class="md-about-tile">
      <h3>Status Normalisasi (Scaler)</h3>
      <p>${scalerMode}</p>
    </div>
  `;
}

function exportCSV() {
  const data = state.lastPayload;
  if (!data) return;
  const ts = data.forecast.timestamps;
  const actual = data.forecast.actual_usd;
  const base = data.forecast.models.baseline?.usd;
  const s2s = data.forecast.models.seq2seq?.usd;

  const header = ["Horizon_Hour", "Timestamp"];
  if (actual) header.push("Actual_USD");
  if (base) header.push("Baseline_LSTM_USD");
  if (s2s) header.push("Seq2Seq_LSTM_USD");

  const lines = [header.join(",")];
  ts.forEach((t, i) => {
    const r = [i + 1, t];
    if (actual) r.push(actual[i]);
    if (base) r.push(base[i]);
    if (s2s) r.push(s2s[i]);
    lines.push(r.join(","));
  });

  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "prediksi_bitcoin_24h.csv";
  a.click();
  URL.revokeObjectURL(url);
}

// ---------- Events ----------
els.sourceTabs.forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    state.activeTab = tab;
    els.sourceTabs.forEach((b) => b.classList.toggle("active", b === btn));
    Object.entries(els.subSections).forEach(([k, el]) => el.classList.toggle("active", k === tab));
    if (tab !== "csv") {
      runForecast();
    }
  });
});

els.modelBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    state.selectedModel = btn.dataset.model;
    els.modelBtns.forEach((b) => b.classList.toggle("active", b === btn));
    if (state.activeTab !== "csv" || els.csvFile.files.length > 0) {
      runForecast();
    }
  });
});

els.sampleOffset.addEventListener("input", () => {
  const val = Number(els.sampleOffset.value);
  els.offsetVal.textContent = val === 0 ? "Terbaru" : `-${val} jam`;
});

els.sampleOffset.addEventListener("change", () => runForecast());
els.sampleBacktest.addEventListener("change", () => runForecast());

els.csvFile.addEventListener("change", () => {
  const file = els.csvFile.files[0];
  els.csvFileName.textContent = file ? `File terpilih: ${file.name}` : "Pilih atau seret file .csv ke sini";
});

["dragenter", "dragover"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("dragover");
  })
);

els.dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  els.csvFile.files = dt.files;
  els.csvFileName.textContent = `File terpilih: ${file.name}`;
  runForecast();
});

els.detailTabs.forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.detail;
    els.detailTabs.forEach((b) => b.classList.toggle("active", b === btn));
    els.detailPanels.forEach((p) => p.classList.toggle("active", p.id === `detail-${target}`));
  });
});

els.runBtn.addEventListener("click", runForecast);
els.exportCsvBtn.addEventListener("click", exportCSV);

// Boot
fetchHealth().then(() => runForecast());
