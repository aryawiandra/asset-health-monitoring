<div align="center">

# 🔧 Asset Health Monitoring

**Predictive Maintenance System untuk Armada Mesin Industri**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-686%20MB-lightgrey?style=flat)](https://git-lfs.github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Pertamina EDM — Mini Project Kerja Praktik*

</div>

---

## 📖 Tentang Proyek

Sistem **Asset Health Monitoring** yang mendeteksi anomali dan memproyeksikan potensi kegagalan mesin industri secara proaktif menggunakan pendekatan **unsupervised machine learning**. Dashboard interaktif memvisualisasikan kondisi kesehatan armada 100 mesin secara real-time, dilengkapi fitur **Anomaly Forecasting** yang memprediksi risiko 72 jam dan 7 hari ke depan.

### ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 🔍 **Anomaly Detection** | Isolation Forest + One-Class SVM mendeteksi anomali tanpa labeled data |
| 🏥 **Health Score** | Skor 0–100 per mesin per jam, intuitif dan dapat dikustomisasi |
| 🔮 **Anomaly Forecast** | Proyeksi health score 72h & 7d ke depan + 7-Day Risk Calendar |
| 📊 **Fleet Overview** | Bird's-eye view 100 mesin + forecast KPIs |
| 🔬 **Machine Detail** | Deep-dive per mesin dengan gauge, sensor time series, forecast chart |
| ⚠️ **Anomaly Timeline** | Heatmap temporal seluruh anomali di fleet |

---

## 🏗️ Arsitektur Pipeline

```
Raw CSV Data (5 file)
    │
    ▼
[Notebook 01] Exploratory Data Analysis
    │
    ▼
[Notebook 02] Preprocessing & Feature Engineering
    │  → 44 features: sensor rolling stats, lag features, error counts, maintenance
    │
    ▼
data/processed/features.parquet
    │
    ▼
[Notebook 03] Model Training
    │  → Isolation Forest (n=200, contamination=0.05)
    │  → One-Class SVM (baseline)
    │  → Health Score normalization (IF score → 0–100)
    │
    ▼
data/processed/scored.parquet
    │
    ▼
[Dashboard] Streamlit App
    │  → Fleet Overview + Forecast KPIs
    │  → Machine Detail + Forecast Chart
    │  → Anomaly Timeline
    └  → 🔮 Anomaly Forecast Page (baru)
```

---

## 📁 Struktur Direktori

```
asset-health-monitoring/
├── 📂 data/
│   ├── raw/                        ← Dataset CSV original (5 file)
│   │   ├── PdM_telemetry.csv       ← 76 MB — sensor readings per jam [LFS]
│   │   ├── PdM_errors.csv          ← Error codes mesin
│   │   ├── PdM_failures.csv        ← Catatan kegagalan komponen
│   │   ├── PdM_maint.csv           ← Catatan maintenance
│   │   └── PdM_machines.csv        ← Metadata mesin
│   └── processed/
│       ├── features.parquet        ← Output Notebook 02 [LFS, 503 MB]
│       └── scored.parquet          ← Output Notebook 03 [LFS, 72 MB]
│
├── 📂 models/
│   ├── isolation_forest.pkl        ← Trained IF model [LFS]
│   ├── one_class_svm.pkl           ← Trained OC-SVM model [LFS]
│   ├── scaler.pkl                  ← StandardScaler [LFS]
│   └── feature_cols.json           ← Feature column names
│
├── 📂 notebooks/
│   ├── 01_eda.ipynb                ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      ← Feature Engineering
│   └── 03_modeling.ipynb           ← Model Training & Evaluation
│
├── 📂 dashboard/
│   └── app.py                      ← Streamlit Dashboard (4 halaman)
│
├── 📂 reports/figures/             ← Plot dari notebook (12 gambar)
├── .gitattributes                  ← Git LFS tracking rules
├── .gitignore
├── requirements.txt
├── PROJECT_DOCUMENTATION.md        ← Dokumentasi teknis lengkap
└── README.md                       ← File ini
```

> **Git LFS**: File berukuran besar (`.parquet`, `.pkl`, `PdM_telemetry.csv`) di-track via Git LFS. Pastikan `git-lfs` terinstall sebelum clone.

---

## 🤖 Model & Performa

### Isolation Forest (Model Utama)

```python
IsolationForest(
    n_estimators  = 200,    # 200 trees untuk stabilitas
    contamination = 0.05,   # ekspektasi 5% data anomali
    random_state  = 42,
    n_jobs        = -1      # parallelisasi semua CPU
)
```

### Hasil Evaluasi (Test Set: Okt–Des 2015)

| Metrik | Isolation Forest | One-Class SVM |
|--------|:---:|:---:|
| **Recall (Pre-failure)** | **96%** | 95% |
| Precision (Pre-failure) | 25% | 26% |
| F1-score | 0.40 | 0.40 |
| Average Precision | 0.642 | 0.710 |
| False Negatives (missed) | **188 / 4,221** | 198 / 4,221 |
| Training time | ~12 detik | ~20 detik |

> 💡 **Recall 96%** diprioritaskan karena dalam konteks keselamatan industri, *missed failure* jauh lebih berbahaya daripada false alarm.

### Health Score

```
Health Score = normalize(IF anomaly score) × 100

🟢 70–100  →  Healthy   (operasi normal)
🟡 40–70   →  Warning   (perlu perhatian)
🔴  0–40   →  Critical  (tindakan segera)
```

Threshold dapat dikustomisasi secara interaktif di sidebar dashboard.

---

## 🔮 Anomaly Forecasting

Fitur baru yang memproyeksikan potensi anomali, warnings, dan critical events ke depan menggunakan **two-layer approach**:

### Metode
| Layer | Teknik | Output |
|-------|--------|--------|
| 1 | **Linear Regression** pada health score 14 hari terakhir | Forecast score +72h & +7d |
| 2 | **Exponential Smoothing** (α=0.3) pada daily anomaly rate | Projected anomaly rate |

### Risk Classification
| Level | Kondisi |
|-------|---------|
| 🔴 **Critical** | Forecast score < Critical threshold **atau** anomaly rate > 50% |
| 🟠 **High** | Forecast score < Warning threshold **atau** anomaly rate > 35% |
| 🟡 **Medium** | Forecast score < Warning+10 **atau** anomaly rate > 20% |
| 🟢 **Low** | Tidak memenuhi kondisi di atas |

### Output per Mesin
- `forecast_72h` — prediksi health score 72 jam ke depan
- `forecast_7d` — prediksi health score 7 hari ke depan
- `confidence interval` — ±1–1.5× residual std dari linear fit
- `trend_slope` — kecepatan perubahan (pts/day)
- `days_to_critical` — estimasi hari hingga zona critical
- `risk_level` — Low / Medium / High / Critical

---

## 🚀 Cara Menjalankan

### Prerequisites

```bash
# Install git-lfs (wajib untuk clone file besar)
brew install git-lfs   # macOS
git lfs install
```

### 1. Clone Repository

```bash
git clone https://github.com/aryawiandra/asset-health-monitoring.git
cd asset-health-monitoring
```

> ⚠️ Clone akan otomatis download file LFS (~686 MB). Pastikan koneksi internet stabil.

### 2. Setup Environment

```bash
python3.10 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Jalankan Notebooks (opsional — data sudah tersedia via LFS)

```bash
# Urutan wajib jika ingin re-generate dari scratch:

# EDA (opsional)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/01_eda.ipynb

# Preprocessing (wajib sebelum modeling)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 notebooks/02_preprocessing.ipynb

# Modeling
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 notebooks/03_modeling.ipynb
```

| Notebook | Estimasi Waktu |
|----------|:---:|
| 01 EDA | ~30 detik |
| 02 Preprocessing | ~5–15 menit |
| 03 Modeling | ~1–2 menit |

### 4. Jalankan Dashboard

```bash
streamlit run dashboard/app.py --server.port 8501
```

Buka browser: **http://localhost:8501**

---

## 📊 Dashboard

Dashboard dibagi menjadi **4 halaman navigasi**:

### 🔧 Fleet Overview
- KPI row 1: Total Machines, Critical, Warning, Healthy, Anomaly Events
- KPI row 2 *(forecast)*: High-Risk 72h, Critical Risk 7d, Machines Declining, Avg Fleet Score 7d
- Bar chart health score distribution + donut status breakdown
- Machine Status Table dengan kolom forecast & export CSV

### 🔍 Machine Detail
- Gauge chart health score dengan zona warna
- Sensor time series (Volt, Rotate, Pressure, Vibration) + anomaly markers
- Health Score Timeline + anomaly overlay
- **🔮 Forecast Section**: Score 72h/7d, Trend, Days to Critical, Forecast Chart dengan confidence interval

### ⚠️ Anomaly Timeline
- Scatter heatmap: tanggal × Machine ID, ukuran = jumlah anomali/hari
- Fleet-level daily chart: anomaly count + avg health score
- Top 10 Machines by Anomaly Count

### 🔮 Anomaly Forecast *(baru)*
- Forecast Summary KPIs (Critical/High/Medium/Low risk count)
- Forecast Risk Table — sortable, exportable CSV
- Risk Distribution Donut + Now vs 7d Scatter
- Projected Health Score Chart — top 10 at-risk machines
- **7-Day Risk Calendar** — heatmap mesin × hari, warna-coded
- Anomaly Rate Bar Chart — current vs projected

---

## 📦 Dataset

**Microsoft Azure AI Gallery — Predictive Maintenance Dataset**

| File | Ukuran | Deskripsi |
|------|:---:|-----------|
| `PdM_telemetry.csv` | 76 MB | Sensor readings per jam: volt, rotate, pressure, vibration |
| `PdM_errors.csv` | 128 KB | Error codes (error1–error5) |
| `PdM_failures.csv` | 24 KB | Catatan kegagalan komponen (comp1–comp4) |
| `PdM_maint.csv` | 104 KB | Catatan maintenance per komponen |
| `PdM_machines.csv` | 4 KB | Metadata: model (model1–model4), age |

**Statistik:** 100 mesin × 8,761 jam = **876,100 baris** | Jan 2015 – Jan 2016

---

## 🔧 Feature Engineering

**44 features** digunakan sebagai input model:

```
Raw sensors (4):       volt, rotate, pressure, vibration
Rolling mean (8):      {sensor}_mean3h, {sensor}_mean24h
Rolling std (8):       {sensor}_std3h, {sensor}_std24h
Rate of change (4):    {sensor}_roc
Lag features (12):     {sensor}_lag1h, _lag3h, _lag6h
Error counts (6):      error{1-5}_count24h, total_error_count24h
Maintenance (1):       days_since_maint
Machine metadata (2):  model_code, age
```

---

## 📋 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
pyarrow
joblib
```

Lihat [`requirements.txt`](requirements.txt) untuk versi lengkap.

---

## 📈 Potensi Pengembangan

- [ ] LSTM/temporal model untuk menangkap sequential patterns
- [ ] Per-component failure prediction (comp1–comp4 separately)
- [ ] Real-time streaming data integration
- [ ] Alert system (email/Slack notification)
- [ ] Root cause analysis per anomali
- [ ] Tuning `contamination` parameter untuk reduce false positives
- [x] **Anomaly Forecasting** — proyeksi 72h & 7d ✅

---

## 📄 Lisensi

MIT License — lihat [LICENSE](LICENSE) untuk detail.

---

<div align="center">
  <sub>Built with ❤️ for Pertamina EDM Mini Project KP</sub>
</div>
