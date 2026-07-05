# Asset Health Monitoring — Dokumentasi End-to-End

> **Project:** Predictive Maintenance & Asset Health Monitoring  
> **Stack:** Python 3.10 · Pandas · Scikit-learn · Streamlit · Plotly  
> **Dataset:** Microsoft Azure AI Gallery — Predictive Maintenance (PdM)

---

## Daftar Isi

1. [Overview Proyek](#1-overview-proyek)
2. [Struktur Direktori](#2-struktur-direktori)
3. [Dataset](#3-dataset)
4. [Notebook 01 — Exploratory Data Analysis (EDA)](#4-notebook-01--exploratory-data-analysis-eda)
5. [Notebook 02 — Preprocessing & Feature Engineering](#5-notebook-02--preprocessing--feature-engineering)
6. [Notebook 03 — Modeling & Health Scoring](#6-notebook-03--modeling--health-scoring)
7. [Dashboard Streamlit](#7-dashboard-streamlit)
8. [Cara Menjalankan](#8-cara-menjalankan)
9. [Hasil & Evaluasi Model](#9-hasil--evaluasi-model)

---

## 1. Overview Proyek

Proyek ini membangun sistem **Asset Health Monitoring** untuk armada mesin industri. Tujuannya adalah mendeteksi anomali dan memprediksi potensi kegagalan mesin sebelum terjadi, sehingga tim maintenance bisa bertindak proaktif (*predictive maintenance*) daripada reaktif.

### Pendekatan
- **Unsupervised anomaly detection** menggunakan **Isolation Forest** sebagai model utama
- **One-Class SVM** sebagai baseline pembanding
- Model dilatih **hanya pada data normal** (tanpa label failure) — murni unsupervised
- Label failure (`will_fail_24h`) digunakan **hanya untuk evaluasi**, bukan training
- Output akhir: **Health Score 0–100** per mesin per jam, divisualisasikan di dashboard

### Pipeline Singkat
```
Raw CSV Data
    ↓ [Notebook 01] EDA — Eksplorasi & Visualisasi
    ↓ [Notebook 02] Preprocessing — Feature Engineering
    ↓ data/processed/features.parquet
    ↓ [Notebook 03] Modeling — Isolation Forest + Health Score
    ↓ data/processed/scored.parquet
    ↓ [Dashboard] Streamlit App — Monitoring Real-time
```

---

## 2. Struktur Direktori

```
asset-health-monitoring/
├── data/
│   ├── raw/                    ← Dataset CSV original (5 file)
│   └── processed/
│       ├── features.parquet    ← Output Notebook 02
│       └── scored.parquet      ← Output Notebook 03 (+ anomaly scores)
├── models/
│   ├── isolation_forest.pkl    ← Trained IF model
│   ├── one_class_svm.pkl       ← Trained OC-SVM model
│   ├── scaler.pkl              ← StandardScaler + feature list
│   └── feature_cols.json       ← Daftar nama feature columns
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  ← Feature Engineering
│   └── 03_modeling.ipynb       ← Model Training & Evaluation
├── dashboard/
│   └── app.py                  ← Streamlit Dashboard
├── reports/
│   └── figures/                ← Plot yang disimpan dari notebook
├── requirements.txt
└── PROJECT_DOCUMENTATION.md    ← File ini
```

---

## 3. Dataset

### Sumber
Dataset **Microsoft Azure AI Gallery — Predictive Maintenance** (Kaggle PdM Dataset). Dataset ini mensimulasikan operasi armada 100 mesin industri selama 1 tahun (Jan 2015 – Jan 2016).

### File Raw (5 file CSV)

| File | Ukuran | Deskripsi |
|------|--------|-----------|
| `PdM_telemetry.csv` | ~80 MB | Sensor readings setiap jam: volt, rotate, pressure, vibration |
| `PdM_errors.csv` | ~129 KB | Error codes yang dihasilkan mesin (error1–error5) |
| `PdM_failures.csv` | ~24 KB | Catatan kegagalan komponen (comp1–comp4) |
| `PdM_maint.csv` | ~105 KB | Catatan maintenance per komponen |
| `PdM_machines.csv` | ~2 KB | Metadata mesin: model (model1–model4), age |

### Statistik Dataset
- **100 mesin** (machineID 1–100)
- **876,100 baris** total setelah preprocessing (8,761 jam × 100 mesin)
- **Date range:** 2015-01-01 → 2016-01-01
- **Sensor:** 4 sensor (volt, rotate, pressure, vibration), frekuensi 1 jam
- **Imbalance:** hanya **2.04%** baris yang merupakan pre-failure window (24 jam sebelum failure)

### Sensor Readings
| Sensor | Unit | Deskripsi |
|--------|------|-----------|
| `volt` | V | Tegangan operasi |
| `rotate` | rpm | Kecepatan rotasi |
| `pressure` | psi | Tekanan operasi |
| `vibration` | mm/s | Level vibrasi |

---

## 4. Notebook 01 — Exploratory Data Analysis (EDA)

**File:** `notebooks/01_eda.ipynb`  
**Output:** Visualisasi & insight, plot disimpan di `reports/figures/`

### Yang Dilakukan

#### 4.1 Setup & Load Data
Membaca kelima file CSV, parsing datetime, dan eksplorasi awal setiap tabel.

#### 4.2 Telemetry Overview
- Distribusi statistik (mean, std, min, max) untuk setiap sensor
- Time series plot untuk beberapa mesin sampel
- Deteksi outlier awal

#### 4.3 Failure Analysis
- Distribusi failure per komponen (comp1–comp4)
- Frekuensi failure per mesin
- Temporal pattern — kapan failure paling sering terjadi

#### 4.4 Error Pattern Analysis
- Distribusi error codes (error1–error5)
- Korelasi antara error codes dan failure events
- Error frequency sebelum dan sesudah failure

#### 4.5 Maintenance Analysis
- Frekuensi maintenance per komponen
- Hubungan antara maintenance interval dan failure rate

#### 4.6 Machine Metadata Analysis
- Distribusi mesin per model (model1–model4)
- Korelasi umur mesin (`age`) dengan failure rate

#### 4.7 Sensor vs Failure Correlation
- Boxplot sensor readings: normal vs pre-failure window
- Visualisasi perubahan sensor menjelang failure

---

## 5. Notebook 02 — Preprocessing & Feature Engineering

**File:** `notebooks/02_preprocessing.ipynb`  
**Input:** 5 file CSV di `data/raw/`  
**Output:** `data/processed/features.parquet`, `models/scaler.pkl`, `models/feature_cols.json`

### Pipeline Feature Engineering

#### 5.1 Merge & Sort
Semua 5 tabel digabungkan. Basis adalah `PdM_telemetry.csv`, kemudian di-merge dengan:
- `PdM_machines.csv` → tambah `model`, `age`, `model_code`

Data di-sort berdasarkan `machineID` + `datetime` — **kritis** untuk rolling calculation.

#### 5.2 Rolling Window Features
Untuk setiap sensor (`volt`, `rotate`, `pressure`, `vibration`), dihitung:

| Feature | Formula | Tujuan |
|---------|---------|--------|
| `{sensor}_mean3h` | Rolling mean 3 jam | Tren jangka pendek |
| `{sensor}_std3h` | Rolling std 3 jam | Volatilitas jangka pendek |
| `{sensor}_mean24h` | Rolling mean 24 jam | Tren jangka menengah |
| `{sensor}_std24h` | Rolling std 24 jam | Volatilitas jangka menengah |
| `{sensor}_roc` | Diff 1 step | Rate of change |

> **Penting:** Rolling dihitung **per mesin** (`groupby machineID`) agar tidak ada data leakage antar mesin.

Total: **20 rolling features** (4 sensor × 5 statistik)

#### 5.3 Lag Features
Nilai sensor pada t-1, t-3, dan t-6 jam sebelumnya:

```
{sensor}_lag1h, {sensor}_lag3h, {sensor}_lag6h
```

Total: **12 lag features** (4 sensor × 3 lag)

NaN akibat lag diisi dengan forward-fill lalu backward-fill.

#### 5.4 Error Count Features
Untuk setiap error type (error1–error5), dihitung berapa kali muncul dalam **24 jam terakhir** per mesin:

```
error1_count24h, error2_count24h, ..., error5_count24h
total_error_count24h
```

Total: **6 error features**

#### 5.5 Days Since Last Maintenance
Untuk setiap timestamp per mesin, dihitung berapa hari sejak maintenance terakhir:
```
days_since_maint = (current_time - last_maintenance_time) / 86400
```
Di-clip ke range [0, 365]. Jika belum pernah ada maintenance, nilainya 999 → di-clip ke 365.

#### 5.6 Label Engineering
Dibuat dua kolom label (bukan untuk training, hanya untuk evaluasi):

| Kolom | Definisi |
|-------|----------|
| `hours_to_failure` | Jam hingga failure berikutnya (∞ jika tidak ada) |
| `will_fail_24h` | 1 jika failure dalam 24 jam ke depan, 0 jika tidak |
| `will_fail_48h` | 1 jika failure dalam 48 jam ke depan, 0 jika tidak |

**Class imbalance:** hanya ~2.04% baris yang `will_fail_24h = 1`

#### 5.7 Feature Set Final
44 features total, tidak termasuk kolom label dan metadata:

```
Raw sensors (4):       volt, rotate, pressure, vibration
Rolling mean (8):      {sensor}_mean3h, {sensor}_mean24h
Rolling std (8):       {sensor}_std3h, {sensor}_std24h
Rate of change (4):    {sensor}_roc
Lag features (12):     {sensor}_lag1h, lag3h, lag6h
Error counts (6):      error{1-5}_count24h, total_error_count24h
Maintenance (1):       days_since_maint
Machine metadata (2):  model_code, age
```

#### 5.8 Normalisasi
`StandardScaler` di-fit pada **seluruh dataset** dan disimpan di `models/scaler.pkl`. Setiap feature disimpan dalam dua versi: raw dan scaled (`{feature}_scaled`).

#### 5.9 Output
- `data/processed/features.parquet` — dataset lengkap (~876K baris, 96 kolom)
- `models/scaler.pkl` — artifact berisi `scaler` + `feature_cols` + `n_features`
- `models/feature_cols.json` — list nama feature columns

---

## 6. Notebook 03 — Modeling & Health Scoring

**File:** `notebooks/03_modeling.ipynb`  
**Input:** `data/processed/features.parquet`  
**Output:** `models/isolation_forest.pkl`, `models/one_class_svm.pkl`, `data/processed/scored.parquet`

### 6.1 Train/Test Split

**Strategi: Time-based split** (menghindari data leakage temporal)

| Set | Period | Baris |
|-----|--------|-------|
| Training (normal only) | Jan–Sep 2015 | 615,203 |
| Test | Okt–Des 2015 | 221,500 |

**Training hanya menggunakan data normal:** baris dengan `hours_to_failure > 72` (lebih dari 72 jam sebelum failure). Model harus belajar distribusi "normal" tanpa melihat pre-failure data sama sekali.

### 6.2 Isolation Forest (Model Utama)

#### Cara Kerja
Isolation Forest secara random membagi data menggunakan decision trees. Anomali lebih mudah "diisolasi" karena mereka jauh dari mayoritas data — butuh lebih sedikit splits.

**Score = rata-rata kedalaman isolasi.** Makin dangkal → makin anomali.

#### Parameter
```python
IsolationForest(
    n_estimators  = 200,   # 200 trees untuk stabilitas
    contamination = 0.05,  # ekspektasi 5% data adalah anomali
    max_samples   = 'auto',
    random_state  = 42,
    n_jobs        = -1     # parallelisasi semua CPU
)
```

#### Output Kolom
| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `if_score` | float | Raw anomaly score (higher = more normal) |
| `if_anomaly` | int | -1 = anomali, 1 = normal |
| `if_anomaly_01` | int | 0/1 encoding (1 = anomali) |

**Anomali terdeteksi:** 63,771 baris (7.3% dari semua data)

### 6.3 One-Class SVM (Baseline)

One-Class SVM mencari hyperplane yang memisahkan data normal dari asal koordinat. Lebih lambat (O(n²)) sehingga dilatih pada subsample 20,000 baris.

```python
OneClassSVM(
    kernel = 'rbf',
    nu     = 0.05,    # mirip contamination di IF
    gamma  = 'scale'
)
```

### 6.4 Evaluasi Model

Dievaluasi pada test set (Okt–Des 2015) menggunakan `will_fail_24h` sebagai ground truth.

#### Hasil Evaluasi

| Metrik | Isolation Forest | One-Class SVM |
|--------|-----------------|---------------|
| **Recall (Pre-failure)** | **96%** | **95%** |
| Precision (Pre-failure) | 25% | 26% |
| F1-score (Pre-failure) | 0.40 | 0.40 |
| Average Precision | 0.6421 | 0.7096 |
| TN | 205,413 | 205,594 |
| **FN (Missed failures)** | **188** | **198** |
| TP (Caught failures) | 4,033 | 4,023 |
| FP (False alarms) | 11,866 | 11,685 |

> **Interpretasi:** Recall ~96% artinya model berhasil mendeteksi 96% dari semua pre-failure events. False positive rate tinggi adalah trade-off yang diterima — lebih baik false alarm daripada missed failure pada konteks predictive maintenance.

### 6.5 Lead Time Analysis

Analisis berapa jam **sebelum** failure anomali mulai terdeteksi. Ini mengukur seberapa dini sistem memberikan peringatan.

### 6.6 Asset Health Score (0–100)

Health Score adalah transformasi dari `if_score` ke skala 0–100 yang intuitif:

```python
# Normalisasi IF score ke [0, 1]
score_norm = (if_score - score_min) / (score_max - score_min)

# Invert: 0 = paling anomali, 1 = paling normal
health_score = score_norm × 100
```

**Interpretasi Health Score:**
| Range | Status | Arti |
|-------|--------|------|
| 70–100 | 🟢 **Healthy** | Operasi normal |
| 40–70 | 🟡 **Warning** | Perlu perhatian |
| 0–40 | 🔴 **Critical** | Tindakan segera diperlukan |

(threshold dapat dikustomisasi di dashboard)

### 6.7 Output
- `models/isolation_forest.pkl` — trained IF model
- `models/one_class_svm.pkl` — trained OC-SVM model
- `data/processed/scored.parquet` — dataset lengkap + semua anomaly scores + health scores

---

## 7. Dashboard Streamlit

**File:** `dashboard/app.py`  
**URL:** http://localhost:8501 (saat dijalankan)

Dashboard dibagi menjadi **4 halaman navigasi**:

### 7.1 Fleet Overview

Tampilan bird's-eye seluruh armada mesin.

**KPI Metrics (row 1 — status saat ini):**
- Total Machines
- 🔴 Critical count
- 🟡 Warning count
- 🟢 Healthy count
- ⚠️ Total Anomaly Events

**KPI Metrics (row 2 — forecast outlook):**
- 🚨 High-Risk (72h) — jumlah mesin diprediksi warning/critical dalam 72 jam
- 📅 Critical Risk (7d) — jumlah mesin diprediksi masuk zona critical dalam 7 hari
- 📉 Machines Declining — mesin dengan tren penurunan > 2 pts/day
- 🏥 Avg Fleet Score (7d) — proyeksi rata-rata health score fleet

**Visualisasi:**
- **Bar chart:** Health Score setiap mesin, berwarna merah/kuning/hijau sesuai status
- **Pie/Donut chart:** Proporsi status fleet
- **Machine Status Table:** Tabel lengkap semua mesin dengan kolom: Machine ID, Model, Age, Health Score, Status, Errors 24h, Days Since Maint., **Forecast 72h**, **Forecast 7d**, **Risk (7d)**
- **Export CSV button**

### 7.2 Machine Detail

Deep-dive untuk satu mesin yang dipilih.

**Fitur:**
- **5 KPI cards:** Machine ID, Model, Age, Status, Health Score
- **Gauge chart:** Health score visual dengan zona merah/kuning/hijau
- **Sensor time series (4 panel):** Volt, Rotate, Pressure, Vibration — menampilkan raw reading, 24h rolling mean, dan marker anomali (merah)
- **Health Score Timeline:** Grafik health score over time + anomaly markers
- **🔮 Forecast Section** *(baru)*:
  - 4 metric cards: Score 72h (±CI), Score 7d (±CI), Trend pts/day, Days to Critical
  - Forecast chart: historical 14d + proyeksi 7d + confidence interval shaded band + threshold lines + marker "Now"

### 7.3 Anomaly Timeline

Visualisasi temporal seluruh anomali di fleet.

**Fitur:**
- **Scatter heatmap:** X = tanggal, Y = Machine ID, ukuran titik = jumlah anomali per hari, warna = avg health score
- **Fleet-level daily chart:**
  - Daily anomaly count (bar chart)
  - Fleet average health score (area chart)
- **Top 10 Machines by Anomaly Count** (tabel)

### 7.4 🔮 Anomaly Forecast *(baru)*

Halaman khusus untuk proyeksi potensi anomali, warnings, dan critical events ke depan.

**Metode:**
- **Layer 1 — Linear Regression** pada health score 14 hari terakhir → ekstrapolasi ke +72h dan +7 hari
- **Layer 2 — Exponential Smoothing** (α=0.3) pada daily anomaly rate → proyeksi ke depan
- Confidence interval dihitung dari residual standar deviasi linear fit (±1 std)

**Output per mesin:**

| Kolom | Deskripsi |
|-------|-----------|
| `forecast_72h` | Prediksi health score 72 jam ke depan |
| `forecast_7d` | Prediksi health score 7 hari ke depan |
| `ci_72h / ci_7d` | Confidence interval (±1–1.5× residual std) |
| `trend_slope` | Kecepatan perubahan health score (pts/day) |
| `anom_rate_current` | Anomaly rate terkini (per hari, %) |
| `anom_rate_7d` | Proyeksi anomaly rate 7 hari ke depan |
| `days_to_critical` | Estimasi hari hingga masuk zona critical (jika tren berlanjut) |
| `risk_level` | Low / Medium / High / Critical |

**Visualisasi di halaman Anomaly Forecast:**
- **Forecast Summary KPIs** — count mesin per risk level
- **Forecast Risk Table** — tabel semua mesin, sortable, dengan export CSV
- **Risk Distribution Donut** — proporsi Critical/High/Medium/Low
- **Now vs Forecast 7d Scatter** — current score vs projected score per mesin
- **Projected Health Score Chart** — multi-line top 10 at-risk machines (historical + forecast)
- **7-Day Risk Calendar** — heatmap grid: mesin × hari ke depan, warna merah/kuning/hijau
- **Anomaly Rate Bar Chart** — current vs projected anomaly rate, top 20 at-risk machines

**Risk Level Klasifikasi:**

| Risk | Kondisi |
|------|---------|
| 🔴 Critical | forecast score < Critical threshold ATAU anomaly rate > 50% |
| 🟠 High | forecast score < Warning threshold ATAU anomaly rate > 35% |
| 🟡 Medium | forecast score < Warning+10 ATAU anomaly rate > 20% |
| 🟢 Low | tidak memenuhi kondisi di atas |

### 7.5 Sidebar Filters

| Filter | Tipe | Default |
|--------|------|---------|
| Machine model | Multiselect | Semua model |
| Date range | Date picker | Full range |
| Critical threshold | Slider | 40 |
| Warning threshold | Slider | 70 |

---

## 8. Cara Menjalankan

### Prerequisites
- Python 3.10
- Virtual environment (venv) di `/Users/aryautomo/venv`

### Setup Kernel Jupyter
```bash
# Kernel sudah terdaftar dengan nama "Python (asset-health)"
# Buka notebook → pilih kernel "Python (asset-health)"
```

### Jalankan Notebooks (urutan wajib)
```bash
cd asset-health-monitoring

# 1. EDA (opsional, tidak menghasilkan file yang dibutuhkan notebook berikutnya)
/Users/aryautomo/venv/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/01_eda.ipynb

# 2. Preprocessing (WAJIB sebelum notebook 03)
/Users/aryautomo/venv/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  notebooks/02_preprocessing.ipynb

# 3. Modeling
/Users/aryautomo/venv/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  notebooks/03_modeling.ipynb
```

### Jalankan Dashboard
```bash
cd asset-health-monitoring
/Users/aryautomo/venv/bin/streamlit run dashboard/app.py --server.port 8501
```

Buka browser: **http://localhost:8501**

### Estimasi Waktu Eksekusi
| Notebook | Estimasi |
|----------|----------|
| 01 EDA | ~30 detik |
| 02 Preprocessing | ~5–15 menit |
| 03 Modeling | ~1–2 menit |

---

## 9. Hasil & Evaluasi Model

### Ringkasan Performa

| | Isolation Forest | One-Class SVM |
|--|--|--|
| **Recall (failure detection)** | **96%** | 95% |
| **False Negatives (missed)** | **188 / 4,221** | 198 / 4,221 |
| Average Precision Score | 0.642 | 0.710 |
| Training time | ~12 detik | ~20 detik |

### Kenapa Isolation Forest Dipilih?

1. **Recall tinggi (96%):** Hampir semua pre-failure window berhasil dideteksi. Dalam konteks keselamatan industri, *missed failure* jauh lebih berbahaya daripada false alarm.
2. **Scalable:** Linear complexity O(n), jauh lebih cepat dari OC-SVM (O(n²)) untuk data besar.
3. **Tidak perlu labeled data:** Bekerja murni unsupervised, cocok untuk kasus di mana labeled failure data sedikit atau tidak ada.
4. **Interpretable:** Decision tree-based, relatif mudah dijelaskan.

### Trade-off
- **Precision rendah (25%):** Banyak false positives. Ini adalah trade-off yang disengaja — threshold `contamination=0.05` dan Health Score thresholds dapat di-tuning untuk mengurangi false alarm sesuai kebutuhan operasional.

### Potensi Pengembangan
- [ ] Tuning `contamination` parameter untuk reduce false positives
- [ ] LSTM/temporal model untuk menangkap sequential patterns
- [ ] Per-component failure prediction (comp1–comp4 separately)
- [ ] Real-time streaming data integration
- [ ] Alert system (email/slack notification)
- [ ] Root cause analysis per anomali
- [x] **Anomaly Forecasting** — proyeksi health score & anomaly rate ke depan (72h & 7d) ✅
