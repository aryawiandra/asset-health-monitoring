<div align="center">

# 🔧 Asset Health Monitoring

**Predictive Maintenance System for Industrial Machine Fleets**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-686%20MB-lightgrey?style=flat)](https://git-lfs.github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Pertamina EDM — Internship Mini Project*

</div>

---

# 📖 Overview

**Asset Health Monitoring** is a predictive maintenance system that proactively detects anomalies and estimates potential machine failures using **unsupervised machine learning**.

The interactive dashboard provides real-time fleet health monitoring for 100 industrial machines and includes an **Anomaly Forecasting** module capable of projecting maintenance risks **72 hours** and **7 days** ahead.

---

# ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Anomaly Detection** | Detects abnormal operating conditions using Isolation Forest and One-Class SVM without labeled data |
| 🏥 **Health Score** | Intuitive 0–100 machine health score computed hourly |
| 🔮 **Anomaly Forecasting** | Forecasts health score 72 hours and 7 days ahead with a 7-Day Risk Calendar |
| 📊 **Fleet Overview** | Fleet-wide monitoring dashboard with forecast KPIs |
| 🔬 **Machine Detail** | Individual machine analysis including gauges, sensor trends, and forecast charts |
| ⚠️ **Anomaly Timeline** | Fleet-wide temporal visualization of anomaly events |

---

# 🏗 Pipeline Architecture

```
Raw CSV Data (5 files)
    │
    ▼
Notebook 01 - Exploratory Data Analysis
    │
    ▼
Notebook 02 - Preprocessing & Feature Engineering
    │
    ├── 44 engineered features
    │
    ▼
features.parquet
    │
    ▼
Notebook 03 - Model Training
    │
    ├── Isolation Forest
    ├── One-Class SVM
    └── Health Score Normalization
    │
    ▼
scored.parquet
    │
    ▼
Streamlit Dashboard
    ├── Fleet Overview
    ├── Machine Detail
    ├── Anomaly Timeline
    └── Anomaly Forecasting
```

---

# 📁 Project Structure

```
asset-health-monitoring/
├── data/
│   ├── raw/
│   │   ├── PdM_telemetry.csv
│   │   ├── PdM_errors.csv
│   │   ├── PdM_failures.csv
│   │   ├── PdM_maint.csv
│   │   └── PdM_machines.csv
│   │
│   └── processed/
│       ├── features.parquet
│       └── scored.parquet
│
├── models/
│   ├── isolation_forest.pkl
│   ├── one_class_svm.pkl
│   ├── scaler.pkl
│   └── feature_cols.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── dashboard/
│   └── app.py
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── PROJECT_DOCUMENTATION.md
└── README.md
```

> **Git LFS:** Large files (`.parquet`, `.pkl`, `PdM_telemetry.csv`) are tracked with Git LFS. Please install Git LFS before cloning.

---

# 🤖 Model Performance

## Primary Model — Isolation Forest

```python
IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)
```

### Evaluation (October–December 2015 Test Set)

| Metric | Isolation Forest | One-Class SVM |
|--------|:---:|:---:|
| **Recall (Pre-failure)** | **96%** | 95% |
| Precision | 25% | 26% |
| F1 Score | 0.40 | 0.40 |
| Average Precision | 0.642 | 0.710 |
| False Negatives | **188 / 4,221** | 198 / 4,221 |
| Training Time | ~12 sec | ~20 sec |

> **Recall (96%)** was intentionally prioritized because missing an impending equipment failure is significantly more costly than generating additional false alarms in industrial environments.

---

# 🏥 Health Score

```
Health Score = Normalized Isolation Forest anomaly score × 100

70–100  Healthy
40–70   Warning
0–40    Critical
```

Thresholds can be adjusted interactively from the dashboard sidebar.

---

# 🔮 Anomaly Forecasting

The forecasting module uses a **two-layer prediction approach**.

## Layer 1

**Linear Regression**

Forecasts machine health score using the previous **14 days** of historical data.

Outputs:

- 72-hour forecast
- 7-day forecast
- Trend slope
- Confidence interval
- Estimated days until critical condition

## Layer 2

**Exponential Smoothing (α = 0.3)**

Projects future fleet anomaly rate using daily anomaly statistics.

### Risk Classification

| Level | Condition |
|--------|-----------|
| 🔴 Critical | Forecast score below Critical threshold or anomaly rate >50% |
| 🟠 High | Forecast score below Warning threshold or anomaly rate >35% |
| 🟡 Medium | Forecast score approaching Warning threshold or anomaly rate >20% |
| 🟢 Low | Normal operating condition |

Each machine produces:

- `forecast_72h`
- `forecast_7d`
- `confidence_interval`
- `trend_slope`
- `days_to_critical`
- `risk_level`

---

# 🚀 Getting Started

## Prerequisites

```bash
brew install git-lfs
git lfs install
```

---

## Clone Repository

```bash
git clone https://github.com/aryawiandra/asset-health-monitoring.git

cd asset-health-monitoring
```

Large Git LFS files (~686 MB) will be downloaded automatically.

---

## Create Virtual Environment

```bash
python3.10 -m venv venv

source venv/bin/activate
# Windows
# venv\Scripts\activate

pip install -r requirements.txt
```

---

## Execute Notebooks (Optional)

Preprocessed data and trained models are already included through Git LFS.

To regenerate everything from scratch:

```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/01_eda.ipynb

jupyter nbconvert --execute --to notebook --inplace notebooks/02_preprocessing.ipynb

jupyter nbconvert --execute --to notebook --inplace notebooks/03_modeling.ipynb
```

Approximate execution times:

| Notebook | Runtime |
|-----------|---------|
| EDA | ~30 sec |
| Feature Engineering | 5–15 min |
| Model Training | 1–2 min |

---

## Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open:

```
http://localhost:8501
```

---

# 📊 Dashboard Pages

## Fleet Overview

- Fleet KPIs
- Forecast KPIs
- Health score distribution
- Machine status table
- CSV export

---

## Machine Detail

- Health gauge
- Sensor time-series
- Health timeline
- Forecast visualization
- Confidence interval
- Days-to-critical estimation

---

## Anomaly Timeline

- Fleet anomaly heatmap
- Daily anomaly trend
- Top anomalous machines

---

## Anomaly Forecast

- Fleet risk summary
- Forecast risk table
- Risk distribution
- Projected health scores
- 7-Day Risk Calendar
- Projected anomaly rate

---

# 📦 Dataset

**Microsoft Azure AI Gallery — Predictive Maintenance Dataset**

| File | Size | Description |
|------|------|-------------|
| PdM_telemetry.csv | 76 MB | Hourly sensor readings |
| PdM_errors.csv | 128 KB | Machine error logs |
| PdM_failures.csv | 24 KB | Historical component failures |
| PdM_maint.csv | 104 KB | Maintenance records |
| PdM_machines.csv | 4 KB | Machine metadata |

Dataset Statistics:

- 100 industrial machines
- 876,100 observations
- January 2015 – January 2016

---

# 🔧 Feature Engineering

A total of **44 engineered features** are used.

```
Raw Sensors (4)

Rolling Mean (8)

Rolling Standard Deviation (8)

Rate of Change (4)

Lag Features (12)

Error Counts (6)

Maintenance Features (1)

Machine Metadata (2)
```

---

# 📋 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
pyarrow
joblib
```

See **requirements.txt** for the complete dependency list.

---

# 🚀 Future Improvements

- LSTM / Temporal Deep Learning models
- Component-level failure prediction
- Real-time streaming integration
- Email / Slack alert system
- Root cause analysis
- Contamination parameter optimization
- Explainable AI (SHAP)
- Remaining Useful Life (RUL) estimation

---

# 📄 License

This project is licensed under the MIT License.

See **LICENSE** for details.

---

<div align="center">

Built for the Pertamina EDM Internship Mini Project

</div>
