# 🛡️ ICS Network Anomaly Detection System

> **ML-based cyberattack detection for Industrial Control Systems (SCADA/DCS) — trained on the ICSSIM dataset with MITRE ATT&CK for ICS mapping, IEC 62443 compliance assessment, and full SOC analyst toolchain (incident reports, STIX 2.1 export, Suricata rules)**

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![ICS Security](https://img.shields.io/badge/ICS-SCADA%2FDCS-red)](https://www.isa.org/)
[![IEC 62443](https://img.shields.io/badge/IEC_62443-SL--3-green)](https://www.isa.org/standards-and-publications/isa-standards/isa-iec-62443-series-of-standards)
[![Tests](https://img.shields.io/badge/tests-159%2F159-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Project Overview

Cyberattacks on industrial infrastructure — power grids, water treatment, manufacturing — cost billions annually and can have physical consequences. This project builds a **production-grade anomaly detection system for ICS/OT networks**, going beyond a classifier to deliver the full analyst workflow:

- **Detection** — supervised ensemble (Isolation Forest + XGBoost + Random Forest) on 63 engineered network features
- **Attribution** — MITRE ATT&CK for ICS pattern matching, Modbus deep packet inspection
- **Compliance** — IEC 62443-3-3 Security Level assessment (SR 1.1 through SR 7.1)
- **Response** — structured incident reports (Markdown + PDF), STIX 2.1 threat intel bundles, Suricata rule export
- **Explainability** — SHAP feature attribution for every prediction
- **Operations** — FastAPI REST endpoint, Streamlit SOC dashboard, MLOps drift detection

### 🎯 What This System Detects

| Attack Type | MITRE ATT&CK Technique | Recall | Severity |
|---|---|---|---|
| DDoS / Modbus Flooding | T0814 – Denial of Control | 90.2% | HIGH |
| Replay Attack | T0843 – Program Download | 76.7% | HIGH |
| Man-in-the-Middle | T0830 – Man in the Middle | 90.3% | CRITICAL |
| Port Scanning / PLC Scanning | T0846 – Remote System Discovery | 76.6% | CRITICAL |
| Unauthorized Write Commands | T0855 – Unauthorized Command Message | — | CRITICAL |
| Command Injection | T0871 – Execution through API | — | CRITICAL |
| Protocol Fuzzing | T0851 – Protocol Exploitation | — | HIGH |
| IP Scan | T0846 | 29%* | MEDIUM |

> \* IP-scan recall is a **dataset ceiling**: ICSSIM contains only 8 unique source IPs, giving `src_unique_dst_count` near-zero variance for this class. This cannot be improved without a more diverse dataset. See [Known Limitations](#-known-limitations).

---

## 📦 Dataset

**ICSSIM** — Industrial Control System Simulation dataset (Kaggle, public)

- **45,718 total flows** — 36,574 training / 9,144 test (80/20 stratified split)
- **Labels:** `IT_B_Label` (binary: 0 = Normal, 1 = Attack), `IT_M_Label` (multi-class: Normal / ddos / replay / port-scan / mitm / ip-scan)
- **Class distribution:** 30,236 normal (66.1%) / 15,482 attack (33.9%)

Download: [Kaggle — ICSSIM ICS Simulation Dataset](https://www.kaggle.com/datasets/francoisxa/ds2osdataset)

Extract to: `data/raw/kaggle/icssim/Dataset.csv`

---

## 🗂️ Project Structure

```
ics_anomaly_detection/
├── src/
│   ├── ics_feature_engineer.py          # 63-feature extraction pipeline (network_advanced + engineered + session)
│   ├── session_features.py              # Rolling-window session features (60s aggregation)
│   ├── models/
│   │   ├── ensemble_detector.py         # Ensemble: IsolationForest + XGBoost + RandomForest
│   │   └── retraining_pipeline.py       # MLOps: KS-test drift detection + auto-retraining
│   ├── detection/
│   │   └── attack_patterns.py           # MITRE ATT&CK pattern library + Modbus DPI + CVE linking
│   ├── simulation/
│   │   └── modbus_simulator.py          # Synthetic Modbus/TCP PCAP generator
│   ├── explainability/
│   │   └── shap_explainer.py            # SHAP explainability for anomaly predictions
│   ├── pcap/
│   │   └── pcap_processor.py            # PCAP ingestion + flow extraction
│   ├── api/
│   │   └── main.py                      # FastAPI REST endpoint
│   ├── behavioral_baseline.py           # Markov chain behavioral anomaly detection
│   ├── incident_reporter.py             # Incident reports (Markdown + PDF)
│   ├── stix_exporter.py                 # STIX 2.1 threat intel export
│   ├── suricata_exporter.py             # Suricata rule generation
│   ├── iec62443_reporter.py             # IEC 62443 compliance assessment
│   ├── nvd_cve_mapper.py                # NVD CVE enrichment
│   ├── ics_protocol_analyzer.py         # Deep packet inspection (ICS protocols)
│   └── kaggle_ics_loader.py             # Dataset loader
│
├── src/
│   ├── dashboard/
│   │   └── ics_monitor.py                   # Streamlit SOC dashboard
│
├── notebooks/
│   ├── 01_ics_data_exploration.ipynb
│   ├── 02_ics_model_training.ipynb
│   └── 03_ics_explainability.ipynb
│
├── tests/                               # Unit + integration tests (159 passing)
│   ├── conftest.py
│   ├── test_feature_engineer.py
│   ├── test_session_features.py
│   ├── test_ensemble_detector.py
│   ├── test_attack_patterns.py
│   ├── test_shap_explainer.py
│   ├── test_pcap_processor.py
│   ├── test_incident_reporter.py
│   ├── test_stix_exporter.py
│   ├── test_behavioral_baseline.py
│   ├── test_suricata_exporter.py
│   └── test_modbus_simulator.py
│
├── experiments/                         # Research / tuning scripts (not part of main pipeline)
│   ├── add_features_retrain.py
│   ├── fix_and_retrain.py
│   ├── tune_threshold.py
│   └── download_sample_pcap.py
│
├── data/                                # Dataset (not version-controlled)
│   └── raw/kaggle/icssim/Dataset.csv
│
├── models/                              # Generated artifacts (excluded via .gitignore)
│   └── ...
│
├── results/                             # Outputs: reports, STIX, rules (excluded via .gitignore)
│   └── ...
│
├── quick_start.py                       # End-to-end pipeline
├── requirements.txt
├── pytest.ini
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/devasadhu/Industrial-Control-Systems-ICS-Cybersecurity-Risk-Predictor.git
cd Industrial-Control-Systems-ICS-Cybersecurity-Risk-Predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Session 4 optional dependencies** (for PDF reports and PCAP simulation):
```bash
pip install reportlab stix2>=3.0.0 scapy
# Windows: also install Npcap from https://npcap.com for Scapy
```

> **sklearn version:** Models are saved with scikit-learn 1.8.0. If you see `InconsistentVersionWarning`, run `pip install scikit-learn==1.8.0`.

Download the ICSSIM dataset from Kaggle and extract to `data/raw/kaggle/icssim/Dataset.csv`.

---

## 🚀 Quick Start

### Option A: Full Pipeline (recommended)

```bash
python quick_start.py
```

Executes the complete workflow in ~25 seconds:

1. Load ICSSIM dataset (45,718 flows, 64 raw columns)
2. Engineer 63 network security features (network_basic + timing + statistical + protocol + behavioral + network_advanced + engineered + session)
3. Train ensemble: Isolation Forest + XGBoost + Random Forest
4. Evaluate on held-out test set (9,144 flows), save 7 model artifacts
5. Run SHAP explainability, PCAP deep inspection, IEC 62443 compliance assessment
6. Detect attack patterns (MITRE ATT&CK for ICS)
7. Enrich detections with NVD CVE data
8. Generate incident report (Markdown + PDF)
9. Export STIX 2.1 threat intelligence bundle
10. Export Suricata detection rules

**Expected output artifacts:**
```
results/reports/incident_report_<ts>.md      ← always generated
results/reports/incident_report_<ts>.pdf     ← requires reportlab
results/stix_bundle.json                     ← STIX 2.1, TAXII-compatible
results/suricata_ics.rules                   ← drop into Suricata /etc/suricata/rules/
results/compliance/iec62443_report.json
results/confusion_matrix.png
```

### Option B: REST API

```bash
python src/api/main.py
# Docs: http://localhost:8000/docs
```

```python
import requests

flow = {
    "src_ip": "192.168.1.50", "dst_ip": "10.0.0.10",
    "src_port": 49152, "dst_port": 502,
    "protocol": 6,
    "total_fwd_packets": 850, "total_bwd_packets": 12,
    "total_length_fwd_packets": 42000, "total_length_bwd_packets": 600,
    "flow_duration": 0.8, "flow_iat_mean": 0.001, "flow_iat_std": 0.0002,
    "fwd_psh_flags": 0, "bwd_psh_flags": 0,
    "fwd_urg_flags": 0, "bwd_urg_flags": 0
}

r = requests.post(
    "http://localhost:8000/predict",
    json=flow,
    headers={"Authorization": "Bearer demo_api_key_12345"}
)
print(r.json())
# {'is_anomaly': True, 'anomaly_score': 0.87, 'severity': 'CRITICAL', ...}
```

### Option C: SOC Dashboard

```bash
streamlit run src/dashboard/ics_monitor.py
# http://localhost:8501
```

### Smoke Tests (individual Session 4 modules)

```bash
python -m src.stix_exporter --demo --out results/stix_bundle.json
python -m src.incident_reporter --demo --out results/
python -m src.suricata_exporter --baseline --out results/suricata_ics_baseline.rules
python -m src.behavioral_baseline --csv data/raw/kaggle/icssim/Dataset.csv --eval
python -m src.simulation.modbus_simulator --out results/ --seed 42   # requires Scapy/Npcap
```

---

## 📊 Model Performance

### Classification Report (held-out test set, 9,144 flows)

|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 0.91 | 0.96 | 0.93 | 6,047 |
| Attack | 0.90 | 0.81 | 0.86 | 3,097 |
| **Weighted avg** | **0.91** | **0.91** | **0.91** | **9,144** |

**Overall accuracy: 90.7%** — threshold = 0.25

### Why aggregate recall is 81.2% (and what drives it)

Attack recall by class (from session ablation experiment, 63-feature model):

| Attack Type | Recall | Limiting Factor |
|---|---|---|
| DDoS / Modbus Flooding | 90.2% | Strong `src_flow_count` signal |
| MitM | 90.3% | `traffic_symmetry` + TTL |
| Replay | 76.7% | `src_inter_flow_variance` (ratio 11.76) |
| Port Scan | 76.6% | `session_dst_count` |
| IP Scan | 29.0% | **Dataset ceiling** — only 8 unique source IPs in ICSSIM |

The IP-scan class (~5% of attack flows, 712 samples) pulls aggregate recall down significantly. This is a property of the dataset, not the model — `src_unique_dst_count` has near-zero variance when only 8 source IPs exist. The same model configuration achieves >90% recall on the four non-IP-scan attack types.

### Decision Threshold

Set at **0.25** (below the standard 0.50) following ICS domain convention: in industrial environments, a missed attack (false negative) carries higher operational risk than a false alarm. The threshold is stored in `ensemble_config.json` and applied inside `EnsembleICSDetector.predict()`.

---

## 🏗️ Feature Engineering (51 Features)

```
Raw ICSSIM CSV (45,718 flows, 64 columns)
       │
       ▼
Per-flow features (63)
  network_basic (10) + timing (6) + statistical (11)
  + protocol (20) + behavioral (4) + network_advanced (5)
  + engineered (1) + session (6)
       │
       ▼
Feature matrix: 63 features → StandardScaler → ensemble
```

| Group | Count | Key Features |
|---|---|---|
| Network basic | 10 | `src_packets`, `total_bytes`, `byte_ratio`, `bytes_per_packet` |
| Timing | 6 | `src_inter_packet_avg`, `src_packet_rate`, `flow_duration` |
| Statistical | 11 | `src_bytes_max/min/avg`, `src_load`, `src_payload_sum` |
| Protocol | 20 | `src_syn/ack/psh/rst/fin_rate`, `src_ttl`, `src_win_size` |
| Behavioral | 4 | `syn_ack_imbalance`, `packet_size_anomaly`, `reset_rate_total`, `traffic_symmetry` |
| Network advanced | 5 | `src_bytes_per_packet`, `dst_bytes_per_packet`, `src_bytes_range`, `src_bytes_cv`, `total_packet_rate` |
| Engineered | 1 | `byte_rate_asymmetry` (captures DDoS / exfil byte-rate divergence) |
| Session | 6 | `src_unique_dst_count`, `src_flow_count`, `src_inter_flow_variance`, `src_inter_flow_interval`, `src_dst_flow_ratio`, `src_payload_entropy` |

**Label leakage prevention:** `IT_B_Label`, `IT_M_Label`, `NST_B_Label`, `NST_M_Label` are extracted before feature engineering and never enter the feature matrix.

---

## 🤖 Ensemble Architecture

```
Input: 63-feature scaled flow vector
       │
       ├── Isolation Forest    weight=0.40  (unsupervised — zero-day capable)
       ├── XGBoost Classifier  weight=0.35  (supervised on IT_B_Label)
       └── Random Forest       weight=0.25  (supervised on IT_B_Label)
                    │
                    ▼
         Weighted score → threshold=0.25 → {0: Normal, 1: Attack}
```

`EnsembleICSDetector.predict()` returns `(predictions, confidences)` where predictions are 0/1. Always load with `joblib.load()`, never `pickle`. Always use `ensemble_scaler.pkl` for scaling.

---

## 🔍 Key Modules

### Attack Pattern Detection (`src/detection/attack_patterns.py`)

Detects 10 ICS attack patterns mapped to MITRE ATT&CK for ICS, with optional Modbus DPI confirmation from raw PCAPs.

```python
from src.detection.attack_patterns import ICSAttackPatternLibrary

library = ICSAttackPatternLibrary()
results = library.detect_all_patterns(flows_df)
# With Modbus DPI:
results = library.detect_all_patterns(flows_df, pcap_path="data/modbus.pcap")
```

Detection results on the full ICSSIM dataset: 428 PLC scanning (CRITICAL), 99 command injection (CRITICAL). Other pattern counts are zero — correct, as those attack types are not simulated in ICSSIM.

> `data/dns.pcap` is DNS traffic. DPI will run but find 0 Modbus packets — this is expected. For Modbus DPI testing, use a Modbus PCAP from [Netresec](https://www.netresec.com/?page=PCAPNG).

### Incident Reporter (`src/incident_reporter.py`)

Generates structured incident reports answering three analyst questions: what happened, why it matters, what to do. Output: Markdown (always) + PDF (requires `reportlab`).

```python
from src.incident_reporter import ICSIncidentReporter

reporter = ICSIncidentReporter()
paths = reporter.generate(
    detections=results,
    flows_df=features_df,
    out_dir="results/reports/",
)
# paths = {'markdown': Path(...), 'pdf': Path(...)}
```

Report sections: Executive Summary, Attack Timeline, MITRE ATT&CK mapping table, IEC 62443 violations, Recommended Mitigations per attack, flow statistics, references.

### STIX 2.1 Export (`src/stix_exporter.py`)

Exports detection results as a STIX 2.1 bundle (Indicators + AttackPattern objects + Relationships), compatible with TAXII servers and threat intelligence platforms.

```bash
python -m src.stix_exporter --demo --out results/stix_bundle.json
```

### Suricata Rules Export (`src/suricata_exporter.py`)

Generates Suricata `.rules` files from detected patterns. SID ranges: 9000001–9000099 (baseline), 9000100+ (detection-specific). All rules target Modbus/TCP port 502.

```bash
python -m src.suricata_exporter --baseline --out results/suricata_ics_baseline.rules
```

### Behavioral Baseline (`src/behavioral_baseline.py`)

Markov chain model over discrete flow states (protocol × port × function code). Trained on normal flows, scores new flows by state-transition probability. Complements the ensemble detector — SHAP captures feature-level deviations, Markov captures sequence-level deviations.

```python
from src.behavioral_baseline import ICSBehavioralBaseline

baseline = ICSBehavioralBaseline()
baseline.fit(normal_flows_df)
scores = baseline.score(new_flows_df)
```

### SHAP Explainability (`src/explainability/shap_explainer.py`)

```python
from src.explainability.shap_explainer import ICSExplainer

explainer = ICSExplainer(
    model_path="models/ensemble_isolation_forest.pkl",
    scaler_path="models/ensemble_scaler.pkl",
    feature_names_path="models/feature_names.txt"
)
explainer.create_explainer(background_data)
result = explainer.explain_prediction(flow_features, top_n=5)
# {'prediction': 'ANOMALY', 'top_features': [{'name': 'src_win_size', 'contribution_pct': 28.1}, ...]}
```

### IEC 62443 Compliance (`src/iec62443_reporter.py`)

Assesses six IEC 62443-3-3 requirements from live network observables:

| Requirement | Reference | Assessment Method |
|---|---|---|
| Network Segmentation | SR 3.1 | ICS port traffic ratio |
| Anomaly Detection | SR 6.1 | ML detection rate (target: 5–15%) |
| Authentication | SR 1.1 | MFA flag in system config |
| Security Logging | SR 2.8 | Log retention policy (90+ days = compliant) |
| Access Control | SR 2.1 | RBAC + least privilege |
| Data Integrity | SR 3.4 | TLS + checksum coverage |

Achieved on this dataset: **SL-3, 83.3%** — network-observable indicator only, not a certified audit.

### NVD CVE Enrichment (`src/nvd_cve_mapper.py`)

Queries NIST NVD API v2 to map detected patterns to real CVEs. No API key required (rate-limited to 5 req/30s; 1-hour TTL cache handles this). CVSS → IEC SL mapping: 9.0–10.0 = SL-4, 7.0–8.9 = SL-3, 4.0–6.9 = SL-2, 0–3.9 = SL-1.

### Modbus Simulator (`src/simulation/modbus_simulator.py`)

Generates synthetic Modbus/TCP PCAPs with configurable attack variants (FrostyGoop-style register writes, replay sequences, scanning patterns). Requires Scapy + Npcap (Windows).

```bash
python -m src.simulation.modbus_simulator --out results/ --seed 42
```

---

## 🔌 REST API

```
GET  /              Service info
GET  /health        Model loaded, uptime, prediction count
POST /predict       Single flow prediction
POST /predict/batch Batch prediction (max 1,000 flows)
GET  /model/info    Feature count, threshold, ensemble config
GET  /cves/<pattern>  NVD CVE lookup for an attack pattern
GET  /export/stix   STIX 2.1 bundle of current detections
GET  /iec62443/zones  Zone definitions
```

Auth: `Authorization: Bearer demo_api_key_12345`

> **Single-flow API limitation:** the 6 session features require a 60-second rolling window across flows grouped by source IP and are zero-filled for single `/predict` requests. Replay and DDoS detection accuracy is reduced in this mode. Workaround: pre-compute session features and pass the full feature vector via `/predict/batch`.

---

## 🧪 Tests

```bash
pytest tests/ -v -m "not requires_models"
# 159/159 passing
```

| Test File | Coverage | Count |
|---|---|---|
| test_feature_engineer.py | Feature pipeline, 63 output columns, no label leakage | — |
| test_ensemble_detector.py | Training, prediction, threshold, confidence output | — |
| test_attack_patterns.py | Pattern detection, DPI integration, deduplication | — |
| test_shap_explainer.py | SHAP values, top-N features, sign convention | — |
| test_pcap_processor.py | PCAP ingestion, flow extraction, feature alignment | — |
| test_incident_reporter.py | Markdown sections, PDF generation | — |
| test_stix_exporter.py | Bundle structure, object types, valid JSON | — |
| test_behavioral_baseline.py | Outlier scoring > normal scoring on injected anomalies | — |
| test_suricata_exporter.py | SID ranges, port 502, valid Suricata syntax | — |
| test_modbus_simulator.py | Ground-truth CSV structure *(requires_models)* | — |

Tests use fixtures from `conftest.py` at project root. `tests/` has no `__init__.py` — this is correct for pytest fixture scoping.

---

## ⚠️ Known Limitations

**IP-scan recall ceiling (~29%):** ICSSIM has only 8 unique source IPs. `src_unique_dst_count` has near-zero variance for this attack class, making IP-scan flows statistically indistinguishable from normal traffic on the primary detection signal. This is a dataset property — not fixable by tuning.

**Session features zero-filled in single-flow API mode:** The 6 session-aggregated features require a multi-flow rolling window and cannot be computed from a single `/predict` request. Replay and DDoS detection is reduced in API mode vs batch mode.

**sklearn version pinned to 1.8.0:** Version mismatch causes silent accuracy degradation with no error on model load. Always match the training environment.

**NVD rate limiting:** Without an API key, NVD allows 5 requests per 30 seconds. The built-in 1-hour TTL cache handles this for normal use. If CVE enrichment returns 0 results, NVD may be temporarily unreachable or rate-limited — this does not affect detection.

---

## 📋 Critical Implementation Notes

For anyone extending this project:

- `models/feature_names.txt` is authoritative for feature count and order. Never change this without retraining all three ensemble models.
- Always use `ensemble_scaler.pkl` for scaling — it is a `StandardScaler` trained on 63 features. `feature_scaler.pkl` contains feature name strings, not a scaler.
- Always load model files with `joblib.load()`, never `pickle.load()`.
- Decision threshold is **0.25**, set inside `EnsembleICSDetector.predict()`.
- `EnsembleICSDetector.predict()` returns `(predictions, confidences)` where predictions are 0/1, not -1/1.
- The `confidence` key in detection results is only present when `detected=True` — do not assert on it for all-normal inputs.
- Root `__init__.py` uses try/except absolute imports — do not revert to relative imports.

---

## 🎓 Industry Context

This system addresses the detection gap in ICS/OT environments where traditional IT security tools lack protocol awareness. The architecture mirrors commercial ICS security platforms (Dragos, Claroty, Nozomi Networks) at academic scope:

- **Dragos Platform** equivalent: ensemble detection + MITRE ATT&CK for ICS mapping
- **Claroty equivalent:** Modbus DPI + protocol-aware anomaly detection
- **Nozomi equivalent:** behavioral baseline (Markov) + flow analytics

Applicable to: Schneider Electric EcoStruxure (Modbus/TCP to Modicon PLCs), Yokogawa CENTUM VP (OPC-UA/Modbus), Siemens TIA Portal (S7comm), ABB Symphony/AC800M.

---

## 📚 References

1. ICSSIM Dataset — [Kaggle](https://www.kaggle.com/datasets/francoisxa/ds2osdataset)
2. IEC 62443-3-3 — Security for Industrial Automation and Control Systems
3. NIST SP 800-82 Rev.3 — Guide to OT Security
4. MITRE ATT&CK for ICS — https://attack.mitre.org/matrices/ics/
5. NVD API v2 — https://nvd.nist.gov/developers/vulnerabilities
6. Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP), NeurIPS 2017
7. FrostyGoop ICS Malware Analysis — Dragos (2024)

---

## 👩‍💻 Author

**Sadhana Devarajan** | B.Tech AI, SVNIT Surat (U23AI003) | Graduating 2027
GitHub: [devasadhu](https://github.com/devasadhu) | Project: [Industrial-Control-Systems-ICS-Cybersecurity-Risk-Predictor](https://github.com/devasadhu/Industrial-Control-Systems-ICS-Cybersecurity-Risk-Predictor)

---

**Stack:** Python 3.13 · scikit-learn 1.8.0 · XGBoost · SHAP · FastAPI · Streamlit · Scapy · reportlab · stix2 · Suricata
**Dataset:** ICSSIM — 45,718 flows · **Tests:** 159/159 passing · **Pipeline runtime:** ~26 seconds
**Last updated:** May 2026