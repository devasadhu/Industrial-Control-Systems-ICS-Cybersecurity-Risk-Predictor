"""
ICS Network Anomaly Detection — Streamlit Dashboard
====================================================
Author : Sadhana Devarajan
Version: 2.0.0

Displays:
  • Live model metrics and system health (polls API at /health + /model/info)
  • Dataset overview  — loaded from data/processed/ics_features_v3.csv + ics_labels.csv
  • Attack distribution — results/attack_type_distribution.csv
  • Confusion matrix   — results/confusion_matrix.png
  • SHAP feature importance — top features from processed data
  • IEC 62443 compliance   — results/compliance/iec62443_report.json
  • Incident reports list  — results/reports/
  • Live prediction form   — single-flow POST to API /predict

Run:
    streamlit run src/dashboard/ics_monitor.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICS Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
DATA_DIR      = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "results"
MODELS_DIR    = ROOT / "models"

FEATURES_CSV  = DATA_DIR / "ics_features_v3.csv"
FEATURES_CSV2 = DATA_DIR / "ics_features.csv"          # fallback (from ics_feature_engineer.py CLI)
LABELS_CSV    = DATA_DIR / "ics_labels.csv"
ATK_DIST_CSV  = RESULTS_DIR / "attack_type_distribution.csv"
CM_PNG        = RESULTS_DIR / "confusion_matrix.png"
COMPLIANCE_JSON = RESULTS_DIR / "compliance" / "iec62443_report.json"
REPORTS_DIR   = RESULTS_DIR / "reports"
METADATA_JSON = MODELS_DIR / "model_metadata.json"

API_BASE      = "http://localhost:8000"
API_TOKEN     = "demo_api_key_12345"
API_HEADERS   = {"Authorization": f"Bearer {API_TOKEN}"}

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, .stCode, [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Dark industrial theme */
[data-testid="stAppViewContainer"] {
    background: #0a0e14;
    color: #c9d1d9;
}
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

.stMetric {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 1rem;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.6rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* Section headers */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #388bfd;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* Status badges */
.badge-ok   { background:#1a3a2a; color:#56d364; border:1px solid #2ea043; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-family:'JetBrains Mono',monospace; }
.badge-warn { background:#2d2008; color:#e3b341; border:1px solid #bb8009; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-family:'JetBrains Mono',monospace; }
.badge-err  { background:#3a1a1a; color:#f85149; border:1px solid #da3633; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-family:'JetBrains Mono',monospace; }

/* Attack severity colors */
.sev-critical { color: #f85149; font-weight:700; }
.sev-high     { color: #e3b341; font-weight:600; }
.sev-medium   { color: #58a6ff; }
.sev-low      { color: #56d364; }

/* Prediction result boxes */
.pred-attack {
    background: #3a1a1a; border: 1px solid #da3633;
    border-radius: 8px; padding: 1rem; margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}
.pred-normal {
    background: #1a3a2a; border: 1px solid #2ea043;
    border-radius: 8px; padding: 1rem; margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 6px; }

/* Expander */
details { border: 1px solid #21262d !important; border-radius: 6px !important; }

/* Tabs */
[data-testid="stTabs"] button { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: #0d1117 !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
}
.stButton > button {
    background: #1f6feb; color: #fff; border: none;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    letter-spacing: 0.05em; border-radius: 6px;
    padding: 0.5rem 1.2rem;
}
.stButton > button:hover { background: #388bfd; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_features_and_labels():
    """Load processed feature CSV + labels. Tries v3 first, falls back to flat ics_features.csv."""
    feat_path = FEATURES_CSV if FEATURES_CSV.exists() else FEATURES_CSV2
    if not feat_path.exists():
        return None, None
    try:
        features = pd.read_csv(feat_path)
        labels   = pd.read_csv(LABELS_CSV).squeeze() if LABELS_CSV.exists() else None
        return features, labels
    except Exception:
        return None, None


@st.cache_data(ttl=60)
def load_attack_distribution():
    if not ATK_DIST_CSV.exists():
        return None
    try:
        df = pd.read_csv(ATK_DIST_CSV, index_col=0, header=None)
        df.columns = ["count"]
        return df
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_compliance():
    if not COMPLIANCE_JSON.exists():
        return None
    try:
        return json.loads(COMPLIANCE_JSON.read_text())
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_metadata():
    if not METADATA_JSON.exists():
        return {}
    try:
        return json.loads(METADATA_JSON.read_text())
    except Exception:
        return {}


def api_get(endpoint: str, timeout: int = 4):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", headers=API_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "API offline"
    except Exception as e:
        return None, str(e)


def api_post(endpoint: str, payload: dict, timeout: int = 8):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, headers=API_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "API offline — start with: python src/api/main.py"
    except Exception as e:
        return None, str(e)


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "ok"):
    return f'<span class="badge-{kind}">{text}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ ICS Monitor")
    st.markdown("**ICSSIM · MITRE ATT&CK ICS · IEC 62443**")
    st.divider()

    # API health check
    health, err = api_get("/health", timeout=2)
    if health:
        st.markdown(f'**API** {badge("ONLINE", "ok")}', unsafe_allow_html=True)
        st.caption(f"Model: {health.get('model_type','—')[:35]}")
        st.caption(f"Uptime: {health.get('uptime_seconds',0)/60:.1f} min")
        st.caption(f"Predictions served: {health.get('total_predictions',0):,}")
    else:
        st.markdown(f'**API** {badge("OFFLINE", "err")}', unsafe_allow_html=True)
        st.caption("Start: `python src/api/main.py`")

    st.divider()

    # Model info
    meta = load_metadata()
    if meta:
        st.markdown("**Model**")
        st.caption(f"Type: {meta.get('model_type','Ensemble')}")
        st.caption(f"Threshold: {meta.get('threshold', 0.25)}")
        st.caption(f"Features: {meta.get('n_features', 62)}")
        st.caption(f"Trained on: {meta.get('training_samples',0):,} flows")
    else:
        st.caption("model_metadata.json not found")

    st.divider()
    st.caption(f"Refreshed: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("# 🛡️ ICS Network Anomaly Detection")
st.markdown(
    "**Ensemble Detector** · IF + XGBoost + RF · threshold=0.25 · "
    "ICSSIM dataset · 45,718 flows · IEC 62443 SL-2"
)

tabs = st.tabs([
    "📊 Overview",
    "🤖 Model Performance",
    "🔍 Feature Analysis",
    "📋 Compliance",
    "🎯 Live Prediction",
    "📝 Reports",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    features_df, labels = load_features_and_labels()
    atk_dist = load_attack_distribution()

    # Top metrics
    section("DATASET SUMMARY")
    c1, c2, c3, c4, c5 = st.columns(5)
    total = len(features_df) if features_df is not None else 45718
    n_feat = features_df.shape[1] if features_df is not None else 62

    normal_count = int((labels == 0).sum()) if labels is not None else 30236
    attack_count = int((labels == 1).sum()) if labels is not None else 15482
    attack_pct   = attack_count / total * 100 if total else 33.9

    c1.metric("Total Flows",    f"{total:,}")
    c2.metric("Features",       f"{n_feat}")
    c3.metric("Normal Flows",   f"{normal_count:,}")
    c4.metric("Attack Flows",   f"{attack_count:,}")
    c5.metric("Attack Rate",    f"{attack_pct:.1f}%")

    st.divider()

    col_a, col_b = st.columns([1, 1])

    with col_a:
        section("ATTACK TYPE DISTRIBUTION")
        if atk_dist is not None:
            # Exclude "Normal" from attack chart
            atk_only = atk_dist[atk_dist.index.str.lower() != "normal"].copy()
            if not atk_only.empty:
                atk_only = atk_only.sort_values("count", ascending=True)
                st.bar_chart(atk_only, horizontal=True, color="#f85149")
            else:
                st.info("No attack-type data found.")
        else:
            # Fallback from known ICSSIM counts
            fallback = pd.DataFrame({
                "count": [4300, 4221, 3235, 3014, 712]
            }, index=["replay", "ddos", "port-scan", "mitm", "ip-scan"])
            fallback = fallback.sort_values("count", ascending=True)
            st.bar_chart(fallback, horizontal=True, color="#f85149")
            st.caption("_Showing known ICSSIM counts — results/attack_type_distribution.csv not found_")

    with col_b:
        section("FEATURE GROUP BREAKDOWN")
        feat_groups = {
            "network_basic":     10,
            "network_advanced":   5,
            "timing":             6,
            "statistical":       11,
            "behavioral":         4,
            "protocol":          20,
            "session":            6,
        }
        # Try loading from feature_groups.json
        fg_path = DATA_DIR / "feature_groups.json"
        if fg_path.exists():
            try:
                raw_fg = json.loads(fg_path.read_text())
                feat_groups = {k: len(v) for k, v in raw_fg.items() if v}
            except Exception:
                pass

        fg_df = pd.DataFrame({"features": feat_groups}).sort_values("features")
        st.bar_chart(fg_df, color="#388bfd")

    st.divider()
    section("MODEL WEIGHTS (ENSEMBLE)")
    w_col1, w_col2, w_col3 = st.columns(3)
    w_col1.metric("Isolation Forest", "40%", help="Unsupervised anomaly scorer")
    w_col2.metric("XGBoost",          "35%", help="Gradient-boosted classifier")
    w_col3.metric("Random Forest",    "25%", help="Bagged decision trees")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    meta = load_metadata()

    section("CORE METRICS (threshold=0.25)")
    mc1, mc2, mc3, mc4 = st.columns(4)

    acc  = meta.get("accuracy",  0.908)
    prec = meta.get("precision", 0.906)
    rec  = meta.get("recall",    0.812)
    f1   = meta.get("f1",        0.8563)

    mc1.metric("Accuracy",  f"{acc:.1%}")
    mc2.metric("Precision", f"{prec:.1%}")
    mc3.metric("Recall",    f"{rec:.1%}")
    mc4.metric("F1-Score",  f"{f1:.4f}")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        section("CONFUSION MATRIX")
        if CM_PNG.exists():
            st.image(str(CM_PNG), use_container_width=True)
        else:
            st.info("results/confusion_matrix.png not found — run quick_start.py first.")

    with col_right:
        section("PER-CLASS RESULTS")
        cls_data = pd.DataFrame({
            "Precision": [meta.get("normal_precision", 0.91), meta.get("attack_precision", 0.91)],
            "Recall":    [meta.get("normal_recall",    0.96), meta.get("attack_recall",    0.81)],
            "F1":        [meta.get("normal_f1",        0.93), meta.get("attack_f1",        0.86)],
            "Support":   [meta.get("normal_support",   6047), meta.get("attack_support",   3097)],
        }, index=["Normal", "Attack"])
        st.dataframe(cls_data.style.format({
            "Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}", "Support": "{:,.0f}"
        }), use_container_width=True)

        st.divider()
        section("ICS DOMAIN NOTES")
        st.markdown("""
- **Threshold 0.25** — tuned for ICS: false negatives (missed attacks) cost more than false positives
- **IP-scan recall ≈35%** — ICSSIM ceiling (only 8 unique source IPs in dataset)
- **Replay signal** — `src_inter_flow_variance` (normal/attack ratio = 11.76)
- **Session features zero-filled** for single-flow API requests — batch mode gives full accuracy
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    features_df, labels = load_features_and_labels()

    if features_df is None:
        st.warning("Feature CSV not found. Run `python quick_start.py` first.")
    else:
        section("TOP FEATURES BY ATTACK DISCRIMINABILITY")

        if labels is not None and len(labels) == len(features_df):
            # Compute absolute mean difference between normal/attack for each feature
            feat_numeric = features_df.select_dtypes(include=[np.number])
            normal_mean  = feat_numeric[labels.values == 0].mean()
            attack_mean  = feat_numeric[labels.values == 1].mean()
            diff_ratio   = (attack_mean - normal_mean).abs()
            # Normalise by std to avoid scale bias
            std = feat_numeric.std().replace(0, 1)
            score = (diff_ratio / std).sort_values(ascending=False).head(20)

            score_df = score.reset_index()
            score_df.columns = ["Feature", "Discriminability Score"]
            st.bar_chart(score_df.set_index("Feature"), color="#e3b341")
            st.caption("Score = |attack_mean − normal_mean| / std — higher = stronger attack signal")
        else:
            st.info("Labels not available — showing feature statistics only.")

        st.divider()
        section("FEATURE STATISTICS (SAMPLE)")
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        selected_feats = st.multiselect(
            "Select features to inspect",
            options=numeric_cols,
            default=numeric_cols[:6],
        )
        if selected_feats:
            desc = features_df[selected_feats].describe().T
            st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

        st.divider()
        section("NORMAL vs ATTACK DISTRIBUTIONS")
        if labels is not None:
            dist_feat = st.selectbox(
                "Feature to compare",
                options=numeric_cols,
                index=numeric_cols.index("src_inter_flow_variance")
                      if "src_inter_flow_variance" in numeric_cols else 0,
            )
            normal_vals = features_df.loc[labels.values == 0, dist_feat].clip(
                lower=features_df[dist_feat].quantile(0.01),
                upper=features_df[dist_feat].quantile(0.99),
            )
            attack_vals = features_df.loc[labels.values == 1, dist_feat].clip(
                lower=features_df[dist_feat].quantile(0.01),
                upper=features_df[dist_feat].quantile(0.99),
            )
            hist_df = pd.DataFrame({
                "Normal": pd.cut(normal_vals, bins=50).value_counts().sort_index().values,
                "Attack": pd.cut(attack_vals, bins=50).value_counts().sort_index().values,
            })
            st.area_chart(hist_df, color=["#56d364", "#f85149"])
            st.caption(f"Distribution of **{dist_feat}** across Normal (green) vs Attack (red) flows")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — IEC 62443 COMPLIANCE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    compliance = load_compliance()

    section("IEC 62443 COMPLIANCE ASSESSMENT")

    if compliance:
        sl     = compliance.get("security_level", "SL-2")
        score  = compliance.get("overall_score",  78.3)
        ts     = compliance.get("timestamp", "—")

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Security Level",  sl)
        cc2.metric("Overall Score",   f"{score:.1f}%")
        cc3.metric("Assessment Date", ts[:10] if len(ts) > 10 else ts)

        st.divider()
        section("REQUIREMENT SCORES")

        reqs = compliance.get("requirements", {})
        if reqs:
            req_df = pd.DataFrame([
                {"Requirement": k, "Score": v.get("score", 0), "Status": v.get("status", "—")}
                for k, v in reqs.items()
            ])
            req_df = req_df.sort_values("Score")

            def color_status(val):
                if val == "compliant":   return "color: #56d364"
                if val == "partial":     return "color: #e3b341"
                return "color: #f85149"

            st.dataframe(
                req_df.style
                    .format({"Score": "{:.1f}%"})
                    .applymap(color_status, subset=["Status"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No per-requirement data in compliance report.")

        with st.expander("📄 Raw compliance JSON"):
            st.json(compliance)
    else:
        st.info("Compliance report not found at `results/compliance/iec62443_report.json`")
        st.markdown("Run `python quick_start.py` to generate it.")

    st.divider()
    section("IEC 62443 ZONE MAP")
    zones_data, zerr = api_get("/iec62443/zones", timeout=2)
    if zones_data:
        zones = zones_data.get("zones", [])
        zone_df = pd.DataFrame([{
            "Level":       z["level"],
            "Name":        z["name"],
            "Description": z["description"],
            "Critical Ports": ", ".join(str(p) for p in z.get("critical_ports", [])),
        } for z in zones])
        st.dataframe(zone_df, use_container_width=True, hide_index=True)
    else:
        # Static fallback
        zone_df = pd.DataFrame([
            {"Level": "Level 0", "Name": "Physical Process",     "Description": "Sensors, actuators, PLCs",              "Critical Ports": "502, 20000, 44818"},
            {"Level": "Level 1", "Name": "Basic Control",        "Description": "Process control, field devices",         "Critical Ports": "502, 2222, 44818"},
            {"Level": "Level 2", "Name": "Supervisory Control",  "Description": "SCADA, HMI, Engineering workstations",   "Critical Ports": "80, 443, 8080, 3389"},
            {"Level": "Level 3", "Name": "Site Operations",      "Description": "MES, data historians",                   "Critical Ports": "1433, 3306, 5432"},
            {"Level": "Level 4", "Name": "Enterprise Network",   "Description": "Corporate IT systems",                   "Critical Ports": "22, 23, 21, 445"},
        ])
        st.dataframe(zone_df, use_container_width=True, hide_index=True)
        if zerr:
            st.caption(f"_API offline — showing static zone map_")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section("SINGLE-FLOW PREDICTION")
    st.caption(
        "Submits a flow to the API (`POST /predict`). "
        "Session features are zero-filled (see /model/info). "
        "Start the API first: `python src/api/main.py`"
    )

    with st.form("predict_form"):
        pc1, pc2 = st.columns(2)
        with pc1:
            src_ip          = st.text_input("Source IP",      value="192.168.1.100")
            dst_ip          = st.text_input("Dest IP",        value="192.168.1.10")
            src_port        = st.number_input("Src Port",     value=1024, min_value=0, max_value=65535)
            dst_port        = st.number_input("Dst Port",     value=502,  min_value=0, max_value=65535,
                                              help="502=Modbus, 44818=EtherNet/IP, 20000=DNP3")
            proto_map       = {"TCP (6)": 6, "UDP (17)": 17, "ICMP (1)": 1}
            proto_label     = st.selectbox("Protocol", list(proto_map.keys()))
            protocol        = proto_map[proto_label]
        with pc2:
            fwd_packets     = st.number_input("Fwd Packets",       value=9,    min_value=0)
            bwd_packets     = st.number_input("Bwd Packets",       value=17,   min_value=0)
            fwd_bytes       = st.number_input("Fwd Bytes",         value=585,  min_value=0)
            bwd_bytes       = st.number_input("Bwd Bytes",         value=1200, min_value=0)
            flow_duration   = st.number_input("Flow Duration (s)", value=0.5,  min_value=0.0, format="%.3f")

        pc3, pc4 = st.columns(2)
        with pc3:
            flow_iat_mean   = st.number_input("Flow IAT Mean (s)", value=0.05, min_value=0.0, format="%.4f")
            flow_iat_std    = st.number_input("Flow IAT Std (s)",  value=0.02, min_value=0.0, format="%.4f")
        with pc4:
            fwd_psh_flags   = st.number_input("Fwd PSH Flags", value=0, min_value=0)
            bwd_psh_flags   = st.number_input("Bwd PSH Flags", value=0, min_value=0)
            fwd_urg_flags   = st.number_input("Fwd URG Flags", value=0, min_value=0)
            bwd_urg_flags   = st.number_input("Bwd URG Flags", value=0, min_value=0)

        submitted = st.form_submit_button("🔍 Run Detection", use_container_width=True)

    if submitted:
        payload = {
            "src_ip":                      src_ip,
            "dst_ip":                      dst_ip,
            "src_port":                    int(src_port),
            "dst_port":                    int(dst_port),
            "protocol":                    int(protocol),
            "total_fwd_packets":           int(fwd_packets),
            "total_bwd_packets":           int(bwd_packets),
            "total_length_fwd_packets":    int(fwd_bytes),
            "total_length_bwd_packets":    int(bwd_bytes),
            "flow_duration":               float(flow_duration),
            "flow_iat_mean":               float(flow_iat_mean),
            "flow_iat_std":                float(flow_iat_std),
            "fwd_psh_flags":               int(fwd_psh_flags),
            "bwd_psh_flags":               int(bwd_psh_flags),
            "fwd_urg_flags":               int(fwd_urg_flags),
            "bwd_urg_flags":               int(bwd_urg_flags),
        }

        with st.spinner("Querying API..."):
            result, err = api_post("/predict", payload)

        if err:
            st.error(f"❌ {err}")
        elif result:
            is_attack = result.get("is_anomaly", False)
            severity  = result.get("severity", "LOW")
            score     = result.get("anomaly_score", 0.0)
            conf      = result.get("confidence", 0.0)
            zone      = result.get("iec62443_zone", "—")
            action    = result.get("recommended_action", "—")

            box_class = "pred-attack" if is_attack else "pred-normal"
            verdict   = "⚠️ ATTACK DETECTED" if is_attack else "✅ NORMAL TRAFFIC"
            sev_class = f"sev-{severity.lower()}"

            st.markdown(f"""
<div class="{box_class}">
  <div style="font-size:1.1rem; font-weight:700; margin-bottom:0.5rem">{verdict}</div>
  <div>Severity:       <span class="{sev_class}">{severity}</span></div>
  <div>Anomaly Score:  <code>{score:.4f}</code></div>
  <div>Confidence:     <code>{conf:.4f}</code></div>
  <div>IEC 62443 Zone: <code>{zone}</code></div>
  <div style="margin-top:0.5rem">🔧 {action}</div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    section("QUICK-TEST PRESETS")
    preset_col1, preset_col2, preset_col3 = st.columns(3)

    with preset_col1:
        st.markdown("**Normal Modbus poll**")
        st.code("dst_port=502\nfwd_pkt=9, bwd_pkt=17\nfwd_bytes=585", language="yaml")

    with preset_col2:
        st.markdown("**PLC scan pattern**")
        st.code("dst_port=502\nfwd_pkt=256, bwd_pkt=1\nfwd_bytes=8192", language="yaml")

    with preset_col3:
        st.markdown("**DDoS burst**")
        st.code("dst_port=502\nfwd_pkt=9800, bwd_pkt=2\nflow_dur=0.01", language="yaml")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section("INCIDENT REPORTS")

    if REPORTS_DIR.exists():
        md_reports  = sorted(REPORTS_DIR.glob("incident_report_*.md"),  reverse=True)
        pdf_reports = sorted(REPORTS_DIR.glob("incident_report_*.pdf"), reverse=True)

        if md_reports:
            st.markdown(f"Found **{len(md_reports)}** report(s) in `results/reports/`")

            for md_path in md_reports[:5]:   # show latest 5
                ts_str = md_path.stem.replace("incident_report_", "")
                try:
                    ts_display = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts_display = ts_str

                with st.expander(f"📄 Report — {ts_display}"):
                    try:
                        content = md_path.read_text(encoding="utf-8")
                        st.markdown(content, unsafe_allow_html=False)
                    except Exception as e:
                        st.error(f"Could not read report: {e}")

                    # Corresponding PDF
                    pdf_match = REPORTS_DIR / md_path.name.replace(".md", ".pdf")
                    if pdf_match.exists():
                        st.markdown(f"📎 PDF also available: `{pdf_match.name}`")
        else:
            st.info("No incident reports found.")
    else:
        st.info("`results/reports/` directory not found. Run `python quick_start.py` to generate reports.")

    st.divider()
    section("OTHER ARTIFACTS")

    art_col1, art_col2 = st.columns(2)

    with art_col1:
        st.markdown("**STIX 2.1 Bundle**")
        stix_path = RESULTS_DIR / "stix_bundle.json"
        if stix_path.exists():
            st.markdown(badge("EXISTS", "ok"), unsafe_allow_html=True)
            with st.expander("Preview STIX bundle"):
                try:
                    stix_data = json.loads(stix_path.read_text())
                    st.json(stix_data)
                except Exception:
                    st.code(stix_path.read_text()[:2000])
        else:
            st.markdown(badge("NOT FOUND", "warn"), unsafe_allow_html=True)

    with art_col2:
        st.markdown("**Suricata Rules**")
        rules_path = RESULTS_DIR / "suricata_ics.rules"
        if rules_path.exists():
            st.markdown(badge("EXISTS", "ok"), unsafe_allow_html=True)
            with st.expander("Preview rules"):
                st.code(rules_path.read_text()[:3000], language="text")
        else:
            st.markdown(badge("NOT FOUND", "warn"), unsafe_allow_html=True)