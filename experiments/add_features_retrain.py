"""
add_features_retrain.py
-----------------------
Adds 4 targeted features to address per-attack-type recall gaps:

  1. inter_packet_timing_variance  — catches replay (abnormally LOW variance)
  2. payload_repetition_score      — catches replay (repeated payload sizes)
  3. unique_dst_ratio              — catches ip-scan (many unique destinations)
  4. flow_burstiness               — catches low-rate DDoS and slow port-scan

Then retrains the full supervised ensemble and compares per-attack recall
before vs after.

Run from project root:
    python add_features_retrain.py

Author: Sadhana Devarajan
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report
)

logging.basicConfig(level=logging.WARNING)  # suppress ensemble INFO spam
log = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO)

sys.path.insert(0, './src')

RAW_DATA   = Path('./data/raw/kaggle/icssim/Dataset.csv')
FEATURES   = Path('./data/processed/ics_features.csv')
LABELS     = Path('./data/processed/ics_labels.csv')
MODELS_DIR = Path('./models')
FEAT_OUT   = Path('./data/processed/ics_features_v2.csv')


# ── New feature engineering ────────────────────────────────────────────────────

def add_targeted_features(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Add 4 features targeting the specific attack types with low recall.
    All features are derived from columns already present in the raw ICSSIM dataset.
    """
    feat = features.copy()

    # 1. inter_packet_timing_variance
    #    Replay attacks retransmit at near-constant intervals → very LOW variance.
    #    Normal ICS polling also has low variance, but replay is even lower AND
    #    combined with high packet count. Computed as coefficient of variation (CV)
    #    of inter-packet times. CV = std/mean — scale-invariant.
    if 'sInterPacketAvg' in raw.columns and 'rInterPacketAvg' in raw.columns:
        s_avg = raw['sInterPacketAvg'].fillna(0)
        r_avg = raw['rInterPacketAvg'].fillna(0)
        # Proxy for variance: absolute difference between src and dst inter-packet avg.
        # In replay, retransmitted packets come from one direction at fixed intervals
        # while the other direction shows normal variance → large asymmetry.
        feat['inter_packet_timing_asymmetry'] = (s_avg - r_avg).abs()
        # CV proxy: ratio of inter-packet avg to total duration / packet count
        # Low CV (near 0) = very regular timing = replay signature
        if 'duration' in raw.columns and 'sPackets' in raw.columns:
            duration_safe = raw['duration'].replace(0, 0.001)
            expected_interval = duration_safe / (raw['sPackets'].fillna(1) + 1)
            feat['timing_regularity'] = 1.0 / (
                (s_avg - expected_interval).abs() + 0.001
            )
            # Cap to prevent inf
            feat['timing_regularity'] = feat['timing_regularity'].clip(upper=1000)
        print("   ✅ inter_packet_timing features added (targets replay)")
    else:
        print("   ⚠️  sInterPacketAvg/rInterPacketAvg not found — skipping timing features")

    # 2. payload_repetition_score
    #    Replay attacks retransmit the same payload repeatedly → very low variance
    #    in payload size. Use (payload_avg / payload_max) as repetition proxy:
    #    if avg ≈ max, all packets are the same size → likely replay.
    if 'sPayloadAvg' in raw.columns and 'sBytesMax' in raw.columns:
        p_avg = raw['sPayloadAvg'].fillna(0)
        b_max = raw['sBytesMax'].replace(0, 1).fillna(1)
        feat['payload_size_consistency'] = p_avg / b_max
        print("   ✅ payload_size_consistency added (targets replay)")
    else:
        print("   ⚠️  sPayloadAvg/sBytesMax not found — skipping payload feature")

    # 3. unique_dst_ratio
    #    IP-scan: one source probes many destinations → high unique-dst count
    #    relative to total packets. In ICSSIM flow data, this is approximated by
    #    low bytes-per-packet + low duration + many packets (scanning signature).
    #    Direct unique-dst count isn't in per-flow features, but we can build a
    #    proxy: flows with very short duration, many packets, and small payloads
    #    are almost exclusively scans.
    if 'duration' in raw.columns and 'sPackets' in raw.columns and 'sBytesAvg' in raw.columns:
        duration = raw['duration'].fillna(0)
        packets  = raw['sPackets'].fillna(0)
        avg_size = raw['sBytesAvg'].fillna(0)

        # scan_score: high when short duration + many packets + small payload
        # Normalized so values are roughly [0, 1]
        duration_norm = 1.0 / (duration + 0.001)
        packet_density = packets * duration_norm
        size_penalty   = 1.0 / (avg_size + 1)

        feat['scan_signature_score'] = (
            packet_density * size_penalty
        ).clip(upper=1000)
        print("   ✅ scan_signature_score added (targets ip-scan, port-scan)")
    else:
        print("   ⚠️  duration/sPackets/sBytesAvg not found — skipping scan feature")

    # 4. flow_burstiness
    #    Low-rate DDoS and slow port-scan look like normal traffic per-flow but
    #    have unusual burstiness: packets arrive in short bursts with gaps.
    #    Burstiness = (max_bytes - avg_bytes) / (avg_bytes + 1) — high when
    #    there are spikes relative to the mean, which is characteristic of
    #    slow flood patterns.
    if 'sBytesMax' in raw.columns and 'sBytesAvg' in raw.columns:
        b_max = raw['sBytesMax'].fillna(0)
        b_avg = raw['sBytesAvg'].fillna(0)
        feat['flow_burstiness'] = (b_max - b_avg) / (b_avg + 1)
        print("   ✅ flow_burstiness added (targets low-rate DDoS, slow port-scan)")
    else:
        print("   ⚠️  sBytesMax/sBytesAvg not found — skipping burstiness feature")

    # Final cleanup
    feat = feat.fillna(0).replace([np.inf, -np.inf], 0)
    return feat


# ── Training + evaluation ──────────────────────────────────────────────────────

def train_and_evaluate(features: pd.DataFrame, labels: np.ndarray,
                       raw: pd.DataFrame, label: str):
    from src.models.ensemble_detector import EnsembleICSDetector

    X = features.values
    y = labels

    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
        X, y, raw.index, test_size=0.2, random_state=42, stratify=y
    )

    ensemble = EnsembleICSDetector(random_seed=42)
    ensemble.feature_names = list(features.columns)
    ensemble.train(X_train, y_train)

    # Use threshold=0.25 — from sweep analysis, best F1/F2 with this ensemble
    Xs = ensemble.scaler.transform(X_test)
    weights = ensemble.weights
    scores = np.zeros(len(X_test))
    scores += (ensemble.isolation_forest.predict(Xs) == -1).astype(int) * weights['isolation_forest']
    if ensemble.xgb_model:
        scores += ensemble.xgb_model.predict(Xs) * weights.get('xgboost', 0)
    if ensemble.rf_model:
        scores += ensemble.rf_model.predict(Xs) * weights.get('random_forest', 0)

    preds = (scores >= 0.25).astype(int)

    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

    # Per-attack-type recall
    if 'IT_M_Label' in raw.columns:
        attack_labels = raw.loc[idx_test, 'IT_M_Label'].values
        attack_mask   = y_test == 1
        print(f"\n  Per-attack recall (threshold=0.25):")
        for attack_type in ['replay', 'ddos', 'port-scan', 'ip-scan', 'mitm']:
            mask = attack_mask & (attack_labels == attack_type)
            if mask.sum() == 0:
                continue
            caught = preds[mask].sum()
            total  = mask.sum()
            print(f"    {attack_type:<12} {caught:>4}/{total:<4}  ({caught/total:.1%})")

    return ensemble, {'precision': prec, 'recall': rec, 'f1': f1}


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*60)
    print("TARGETED FEATURE ADDITION + RETRAIN")
    print("="*60)

    print("\n📥 Loading data...")
    raw      = pd.read_csv(RAW_DATA, low_memory=False)
    features = pd.read_csv(FEATURES)
    labels   = pd.read_csv(LABELS)['label'].values

    print(f"   Features v1: {features.shape[1]} columns")

    # Baseline — train on existing 51 features at threshold 0.25
    print("\n📊 BASELINE (51 features, threshold=0.25):")
    import warnings
    warnings.filterwarnings('ignore')
    _, baseline_metrics = train_and_evaluate(features, labels, raw, "BASELINE — 51 features")

    # Add new features
    print("\n⚙️  Adding targeted features...")
    features_v2 = add_targeted_features(raw, features)
    new_cols = [c for c in features_v2.columns if c not in features.columns]
    print(f"\n   New features added: {new_cols}")
    print(f"   Total features v2: {features_v2.shape[1]}")

    # Retrain on v2 features
    print("\n🚀 RETRAIN (v2 features):")
    ensemble_v2, v2_metrics = train_and_evaluate(features_v2, labels, raw, f"V2 — {features_v2.shape[1]} features")

    # Delta summary
    print("\n" + "="*60)
    print("  IMPROVEMENT SUMMARY")
    print("="*60)
    print(f"  {'Metric':<12} {'Baseline':>10} {'V2':>10} {'Delta':>10}")
    print(f"  {'-'*42}")
    for m in ['precision', 'recall', 'f1']:
        delta = v2_metrics[m] - baseline_metrics[m]
        sign  = '+' if delta >= 0 else ''
        print(f"  {m:<12} {baseline_metrics[m]:>10.4f} {v2_metrics[m]:>10.4f} {sign}{delta:>9.4f}")

    # Save v2 features and retrained models only if recall improved
    if v2_metrics['recall'] > baseline_metrics['recall']:
        print(f"\n✅ Recall improved — saving v2 features and models...")
        features_v2.to_csv(FEAT_OUT, index=False)
        print(f"   Saved: {FEAT_OUT}")

        # Update feature_names.txt for downstream consumers
        with open(MODELS_DIR / 'feature_names.txt', 'w') as f:
            f.write('\n'.join(features_v2.columns))

        # Save ensemble_v2 models (overwrites existing)
        ensemble_v2.save(str(MODELS_DIR))

        # Update metadata
        meta_path = MODELS_DIR / 'model_metadata.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta['n_features'] = features_v2.shape[1]
        meta['feature_version'] = 'v2'
        meta['new_features'] = new_cols
        meta['decision_threshold'] = 0.25
        meta['performance'] = {
            'precision': round(v2_metrics['precision'], 6),
            'recall':    round(v2_metrics['recall'], 6),
            'f1_score':  round(v2_metrics['f1'], 6),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"   Updated: model_metadata.json")

        print(f"""
Next steps:
  1. Replace data/processed/ics_features.csv with ics_features_v2.csv
     (or update all loader paths to point to _v2)
  2. In ensemble_detector.py predict(), change threshold 0.5 → 0.25
  3. Update README with new feature count and metrics
""")
    else:
        print(f"\n⚠️  Recall did not improve — new features not saved.")
        print(f"   This suggests the attack types require sequence-level features")
        print(f"   (session aggregation across flows) rather than per-flow statistics.")
        print(f"   Consider: group flows by src_ip+dst_ip within time windows.")
