"""
ics_feature_engineer.py
-----------------------
Feature engineering pipeline for ICS anomaly detection (ICSSIM dataset).

Author: Sadhana Devarajan
Version: 3.1.0

Feature groups (63 total) — quick_start.py / create_all_features() path:
  network_basic  : 10 (src/dst packets, bytes, totals, packet_ratio, byte_ratio, bytes_per_packet)
  timing         : 6  (inter-packet avg, packet/byte rates)
  statistical    : 11 (byte stats, load, payload)
  protocol       : 20 (TCP flags x12, TTL/win/fragment/ack_delay x8)
  behavioral     : 4  (syn_ack_imbalance, packet_size_anomaly, reset_rate_total, traffic_symmetry)
  engineered     : 1  (byte_rate_asymmetry)
  session        : 6  (unique_dst, flow_count, inter_flow_interval, inter_flow_variance,
                       src_dst_flow_ratio, payload_entropy)

To use this 63-feature path add create_v2_per_flow_features() inside create_all_features()
(between behavioral and engineered) then retrain. See Section 3 of HANDOFF.md.

Fixes applied in v3.1.0:
  - Added 'engineered' feature group (previously printed as 0 in quick_start.py)
  - save_features() now also writes ics_features_v2.csv (pre-session cols) so
    session_feature_experiment.py can run its ablation without a FileNotFoundError
  - Removed label leakage, redundant timestamps, rPayloadSum (unchanged from v3.0)
  - Session features: early-return now fills zeros instead of returning empty DataFrame
    so concat never silently drops rows
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Columns never allowed in the feature matrix ──────────────────────────────
LABEL_COLUMNS = {"IT_B_Label", "IT_M_Label", "NST_B_Label", "NST_M_Label"}
REDUNDANT_TIMESTAMP_COLS = {"start", "end", "startOffset", "endOffset"}
REDUNDANT_PAYLOAD_COLS = {"rPayloadSum"}


class ICSFeatureEngineer:
    """
    End-to-end feature engineering for ICS network anomaly detection.

    Usage
    -----
    engineer = ICSFeatureEngineer(random_seed=42)
    features, labels = engineer.create_all_features(df)
    engineer.save_features(features, labels, Path('./data/processed'))
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        np.random.seed(random_seed)

        # All groups defined upfront so quick_start.py prints correct counts
        self.feature_groups: dict[str, list[str]] = {
            "network_basic": [],
            "timing": [],
            "statistical": [],
            "protocol": [],
            "behavioral": [],
            "engineered": [],   # ← was missing, causing always-0 print
            "session": [],
        }

    # =========================================================================
    # BASIC NETWORK FEATURES  (9)
    # =========================================================================

    def create_basic_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating basic network features...")
        features = pd.DataFrame(index=df.index)

        raw_map = [
            ("sPackets",  "src_packets"),
            ("rPackets",  "dst_packets"),
            ("sBytesSum", "src_bytes"),
            ("rBytesSum", "dst_bytes"),
            ("duration",  "flow_duration"),
        ]
        for src, dst in raw_map:
            if src in df.columns:
                features[dst] = df[src].fillna(0)
                self.feature_groups["network_basic"].append(dst)

        if {"src_packets", "dst_packets"}.issubset(features.columns):
            features["total_packets"] = features["src_packets"] + features["dst_packets"]
            features["packet_ratio"]  = features["src_packets"] / (features["dst_packets"] + 1)
            self.feature_groups["network_basic"].extend(["total_packets", "packet_ratio"])

        if {"src_bytes", "dst_bytes"}.issubset(features.columns):
            features["total_bytes"] = features["src_bytes"] + features["dst_bytes"]
            features["byte_ratio"]  = features["src_bytes"] / (features["dst_bytes"] + 1)
            self.feature_groups["network_basic"].extend(["total_bytes", "byte_ratio"])

        if {"total_bytes", "total_packets"}.issubset(features.columns):
            features["bytes_per_packet"] = features["total_bytes"] / (features["total_packets"] + 1)
            self.feature_groups["network_basic"].append("bytes_per_packet")

        logger.info(f"   Created {len(features.columns)} basic network features")
        return features

    # =========================================================================
    # TIMING FEATURES  (6)
    # =========================================================================

    def create_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating timing features...")
        features = pd.DataFrame(index=df.index)

        if "sInterPacketAvg" in df.columns:
            features["src_inter_packet_avg"] = df["sInterPacketAvg"].fillna(0)
            self.feature_groups["timing"].append("src_inter_packet_avg")

        if "rInterPacketAvg" in df.columns:
            features["dst_inter_packet_avg"] = df["rInterPacketAvg"].fillna(0)
            self.feature_groups["timing"].append("dst_inter_packet_avg")

        if "duration" in df.columns and df["duration"].notna().any():
            duration_safe = df["duration"].replace(0, 0.001)
            rate_map = [
                ("sPackets",  "src_packet_rate"),
                ("rPackets",  "dst_packet_rate"),
                ("sBytesSum", "src_byte_rate"),
                ("rBytesSum", "dst_byte_rate"),
            ]
            for src, dst in rate_map:
                if src in df.columns:
                    features[dst] = df[src] / duration_safe
                    self.feature_groups["timing"].append(dst)

        logger.info(f"   Created {len(features.columns)} timing features")
        return features

    # =========================================================================
    # STATISTICAL FEATURES  (11)
    # =========================================================================

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating statistical features...")
        features = pd.DataFrame(index=df.index)

        stat_map = [
            ("sBytesMax",    "src_bytes_max"),
            ("rBytesMax",    "dst_bytes_max"),
            ("sBytesMin",    "src_bytes_min"),
            ("rBytesMin",    "dst_bytes_min"),
            ("sBytesAvg",    "src_bytes_avg"),
            ("rBytesAvg",    "dst_bytes_avg"),
            ("sLoad",        "src_load"),
            ("rLoad",        "dst_load"),
            ("sPayloadSum",  "src_payload_sum"),
            ("sPayloadAvg",  "src_payload_avg"),
            ("rPayloadAvg",  "dst_payload_avg"),
        ]
        for src, dst in stat_map:
            if src in df.columns:
                features[dst] = df[src].fillna(0)
                self.feature_groups["statistical"].append(dst)

        logger.info(f"   Created {len(features.columns)} statistical features")
        return features

    # =========================================================================
    # PROTOCOL FEATURES  (20)
    # =========================================================================

    def create_protocol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating protocol features...")
        features = pd.DataFrame(index=df.index)

        flag_map = [
            ("sAckRate",      "src_ack_rate"),
            ("rAckRate",      "dst_ack_rate"),
            ("sSynRate",      "src_syn_rate"),
            ("rSynRate",      "dst_syn_rate"),
            ("sFinRate",      "src_fin_rate"),
            ("rFinRate",      "dst_fin_rate"),
            ("sRstRate",      "src_rst_rate"),
            ("rRstRate",      "dst_rst_rate"),
            ("sPshRate",      "src_psh_rate"),
            ("rPshRate",      "dst_psh_rate"),
            ("sUrgRate",      "src_urg_rate"),
            ("rUrgRate",      "dst_urg_rate"),
        ]
        for src, dst in flag_map:
            if src in df.columns:
                features[dst] = df[src].fillna(0)
                self.feature_groups["protocol"].append(dst)

        # (src_col, dst_col, fill_value)
        proto_map = [
            ("sttl",           "src_ttl",           64),
            ("rttl",           "dst_ttl",           64),
            ("sWinTCP",        "src_win_size",       0),
            ("rWinTCP",        "dst_win_size",       0),
            ("sFragmentRate",  "src_fragment_rate",  0),
            ("rFragmentRate",  "dst_fragment_rate",  0),
            ("sAckDelayAvg",   "src_ack_delay",      0),
            ("rAckDelayAvg",   "dst_ack_delay",      0),
        ]
        for src, dst, default in proto_map:
            if src in df.columns:
                features[dst] = df[src].fillna(default)
                self.feature_groups["protocol"].append(dst)

        logger.info(f"   Created {len(features.columns)} protocol features")
        return features

    # =========================================================================
    # BEHAVIORAL FEATURES  (4)
    # =========================================================================

    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating behavioral features...")
        features = pd.DataFrame(index=df.index)

        if {"sSynRate", "sAckRate"}.issubset(df.columns):
            features["syn_ack_imbalance"] = df["sSynRate"] - df["sAckRate"]
            self.feature_groups["behavioral"].append("syn_ack_imbalance")

        if {"sPackets", "sBytesSum"}.issubset(df.columns):
            avg_pkt = df["sBytesSum"] / (df["sPackets"] + 1)
            features["packet_size_anomaly"] = (avg_pkt - avg_pkt.mean()).abs()
            self.feature_groups["behavioral"].append("packet_size_anomaly")

        if {"sRstRate", "rRstRate"}.issubset(df.columns):
            features["reset_rate_total"] = df["sRstRate"] + df["rRstRate"]
            self.feature_groups["behavioral"].append("reset_rate_total")

        if {"sPackets", "rPackets"}.issubset(df.columns):
            total = df["sPackets"] + df["rPackets"] + 1
            features["traffic_symmetry"] = 1 - (df["sPackets"] - df["rPackets"]).abs() / total
            self.feature_groups["behavioral"].append("traffic_symmetry")

        logger.info(f"   Created {len(features.columns)} behavioral features")
        return features

    # =========================================================================
    # ENGINEERED FEATURES  (1)  ← fixes the always-0 bug
    # =========================================================================

    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Higher-order features derived from combinations of raw columns.
        Currently: byte_rate_asymmetry — captures DDoS / exfil patterns
        where outbound and inbound byte rates diverge sharply.
        """
        logger.info("Creating engineered features...")
        features = pd.DataFrame(index=df.index)

        if {"sBytesSum", "rBytesSum", "duration"}.issubset(df.columns):
            dur = df["duration"].replace(0, 0.001)
            src_rate = df["sBytesSum"] / dur
            dst_rate = df["rBytesSum"] / dur
            features["byte_rate_asymmetry"] = (src_rate - dst_rate).abs() / (src_rate + dst_rate + 1)
            self.feature_groups["engineered"].append("byte_rate_asymmetry")

        logger.info(f"   Created {len(features.columns)} engineered features")
        return features

    # =========================================================================
    # V2 PER-FLOW FEATURES  (4)  ← upgrade path: adds network_advanced group
    # Call create_all_features_v2() instead of create_all_features() to get
    # 62 features total. Requires retraining all models + scaler.
    # =========================================================================

    def create_v2_per_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Additional per-flow packet-level statistics that form the
        'network_advanced' group (5 features).

        Not included in create_all_features() by default because the trained
        ensemble was built on 58 features. To use this 63-feature path:
          1. Call create_all_features_v2() (below) instead of create_all_features()
          2. Re-run quick_start.py — retrains all three models + scaler on 63 features
          3. Update models/model_metadata.json → "n_features": 63
        """
        logger.info("Creating v2 per-flow features...")
        features = pd.DataFrame(index=df.index)

        if {"sPackets", "sBytesSum"}.issubset(df.columns):
            features["src_bytes_per_packet"] = df["sBytesSum"] / (df["sPackets"] + 1)
            self.feature_groups.setdefault("network_advanced", []).append("src_bytes_per_packet")

        if {"rPackets", "rBytesSum"}.issubset(df.columns):
            features["dst_bytes_per_packet"] = df["rBytesSum"] / (df["rPackets"] + 1)
            self.feature_groups.setdefault("network_advanced", []).append("dst_bytes_per_packet")

        if {"sBytesAvg", "sBytesMax", "sBytesMin"}.issubset(df.columns):
            range_ = df["sBytesMax"] - df["sBytesMin"]
            features["src_bytes_range"] = range_
            features["src_bytes_cv"] = range_ / (df["sBytesAvg"] + 1)
            self.feature_groups.setdefault("network_advanced", []).extend(
                ["src_bytes_range", "src_bytes_cv"]
            )

        if {"sPackets", "rPackets", "duration"}.issubset(df.columns):
            dur = df["duration"].replace(0, 0.001)
            features["total_packet_rate"] = (df["sPackets"] + df["rPackets"]) / dur
            self.feature_groups.setdefault("network_advanced", []).append("total_packet_rate")

        logger.info(f"   Created {len(features.columns)} v2 per-flow features")
        return features

    def create_all_features_v2(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        63-feature pipeline: same as create_all_features() but inserts
        create_v2_per_flow_features() between behavioral and engineered.

        Run quick_start.py after switching to this path to retrain models on 63 features.
        """
        logger.info("\n" + "=" * 80)
        logger.info("ICS FEATURE ENGINEERING v3.1  [63-feature path]")
        logger.info("=" * 80)

        basic        = self.create_basic_network_features(df)
        timing       = self.create_timing_features(df)
        statistical  = self.create_statistical_features(df)
        protocol     = self.create_protocol_features(df)
        behavioral   = self.create_behavioral_features(df)
        network_adv  = self.create_v2_per_flow_features(df)      # ← +4 features
        engineered   = self.create_engineered_features(df)
        session      = self.create_session_features(df)

        all_features = pd.concat(
            [basic, timing, statistical, protocol, behavioral, network_adv, engineered, session],
            axis=1,
        )

        leaked = LABEL_COLUMNS & set(all_features.columns)
        if leaked:
            all_features.drop(columns=list(leaked), inplace=True)

        drop_cols = (REDUNDANT_TIMESTAMP_COLS | REDUNDANT_PAYLOAD_COLS) & set(all_features.columns)
        if drop_cols:
            all_features.drop(columns=list(drop_cols), inplace=True)

        all_features = all_features.fillna(0).replace([np.inf, -np.inf], 0)
        labels = self._extract_labels(df)

        logger.info("\n" + "=" * 80)
        logger.info("FEATURE ENGINEERING SUMMARY  [63-feature path]")
        logger.info("=" * 80)
        logger.info(f"Total features : {len(all_features.columns)}")
        logger.info(f"Total samples  : {len(all_features)}")
        logger.info("\nFeature breakdown:")
        for group, feats in self.feature_groups.items():
            if feats:
                logger.info(f"  • {group:<15}: {len(feats)}")
        normal_count = (labels == 0).sum()
        attack_count = (labels == 1).sum()
        logger.info(f"\nLabel distribution:")
        logger.info(f"  Normal : {normal_count:,}")
        logger.info(f"  Attack : {attack_count:,}")
        logger.info("=" * 80 + "\n")

        return all_features, labels

    # =========================================================================
    # SESSION FEATURES  (6)
    # =========================================================================

    def create_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Temporal / cross-flow behavioral intelligence.
        Strongest single feature: src_inter_flow_variance (ratio ~11.76 for replay).
        Falls back to zeros if required columns are missing (never returns empty DataFrame).
        """
        logger.info("Creating session-level behavioral features...")

        # Zero-filled fallback keeps concat safe if columns are absent
        zero_cols = [
            "src_unique_dst_count",
            "src_flow_count",
            "src_inter_flow_interval",
            "src_inter_flow_variance",
            "src_dst_flow_ratio",
            "src_payload_entropy",
        ]
        features = pd.DataFrame(0.0, index=df.index, columns=zero_cols)
        for col in zero_cols:
            self.feature_groups["session"].append(col)

        required = {"sIPs", "rIPs"}
        if not required.issubset(df.columns):
            logger.warning(f"Missing columns {required - set(df.columns)} — session features set to 0")
            return features

        # Resolve timestamp column
        time_col = next((c for c in ("timestamp", "start") if c in df.columns), None)
        if time_col is None:
            logger.warning("No timestamp column found — session features set to 0")
            return features

        timestamps = pd.to_datetime(df[time_col], errors="coerce")

        working = pd.DataFrame(
            {"srcIP": df["sIPs"], "dstIP": df["rIPs"], "ts": timestamps},
            index=df.index,
        ).sort_values("ts")

        # Unique destination count per source IP
        features["src_unique_dst_count"] = (
            working.groupby("srcIP")["dstIP"].transform("nunique")
        )

        # Flow count per source IP
        flow_count = working.groupby("srcIP")["srcIP"].transform("count")
        features["src_flow_count"] = flow_count

        # Inter-flow interval (seconds between successive flows from same IP)
        inter_times = (
            working.groupby("srcIP")["ts"]
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )
        features["src_inter_flow_interval"] = inter_times

        # Rolling variance of inter-flow intervals (primary replay signal)
        variance = inter_times.groupby(working["srcIP"]).transform(
            lambda x: x.rolling(10, min_periods=2).var()
        )
        features["src_inter_flow_variance"] = variance.fillna(0)

        # Src-dst pair flow ratio
        pair_counts = working.groupby(["srcIP", "dstIP"])["srcIP"].transform("count")
        features["src_dst_flow_ratio"] = pair_counts / (flow_count + 1)

        # Payload entropy approximation
        if "sPayloadSum" in df.columns:
            features["src_payload_entropy"] = np.log1p(df["sPayloadSum"].fillna(0))
        # else stays 0 from initialisation

        # Re-align to original index order, fill any NaNs introduced by groupby
        features = features.reindex(df.index).fillna(0).replace([np.inf, -np.inf], 0)

        logger.info(f"   Created {len(features.columns)} session features")
        return features

    # =========================================================================
    # LABEL EXTRACTION
    # =========================================================================

    def _extract_labels(self, df: pd.DataFrame) -> pd.Series:
        if "IT_B_Label" in df.columns:
            return df["IT_B_Label"].fillna(0).astype(int)
        if "NST_B_Label" in df.columns:
            return df["NST_B_Label"].fillna(0).astype(int)
        if "IT_M_Label" in df.columns:
            return (df["IT_M_Label"].fillna("Normal") != "Normal").astype(int)
        logger.warning("No label column found — all labels set to 0")
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    # =========================================================================
    # CREATE ALL FEATURES  (main entry point)
    # =========================================================================

    def create_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("\n" + "=" * 80)
        logger.info("ICS FEATURE ENGINEERING v3.1")
        logger.info("=" * 80)

        basic      = self.create_basic_network_features(df)
        timing     = self.create_timing_features(df)
        statistical = self.create_statistical_features(df)
        protocol   = self.create_protocol_features(df)
        behavioral = self.create_behavioral_features(df)
        engineered = self.create_engineered_features(df)   # ← new
        session    = self.create_session_features(df)

        all_features = pd.concat(
            [basic, timing, statistical, protocol, behavioral, engineered, session],
            axis=1,
        )

        # Remove label leakage
        leaked = LABEL_COLUMNS & set(all_features.columns)
        if leaked:
            logger.warning(f"Removing leaked labels: {leaked}")
            all_features.drop(columns=list(leaked), inplace=True)

        # Remove redundant timestamps & duplicate payload
        drop_cols = (REDUNDANT_TIMESTAMP_COLS | REDUNDANT_PAYLOAD_COLS) & set(all_features.columns)
        if drop_cols:
            all_features.drop(columns=list(drop_cols), inplace=True)

        all_features = all_features.fillna(0).replace([np.inf, -np.inf], 0)

        labels = self._extract_labels(df)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total features : {len(all_features.columns)}")
        logger.info(f"Total samples  : {len(all_features)}")
        logger.info("\nFeature breakdown:")
        for group, feats in self.feature_groups.items():
            if feats:
                logger.info(f"  • {group:<15}: {len(feats)}")
        normal_count = (labels == 0).sum()
        attack_count = (labels == 1).sum()
        logger.info(f"\nLabel distribution:")
        logger.info(f"  Normal : {normal_count:,}")
        logger.info(f"  Attack : {attack_count:,}")
        logger.info("=" * 80 + "\n")

        return all_features, labels

    # =========================================================================
    # SAVE FEATURES
    # =========================================================================

    def save_features(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        output_dir: Path,
    ):
        """
        Saves:
          ics_features_v3.csv   — full 63-feature matrix (used by quick_start.py)
          ics_features_v2.csv   — pre-session 57-feature matrix (used by session ablation)
          ics_labels.csv
          feature_groups.json
        """
        label_cols = {"IT_B_Label","IT_M_Label","NST_B_Label","NST_M_Label"}
        found = label_cols & set(features.columns)
        if found:
            raise ValueError(f"Label leakage: {found} found in features")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full v3 features
        v3_path = output_dir / "ics_features_v3.csv"
        features.to_csv(v3_path, index=False)
        logger.info(f"✅ Saved features v3 : {v3_path}")

        # Pre-session v2 features (all groups except 'session')
        session_cols = set(self.feature_groups.get("session", []))
        v2_cols = [c for c in features.columns if c not in session_cols]
        v2_path = output_dir / "ics_features_v2.csv"
        features[v2_cols].to_csv(v2_path, index=False)
        logger.info(f"✅ Saved features v2 : {v2_path}  ({len(v2_cols)} cols, no session)")

        # Labels
        labels_path = output_dir / "ics_labels.csv"
        labels.to_csv(labels_path, index=False, header=["label"])
        logger.info(f"✅ Saved labels      : {labels_path}")

        # Feature group metadata
        meta_path = output_dir / "feature_groups.json"
        meta_path.write_text(json.dumps(self.feature_groups, indent=2))
        logger.info(f"✅ Saved metadata    : {meta_path}")

        return v3_path, labels_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ICS FEATURE ENGINEER v3.1")
    print("=" * 80)

    data_path = Path("./data/raw/kaggle/icssim/Dataset.csv")

    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
    else:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} flows, {len(df.columns)} raw columns")

        engineer = ICSFeatureEngineer(random_seed=42)
        features, labels = engineer.create_all_features(df)
        engineer.save_features(features, labels, Path("./data/processed"))

        print(f"\nFinal feature count : {features.shape[1]}")
        print(f"Sample rows:\n{features.head(3)}")
        print("\n✅ Ready for quick_start.py")
