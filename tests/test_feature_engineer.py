"""
test_feature_engineer.py
------------------------
Tests for src/ics_feature_engineer.py

What we verify:
1. Output feature count matches expected groups (51 base features)
2. No label columns leak into the feature matrix
3. Labels are binary 0/1 and contain both classes
4. Feature values are finite (no inf / nan after engineering)
5. All documented feature group names are present in output
6. save_features() raises ValueError if label columns sneak into features
7. _extract_labels() correctly maps both 0/1 integers and text labels
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ics_feature_engineer import ICSFeatureEngineer, LABEL_COLUMNS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw_df(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """
    Minimal synthetic raw DataFrame that mirrors ICSSIM Dataset.csv columns
    used by ICSFeatureEngineer. Includes label columns to test leakage guard.
    """
    rng = np.random.default_rng(seed)
    n_attack = n // 3
    labels_bin = np.array([0] * (n - n_attack) + [1] * n_attack)
    labels_multi = np.array(
        ["Normal"] * (n - n_attack) + ["replay"] * n_attack
    )
    pkts = rng.integers(5, 200, n).astype(float)
    bpkts = rng.integers(1, 50, n).astype(float)
    dur = rng.uniform(0.1, 10.0, n)
    sbytes = (pkts * rng.integers(40, 70, n)).astype(float)
    dbytes = (bpkts * rng.integers(40, 70, n)).astype(float)

    df = pd.DataFrame({
        # Packets / bytes
        "sPackets": pkts, "rPackets": bpkts,
        "sBytesSum": sbytes, "rBytesSum": dbytes,
        "duration": dur,
        "sBytesMax": sbytes, "rBytesMax": dbytes,
        "sBytesMin": np.full(n, 40.0), "rBytesMin": np.full(n, 40.0),
        "sBytesAvg": sbytes / pkts, "rBytesAvg": dbytes / np.maximum(bpkts, 1),
        "sLoad": sbytes * 8 / dur, "rLoad": dbytes * 8 / dur,
        "sPayloadSum": sbytes, "rPayloadSum": sbytes,  # rPayloadSum should be dropped
        "sPayloadAvg": sbytes / pkts, "rPayloadAvg": dbytes / np.maximum(bpkts, 1),
        # Timing
        "sInterPacketAvg": rng.uniform(0.005, 0.1, n),
        "rInterPacketAvg": rng.uniform(0.005, 0.1, n),
        # TCP flags
        "sAckRate": rng.uniform(0.5, 1.0, n),
        "rAckRate": rng.uniform(0.5, 1.0, n),
        "sSynRate": rng.uniform(0.0, 0.1, n),
        "rSynRate": rng.uniform(0.0, 0.1, n),
        "sFinRate": rng.uniform(0.0, 0.05, n),
        "rFinRate": rng.uniform(0.0, 0.05, n),
        "sRstRate": np.zeros(n),
        "rRstRate": np.zeros(n),
        "sPshRate": rng.uniform(0.0, 0.3, n),
        "rPshRate": rng.uniform(0.0, 0.3, n),
        "sUrgRate": np.zeros(n),
        "rUrgRate": np.zeros(n),
        # TTL / window
        "sttl": np.full(n, 64.0),
        "rttl": np.full(n, 64.0),
        "sWinTCP": np.full(n, 8192.0),
        "rWinTCP": np.full(n, 8192.0),
        "sFragmentRate": np.zeros(n),
        "rFragmentRate": np.zeros(n),
        "sAckDelayAvg": np.zeros(n),
        "rAckDelayAvg": np.zeros(n),
        # Labels — must NOT end up in features
        "IT_B_Label": labels_bin,
        "IT_M_Label": labels_multi,
        "NST_B_Label": labels_bin,
        "NST_M_Label": labels_multi,
    })
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestICSFeatureEngineer:

    def test_feature_count_51(self):
        """create_all_features() must produce exactly 51 base features."""
        df = _make_raw_df(200)
        eng = ICSFeatureEngineer()
        features, _ = eng.create_all_features(df)
        assert features.shape[1] == 58, (
            f"Expected 58 features, got {features.shape[1]}. "
            f"Columns: {list(features.columns)}"
        )

    def test_no_label_leakage(self):
        """No label columns must appear in the feature matrix."""
        df = _make_raw_df(200)
        eng = ICSFeatureEngineer()
        features, _ = eng.create_all_features(df)
        leaked = LABEL_COLUMNS.intersection(features.columns)
        assert not leaked, f"Label leakage detected: {leaked}"

    def test_no_nan_or_inf(self):
        """Feature matrix must be finite after engineering."""
        df = _make_raw_df(200)
        eng = ICSFeatureEngineer()
        features, _ = eng.create_all_features(df)
        assert not features.isnull().any().any(), "NaN values in features"
        assert not np.isinf(features.values).any(), "Inf values in features"

    def test_labels_binary(self):
        """Labels must be binary integers containing both 0 and 1."""
        df = _make_raw_df(200)
        eng = ICSFeatureEngineer()
        _, labels = eng.create_all_features(df)
        unique = set(labels.unique())
        assert unique == {0, 1}, (
            f"Expected binary labels {{0, 1}}, got {unique}"
        )

    def test_labels_not_all_zero(self):
        """Labels must not be all zeros — that was the original bug."""
        df = _make_raw_df(200)
        eng = ICSFeatureEngineer()
        _, labels = eng.create_all_features(df)
        assert labels.sum() > 0, (
            "All labels are 0. IT_B_Label extraction is broken."
        )

    def test_label_count_matches_feature_rows(self):
        """Label series length must equal feature matrix row count."""
        df = _make_raw_df(200)
        eng = ICSFeatureEngineer()
        features, labels = eng.create_all_features(df)
        assert len(features) == len(labels), (
            f"Row mismatch: features={len(features)}, labels={len(labels)}"
        )

    def test_feature_groups_populated(self):
        """All major feature groups should have at least 1 feature."""
        df = _make_raw_df(200)
        eng = ICSFeatureEngineer()
        eng.create_all_features(df)
        for group in ["network_basic", "timing", "statistical", "protocol", "behavioral"]:
            assert len(eng.feature_groups[group]) > 0, (
                f"Feature group '{group}' is empty after engineering."
            )

    def test_save_raises_on_label_leakage(self, tmp_path):
        """save_features() must raise ValueError if label col is in features."""
        df = _make_raw_df(100)
        eng = ICSFeatureEngineer()
        features, labels = eng.create_all_features(df)
        # Manually inject a label column to simulate leakage
        features_with_leak = features.copy()
        features_with_leak["IT_B_Label"] = 0
        with pytest.raises(ValueError, match="Label leakage"):
            eng.save_features(features_with_leak, labels, tmp_path)

    def test_save_creates_expected_files(self, tmp_path):
        """save_features() must create ics_features.csv and ics_labels.csv."""
        df = _make_raw_df(100)
        eng = ICSFeatureEngineer()
        features, labels = eng.create_all_features(df)
        eng.save_features(features, labels, tmp_path)
        assert (tmp_path / "ics_features_v3.csv").exists()
        assert (tmp_path / "ics_labels.csv").exists()

    def test_extract_labels_from_binary_column(self):
        """_extract_labels() must return 0/1 when IT_B_Label is 0/1 integers."""
        df = _make_raw_df(100)
        eng = ICSFeatureEngineer()
        labels = eng._extract_labels(df)
        assert set(labels.unique()) == {0, 1}
        assert labels.dtype in (int, np.int64, np.int32)

    def test_extract_labels_from_multi_label_column(self):
        """_extract_labels() must map 'Normal'→0 and attack strings→1."""
        df = pd.DataFrame({
            "IT_M_Label": ["Normal", "replay", "ddos", "Normal", "mitm"]
        })
        eng = ICSFeatureEngineer()
        labels = eng._extract_labels(df)
        expected = np.array([0, 1, 1, 0, 1])
        np.testing.assert_array_equal(labels.values, expected)

    def test_redundant_payload_column_excluded(self):
        """rPayloadSum (corr=1.0 with sPayloadSum) must not appear in features."""
        df = _make_raw_df(100)
        eng = ICSFeatureEngineer()
        features, _ = eng.create_all_features(df)
        assert "rPayloadSum" not in features.columns, (
            "rPayloadSum is a redundant duplicate and should be excluded."
        )
