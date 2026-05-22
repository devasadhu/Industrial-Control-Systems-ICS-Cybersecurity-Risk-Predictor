"""
test_session_features.py
------------------------
Tests for session_features.py — compute_session_features()

What we verify:
1. Output has exactly 6 new columns with correct names
2. Row count matches input (left-join semantics)
3. All values are finite
4. src_inter_flow_variance is lower for mechanically-timed (replay) IPs
   than for randomly-timed (normal) IPs — the key discriminator (ratio=11.76)
5. Single-flow IPs get interval=0 and variance=0 (no division errors)
6. src_unique_dst_count counts unique destinations per time window correctly
7. src_payload_entropy is non-negative
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from session_features import compute_session_features

SESSION_COLS = [
    "src_unique_dst_count",
    "src_flow_count",
    "src_inter_flow_interval",
    "src_inter_flow_variance",
    "src_dst_flow_ratio",
    "src_payload_entropy",
]


class TestComputeSessionFeatures:

    def test_output_column_names(self, session_raw_df):
        """Output must have exactly the 6 documented column names."""
        result = compute_session_features(session_raw_df)
        assert list(result.columns) == SESSION_COLS, (
            f"Expected columns {SESSION_COLS}, got {list(result.columns)}"
        )

    def test_row_count_preserved(self, session_raw_df):
        """Output must have the same number of rows as input."""
        result = compute_session_features(session_raw_df)
        assert len(result) == len(session_raw_df), (
            f"Row count changed: input={len(session_raw_df)}, "
            f"output={len(result)}"
        )

    def test_no_nan_or_inf(self, session_raw_df):
        """All output values must be finite."""
        result = compute_session_features(session_raw_df)
        assert not result.isnull().any().any(), "NaN values in session features"
        assert not np.isinf(result.values).any(), "Inf values in session features"

    def test_variance_lower_for_replay_ip(self):
        """
        An IP that sends flows at perfectly regular intervals must have
        lower src_inter_flow_variance than one with random intervals.

        This is the key property that drove replay recall from 49.5% → 97.5%.
        Normal/attack variance ratio in production data is 11.76.
        """
        # Replay IP: flows every exactly 1 second for 120 seconds
        replay_times = np.arange(0, 120, 1.0)
        # Normal IP: random times across the same window
        rng = np.random.default_rng(0)
        normal_times = np.sort(rng.uniform(0, 120, len(replay_times)))

        n_replay = len(replay_times)
        n_normal = len(normal_times)

        df = pd.DataFrame({
            "sIPs": (["replay_ip"] * n_replay + ["normal_ip"] * n_normal),
            "rIPs": ["192.168.1.1"] * (n_replay + n_normal),
            "start": np.concatenate([replay_times, normal_times]),
            "sPayloadAvg": np.full(n_replay + n_normal, 60.0),
            "duration": np.ones(n_replay + n_normal),
            "sPackets": np.ones(n_replay + n_normal),
            "sBytesAvg": np.full(n_replay + n_normal, 60.0),
        })

        result = compute_session_features(df)
        result["sIP"] = df["sIPs"].values

        replay_var = result[result["sIP"] == "replay_ip"]["src_inter_flow_variance"].mean()
        normal_var = result[result["sIP"] == "normal_ip"]["src_inter_flow_variance"].mean()

        assert replay_var < normal_var, (
            f"Replay IP variance ({replay_var:.4f}) should be < "
            f"normal IP variance ({normal_var:.4f}). "
            "Session feature discriminator is broken."
        )

    def test_single_flow_ip_gets_zero_variance(self):
        """An IP with exactly one flow in the window gets interval=0, variance=0."""
        df = pd.DataFrame({
            "sIPs": ["10.0.0.1"],
            "rIPs": ["192.168.1.1"],
            "start": [0.0],
            "sPayloadAvg": [60.0],
            "duration": [1.0],
            "sPackets": [10.0],
            "sBytesAvg": [60.0],
        })
        result = compute_session_features(df)
        assert result["src_inter_flow_interval"].iloc[0] == 0.0
        assert result["src_inter_flow_variance"].iloc[0] == 0.0

    def test_unique_dst_count(self):
        """
        src_unique_dst_count must equal the number of unique destinations
        contacted by a source IP within the time window.
        """
        # 1 source IP, 3 different destinations, all within 60s window
        df = pd.DataFrame({
            "sIPs": ["attacker"] * 6,
            "rIPs": ["dst1", "dst2", "dst3", "dst1", "dst2", "dst3"],
            "start": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "sPayloadAvg": [60.0] * 6,
            "duration": [1.0] * 6,
            "sPackets": [1.0] * 6,
            "sBytesAvg": [60.0] * 6,
        })
        result = compute_session_features(df)
        # All flows are within 60s of each other, so window captures all 3 dsts
        assert result["src_unique_dst_count"].iloc[0] == 3.0

    def test_payload_entropy_non_negative(self, session_raw_df):
        """Shannon entropy must be >= 0 for all flows."""
        result = compute_session_features(session_raw_df)
        assert (result["src_payload_entropy"] >= 0).all(), (
            "Negative entropy detected — log computation error."
        )

    def test_missing_start_column_raises(self):
        """compute_session_features() must raise ValueError if 'start' is absent."""
        df = pd.DataFrame({
            "sIPs": ["10.0.0.1"],
            "rIPs": ["192.168.1.1"],
            "sPayloadAvg": [60.0],
        })
        with pytest.raises(ValueError, match="'start' column not found"):
            compute_session_features(df)

    def test_dst_flow_ratio_at_least_one(self, session_raw_df):
        """src_dst_flow_ratio must be >= 1.0 (flows >= unique destinations)."""
        result = compute_session_features(session_raw_df)
        assert (result["src_dst_flow_ratio"] >= 1.0).all(), (
            "src_dst_flow_ratio < 1.0 — division error."
        )
