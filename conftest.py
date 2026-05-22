"""
conftest.py
-----------
Shared pytest fixtures.

Design principles:
- No real model files, no real dataset required — CI runs everywhere.
- Synthetic data mirrors the actual ICSSIM feature schema (62 features after
  session aggregation) so tests exercise real code paths.
- Fixtures are scoped at 'session' level where possible to avoid redundant
  training across test files.
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── make src/ importable without installing the package ──────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ── Feature name constants ────────────────────────────────────────────────────

BASE_51_FEATURES = [
    # network_basic (10)
    "src_packets", "dst_packets", "src_bytes", "dst_bytes", "flow_duration",
    "total_packets", "packet_ratio", "total_bytes", "byte_ratio", "bytes_per_packet",
    # timing (6)
    "src_inter_packet_avg", "dst_inter_packet_avg", "src_packet_rate",
    "dst_packet_rate", "src_byte_rate", "dst_byte_rate",
    # statistical (11)
    "src_bytes_max", "dst_bytes_max", "src_bytes_min", "dst_bytes_min",
    "src_bytes_avg", "dst_bytes_avg", "src_load", "dst_load",
    "src_payload_sum", "src_payload_avg", "dst_payload_avg",
    # protocol (20)
    "src_ack_rate", "dst_ack_rate", "src_syn_rate", "dst_syn_rate",
    "src_fin_rate", "dst_fin_rate", "src_rst_rate", "dst_rst_rate",
    "src_psh_rate", "dst_psh_rate", "src_urg_rate", "dst_urg_rate",
    "src_ttl", "dst_ttl", "src_win_size", "dst_win_size",
    "src_fragment_rate", "dst_fragment_rate", "src_ack_delay", "dst_ack_delay",
    # behavioral (4)
    "syn_ack_imbalance", "packet_size_anomaly", "reset_rate_total", "traffic_symmetry",
]

V2_EXTRA_FEATURES = [
    "inter_packet_timing_asymmetry", "timing_regularity",
    "payload_size_consistency", "scan_signature_score", "flow_burstiness",
]

SESSION_FEATURES = [
    "src_unique_dst_count", "src_flow_count", "src_inter_flow_interval",
    "src_inter_flow_variance", "src_dst_flow_ratio", "src_payload_entropy",
]

ALL_62_FEATURES = BASE_51_FEATURES + V2_EXTRA_FEATURES + SESSION_FEATURES

# ── Marker registration ───────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_models: mark test as requiring trained model artifacts in models/ "
        "(skipped in CI where models/ is git-ignored)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running (skip with -m 'not slow')",
    )


def pytest_collection_modifyitems(config, items):
    """
    Auto-skip tests marked @pytest.mark.requires_models when the real model
    artifacts are absent. Checks for ensemble_isolation_forest.pkl as the
    canonical sentinel — if it's missing, all model-dependent tests are skipped
    with a clear message rather than failing with a FileNotFoundError.
    """
    models_dir = Path(__file__).parent / "models"
    sentinel = models_dir / "ensemble_isolation_forest.pkl"
    models_present = sentinel.exists()

    if not models_present:
        skip_reason = pytest.mark.skip(
            reason=(
                "Real model artifacts not found (models/ensemble_isolation_forest.pkl missing). "
                "Run quick_start.py locally to train, or remove @pytest.mark.requires_models "
                "if this test should use the synthetic fixture instead."
            )
        )
        for item in items:
            if item.get_closest_marker("requires_models"):
                item.add_marker(skip_reason)


# ── Synthetic data generators ─────────────────────────────────────────────────

def _make_normal_row(rng: np.random.Generator) -> dict:
    """Generate one realistic-looking normal ICS flow."""
    pkts = rng.integers(5, 200)
    bpkts = rng.integers(1, pkts)
    dur = rng.uniform(0.1, 5.0)
    sbytes = int(pkts * rng.integers(40, 70))
    dbytes = int(bpkts * rng.integers(40, 70))
    return {
        "src_packets": pkts, "dst_packets": bpkts,
        "src_bytes": sbytes, "dst_bytes": dbytes,
        "flow_duration": dur,
        "total_packets": pkts + bpkts,
        "packet_ratio": pkts / max(bpkts, 1),
        "total_bytes": sbytes + dbytes,
        "byte_ratio": sbytes / max(dbytes, 1),
        "bytes_per_packet": (sbytes + dbytes) / max(pkts + bpkts, 1),
        "src_inter_packet_avg": rng.uniform(0.005, 0.1),
        "dst_inter_packet_avg": rng.uniform(0.005, 0.1),
        "src_packet_rate": pkts / dur,
        "dst_packet_rate": bpkts / dur,
        "src_byte_rate": sbytes / dur,
        "dst_byte_rate": dbytes / dur,
        "src_bytes_max": sbytes, "dst_bytes_max": dbytes,
        "src_bytes_min": 40, "dst_bytes_min": 40,
        "src_bytes_avg": sbytes / max(pkts, 1),
        "dst_bytes_avg": dbytes / max(bpkts, 1),
        "src_load": sbytes * 8 / dur, "dst_load": dbytes * 8 / dur,
        "src_payload_sum": sbytes,
        "src_payload_avg": sbytes / max(pkts, 1),
        "dst_payload_avg": dbytes / max(bpkts, 1),
        "src_ack_rate": rng.uniform(0.7, 1.0),
        "dst_ack_rate": rng.uniform(0.7, 1.0),
        "src_syn_rate": rng.uniform(0.0, 0.05),
        "dst_syn_rate": rng.uniform(0.0, 0.05),
        "src_fin_rate": rng.uniform(0.0, 0.02),
        "dst_fin_rate": rng.uniform(0.0, 0.02),
        "src_rst_rate": 0.0, "dst_rst_rate": 0.0,
        "src_psh_rate": rng.uniform(0.0, 0.2),
        "dst_psh_rate": rng.uniform(0.0, 0.2),
        "src_urg_rate": 0.0, "dst_urg_rate": 0.0,
        "src_ttl": 64.0, "dst_ttl": 64.0,
        "src_win_size": 8192.0, "dst_win_size": 8192.0,
        "src_fragment_rate": 0.0, "dst_fragment_rate": 0.0,
        "src_ack_delay": 0.0, "dst_ack_delay": 0.0,
        "syn_ack_imbalance": rng.uniform(-0.1, 0.1),
        "packet_size_anomaly": rng.uniform(0, 50),
        "reset_rate_total": 0.0,
        "traffic_symmetry": rng.uniform(0.6, 1.0),
        "inter_packet_timing_asymmetry": rng.uniform(0, 0.1),
        "timing_regularity": rng.uniform(0.5, 1.0),
        "payload_size_consistency": rng.uniform(0.5, 1.0),
        "scan_signature_score": rng.uniform(0, 0.1),
        "flow_burstiness": rng.uniform(0, 0.2),
        # session: normal = high inter-flow variance (bursty, event-driven)
        "src_unique_dst_count": float(rng.integers(1, 4)),
        "src_flow_count": float(rng.integers(1, 10)),
        "src_inter_flow_interval": rng.uniform(0.5, 5.0),
        "src_inter_flow_variance": rng.uniform(0.5, 10.0),
        "src_dst_flow_ratio": rng.uniform(1.0, 3.0),
        "src_payload_entropy": rng.uniform(2.0, 3.5),
    }


def _make_attack_row(rng: np.random.Generator) -> dict:
    """Generate one obvious attack flow (replay-like: near-zero inter-flow variance)."""
    pkts = rng.integers(500, 3000)
    dur = rng.uniform(1.0, 10.0)
    sbytes = pkts * 60
    dbytes = pkts * 60  # symmetric — replay mirrors src exactly
    return {
        "src_packets": pkts, "dst_packets": pkts,
        "src_bytes": sbytes, "dst_bytes": dbytes,
        "flow_duration": dur,
        "total_packets": pkts * 2,
        "packet_ratio": 1.0,
        "total_bytes": sbytes + dbytes,
        "byte_ratio": 1.0,
        "bytes_per_packet": 60.0,
        "src_inter_packet_avg": 0.0005,
        "dst_inter_packet_avg": 0.0005,
        "src_packet_rate": pkts / dur,
        "dst_packet_rate": pkts / dur,
        "src_byte_rate": sbytes / dur,
        "dst_byte_rate": dbytes / dur,
        "src_bytes_max": sbytes, "dst_bytes_max": dbytes,
        "src_bytes_min": 60, "dst_bytes_min": 60,
        "src_bytes_avg": 60.0, "dst_bytes_avg": 60.0,
        "src_load": sbytes * 8 / dur, "dst_load": dbytes * 8 / dur,
        "src_payload_sum": sbytes,
        "src_payload_avg": 60.0, "dst_payload_avg": 60.0,
        "src_ack_rate": 1.0, "dst_ack_rate": 1.0,
        "src_syn_rate": 0.6, "dst_syn_rate": 0.0,
        "src_fin_rate": 0.0, "dst_fin_rate": 0.0,
        "src_rst_rate": 0.1, "dst_rst_rate": 0.0,
        "src_psh_rate": 0.8, "dst_psh_rate": 0.0,
        "src_urg_rate": 0.0, "dst_urg_rate": 0.0,
        "src_ttl": 64.0, "dst_ttl": 64.0,
        "src_win_size": 8192.0, "dst_win_size": 8192.0,
        "src_fragment_rate": 0.0, "dst_fragment_rate": 0.0,
        "src_ack_delay": 0.0, "dst_ack_delay": 0.0,
        "syn_ack_imbalance": 0.6,
        "packet_size_anomaly": 0.0,
        "reset_rate_total": 0.05,
        "traffic_symmetry": 0.99,
        "inter_packet_timing_asymmetry": 0.0,
        "timing_regularity": 0.99,
        "payload_size_consistency": 0.99,
        "scan_signature_score": 0.8,
        "flow_burstiness": 0.9,
        # session: attack = near-zero variance (mechanically regular timing)
        "src_unique_dst_count": float(rng.integers(1, 3)),
        "src_flow_count": float(rng.integers(20, 60)),
        "src_inter_flow_interval": rng.uniform(0.01, 0.1),
        "src_inter_flow_variance": rng.uniform(0.0, 0.01),
        "src_dst_flow_ratio": rng.uniform(10.0, 40.0),
        "src_payload_entropy": rng.uniform(0.0, 0.5),
    }


def make_synthetic_dataset(
    n_normal: int = 300,
    n_attack: int = 200,
    seed: int = 42,
    feature_list: list = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Return (X_df, y) where X_df columns match feature_list
    (defaults to ALL_62_FEATURES) and y is binary 0/1.
    """
    if feature_list is None:
        feature_list = ALL_62_FEATURES
    rng = np.random.default_rng(seed)
    rows = (
        [_make_normal_row(rng) for _ in range(n_normal)]
        + [_make_attack_row(rng) for _ in range(n_attack)]
    )
    df = pd.DataFrame(rows)[feature_list].fillna(0.0)
    y = np.array([0] * n_normal + [1] * n_attack)
    return df, y


# ── Session-scoped fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_62():
    """62-feature dataset — 300 normal + 200 attack rows."""
    return make_synthetic_dataset(n_normal=300, n_attack=200, seed=42)


@pytest.fixture(scope="session")
def synthetic_51():
    """51-feature dataset — base + v2 features only, no session features."""
    return make_synthetic_dataset(
        n_normal=300, n_attack=200, seed=42, feature_list=BASE_51_FEATURES
    )


@pytest.fixture(scope="session")
def feature_names_62():
    return list(ALL_62_FEATURES)


@pytest.fixture(scope="session")
def trained_ensemble(synthetic_62, tmp_path_factory):
    """
    Train a tiny EnsembleICSDetector on synthetic 62-feature data once per
    session. Returns (detector, save_dir).

    Used by:
      - test_ensemble_detector.py  — predict(), confidence shape, threshold
      - test_shap_explainer.py     — loads IF from save_dir
      - test_attack_patterns.py    — needs a trained detector for integration path

    NOT marked requires_models — this fixture trains its own model from scratch
    and does not touch models/ on disk.
    """
    from models.ensemble_detector import EnsembleICSDetector

    X_df, y = synthetic_62
    save_dir = tmp_path_factory.mktemp("models")

    detector = EnsembleICSDetector(random_seed=42)
    detector.feature_names = list(X_df.columns)
    detector.train(X_df.values, y)
    detector.save(str(save_dir))

    return detector, save_dir


@pytest.fixture(scope="session")
def session_raw_df():
    """
    Minimal raw DataFrame that mirrors the ICSSIM columns consumed by
    ICSFeatureEngineer.compute_session_features():
      sIPs, rIPs, start, sPayloadAvg, duration, sPackets, sBytesAvg

    500 rows, 5 source IPs, timestamps spread over 300 seconds.
    Designed so 60-second windows produce non-trivial session feature values.
    """
    rng = np.random.default_rng(99)
    n = 500
    src_ips = rng.choice(
        ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5"], n
    )
    dst_ips = rng.choice(["192.168.1.1", "192.168.1.2", "192.168.1.3"], n)
    starts = np.sort(rng.uniform(0, 300, n))
    payloads = rng.uniform(40, 80, n)
    return pd.DataFrame(
        {
            "sIPs": src_ips,
            "rIPs": dst_ips,
            "start": starts,
            "sPayloadAvg": payloads,
            "duration": rng.uniform(0.1, 2.0, n),
            "sPackets": rng.integers(1, 100, n),
            "sBytesAvg": payloads,
        }
    )


@pytest.fixture(scope="session")
def sample_flows_df(synthetic_62):
    """
    Convenience fixture: just the feature DataFrame (no labels).
    Used by test_attack_patterns.py and test_shap_explainer.py.
    """
    X_df, _ = synthetic_62
    return X_df.copy()
