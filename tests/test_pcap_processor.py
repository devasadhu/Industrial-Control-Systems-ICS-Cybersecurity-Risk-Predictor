"""
test_pcap_processor.py
-----------------------
Tests for src/pcap/pcap_processor.py

What we verify:
1. extract_flows() returns a dict keyed by (src_ip, dst_ip, sport, dport, proto)
2. engineer_flow_features() produces a DataFrame with expected columns
3. process_pcap_file() returns a (DataFrame, dict) tuple — not a single value
4. _calculate_severity() maps is_anomaly=False → 'NORMAL'
5. _calculate_severity() maps is_anomaly=True with various scores → correct labels
6. detect_ics_protocols() returns counts for known ICS port names
7. Engineer output has no NaN / Inf values
8. engineer_flow_features() handles an empty flows dict gracefully
9. src_win_size and dst_win_size are numeric (FIX 3 — no longer hardcoded 0)
10. process_pcap_file() on a real pcap (data/dns.pcap) if it exists
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pcap.pcap_processor import PCAPProcessor, ICS_PORTS


# ── Synthetic packet / flow helpers ──────────────────────────────────────────

def _make_synthetic_flow(
    src_ip="10.0.0.1", dst_ip="192.168.1.1",
    src_port=1234, dst_port=502,
    n_fwd=20, n_bwd=15,
    fwd_pkt_size=60, bwd_pkt_size=60,
    start_time=0.0,
) -> dict:
    """Return one raw flow dict as extract_flows() would produce."""
    iats = [start_time + i * 0.01 for i in range(n_fwd + n_bwd)]
    return {
        "fwd_packets": n_fwd, "bwd_packets": n_bwd,
        "fwd_bytes": n_fwd * fwd_pkt_size,
        "bwd_bytes": n_bwd * bwd_pkt_size,
        "fwd_psh_flags": 2, "bwd_psh_flags": 1,
        "fwd_urg_flags": 0, "bwd_urg_flags": 0,
        "fwd_syn": 1, "fwd_ack": n_fwd - 1, "fwd_fin": 1, "fwd_rst": 0,
        "bwd_syn": 0, "bwd_ack": n_bwd, "bwd_fin": 1, "bwd_rst": 0,
        "timestamps": iats,
        "start_time": start_time, "end_time": start_time + 0.01 * (n_fwd + n_bwd - 1),
        "protocol": 6,  # TCP
        "src_ip": src_ip, "dst_ip": dst_ip,
        "src_port": src_port, "dst_port": dst_port,
        "src_ttl": 64, "dst_ttl": 64,
        "fwd_pkt_sizes": [fwd_pkt_size] * n_fwd,
        "bwd_pkt_sizes": [bwd_pkt_size] * n_bwd,
        "fwd_win_size": 8192,   # FIX 3: real TCP window
        "bwd_win_size": 8192,
    }


def _make_flow_dict(n_flows: int = 5) -> dict:
    """Return a dict of n_flows synthetic flows."""
    flows = {}
    for i in range(n_flows):
        key = (f"10.0.0.{i+1}", "192.168.1.1", 1024 + i, 502, 6)
        flows[key] = _make_synthetic_flow(
            src_ip=f"10.0.0.{i+1}",
            src_port=1024 + i,
        )
    return flows


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPCAPProcessor:

    @pytest.fixture(autouse=True)
    def proc(self):
        self.proc = PCAPProcessor()

    # ── engineer_flow_features ────────────────────────────────────────────────

    def test_engineer_features_returns_dataframe(self):
        flows = _make_flow_dict(3)
        df = self.proc.engineer_flow_features(flows)
        assert isinstance(df, pd.DataFrame)

    def test_engineer_features_row_count(self):
        """Row count must equal number of flows."""
        n = 5
        flows = _make_flow_dict(n)
        df = self.proc.engineer_flow_features(flows)
        assert len(df) == n, f"Expected {n} rows, got {len(df)}"

    def test_engineer_features_expected_columns(self):
        """Output must contain the core feature columns used by the model."""
        required = [
            "src_packets", "dst_packets", "src_bytes", "dst_bytes",
            "flow_duration", "total_packets", "packet_ratio",
            "src_packet_rate", "src_inter_packet_avg",
            "syn_ack_imbalance", "traffic_symmetry",
            "src_psh_rate", "src_syn_rate",
        ]
        flows = _make_flow_dict(3)
        df = self.proc.engineer_flow_features(flows)
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_engineer_features_no_nan(self):
        """Feature matrix must not contain NaN or Inf."""
        flows = _make_flow_dict(10)
        df = self.proc.engineer_flow_features(flows)
        numeric = df.select_dtypes(include=[np.number])
        assert not numeric.isnull().any().any(), "NaN in engineered features"
        assert not np.isinf(numeric.values).any(), "Inf in engineered features"

    def test_win_size_is_numeric_not_hardcoded_zero(self):
        """
        src_win_size and dst_win_size must use the actual TCP window from
        the flow (FIX 3). The old bug hardcoded them to 0 for all flows.
        """
        flows = _make_flow_dict(3)
        df = self.proc.engineer_flow_features(flows)
        assert "src_win_size" in df.columns
        assert "dst_win_size" in df.columns
        # Our synthetic flows have fwd_win_size=8192 — verify it's used
        assert (df["src_win_size"] == 8192).all(), (
            "src_win_size should be 8192 for TCP flows with window=8192. "
            "FIX 3 (real TCP window extraction) may be broken."
        )

    def test_engineer_empty_flows(self):
        """engineer_flow_features({}) must return an empty DataFrame, not crash."""
        df = self.proc.engineer_flow_features({})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    # ── _calculate_severity ───────────────────────────────────────────────────

    def test_severity_normal_flow(self):
        """is_anomaly=False must always return 'NORMAL'."""
        for score in [-0.8, -0.5, -0.3, 0.0, 0.1]:
            result = self.proc._calculate_severity(score, is_anomaly=False)
            assert result == "NORMAL", (
                f"Normal flow (score={score}) got severity '{result}', expected 'NORMAL'"
            )

    def test_severity_critical(self):
        """is_anomaly=True, abs(score) > 0.6 → CRITICAL."""
        assert self.proc._calculate_severity(-0.7, is_anomaly=True) == "CRITICAL"
        assert self.proc._calculate_severity(-0.9, is_anomaly=True) == "CRITICAL"

    def test_severity_high(self):
        """is_anomaly=True, 0.5 < abs(score) <= 0.6 → HIGH."""
        assert self.proc._calculate_severity(-0.55, is_anomaly=True) == "HIGH"

    def test_severity_medium(self):
        """is_anomaly=True, 0.4 < abs(score) <= 0.5 → MEDIUM."""
        assert self.proc._calculate_severity(-0.45, is_anomaly=True) == "MEDIUM"

    def test_severity_low(self):
        """is_anomaly=True, abs(score) <= 0.4 → LOW."""
        assert self.proc._calculate_severity(-0.2, is_anomaly=True) == "LOW"

    # ── detect_ics_protocols ──────────────────────────────────────────────────

    def test_detect_ics_protocols_counts_modbus(self):
        """Flows to port 502 must be counted as Modbus/TCP."""
        flows = {
            ("10.0.0.1", "192.168.1.1", 1024, 502, 6): _make_synthetic_flow(dst_port=502),
            ("10.0.0.2", "192.168.1.1", 1025, 502, 6): _make_synthetic_flow(dst_port=502),
        }
        counts = self.proc.detect_ics_protocols(flows)
        assert counts["Modbus/TCP"] == 2, (
            f"Expected 2 Modbus/TCP flows, got {counts['Modbus/TCP']}"
        )

    def test_detect_ics_protocols_non_ics_port(self):
        """Flows to non-ICS ports must not be counted."""
        flows = {
            ("10.0.0.1", "192.168.1.1", 1234, 53, 17): _make_synthetic_flow(dst_port=53),
        }
        counts = self.proc.detect_ics_protocols(flows)
        assert all(v == 0 for v in counts.values()), (
            f"DNS port 53 should not be counted, but got: {counts}"
        )

    def test_detect_ics_protocols_all_known_ports(self):
        """detect_ics_protocols() must return a key for every ICS_PORTS entry."""
        flows = _make_flow_dict(1)
        counts = self.proc.detect_ics_protocols(flows)
        for proto_name in ICS_PORTS.values():
            assert proto_name in counts, (
                f"Protocol '{proto_name}' missing from detect_ics_protocols() output"
            )

    # ── process_pcap_file ─────────────────────────────────────────────────────

    def test_process_pcap_returns_tuple(self):
        """
        process_pcap_file() must return a 2-tuple (DataFrame, dict).
        FIX 1: previously it returned only a DataFrame.

        We mock rdpcap to avoid needing a real PCAP file in CI.
        """
        with patch("pcap.pcap_processor.rdpcap") as mock_rdpcap:
            mock_rdpcap.return_value = []  # empty packet list
            result = self.proc.process_pcap_file("fake.pcap")
        assert isinstance(result, tuple), (
            "process_pcap_file() must return a tuple (features_df, flows)"
        )
        assert len(result) == 2, (
            f"Expected 2-tuple, got {len(result)}-tuple"
        )

    def test_process_pcap_empty_returns_empty_df_and_dict(self):
        """An empty PCAP should return (empty DataFrame, empty dict)."""
        with patch("pcap.pcap_processor.rdpcap") as mock_rdpcap:
            mock_rdpcap.return_value = []
            df, flows = self.proc.process_pcap_file("fake.pcap")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert isinstance(flows, dict)

    @pytest.mark.skipif(
        not Path("data/dns.pcap").exists(),
        reason="data/dns.pcap not present — skipping real PCAP test"
    )
    def test_process_real_dns_pcap(self):
        """Smoke test on the actual data/dns.pcap included in the repo."""
        df, flows = self.proc.process_pcap_file("data/dns.pcap")
        assert isinstance(df, pd.DataFrame)
        assert isinstance(flows, dict)
        assert len(df) > 0, "dns.pcap should produce at least 1 flow"
        assert len(flows) == len(df), "flows dict and features_df must have same count"
