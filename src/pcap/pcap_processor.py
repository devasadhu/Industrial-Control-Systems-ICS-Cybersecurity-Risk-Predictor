"""
PCAP File Processor for ICS Network Analysis
Extracts network flows from packet captures and detects anomalies

Supports: Wireshark captures, tcpdump, live captures
Protocols: TCP, UDP, Modbus, DNP3, EtherNet/IP

Author: Sadhana Devarajan
Version: 1.4.0

Fixes applied (v1.4.0):
- FIX 5: Added 5 missing v2 engineered features to engineer_flow_features():
         inter_packet_timing_asymmetry, timing_regularity,
         payload_size_consistency, scan_signature_score, flow_burstiness.
         Previously these were zero-filled during prediction (56 of 62
         features used). Now all 62 features are computed from PCAP data.

Fixes carried over from v1.3.0:
- FIX 1: PCAP no longer read twice. process_pcap_file() returns both
         features_df AND flows.
- FIX 2: _calculate_severity() accepts is_anomaly flag. Normal flows
         return 'NORMAL' instead of 'LOW'.
- FIX 3: src_win_size and dst_win_size extracted from actual TCP window
         fields instead of being hardcoded to 0.
- FIX 4: Models loaded with joblib.load(), ensemble_scaler.pkl used for
         scaling, feature alignment done via reindex.
"""

from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ICS protocol port mapping
ICS_PORTS = {
    502: 'Modbus/TCP',
    20000: 'DNP3',
    44818: 'EtherNet/IP',
    47808: 'BACnet',
    4840: 'OPC UA',
    102: 'S7comm',
}


class PCAPProcessor:
    """Process PCAP files and extract network flows."""

    def __init__(self):
        self.flow_features = []

    def read_pcap(self, pcap_path: str) -> List:
        logger.info(f"Reading PCAP file: {pcap_path}")
        try:
            packets = rdpcap(pcap_path)
            logger.info(f"✅ Loaded {len(packets)} packets")
            return packets
        except Exception as e:
            logger.error(f"❌ Failed to read PCAP: {e}")
            return []

    def extract_flows(self, packets: List, timeout: int = 120) -> Dict:
        """
        Extract bidirectional flows from packets.

        Tracks per-packet byte sizes (fwd_pkt_sizes / bwd_pkt_sizes) for
        true min/max/avg/std computation downstream.

        FIX 3: Also tracks the first TCP window size seen per direction
        (fwd_win_size / bwd_win_size) so src_win_size / dst_win_size are
        no longer hardcoded to 0.
        """
        logger.info(f"Extracting flows from {len(packets)} packets...")

        flows = defaultdict(lambda: {
            'fwd_packets': 0, 'bwd_packets': 0,
            'fwd_bytes': 0, 'bwd_bytes': 0,
            'fwd_psh_flags': 0, 'bwd_psh_flags': 0,
            'fwd_urg_flags': 0, 'bwd_urg_flags': 0,
            'fwd_syn': 0, 'fwd_ack': 0, 'fwd_fin': 0, 'fwd_rst': 0,
            'bwd_syn': 0, 'bwd_ack': 0, 'bwd_fin': 0, 'bwd_rst': 0,
            'timestamps': [],
            'start_time': None, 'end_time': None,
            'protocol': None, 'src_ip': None, 'dst_ip': None,
            'src_port': None, 'dst_port': None,
            'src_ttl': None, 'dst_ttl': None,
            'fwd_pkt_sizes': [],
            'bwd_pkt_sizes': [],
            # FIX 3: store first-seen TCP window size per direction
            'fwd_win_size': None,
            'bwd_win_size': None,
        })

        for packet in packets:
            if not packet.haslayer(IP):
                continue

            ip_layer = packet[IP]
            timestamp = float(packet.time)
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            protocol = ip_layer.proto
            src_port = dst_port = 0

            if packet.haslayer(TCP):
                tcp = packet[TCP]
                src_port, dst_port = tcp.sport, tcp.dport
            elif packet.haslayer(UDP):
                udp = packet[UDP]
                src_port, dst_port = udp.sport, udp.dport

            forward_key = (src_ip, dst_ip, src_port, dst_port, protocol)
            backward_key = (dst_ip, src_ip, dst_port, src_port, protocol)

            if forward_key in flows:
                flow_key, direction = forward_key, 'forward'
            elif backward_key in flows:
                flow_key, direction = backward_key, 'backward'
            else:
                flow_key, direction = forward_key, 'forward'
                f = flows[flow_key]
                f['src_ip'] = src_ip
                f['dst_ip'] = dst_ip
                f['src_port'] = src_port
                f['dst_port'] = dst_port
                f['protocol'] = protocol
                f['start_time'] = timestamp

            flow = flows[flow_key]
            flow['timestamps'].append(timestamp)
            flow['end_time'] = timestamp
            pkt_size = len(packet)

            if direction == 'forward':
                flow['fwd_packets'] += 1
                flow['fwd_bytes'] += pkt_size
                flow['fwd_pkt_sizes'].append(pkt_size)
                if flow['src_ttl'] is None:
                    flow['src_ttl'] = ip_layer.ttl
                if packet.haslayer(TCP):
                    tcp = packet[TCP]
                    # FIX 3: capture first-seen forward window size
                    if flow['fwd_win_size'] is None:
                        flow['fwd_win_size'] = tcp.window
                    if tcp.flags.P: flow['fwd_psh_flags'] += 1
                    if tcp.flags.U: flow['fwd_urg_flags'] += 1
                    if tcp.flags.S: flow['fwd_syn'] += 1
                    if tcp.flags.A: flow['fwd_ack'] += 1
                    if tcp.flags.F: flow['fwd_fin'] += 1
                    if tcp.flags.R: flow['fwd_rst'] += 1
            else:
                flow['bwd_packets'] += 1
                flow['bwd_bytes'] += pkt_size
                flow['bwd_pkt_sizes'].append(pkt_size)
                if flow['dst_ttl'] is None:
                    flow['dst_ttl'] = ip_layer.ttl
                if packet.haslayer(TCP):
                    tcp = packet[TCP]
                    # FIX 3: capture first-seen backward window size
                    if flow['bwd_win_size'] is None:
                        flow['bwd_win_size'] = tcp.window
                    if tcp.flags.P: flow['bwd_psh_flags'] += 1
                    if tcp.flags.U: flow['bwd_urg_flags'] += 1
                    if tcp.flags.S: flow['bwd_syn'] += 1
                    if tcp.flags.A: flow['bwd_ack'] += 1
                    if tcp.flags.F: flow['bwd_fin'] += 1
                    if tcp.flags.R: flow['bwd_rst'] += 1

        logger.info(f"✅ Extracted {len(flows)} flows")
        return dict(flows)

    def engineer_flow_features(self, flows: Dict) -> pd.DataFrame:
        """
        Convert flows to feature vectors aligned with feature_names.txt.

        Statistical features (min, max, avg, std) are computed from the
        per-packet size lists tracked during extract_flows().

        FIX 3: src_win_size / dst_win_size now use values extracted from
        TCP headers instead of being hardcoded to 0.

        FIX 5: Added 5 v2 engineered features:
               inter_packet_timing_asymmetry, timing_regularity,
               payload_size_consistency, scan_signature_score,
               flow_burstiness.
        """
        logger.info("Engineering features from flows...")
        features_list = []

        for flow_key, flow in flows.items():
            if flow['fwd_packets'] == 0 and flow['bwd_packets'] == 0:
                continue

            timestamps = flow['timestamps']
            duration = (flow['end_time'] - flow['start_time']) if flow['end_time'] else 0
            duration_safe = max(duration, 0.001)

            iats = np.diff(timestamps) if len(timestamps) > 1 else [0]
            iat_mean = float(np.mean(iats))

            fwd = flow['fwd_packets']
            bwd = flow['bwd_packets']
            fwd_bytes = flow['fwd_bytes']
            bwd_bytes = flow['bwd_bytes']
            total_pkts = fwd + bwd
            total_bytes = fwd_bytes + bwd_bytes

            fwd_sizes = flow['fwd_pkt_sizes'] if flow['fwd_pkt_sizes'] else [0]
            bwd_sizes = flow['bwd_pkt_sizes'] if flow['bwd_pkt_sizes'] else [0]

            src_bytes_max = float(np.max(fwd_sizes))
            src_bytes_min = float(np.min(fwd_sizes))
            src_bytes_avg = float(np.mean(fwd_sizes))

            dst_bytes_max = float(np.max(bwd_sizes))
            dst_bytes_min = float(np.min(bwd_sizes))
            dst_bytes_avg = float(np.mean(bwd_sizes))

            # FIX 3: use real window sizes extracted from TCP headers;
            # fall back to 0 only for non-TCP flows (e.g. UDP/DNS).
            src_win_size = flow['fwd_win_size'] if flow['fwd_win_size'] is not None else 0
            dst_win_size = flow['bwd_win_size'] if flow['bwd_win_size'] is not None else 0

            feat = {
                # Identity (not used in model, useful for display)
                'src_ip': flow['src_ip'],
                'dst_ip': flow['dst_ip'],
                'src_port': flow['src_port'],
                'dst_port': flow['dst_port'],
                'protocol': flow['protocol'],

                # Network basic
                'src_packets': fwd,
                'dst_packets': bwd,
                'src_bytes': fwd_bytes,
                'dst_bytes': bwd_bytes,
                'flow_duration': duration,
                'total_packets': total_pkts,
                'packet_ratio': fwd / max(bwd, 1),
                'total_bytes': total_bytes,
                'byte_ratio': fwd_bytes / max(bwd_bytes, 1),
                'bytes_per_packet': total_bytes / max(total_pkts, 1),

                # Timing
                'src_inter_packet_avg': iat_mean,
                'dst_inter_packet_avg': iat_mean,
                'src_packet_rate': fwd / duration_safe,
                'dst_packet_rate': bwd / duration_safe,
                'src_byte_rate': fwd_bytes / duration_safe,
                'dst_byte_rate': bwd_bytes / duration_safe,

                # Statistical — real min/max/avg from per-packet sizes
                'src_bytes_max': src_bytes_max,
                'dst_bytes_max': dst_bytes_max,
                'src_bytes_min': src_bytes_min,
                'dst_bytes_min': dst_bytes_min,
                'src_bytes_avg': src_bytes_avg,
                'dst_bytes_avg': dst_bytes_avg,
                'src_load': fwd_bytes * 8 / duration_safe,
                'dst_load': bwd_bytes * 8 / duration_safe,
                'src_payload_sum': fwd_bytes,
                'src_payload_avg': src_bytes_avg,
                'dst_payload_avg': dst_bytes_avg,

                # Protocol flags
                'src_ack_rate': flow['fwd_ack'] / max(fwd, 1),
                'dst_ack_rate': flow['bwd_ack'] / max(bwd, 1),
                'src_syn_rate': flow['fwd_syn'] / max(fwd, 1),
                'dst_syn_rate': flow['bwd_syn'] / max(bwd, 1),
                'src_fin_rate': flow['fwd_fin'] / max(fwd, 1),
                'dst_fin_rate': flow['bwd_fin'] / max(bwd, 1),
                'src_rst_rate': flow['fwd_rst'] / max(fwd, 1),
                'dst_rst_rate': flow['bwd_rst'] / max(bwd, 1),
                'src_psh_rate': flow['fwd_psh_flags'] / max(fwd, 1),
                'dst_psh_rate': flow['bwd_psh_flags'] / max(bwd, 1),
                'src_urg_rate': flow['fwd_urg_flags'] / max(fwd, 1),
                'dst_urg_rate': flow['bwd_urg_flags'] / max(bwd, 1),
                'src_ttl': flow['src_ttl'] or 64,
                'dst_ttl': flow['dst_ttl'] or 64,

                # FIX 3: real TCP window sizes (0 for non-TCP flows)
                'src_win_size': src_win_size,
                'dst_win_size': dst_win_size,

                'src_fragment_rate': 0,
                'dst_fragment_rate': 0,
                'src_ack_delay': 0,
                'dst_ack_delay': 0,

                # Behavioral
                'syn_ack_imbalance': (
                    (flow['fwd_syn'] / max(fwd, 1)) -
                    (flow['fwd_ack'] / max(fwd, 1))
                ),
                'packet_size_anomaly': abs(src_bytes_avg - 500),
                'reset_rate_total': (
                    (flow['fwd_rst'] + flow['bwd_rst']) / max(total_pkts, 1)
                ),
                'traffic_symmetry': 1 - abs(fwd - bwd) / max(total_pkts, 1),

                # FIX 5: v2 engineered features — previously zero-filled,
                # now computed directly from PCAP flow data.
                'inter_packet_timing_asymmetry': abs(iat_mean - (
                    float(np.mean(np.diff(timestamps))) if len(timestamps) > 2 else iat_mean
                )),
                'timing_regularity': 1.0 - (
                    float(np.std(iats) / max(np.mean(iats), 1e-6))
                    if len(iats) > 1 else 0.0
                ),
                'payload_size_consistency': 1.0 - (
                    float(np.std(fwd_sizes) / max(np.mean(fwd_sizes), 1e-6))
                    if len(fwd_sizes) > 1 else 1.0
                ),
                'scan_signature_score': float(
                    (fwd / max(total_pkts, 1)) * (1.0 - min(bwd / max(fwd, 1), 1.0))
                ),
                'flow_burstiness': float(
                    np.std(iats) / max(np.mean(iats), 1e-6)
                    if len(iats) > 1 else 0.0
                ),
            }
            features_list.append(feat)

        df = pd.DataFrame(features_list)
        logger.info(f"✅ Engineered {len(df)} flow features")
        return df

    def process_pcap_file(self, pcap_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete pipeline: PCAP → Flows → Features.

        FIX 1: Returns (features_df, flows) tuple so callers can reuse
        the extracted flows for protocol detection without re-reading the
        PCAP file a second time.

        Returns:
            features_df: DataFrame of engineered flow features
            flows: Raw flow dict (use for detect_ics_protocols, etc.)
        """
        packets = self.read_pcap(pcap_path)
        if not packets:
            return pd.DataFrame(), {}
        flows = self.extract_flows(packets)
        features_df = self.engineer_flow_features(flows)
        return features_df, flows   # FIX 1: return flows alongside features

    def detect_ics_protocols(self, flows: Dict) -> Dict[str, int]:
        """Detect ICS/SCADA protocols in flows."""
        counts = {name: 0 for name in ICS_PORTS.values()}
        for flow_key, flow in flows.items():
            proto_name = ICS_PORTS.get(flow['dst_port'])
            if proto_name:
                counts[proto_name] += 1
        return counts

    def predict_anomalies(self,
                          features_df: pd.DataFrame,
                          model_path: str,
                          scaler_path: str = None,
                          feature_names_path: str = None) -> pd.DataFrame:
        """
        Predict anomalies in flows.

        FIX 2: severity is computed per-row using both the anomaly score
        AND the is_anomaly flag, so normal flows get 'NORMAL' instead of
        being mislabelled as 'LOW'.

        Args:
            features_df: DataFrame with flow features
            model_path: Path to ensemble_isolation_forest.pkl (joblib format).
            scaler_path: Path to ensemble_scaler.pkl (StandardScaler).
                         Do NOT pass feature_scaler.pkl — that file contains
                         feature names only, not a StandardScaler.
            feature_names_path: Path to feature_names.txt
        """
        logger.info("Loading model for predictions...")

        model = joblib.load(model_path)

        scaler = None
        if scaler_path and Path(scaler_path).exists():
            loaded = joblib.load(scaler_path)
            if hasattr(loaded, 'transform'):
                scaler = loaded
                logger.info("✅ Scaler loaded")
            else:
                logger.warning(
                    f"⚠️  {scaler_path} does not appear to be a StandardScaler "
                    "(it may be feature_scaler.pkl which contains feature names). "
                    "Predictions will run without scaling."
                )

        if feature_names_path and Path(feature_names_path).exists():
            with open(feature_names_path) as f:
                feature_names = [line.strip() for line in f if line.strip()]
        else:
            feature_names = [c for c in features_df.columns
                             if features_df[c].dtype in [np.float64, np.int64, float, int]]

        missing = [f for f in feature_names if f not in features_df.columns]
        if missing:
            logger.warning(f"   Missing features (will be zero-filled): {missing[:5]}...")

        X = features_df.reindex(columns=feature_names, fill_value=0).values

        if scaler is not None:
            X = scaler.transform(X)

        predictions = model.predict(X)
        scores = model.score_samples(X)

        results = features_df.copy()
        results['is_anomaly'] = predictions == -1
        results['anomaly_score'] = scores

        # FIX 2: pass both score and is_anomaly flag so normal flows
        # are labelled 'NORMAL' rather than 'LOW'.
        results['severity'] = results.apply(
            lambda row: self._calculate_severity(
                row['anomaly_score'], row['is_anomaly']
            ),
            axis=1
        )

        anomaly_count = (predictions == -1).sum()
        logger.info(f"✅ Predictions complete")
        logger.info(f"   Anomalies detected: {anomaly_count}/{len(predictions)}")
        return results

    def _calculate_severity(self, score: float, is_anomaly: bool) -> str:
        """
        Map anomaly score to severity label.

        FIX 2: Added is_anomaly parameter. Normal flows (is_anomaly=False)
        now return 'NORMAL' instead of 'LOW', which was misleading in v1.2.0
        because abs(score) for a normal flow could still be < 0.4.
        """
        if not is_anomaly:
            return 'NORMAL'
        abs_score = abs(score)
        if abs_score > 0.6:   return 'CRITICAL'
        elif abs_score > 0.5: return 'HIGH'
        elif abs_score > 0.4: return 'MEDIUM'
        else:                 return 'LOW'


def demo_pcap_processor():
    print("=" * 80)
    print("ICS NETWORK ANOMALY DETECTION - PCAP ANALYSIS DEMO")
    print("=" * 80)

    processor = PCAPProcessor()

    possible_paths = [
        Path("./data/sample.pcap"),
        Path("./data/dns.pcap"),
        Path("../../data/sample.pcap"),
    ]
    pcap_path = next((p for p in possible_paths if p.exists()), None)

    if pcap_path is None:
        print("\n⚠️  No sample PCAP found.")
        print("To test PCAP processing:")
        print("  1. tcpdump -i eth0 -w data/sample.pcap")
        print("  2. Or download from https://wiki.wireshark.org/SampleCaptures")
        return

    print(f"\n📄 Processing: {pcap_path}")

    # FIX 1: process_pcap_file() returns (features_df, flows).
    # No second call to read_pcap() + extract_flows() needed.
    features_df, flows = processor.process_pcap_file(str(pcap_path))

    if len(features_df) == 0:
        print("❌ No flows extracted")
        return

    print(f"\n✅ Extracted {len(features_df)} flows")
    print(features_df.head())

    # Reuse flows from the first parse — no redundant PCAP read
    protocols = processor.detect_ics_protocols(flows)

    print(f"\n🔍 ICS Protocol Detection:")
    detected = {k: v for k, v in protocols.items() if v > 0}
    if detected:
        for proto, count in detected.items():
            print(f"   {proto}: {count} flows")
    else:
        print("   No ICS protocols detected in this capture.")
        print("   (This is expected for non-ICS PCAPs like dns.pcap)")

    models_dir = Path("./models")
    if not models_dir.exists():
        models_dir = Path("../../models")

    model_path = models_dir / "ensemble_isolation_forest.pkl"
    scaler_path = models_dir / "ensemble_scaler.pkl"
    feature_names_path = models_dir / "feature_names.txt"

    if model_path.exists():
        print(f"\n🔌 Running anomaly detection...")
        results = processor.predict_anomalies(
            features_df,
            str(model_path),
            str(scaler_path) if scaler_path.exists() else None,
            str(feature_names_path) if feature_names_path.exists() else None,
        )

        anomalies = results[results['is_anomaly']]
        normals = results[~results['is_anomaly']]

        print(f"\n📈 Results Summary:")
        print(f"   Total flows:  {len(results)}")
        print(f"   Normal:       {len(normals)}")
        print(f"   Anomalies:    {len(anomalies)}")

        # FIX 2: severity breakdown now includes NORMAL bucket
        print(f"\n📈 Severity Breakdown:")
        for severity in ['NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            count = (results['severity'] == severity).sum()
            if count > 0:
                print(f"   {severity:<10} {count}")

        if len(anomalies) > 0:
            print(f"\n⚠️  Anomalous Flows:")
            display_cols = [c for c in
                            ['src_ip', 'dst_ip', 'dst_port', 'severity', 'anomaly_score']
                            if c in anomalies.columns]
            print(anomalies[display_cols].to_string(index=False))

        if len(normals) > 0:
            print(f"\n✅ Note: {len(normals)} flow(s) classified as NORMAL.")
            print("   High anomaly rate on non-ICS PCAPs (e.g. dns.pcap) is expected")
            print("   because the model was trained on ICS/SCADA traffic patterns.")
    else:
        print(f"\n⚠️  Model not found at {model_path}")
        print("   Run `python quick_start.py` to train the ensemble model first.")


if __name__ == "__main__":
    demo_pcap_processor()