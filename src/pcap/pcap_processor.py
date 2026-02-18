"""
PCAP File Processor for ICS Network Analysis
Extracts network flows from packet captures and detects anomalies

Supports: Wireshark captures, tcpdump, live captures
Protocols: TCP, UDP, Modbus, DNP3, EtherNet/IP

Author: Sadhana Devarajan
Version: 1.0.0
"""

from scapy.all import rdpcap, IP, TCP, UDP, wrpcap, Ether
from scapy.layers.inet import ICMP
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCAPProcessor:
    """Process PCAP files and extract network flows."""
    
    def __init__(self):
        """Initialize PCAP processor."""
        self.flows = defaultdict(lambda: {
            'packets': [],
            'fwd_packets': 0,
            'bwd_packets': 0,
            'fwd_bytes': 0,
            'bwd_bytes': 0,
            'timestamps': [],
            'flags': [],
            'start_time': None,
            'end_time': None
        })
        
        self.flow_features = []
    
    def read_pcap(self, pcap_path: str) -> List:
        """
        Read PCAP file and return packets.
        
        Args:
            pcap_path: Path to PCAP file
            
        Returns:
            List of packets
        """
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
        
        Args:
            packets: List of packets from rdpcap
            timeout: Flow timeout in seconds
            
        Returns:
            Dictionary of flows
        """
        logger.info(f"Extracting flows from {len(packets)} packets...")
        
        flows = defaultdict(lambda: {
            'packets': [],
            'fwd_packets': 0,
            'bwd_packets': 0,
            'fwd_bytes': 0,
            'bwd_bytes': 0,
            'fwd_psh_flags': 0,
            'bwd_psh_flags': 0,
            'fwd_urg_flags': 0,
            'bwd_urg_flags': 0,
            'timestamps': [],
            'start_time': None,
            'end_time': None,
            'protocol': None,
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None
        })
        
        for packet in packets:
            if not packet.haslayer(IP):
                continue
            
            ip_layer = packet[IP]
            timestamp = float(packet.time)
            
            # Identify flow (5-tuple)
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            protocol = ip_layer.proto
            
            # Get ports
            src_port = 0
            dst_port = 0
            
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                src_port = tcp_layer.sport
                dst_port = tcp_layer.dport
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                src_port = udp_layer.sport
                dst_port = udp_layer.dport
            
            # Create flow key (bidirectional)
            forward_key = (src_ip, dst_ip, src_port, dst_port, protocol)
            backward_key = (dst_ip, src_ip, dst_port, src_port, protocol)
            
            # Determine flow direction
            if forward_key in flows:
                flow_key = forward_key
                direction = 'forward'
            elif backward_key in flows:
                flow_key = backward_key
                direction = 'backward'
            else:
                # New flow
                flow_key = forward_key
                direction = 'forward'
                flows[flow_key]['src_ip'] = src_ip
                flows[flow_key]['dst_ip'] = dst_ip
                flows[flow_key]['src_port'] = src_port
                flows[flow_key]['dst_port'] = dst_port
                flows[flow_key]['protocol'] = protocol
                flows[flow_key]['start_time'] = timestamp
            
            # Update flow
            flow = flows[flow_key]
            flow['packets'].append(packet)
            flow['timestamps'].append(timestamp)
            flow['end_time'] = timestamp
            
            packet_size = len(packet)
            
            if direction == 'forward':
                flow['fwd_packets'] += 1
                flow['fwd_bytes'] += packet_size
                
                # TCP flags
                if packet.haslayer(TCP):
                    tcp = packet[TCP]
                    if tcp.flags.P:  # PSH flag
                        flow['fwd_psh_flags'] += 1
                    if tcp.flags.U:  # URG flag
                        flow['fwd_urg_flags'] += 1
            else:
                flow['bwd_packets'] += 1
                flow['bwd_bytes'] += packet_size
                
                if packet.haslayer(TCP):
                    tcp = packet[TCP]
                    if tcp.flags.P:
                        flow['bwd_psh_flags'] += 1
                    if tcp.flags.U:
                        flow['bwd_urg_flags'] += 1
        
        logger.info(f"✅ Extracted {len(flows)} flows")
        return dict(flows)
    
    def engineer_flow_features(self, flows: Dict) -> pd.DataFrame:
        """
        Convert flows to feature vectors.
        
        Args:
            flows: Dictionary of flows
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features from flows...")
        
        features_list = []
        
        for flow_key, flow in flows.items():
            if flow['fwd_packets'] == 0 and flow['bwd_packets'] == 0:
                continue
            
            # Calculate timing features
            timestamps = flow['timestamps']
            duration = flow['end_time'] - flow['start_time'] if flow['end_time'] else 0
            
            # Inter-arrival times
            if len(timestamps) > 1:
                iats = np.diff(timestamps)
                iat_mean = np.mean(iats)
                iat_std = np.std(iats)
                iat_max = np.max(iats)
                iat_min = np.min(iats)
            else:
                iat_mean = iat_std = iat_max = iat_min = 0
            
            # Packet rate
            packet_rate = (flow['fwd_packets'] + flow['bwd_packets']) / max(duration, 0.001)
            
            # Byte rate
            byte_rate = (flow['fwd_bytes'] + flow['bwd_bytes']) / max(duration, 0.001)
            
            # Create feature vector
            features = {
                # Basic info
                'src_ip': flow['src_ip'],
                'dst_ip': flow['dst_ip'],
                'src_port': flow['src_port'],
                'dst_port': flow['dst_port'],
                'protocol': flow['protocol'],
                
                # Packet counts
                'total_fwd_packets': flow['fwd_packets'],
                'total_bwd_packets': flow['bwd_packets'],
                'total_packets': flow['fwd_packets'] + flow['bwd_packets'],
                
                # Byte counts
                'total_length_fwd_packets': flow['fwd_bytes'],
                'total_length_bwd_packets': flow['bwd_bytes'],
                'total_bytes': flow['fwd_bytes'] + flow['bwd_bytes'],
                
                # Timing
                'flow_duration': duration,
                'flow_iat_mean': iat_mean,
                'flow_iat_std': iat_std,
                'flow_iat_max': iat_max,
                'flow_iat_min': iat_min,
                
                # Rates
                'packet_rate': packet_rate,
                'byte_rate': byte_rate,
                
                # TCP flags
                'fwd_psh_flags': flow['fwd_psh_flags'],
                'bwd_psh_flags': flow['bwd_psh_flags'],
                'fwd_urg_flags': flow['fwd_urg_flags'],
                'bwd_urg_flags': flow['bwd_urg_flags'],
                
                # Ratios
                'fwd_bwd_packet_ratio': flow['fwd_packets'] / max(flow['bwd_packets'], 1),
                'fwd_bwd_byte_ratio': flow['fwd_bytes'] / max(flow['bwd_bytes'], 1),
            }
            
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        logger.info(f"✅ Engineered {len(df)} flow features")
        
        return df
    
    def process_pcap_file(self, pcap_path: str) -> pd.DataFrame:
        """
        Complete processing pipeline: PCAP → Flows → Features.
        
        Args:
            pcap_path: Path to PCAP file
            
        Returns:
            DataFrame with flow features
        """
        # Read packets
        packets = self.read_pcap(pcap_path)
        if not packets:
            return pd.DataFrame()
        
        # Extract flows
        flows = self.extract_flows(packets)
        
        # Engineer features
        features_df = self.engineer_flow_features(flows)
        
        return features_df
    
    def detect_ics_protocols(self, flows: Dict) -> Dict[str, int]:
        """
        Detect ICS/SCADA protocols in flows.
        
        Returns:
            Dictionary with protocol counts
        """
        protocol_counts = {
            'Modbus/TCP': 0,      # Port 502
            'DNP3': 0,            # Port 20000
            'EtherNet/IP': 0,     # Port 44818
            'BACnet': 0,          # Port 47808
            'OPC UA': 0,          # Port 4840
            'S7comm': 0,          # Port 102
        }
        
        for flow_key, flow in flows.items():
            dst_port = flow['dst_port']
            
            if dst_port == 502:
                protocol_counts['Modbus/TCP'] += 1
            elif dst_port == 20000:
                protocol_counts['DNP3'] += 1
            elif dst_port == 44818:
                protocol_counts['EtherNet/IP'] += 1
            elif dst_port == 47808:
                protocol_counts['BACnet'] += 1
            elif dst_port == 4840:
                protocol_counts['OPC UA'] += 1
            elif dst_port == 102:
                protocol_counts['S7comm'] += 1
        
        return protocol_counts
    
    def predict_anomalies(self, 
                         features_df: pd.DataFrame,
                         model_path: str,
                         scaler_path: str) -> pd.DataFrame:
        """
        Predict anomalies in flows.
        
        Args:
            features_df: DataFrame with flow features
            model_path: Path to trained model
            scaler_path: Path to feature scaler
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Loading model for predictions...")
        
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Select numeric features
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_cols].fillna(0)
        
        # Ensure we have 52 features (pad or truncate)
        if X.shape[1] < 52:
            padding = np.zeros((len(X), 52 - X.shape[1]))
            X = np.concatenate([X.values, padding], axis=1)
        elif X.shape[1] > 52:
            X = X.iloc[:, :52].values
        else:
            X = X.values
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        scores = model.score_samples(X_scaled)
        
        # Add predictions to dataframe
        results = features_df.copy()
        results['is_anomaly'] = predictions == -1
        results['anomaly_score'] = scores
        results['severity'] = results['anomaly_score'].apply(self._calculate_severity)
        
        logger.info(f"✅ Predictions complete")
        logger.info(f"   Anomalies detected: {(predictions == -1).sum()}/{len(predictions)}")
        
        return results
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate severity from anomaly score."""
        abs_score = abs(score)
        
        if abs_score > 0.6:
            return 'CRITICAL'
        elif abs_score > 0.5:
            return 'HIGH'
        elif abs_score > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'


def demo_pcap_processor():
    """Demo the PCAP processor."""
    print("="*80)
    print("ICS NETWORK ANOMALY DETECTION - PCAP ANALYSIS DEMO")
    print("="*80)
    
    processor = PCAPProcessor()
    
    # Check multiple possible PCAP locations
    possible_paths = [
        Path("./data/sample.pcap"),
        Path("./data/dns.pcap"),
        Path("../../data/sample.pcap"),
        Path("../../data/dns.pcap")
    ]
    
    pcap_path = None
    for path in possible_paths:
        if path.exists():
            pcap_path = path
            break
    
    if pcap_path is None:
        pcap_path = Path("./data/sample.pcap")
    
    if not pcap_path.exists():
        print(f"\n⚠️  No sample PCAP found at {pcap_path}")
        print("To test PCAP processing:")
        print("1. Capture network traffic: tcpdump -i eth0 -w sample.pcap")
        print("2. Or download sample: https://wiki.wireshark.org/SampleCaptures")
        return
    
    # Process PCAP
    print(f"\n📦 Processing: {pcap_path}")
    features_df = processor.process_pcap_file(str(pcap_path))
    
    if len(features_df) == 0:
        print("❌ No flows extracted")
        return
    
    print(f"\n✅ Extracted {len(features_df)} flows")
    print(f"\nSample features:")
    print(features_df.head())
    
    # Detect ICS protocols
    packets = processor.read_pcap(str(pcap_path))
    flows = processor.extract_flows(packets)
    protocols = processor.detect_ics_protocols(flows)
    
    print(f"\n🏭 ICS Protocol Detection:")
    for proto, count in protocols.items():
        if count > 0:
            print(f"   {proto}: {count} flows")
    
    # Predict anomalies
    models_dir = Path("../../models")
    if (models_dir / "isolation_forest_ics_detector.pkl").exists():
        print(f"\n🔍 Running anomaly detection...")
        results = processor.predict_anomalies(
            features_df,
            str(models_dir / "isolation_forest_ics_detector.pkl"),
            str(models_dir / "feature_scaler.pkl")
        )
        
        anomalies = results[results['is_anomaly']]
        print(f"\n⚠️  Anomalies detected: {len(anomalies)}")
        
        if len(anomalies) > 0:
            print(f"\nTop 5 anomalies:")
            print(anomalies[['src_ip', 'dst_ip', 'dst_port', 'severity', 'anomaly_score']].head())


if __name__ == "__main__":
    demo_pcap_processor()