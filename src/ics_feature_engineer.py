"""
ICS Feature Engineer
Extracts and creates ML features from real ICSSIM network traffic data

Author: Sadhana Devarajan
Date: December 2025
Purpose: Feature engineering for ICS anomaly detection (Schneider Electric / Yokogawa)

Features Created:
1. Network features (packets, bytes, duration)
2. Protocol features (TCP flags, ports)
3. Timing features (inter-packet intervals)
4. Statistical features (mean, std, entropy)
5. Behavioral features (rate metrics, patterns)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ICSFeatureEngineer:
    """
    Feature engineering for ICS network traffic data.
    
    Transforms raw network flows into ML-ready features for
    anomaly detection in Industrial Control Systems.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        np.random.seed(random_seed)
        
        # Feature groups for organization
        self.feature_groups = {
            'network_basic': [],
            'network_advanced': [],
            'timing': [],
            'statistical': [],
            'behavioral': [],
            'protocol': []
        }
    
    def create_basic_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic network traffic features.
        
        Features:
        - Packet counts (source/destination)
        - Byte sums (source/destination)
        - Duration
        - Packet/byte ratios
        """
        logger.info("Creating basic network features...")
        
        features = pd.DataFrame(index=df.index)
        
        # Direct features (if available)
        if 'sPackets' in df.columns:
            features['src_packets'] = df['sPackets'].fillna(0)
            self.feature_groups['network_basic'].append('src_packets')
        
        if 'rPackets' in df.columns:
            features['dst_packets'] = df['rPackets'].fillna(0)
            self.feature_groups['network_basic'].append('dst_packets')
        
        if 'sBytesSum' in df.columns:
            features['src_bytes'] = df['sBytesSum'].fillna(0)
            self.feature_groups['network_basic'].append('src_bytes')
        
        if 'rBytesSum' in df.columns:
            features['dst_bytes'] = df['rBytesSum'].fillna(0)
            self.feature_groups['network_basic'].append('dst_bytes')
        
        if 'duration' in df.columns:
            features['flow_duration'] = df['duration'].fillna(0)
            self.feature_groups['network_basic'].append('flow_duration')
        
        # Derived features
        if 'src_packets' in features.columns and 'dst_packets' in features.columns:
            features['total_packets'] = features['src_packets'] + features['dst_packets']
            features['packet_ratio'] = features['src_packets'] / (features['dst_packets'] + 1)
            self.feature_groups['network_basic'].extend(['total_packets', 'packet_ratio'])
        
        if 'src_bytes' in features.columns and 'dst_bytes' in features.columns:
            features['total_bytes'] = features['src_bytes'] + features['dst_bytes']
            features['byte_ratio'] = features['src_bytes'] / (features['dst_bytes'] + 1)
            self.feature_groups['network_basic'].extend(['total_bytes', 'byte_ratio'])
        
        # Bytes per packet
        if 'total_bytes' in features.columns and 'total_packets' in features.columns:
            features['bytes_per_packet'] = features['total_bytes'] / (features['total_packets'] + 1)
            self.feature_groups['network_basic'].append('bytes_per_packet')
        
        logger.info(f"   Created {len(features.columns)} basic network features")
        return features
    
    def create_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create timing-based features.
        
        Features:
        - Inter-packet arrival times
        - Flow rates (packets/sec, bytes/sec)
        - Timing statistics
        """
        logger.info("Creating timing features...")
        
        features = pd.DataFrame(index=df.index)
        
        # Inter-packet arrival times
        if 'sInterPacketAvg' in df.columns:
            features['src_inter_packet_avg'] = df['sInterPacketAvg'].fillna(0)
            self.feature_groups['timing'].append('src_inter_packet_avg')
        
        if 'rInterPacketAvg' in df.columns:
            features['dst_inter_packet_avg'] = df['rInterPacketAvg'].fillna(0)
            self.feature_groups['timing'].append('dst_inter_packet_avg')
        
        # Flow rates (packets and bytes per second)
        if 'duration' in df.columns and df['duration'].notna().any():
            duration_safe = df['duration'].replace(0, 0.001)  # Avoid division by zero
            
            if 'sPackets' in df.columns:
                features['src_packet_rate'] = df['sPackets'] / duration_safe
                self.feature_groups['timing'].append('src_packet_rate')
            
            if 'rPackets' in df.columns:
                features['dst_packet_rate'] = df['rPackets'] / duration_safe
                self.feature_groups['timing'].append('dst_packet_rate')
            
            if 'sBytesSum' in df.columns:
                features['src_byte_rate'] = df['sBytesSum'] / duration_safe
                self.feature_groups['timing'].append('src_byte_rate')
            
            if 'rBytesSum' in df.columns:
                features['dst_byte_rate'] = df['rBytesSum'] / duration_safe
                self.feature_groups['timing'].append('dst_byte_rate')
        
        logger.info(f"   Created {len(features.columns)} timing features")
        return features
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features from packet/byte distributions.
        
        Features:
        - Min/max/avg packet sizes
        - Standard deviations
        - Load metrics
        """
        logger.info("Creating statistical features...")
        
        features = pd.DataFrame(index=df.index)
        
        # Packet size statistics
        stat_features = [
            ('sBytesMax', 'src_bytes_max'),
            ('rBytesMax', 'dst_bytes_max'),
            ('sBytesMin', 'src_bytes_min'),
            ('rBytesMin', 'dst_bytes_min'),
            ('sBytesAvg', 'src_bytes_avg'),
            ('rBytesAvg', 'dst_bytes_avg')
        ]
        
        for old_name, new_name in stat_features:
            if old_name in df.columns:
                features[new_name] = df[old_name].fillna(0)
                self.feature_groups['statistical'].append(new_name)
        
        # Load (throughput)
        if 'sLoad' in df.columns:
            features['src_load'] = df['sLoad'].fillna(0)
            self.feature_groups['statistical'].append('src_load')
        
        if 'rLoad' in df.columns:
            features['dst_load'] = df['rLoad'].fillna(0)
            self.feature_groups['statistical'].append('dst_load')
        
        # Payload statistics
        payload_features = [
            ('sPayloadSum', 'src_payload_sum'),
            ('rPayloadSum', 'dst_payload_sum'),
            ('sPayloadAvg', 'src_payload_avg'),
            ('rPayloadAvg', 'dst_payload_avg')
        ]
        
        for old_name, new_name in payload_features:
            if old_name in df.columns:
                features[new_name] = df[old_name].fillna(0)
                self.feature_groups['statistical'].append(new_name)
        
        logger.info(f"   Created {len(features.columns)} statistical features")
        return features
    
    def create_protocol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create protocol-specific features.
        
        Features:
        - TCP flags (ACK, SYN, FIN, RST, PSH, URG)
        - TTL values
        - Window sizes
        - Fragment rates
        """
        logger.info("Creating protocol features...")
        
        features = pd.DataFrame(index=df.index)
        
        # TCP flags
        tcp_flags = [
            ('sAckRate', 'src_ack_rate'),
            ('rAckRate', 'dst_ack_rate'),
            ('sSynRate', 'src_syn_rate'),
            ('rSynRate', 'dst_syn_rate'),
            ('sFinRate', 'src_fin_rate'),
            ('rFinRate', 'dst_fin_rate'),
            ('sRstRate', 'src_rst_rate'),
            ('rRstRate', 'dst_rst_rate'),
            ('sPshRate', 'src_psh_rate'),
            ('rPshRate', 'dst_psh_rate'),
            ('sUrgRate', 'src_urg_rate'),
            ('rUrgRate', 'dst_urg_rate')
        ]
        
        for old_name, new_name in tcp_flags:
            if old_name in df.columns:
                features[new_name] = df[old_name].fillna(0)
                self.feature_groups['protocol'].append(new_name)
        
        # TTL
        if 'sttl' in df.columns:
            features['src_ttl'] = df['sttl'].fillna(64)  # Default TTL
            self.feature_groups['protocol'].append('src_ttl')
        
        if 'rttl' in df.columns:
            features['dst_ttl'] = df['rttl'].fillna(64)
            self.feature_groups['protocol'].append('dst_ttl')
        
        # TCP window size
        if 'sWinTCP' in df.columns:
            features['src_win_size'] = df['sWinTCP'].fillna(0)
            self.feature_groups['protocol'].append('src_win_size')
        
        if 'rWinTCP' in df.columns:
            features['dst_win_size'] = df['rWinTCP'].fillna(0)
            self.feature_groups['protocol'].append('dst_win_size')
        
        # Fragment rate
        if 'sFragmentRate' in df.columns:
            features['src_fragment_rate'] = df['sFragmentRate'].fillna(0)
            self.feature_groups['protocol'].append('src_fragment_rate')
        
        if 'rFragmentRate' in df.columns:
            features['dst_fragment_rate'] = df['rFragmentRate'].fillna(0)
            self.feature_groups['protocol'].append('dst_fragment_rate')
        
        # ACK delay
        if 'sAckDelayAvg' in df.columns:
            features['src_ack_delay'] = df['sAckDelayAvg'].fillna(0)
            self.feature_groups['protocol'].append('src_ack_delay')
        
        if 'rAckDelayAvg' in df.columns:
            features['dst_ack_delay'] = df['rAckDelayAvg'].fillna(0)
            self.feature_groups['protocol'].append('dst_ack_delay')
        
        logger.info(f"   Created {len(features.columns)} protocol features")
        return features
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features that capture attack patterns.
        
        Features:
        - Connection patterns
        - Rate anomalies
        - Communication patterns
        """
        logger.info("Creating behavioral features...")
        
        features = pd.DataFrame(index=df.index)
        
        # Calculate features based on existing data
        # High SYN rate without ACK (potential SYN flood)
        if 'sSynRate' in df.columns and 'sAckRate' in df.columns:
            features['syn_ack_imbalance'] = df['sSynRate'] - df['sAckRate']
            self.feature_groups['behavioral'].append('syn_ack_imbalance')
        
        # Unusual packet/byte ratio (potential padding attacks)
        if 'sPackets' in df.columns and 'sBytesSum' in df.columns:
            avg_packet_size = df['sBytesSum'] / (df['sPackets'] + 1)
            features['packet_size_anomaly'] = (avg_packet_size - avg_packet_size.mean()).abs()
            self.feature_groups['behavioral'].append('packet_size_anomaly')
        
        # High RST rate (potential connection disruption)
        if 'sRstRate' in df.columns and 'rRstRate' in df.columns:
            features['reset_rate_total'] = df['sRstRate'] + df['rRstRate']
            self.feature_groups['behavioral'].append('reset_rate_total')
        
        # Bidirectional traffic imbalance
        if 'sPackets' in df.columns and 'rPackets' in df.columns:
            total_packets = df['sPackets'] + df['rPackets'] + 1
            features['traffic_symmetry'] = 1 - abs(df['sPackets'] - df['rPackets']) / total_packets
            self.feature_groups['behavioral'].append('traffic_symmetry')
        
        logger.info(f"   Created {len(features.columns)} behavioral features")
        return features
    
    def create_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create all feature groups and prepare for ML.
        
        Args:
            df: Raw ICSSIM dataset
            
        Returns:
            features: Feature matrix
            labels: Target labels
        """
        logger.info("\n" + "="*80)
        logger.info("ICS FEATURE ENGINEERING")
        logger.info("="*80 + "\n")
        
        # Create each feature group
        basic = self.create_basic_network_features(df)
        timing = self.create_timing_features(df)
        statistical = self.create_statistical_features(df)
        protocol = self.create_protocol_features(df)
        behavioral = self.create_behavioral_features(df)
        
        # Combine all features
        all_features = pd.concat([basic, timing, statistical, protocol, behavioral], axis=1)
        
        # Handle any remaining NaN values
        all_features = all_features.fillna(0)
        
        # Replace infinity values
        all_features = all_features.replace([np.inf, -np.inf], 0)
        
        # Extract labels
        label_col = 'IT_B_Label' if 'IT_B_Label' in df.columns else 'IT_M_Label'
        labels = df[label_col].fillna('Normal')
        
        # Convert labels to binary (0=Normal, 1=Attack)
        labels_binary = (labels != 'Normal').astype(int)
        
        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("="*80)
        logger.info(f"\nTotal Features Created: {len(all_features.columns)}")
        logger.info(f"Total Samples: {len(all_features)}")
        logger.info("\nFeature Breakdown:")
        for group, features in self.feature_groups.items():
            if features:
                logger.info(f"  • {group}: {len(features)} features")
        
        logger.info(f"\nLabel Distribution:")
        logger.info(f"  Normal: {(labels_binary == 0).sum()} ({(labels_binary == 0).sum()/len(labels_binary)*100:.1f}%)")
        logger.info(f"  Attack: {(labels_binary == 1).sum()} ({(labels_binary == 1).sum()/len(labels_binary)*100:.1f}%)")
        logger.info("="*80 + "\n")
        
        return all_features, labels_binary
    
    def save_features(self, features: pd.DataFrame, labels: pd.Series, output_dir: Path):
        """Save engineered features to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features_path = output_dir / 'ics_features.csv'
        features.to_csv(features_path, index=False)
        logger.info(f"✅ Saved features: {features_path}")
        
        # Save labels
        labels_path = output_dir / 'ics_labels.csv'
        labels.to_csv(labels_path, index=False, header=['label'])
        logger.info(f"✅ Saved labels: {labels_path}")
        
        # Save feature groups metadata
        import json
        metadata_path = output_dir / 'feature_groups.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.feature_groups, f, indent=2)
        logger.info(f"✅ Saved feature groups: {metadata_path}")
        
        return features_path, labels_path


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    print("\n" + "="*80)
    print("ICS FEATURE ENGINEER - DEMO")
    print("="*80 + "\n")
    
    # Load ICSSIM data
    data_path = Path('./data/raw/kaggle/icssim/Dataset.csv')
    
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        print("Please run manual_ics_loader.py first")
    else:
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} flows\n")
        
        # Create feature engineer
        engineer = ICSFeatureEngineer(random_seed=42)
        
        # Generate features
        features, labels = engineer.create_all_features(df)
        
        # Save
        engineer.save_features(features, labels, Path('./data/processed'))
        
        print("\n" + "="*80)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*80)
        print("\nSample features:")
        print(features.head())
        print("\n✅ Ready for model training!")
