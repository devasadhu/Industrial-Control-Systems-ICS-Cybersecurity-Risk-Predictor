"""
ICS Attack Pattern Detection Library
Detects known attack patterns in industrial control systems

Based on MITRE ATT&CK for ICS and real-world incidents:
- Stuxnet, Triton, BlackEnergy, Industroyer

Author: Sadhana Devarajan
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AttackPattern:
    """Attack pattern definition."""
    name: str
    mitre_technique: str
    description: str
    indicators: Dict
    severity: str
    mitigation: str
    real_world_example: str


class ICSAttackPatternLibrary:
    """Library of known ICS attack patterns."""
    
    def __init__(self):
        """Initialize attack pattern library."""
        self.patterns = self._load_patterns()
        self.detected_patterns = []
    
    def _load_patterns(self) -> Dict[str, AttackPattern]:
        """Load known attack patterns."""
        return {
            'modbus_flooding': AttackPattern(
                name='Modbus Traffic Flooding',
                mitre_technique='T0823 - Denial of Service',
                description='Overwhelming Modbus/TCP service with excessive requests',
                indicators={
                    'dst_port': 502,
                    'packet_rate_threshold': 100,  # packets/second
                    'unique_transactions': 50       # different requests
                },
                severity='HIGH',
                mitigation='Implement rate limiting on Modbus gateway, deploy IPS rules',
                real_world_example='Used in BlackEnergy attacks against Ukrainian power grid'
            ),
            
            'plc_scanning': AttackPattern(
                name='PLC Network Scanning',
                mitre_technique='T0846 - Remote System Discovery',
                description='Sequential scanning of ICS ports to discover PLCs/RTUs',
                indicators={
                    'dst_ports': [502, 20000, 44818, 102],
                    'scan_pattern': 'sequential',
                    'time_window': 60,  # seconds
                    'unique_destinations': 10
                },
                severity='CRITICAL',
                mitigation='Deploy network segmentation, enable port-based access control',
                real_world_example='Pre-attack reconnaissance in Stuxnet campaign'
            ),
            
            'unauthorized_write': AttackPattern(
                name='Unauthorized PLC Write Command',
                mitre_technique='T0836 - Modify Control Logic',
                description='Unauthorized write to PLC registers or memory',
                indicators={
                    'modbus_function_codes': [0x05, 0x06, 0x0F, 0x10],  # Write functions
                    'source_unauthorized': True,
                    'outside_maintenance_window': True
                },
                severity='CRITICAL',
                mitigation='Enforce write access controls, implement change management',
                real_world_example='Core technique in Stuxnet and Triton attacks'
            ),
            
            'protocol_fuzzing': AttackPattern(
                name='Protocol Fuzzing/Malformed Packets',
                mitre_technique='T0851 - Protocol Exploitation',
                description='Sending malformed protocol packets to crash systems',
                indicators={
                    'malformed_packets': True,
                    'invalid_function_codes': True,
                    'unexpected_payload_sizes': True
                },
                severity='HIGH',
                mitigation='Update firmware, deploy protocol-aware firewall',
                real_world_example='ICS vulnerability discovery technique'
            ),
            
            'man_in_the_middle': AttackPattern(
                name='Man-in-the-Middle Attack',
                mitre_technique='T0830 - Adversary-in-the-Middle',
                description='Intercepting and modifying ICS communications',
                indicators={
                    'duplicate_ips': True,
                    'arp_spoofing': True,
                    'unexpected_gateway': True
                },
                severity='CRITICAL',
                mitigation='Enable TLS/SSL, deploy network monitoring, use certificates',
                real_world_example='Used in advanced APT campaigns'
            ),
            
            'command_injection': AttackPattern(
                name='Command Injection',
                mitre_technique='T0871 - Execution through API',
                description='Injecting malicious commands through ICS protocols',
                indicators={
                    'unusual_command_sequences': True,
                    'suspicious_payloads': True,
                    'rapid_successive_writes': True
                },
                severity='CRITICAL',
                mitigation='Input validation, command whitelisting, audit logging',
                real_world_example='Triton safety system attack'
            ),
            
            'time_based_attack': AttackPattern(
                name='Time-Based Logic Attack',
                mitre_technique='T0889 - Modify Program',
                description='Exploiting timing in control logic (à la Stuxnet)',
                indicators={
                    'periodic_writes': True,
                    'specific_time_patterns': True,
                    'target_specific_processes': True
                },
                severity='CRITICAL',
                mitigation='Monitor process timing, detect logic modifications',
                real_world_example='Stuxnet centrifuge attack'
            ),
            
            'replay_attack': AttackPattern(
                name='Replay Attack',
                mitre_technique='T0843 - Replay Attack',
                description='Replaying captured legitimate commands',
                indicators={
                    'duplicate_transactions': True,
                    'identical_sequences': True,
                    'stale_timestamps': True
                },
                severity='HIGH',
                mitigation='Implement nonces, timestamps, sequence numbers',
                real_world_example='Common in unsecured SCADA systems'
            ),
            
            'credential_stuffing': AttackPattern(
                name='Credential Stuffing',
                mitre_technique='T0859 - Valid Accounts',
                description='Using stolen credentials to access ICS systems',
                indicators={
                    'multiple_failed_logins': True,
                    'login_from_unusual_location': True,
                    'off_hours_access': True
                },
                severity='HIGH',
                mitigation='Enforce MFA, monitor authentication logs, lockout policies',
                real_world_example='Common initial access vector'
            ),
            
            'firmware_modification': AttackPattern(
                name='Firmware Modification',
                mitre_technique='T0857 - System Firmware',
                description='Unauthorized modification of device firmware',
                indicators={
                    'firmware_update_commands': True,
                    'unauthorized_source': True,
                    'integrity_check_failure': True
                },
                severity='CRITICAL',
                mitigation='Code signing, firmware validation, secure boot',
                real_world_example='BlackEnergy firmware attacks'
            )
        }
    
    def detect_modbus_flooding(self, flows_df: pd.DataFrame) -> List[Dict]:
        """Detect Modbus flooding attacks."""
        detections = []
        
        if 'dst_port' not in flows_df.columns:
            return detections
        
        modbus_flows = flows_df[flows_df['dst_port'] == 502]
        
        if len(modbus_flows) == 0:
            return detections
        
        # Calculate packet rate
        if 'flow_duration' in modbus_flows.columns and 'total_packets' in modbus_flows.columns:
            modbus_flows_copy = modbus_flows.copy()
            modbus_flows_copy['packet_rate'] = modbus_flows_copy['total_packets'] / modbus_flows_copy['flow_duration'].clip(lower=0.001)
            
            # Detect flooding
            high_rate_flows = modbus_flows_copy[modbus_flows_copy['packet_rate'] > 100]
            
            for _, flow in high_rate_flows.iterrows():
                detections.append({
                    'pattern': 'modbus_flooding',
                    'severity': 'HIGH',
                    'confidence': min(flow['packet_rate'] / 200, 1.0),
                    'details': {
                        'packet_rate': flow['packet_rate'],
                        'dst_ip': flow.get('dst_ip', 'unknown'),
                        'dst_port': 502
                    }
                })
        
        return detections
    
    def detect_plc_scanning(self, flows_df: pd.DataFrame, time_window: int = 60) -> List[Dict]:
        """Detect PLC network scanning."""
        detections = []
        
        if 'dst_port' not in flows_df.columns or 'src_ip' not in flows_df.columns:
            return detections
        
        ics_ports = [502, 20000, 44818, 102, 4840]
        
        # Group by source IP
        for src_ip, group in flows_df.groupby('src_ip'):
            # Check if scanning ICS ports
            scanned_ports = group[group['dst_port'].isin(ics_ports)]['dst_port'].unique()
            unique_dests = group['dst_ip'].nunique() if 'dst_ip' in group.columns else 0
            
            # Scanning pattern: multiple ICS ports, multiple destinations
            if len(scanned_ports) >= 2 and unique_dests >= 5:
                detections.append({
                    'pattern': 'plc_scanning',
                    'severity': 'CRITICAL',
                    'confidence': min((len(scanned_ports) * unique_dests) / 50, 1.0),
                    'details': {
                        'src_ip': src_ip,
                        'scanned_ports': scanned_ports.tolist(),
                        'unique_targets': unique_dests
                    }
                })
        
        return detections
    
    def detect_unauthorized_writes(self, flows_df: pd.DataFrame, protocol_analysis: List[Dict]) -> List[Dict]:
        """Detect unauthorized write commands."""
        detections = []
        
        # Check protocol analysis for write commands
        for analysis in protocol_analysis:
            if analysis.get('protocol') == 'Modbus/TCP' and analysis.get('is_write'):
                # Check if from unauthorized source
                src_ip = analysis.get('details', {}).get('src_ip', '')
                
                # Simple heuristic: 192.168.1.x are workstations (authorized)
                # Other IPs are suspicious
                authorized = src_ip.startswith('192.168.1.')
                
                if not authorized or analysis.get('severity') in ['HIGH', 'CRITICAL']:
                    detections.append({
                        'pattern': 'unauthorized_write',
                        'severity': 'CRITICAL',
                        'confidence': 0.8,
                        'details': {
                            'function': analysis.get('function_name'),
                            'register': analysis.get('details', {}).get('register_address'),
                            'src_ip': src_ip
                        }
                    })
        
        return detections
    
    def detect_all_patterns(self, 
                           flows_df: pd.DataFrame,
                           protocol_analysis: List[Dict] = None) -> Dict:
        """
        Detect all attack patterns in the data.
        
        Args:
            flows_df: Network flows DataFrame
            protocol_analysis: Optional protocol analysis results
            
        Returns:
            Dictionary of detected patterns
        """
        logger.info("Scanning for known attack patterns...")
        
        all_detections = {
            'modbus_flooding': self.detect_modbus_flooding(flows_df),
            'plc_scanning': self.detect_plc_scanning(flows_df),
        }
        
        if protocol_analysis:
            all_detections['unauthorized_writes'] = self.detect_unauthorized_writes(flows_df, protocol_analysis)
        
        # Count detections
        total_detections = sum(len(v) for v in all_detections.values())
        
        # Get severity counts
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for pattern_detections in all_detections.values():
            for detection in pattern_detections:
                severity_counts[detection['severity']] += 1
        
        results = {
            'total_detections': total_detections,
            'patterns_found': [k for k, v in all_detections.items() if v],
            'detections_by_pattern': all_detections,
            'severity_breakdown': severity_counts
        }
        
        logger.info(f"✅ Pattern detection complete")
        logger.info(f"   Total detections: {total_detections}")
        logger.info(f"   Critical: {severity_counts['CRITICAL']}")
        logger.info(f"   High: {severity_counts['HIGH']}")
        
        return results
    
    def get_pattern_info(self, pattern_name: str) -> Optional[AttackPattern]:
        """Get information about a specific attack pattern."""
        return self.patterns.get(pattern_name)
    
    def list_all_patterns(self) -> List[str]:
        """List all available attack patterns."""
        return list(self.patterns.keys())
    
    def generate_threat_report(self, detection_results: Dict) -> str:
        """Generate human-readable threat report."""
        report = []
        report.append("="*80)
        report.append("ICS ATTACK PATTERN DETECTION REPORT")
        report.append("="*80)
        report.append(f"\nTotal Detections: {detection_results['total_detections']}")
        
        if detection_results['total_detections'] == 0:
            report.append("\n✅ No known attack patterns detected")
            return "\n".join(report)
        
        report.append(f"\n⚠️  THREATS DETECTED:")
        report.append("\nSeverity Breakdown:")
        for severity, count in detection_results['severity_breakdown'].items():
            if count > 0:
                icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}[severity]
                report.append(f"  {icon} {severity}: {count}")
        
        report.append(f"\n{'='*80}")
        report.append("DETECTED ATTACK PATTERNS")
        report.append("="*80)
        
        for pattern_name, detections in detection_results['detections_by_pattern'].items():
            if not detections:
                continue
            
            pattern_info = self.get_pattern_info(pattern_name)
            
            report.append(f"\n🚨 {pattern_info.name if pattern_info else pattern_name.upper()}")
            report.append(f"   Count: {len(detections)}")
            
            if pattern_info:
                report.append(f"   MITRE ATT&CK: {pattern_info.mitre_technique}")
                report.append(f"   Description: {pattern_info.description}")
                report.append(f"   Mitigation: {pattern_info.mitigation}")
            
            # Show first few detections
            for i, detection in enumerate(detections[:3], 1):
                report.append(f"\n   Detection {i}:")
                report.append(f"      Severity: {detection['severity']}")
                report.append(f"      Confidence: {detection['confidence']:.1%}")
                for key, value in detection['details'].items():
                    report.append(f"      {key}: {value}")
        
        report.append("\n" + "="*80)
        report.append("✅ REPORT COMPLETE")
        report.append("="*80)
        
        return "\n".join(report)


def demo_attack_patterns():
    """Demo the attack pattern library."""
    print("="*80)
    print("ICS ATTACK PATTERN DETECTION DEMO")
    print("="*80)
    
    library = ICSAttackPatternLibrary()
    
    # Show available patterns
    print(f"\n📚 Available Attack Patterns: {len(library.patterns)}")
    for name, pattern in library.patterns.items():
        print(f"\n• {pattern.name}")
        print(f"  MITRE: {pattern.mitre_technique}")
        print(f"  Severity: {pattern.severity}")
    
    # Load sample data
    from pathlib import Path
    
    data_path = Path("./data/processed/ics_features.csv")
    if not data_path.exists():
        data_path = Path("../../data/processed/ics_features.csv")
    
    if data_path.exists():
        print(f"\n{'='*80}")
        print("ANALYZING NETWORK DATA")
        print("="*80)
        
        flows = pd.read_csv(data_path)
        print(f"\nLoaded {len(flows)} network flows")
        
        # Detect patterns
        results = library.detect_all_patterns(flows)
        
        # Generate report
        report = library.generate_threat_report(results)
        print(f"\n{report}")
    else:
        print(f"\n⚠️  No data found - run pipeline first")


if __name__ == "__main__":
    demo_attack_patterns()