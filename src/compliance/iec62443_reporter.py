"""
IEC 62443 Compliance Assessment Tool
Automated security level assessment for ICS/OT networks

Standard: IEC 62443 (Industrial Automation and Control Systems Security)

Author: Sadhana Devarajan
Version: 1.1.0 - Added methodology_note to generate_report() for assessment transparency
"""

import pandas as pd
from typing import Dict, List
from datetime import datetime
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IEC62443ComplianceReporter:
    """Generate IEC 62443 compliance reports for ICS networks."""
    
    SECURITY_LEVELS = {
        'SL-1': {
            'name': 'Protection against casual or coincidental violation',
            'requirements': [
                'User identification and authentication',
                'Use control (authorization)',
                'Data integrity',
                'Resource availability'
            ],
            'target_industries': 'Low-risk manufacturing, building automation'
        },
        'SL-2': {
            'name': 'Protection against intentional violation using simple means',
            'requirements': [
                'All SL-1 requirements',
                'Audit logging',
                'Security event logging',
                'Cryptographic integrity',
                'Network segmentation'
            ],
            'target_industries': 'Food processing, HVAC control'
        },
        'SL-3': {
            'name': 'Protection against intentional violation using sophisticated means',
            'requirements': [
                'All SL-2 requirements',
                'Advanced network segmentation',
                'Anomaly detection systems',
                'Security event management (SIEM)',
                'Intrusion detection/prevention'
            ],
            'target_industries': 'Chemical plants, power generation, water treatment'
        },
        'SL-4': {
            'name': 'Protection against intentional violation using sophisticated means with extended resources',
            'requirements': [
                'All SL-3 requirements',
                'Advanced encryption',
                'Continuous monitoring',
                'Threat intelligence integration',
                'Red team exercises'
            ],
            'target_industries': 'Nuclear facilities, critical infrastructure'
        }
    }
    
    ZONES = {
        'Level 0': {
            'name': 'Physical Process',
            'description': 'Sensors, actuators, field devices',
            'criticality': 'VERY HIGH',
            'typical_protocols': ['Modbus', 'Profibus', '4-20mA']
        },
        'Level 1': {
            'name': 'Basic Control',
            'description': 'PLCs, RTUs, DCS controllers',
            'criticality': 'VERY HIGH',
            'typical_protocols': ['Modbus/TCP', 'DNP3', 'S7comm']
        },
        'Level 2': {
            'name': 'Supervisory Control',
            'description': 'SCADA, HMI, Engineering workstations',
            'criticality': 'HIGH',
            'typical_protocols': ['OPC UA', 'EtherNet/IP', 'HTTP/HTTPS']
        },
        'Level 3': {
            'name': 'Site Operations',
            'description': 'MES, Historians, Asset management',
            'criticality': 'MEDIUM',
            'typical_protocols': ['SQL', 'HTTP/HTTPS', 'FTP']
        },
        'Level 4': {
            'name': 'Enterprise Network',
            'description': 'ERP, Corporate IT systems',
            'criticality': 'LOW',
            'typical_protocols': ['HTTP/HTTPS', 'SMTP', 'RDP']
        }
    }
    
    def __init__(self):
        """Initialize compliance reporter."""
        self.assessment_results = {}
    
    def assess_network_segmentation(self, flows_df: pd.DataFrame) -> Dict:
        """
        Assess network segmentation compliance.
        
        IEC 62443-3-3 SR 3.1: Zone and Conduit
        """
        logger.info("Assessing network segmentation...")
        
        # Check for cross-zone traffic
        critical_ports = [502, 20000, 44818, 102, 4840]
        
        # Count flows to critical ICS ports
        if 'dst_port' in flows_df.columns:
            ics_traffic = flows_df[flows_df['dst_port'].isin(critical_ports)]
            cross_zone_ratio = len(ics_traffic) / len(flows_df) if len(flows_df) > 0 else 0
        else:
            cross_zone_ratio = 0
        
        # Good segmentation = < 20% cross-zone traffic
        score = max(0, 1.0 - (cross_zone_ratio * 2))
        
        status = 'COMPLIANT' if score >= 0.8 else ('PARTIAL' if score >= 0.5 else 'NON-COMPLIANT')
        
        return {
            'component': 'Network Segmentation',
            'iec_reference': 'IEC 62443-3-3 SR 3.1',
            'requirement': 'Zone and Conduit architecture',
            'score': score,
            'status': status,
            'findings': f'ICS traffic ratio: {cross_zone_ratio:.1%}',
            'recommendation': 'Implement VLAN segmentation per Purdue Model levels',
            'priority': 'HIGH'
        }
    
    def assess_anomaly_detection(self, anomaly_rate: float, detection_enabled: bool = True) -> Dict:
        """
        Assess anomaly detection capability.
        
        Required for SL-3 and above.
        IEC 62443-3-3 SR 6.1: Network and Security Event Logging
        """
        logger.info("Assessing anomaly detection...")
        
        # Anomaly detection should be enabled with 5-15% detection rate
        if not detection_enabled:
            return {
                'component': 'Anomaly Detection',
                'iec_reference': 'IEC 62443-3-3 SR 6.1',
                'requirement': 'Continuous monitoring for anomalies',
                'score': 0.0,
                'status': 'NON-COMPLIANT',
                'findings': 'Anomaly detection not enabled',
                'recommendation': 'Deploy IDS/IPS with anomaly detection capability',
                'priority': 'CRITICAL'
            }
        
        # Optimal detection rate: 5-15%
        is_optimal = 0.05 <= anomaly_rate <= 0.15
        
        if is_optimal:
            score = 1.0
            status = 'COMPLIANT'
            finding = f'Detection rate: {anomaly_rate:.1%} (optimal range)'
        elif anomaly_rate < 0.05:
            score = 0.6
            status = 'PARTIAL'
            finding = f'Detection rate: {anomaly_rate:.1%} (may miss threats)'
        else:
            score = 0.7
            status = 'PARTIAL'
            finding = f'Detection rate: {anomaly_rate:.1%} (high false positives)'
        
        return {
            'component': 'Anomaly Detection',
            'iec_reference': 'IEC 62443-3-3 SR 6.1',
            'requirement': 'Continuous monitoring for anomalies',
            'score': score,
            'status': status,
            'findings': finding,
            'recommendation': 'Maintain detection rate between 5-15% with regular tuning',
            'priority': 'HIGH'
        }
    
    def assess_authentication(self, config: Dict) -> Dict:
        """
        Assess user authentication.
        
        Required for SL-1+
        IEC 62443-3-3 SR 1.1: Human User Authentication
        """
        logger.info("Assessing authentication...")
        
        auth_enabled = config.get('auth_enabled', False)
        mfa_enabled = config.get('mfa_enabled', False)
        
        if not auth_enabled:
            score = 0.0
            status = 'NON-COMPLIANT'
            finding = 'Authentication disabled'
            rec = 'Implement mandatory authentication for all users'
        elif mfa_enabled:
            score = 1.0
            status = 'COMPLIANT'
            finding = 'Multi-factor authentication enabled'
            rec = 'Maintain MFA and review access logs regularly'
        else:
            score = 0.6
            status = 'PARTIAL'
            finding = 'Basic authentication only'
            rec = 'Upgrade to multi-factor authentication (MFA)'
        
        return {
            'component': 'User Authentication',
            'iec_reference': 'IEC 62443-3-3 SR 1.1',
            'requirement': 'Unique identification and authentication',
            'score': score,
            'status': status,
            'findings': finding,
            'recommendation': rec,
            'priority': 'CRITICAL'
        }
    
    def assess_logging(self, config: Dict) -> Dict:
        """
        Assess security logging.
        
        Required for SL-2+
        IEC 62443-3-3 SR 2.8: Audit Log Accessibility
        """
        logger.info("Assessing security logging...")
        
        logging_enabled = config.get('logging_enabled', False)
        siem_enabled = config.get('siem_enabled', False)
        log_retention_days = config.get('log_retention_days', 0)
        
        if not logging_enabled:
            score = 0.0
            status = 'NON-COMPLIANT'
            finding = 'Security logging disabled'
        elif siem_enabled and log_retention_days >= 90:
            score = 1.0
            status = 'COMPLIANT'
            finding = f'SIEM enabled with {log_retention_days}-day retention'
        elif log_retention_days >= 90:
            score = 0.7
            status = 'PARTIAL'
            finding = f'Logging enabled, no SIEM integration'
        else:
            score = 0.5
            status = 'PARTIAL'
            finding = f'Insufficient log retention: {log_retention_days} days'
        
        return {
            'component': 'Security Logging',
            'iec_reference': 'IEC 62443-3-3 SR 2.8',
            'requirement': 'Audit log accessibility and management',
            'score': score,
            'status': status,
            'findings': finding,
            'recommendation': 'Implement SIEM with 90+ day retention',
            'priority': 'HIGH'
        }
    
    def assess_access_control(self, config: Dict) -> Dict:
        """
        Assess access control mechanisms.
        
        IEC 62443-3-3 SR 2.1: Authorization Enforcement
        """
        logger.info("Assessing access control...")
        
        rbac_enabled = config.get('rbac_enabled', False)
        least_privilege = config.get('least_privilege', False)
        
        if rbac_enabled and least_privilege:
            score = 1.0
            status = 'COMPLIANT'
            finding = 'RBAC with least privilege enforced'
        elif rbac_enabled:
            score = 0.7
            status = 'PARTIAL'
            finding = 'RBAC enabled, least privilege not fully enforced'
        else:
            score = 0.3
            status = 'NON-COMPLIANT'
            finding = 'No role-based access control'
        
        return {
            'component': 'Access Control',
            'iec_reference': 'IEC 62443-3-3 SR 2.1',
            'requirement': 'Authorization enforcement',
            'score': score,
            'status': status,
            'findings': finding,
            'recommendation': 'Implement RBAC with principle of least privilege',
            'priority': 'HIGH'
        }
    
    def assess_data_integrity(self, config: Dict) -> Dict:
        """
        Assess data integrity protections.
        
        IEC 62443-3-3 SR 3.4: Software and Information Integrity
        """
        logger.info("Assessing data integrity...")
        
        encryption_enabled = config.get('encryption_enabled', False)
        integrity_checks = config.get('integrity_checks', False)
        
        if encryption_enabled and integrity_checks:
            score = 1.0
            status = 'COMPLIANT'
            finding = 'Encryption and integrity verification active'
        elif encryption_enabled or integrity_checks:
            score = 0.6
            status = 'PARTIAL'
            finding = 'Partial data protection implemented'
        else:
            score = 0.2
            status = 'NON-COMPLIANT'
            finding = 'No data integrity protection'
        
        return {
            'component': 'Data Integrity',
            'iec_reference': 'IEC 62443-3-3 SR 3.4',
            'requirement': 'Software and information integrity',
            'score': score,
            'status': status,
            'findings': finding,
            'recommendation': 'Enable TLS/SSL and implement checksums/signatures',
            'priority': 'HIGH'
        }
    
    def generate_report(self, 
                       flows_df: pd.DataFrame,
                       anomaly_rate: float,
                       config: Dict) -> Dict:
        """
        Generate complete IEC 62443 compliance report.
        
        Args:
            flows_df: Network flows data
            anomaly_rate: Anomaly detection rate
            config: System configuration
            
        Returns:
            Complete compliance report
        """
        logger.info("="*70)
        logger.info("GENERATING IEC 62443 COMPLIANCE REPORT")
        logger.info("="*70)
        
        # Run all assessments
        assessments = [
            self.assess_network_segmentation(flows_df),
            self.assess_anomaly_detection(anomaly_rate, config.get('detection_enabled', True)),
            self.assess_authentication(config),
            self.assess_logging(config),
            self.assess_access_control(config),
            self.assess_data_integrity(config)
        ]
        
        # Calculate overall score
        total_score = sum(a['score'] for a in assessments) / len(assessments)
        
        # Determine achieved security level
        if total_score >= 0.95:
            achieved_sl = 'SL-4'
            compliance_level = 'Excellent'
        elif total_score >= 0.80:
            achieved_sl = 'SL-3'
            compliance_level = 'Good'
        elif total_score >= 0.60:
            achieved_sl = 'SL-2'
            compliance_level = 'Acceptable'
        elif total_score >= 0.40:
            achieved_sl = 'SL-1'
            compliance_level = 'Minimal'
        else:
            achieved_sl = 'Below SL-1'
            compliance_level = 'Non-compliant'
        
        report = {
            'report_metadata': {
                'report_date': datetime.now().isoformat(),
                'standard': 'IEC 62443-3-3',
                'report_version': '1.1.0',
                'generated_by': 'ICS Anomaly Detection System'
            },
            'executive_summary': {
                'overall_score': total_score,
                'compliance_level': compliance_level,
                'achieved_security_level': achieved_sl,
                'target_security_level': config.get('target_sl', 'SL-3'),
                'recommendation': self._get_executive_recommendation(achieved_sl, config.get('target_sl', 'SL-3'))
            },
            'detailed_assessments': assessments,
            'summary_statistics': {
                'total_components': len(assessments),
                'compliant': sum(1 for a in assessments if a['status'] == 'COMPLIANT'),
                'partial': sum(1 for a in assessments if a['status'] == 'PARTIAL'),
                'non_compliant': sum(1 for a in assessments if a['status'] == 'NON-COMPLIANT'),
                'critical_priorities': sum(1 for a in assessments if a['priority'] == 'CRITICAL'),
                'high_priorities': sum(1 for a in assessments if a['priority'] == 'HIGH')
            },
            'security_zones': self.ZONES,
            'security_levels': self.SECURITY_LEVELS,
            # ── Assessment transparency: distinguishes data-driven from config-based ──
            'methodology_note': {
                'data_driven_assessments': [
                    'SR 3.1 - Network Segmentation (computed from live flow dst_port distribution)',
                    'SR 6.1 - Anomaly Detection (computed from trained model detection rate)'
                ],
                'config_based_assessments': [
                    'SR 1.1 - Authentication (requires system config input)',
                    'SR 2.8 - Security Logging (requires system config input)',
                    'SR 2.1 - Access Control (requires system config input)',
                    'SR 3.4 - Data Integrity (requires system config input)'
                ],
                'scope_statement': (
                    'Network-observable requirements assessed from 45,718 ICSSIM flows. '
                    'System-level requirements assessed from configuration flags. '
                    'This is a compliance indicator tool, not a certified audit.'
                )
            },
        }
        
        logger.info("✅ Report generation complete")
        return report
    
    def _get_executive_recommendation(self, achieved: str, target: str) -> str:
        """Generate executive recommendation."""
        if achieved == target:
            return f"System meets target security level ({target}). Maintain current controls and conduct regular audits."
        elif achieved > target:
            return f"System exceeds target security level. Current: {achieved}, Target: {target}."
        else:
            return f"System below target. Current: {achieved}, Target: {target}. Prioritize critical and high-priority recommendations."
    
    def print_report(self, report: Dict):
        """Print formatted compliance report."""
        print("\n" + "="*80)
        print("IEC 62443 COMPLIANCE ASSESSMENT REPORT")
        print("="*80)
        
        meta = report['report_metadata']
        print(f"\nReport Date: {meta['report_date']}")
        print(f"Standard: {meta['standard']}")
        
        summary = report['executive_summary']
        print(f"\n{'='*80}")
        print("EXECUTIVE SUMMARY")
        print("="*80)
        print(f"Overall Compliance Score: {summary['overall_score']:.1%}")
        print(f"Compliance Level: {summary['compliance_level']}")
        print(f"Achieved Security Level: {summary['achieved_security_level']}")
        print(f"Target Security Level: {summary['target_security_level']}")
        print(f"\nRecommendation: {summary['recommendation']}")
        
        print(f"\n{'='*80}")
        print("DETAILED ASSESSMENT RESULTS")
        print("="*80)
        
        for assessment in report['detailed_assessments']:
            status_icon = {
                'COMPLIANT': '✅',
                'PARTIAL': '⚠️',
                'NON-COMPLIANT': '❌'
            }[assessment['status']]
            
            print(f"\n{status_icon} {assessment['component']}")
            print(f"   Reference: {assessment['iec_reference']}")
            print(f"   Requirement: {assessment['requirement']}")
            print(f"   Status: {assessment['status']} (Score: {assessment['score']:.0%})")
            print(f"   Findings: {assessment['findings']}")
            print(f"   Recommendation: {assessment['recommendation']}")
            print(f"   Priority: {assessment['priority']}")
        
        stats = report['summary_statistics']
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Total Components Assessed: {stats['total_components']}")
        print(f"✅ Compliant: {stats['compliant']}")
        print(f"⚠️  Partial Compliance: {stats['partial']}")
        print(f"❌ Non-Compliant: {stats['non_compliant']}")
        print(f"\nAction Items:")
        print(f"  🔴 Critical Priority: {stats['critical_priorities']}")
        print(f"  🟠 High Priority: {stats['high_priorities']}")

        # Print methodology note so it surfaces in console output too
        note = report.get('methodology_note', {})
        if note:
            print(f"\n{'='*80}")
            print("ASSESSMENT METHODOLOGY")
            print("="*80)
            print("\n📊 Data-Driven (from network flows):")
            for item in note.get('data_driven_assessments', []):
                print(f"   • {item}")
            print("\n⚙️  Config-Based (from system flags):")
            for item in note.get('config_based_assessments', []):
                print(f"   • {item}")
            print(f"\n⚠️  Scope: {note.get('scope_statement', '')}")

        print("="*80 + "\n")
    
    def export_json(self, report: Dict, output_path: str):
        """Export report as JSON."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"✅ Report exported to {output_path}")
    
    def export_csv(self, report: Dict, output_path: str):
        """Export assessments as CSV."""
        df = pd.DataFrame(report['detailed_assessments'])
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Report exported to {output_path}")


def demo_compliance_reporter():
    """Demo the compliance reporter."""
    print("="*80)
    print("IEC 62443 COMPLIANCE ASSESSMENT DEMO")
    print("="*80)
    
    reporter = IEC62443ComplianceReporter()
    
    # Load sample data
    data_path = Path("./data/processed/ics_features.csv")
    if not data_path.exists():
        data_path = Path("../../data/processed/ics_features.csv")
    
    if data_path.exists():
        flows = pd.read_csv(data_path)
        logger.info(f"Loaded {len(flows)} network flows")
    else:
        # Generate sample data
        flows = pd.DataFrame({
            'dst_port': [80, 443, 502, 20000, 22] * 100
        })
    
    # Sample configuration
    config = {
        'auth_enabled': True,
        'mfa_enabled': False,
        'logging_enabled': True,
        'siem_enabled': False,
        'log_retention_days': 90,
        'rbac_enabled': True,
        'least_privilege': False,
        'encryption_enabled': True,
        'integrity_checks': True,
        'detection_enabled': True,
        'target_sl': 'SL-3'
    }
    
    # Generate report
    report = reporter.generate_report(
        flows_df=flows,
        anomaly_rate=0.096,
        config=config
    )
    
    # Print report
    reporter.print_report(report)
    
    # Export
    output_dir = Path("./results/compliance")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reporter.export_json(report, str(output_dir / "iec62443_report.json"))
    reporter.export_csv(report, str(output_dir / "iec62443_assessment.csv"))
    
    print("\n📄 Reports saved to: results/compliance/")


if __name__ == "__main__":
    demo_compliance_reporter()