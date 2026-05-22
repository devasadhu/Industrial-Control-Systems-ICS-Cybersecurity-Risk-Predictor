"""
ICS Network Anomaly Detection System

ML-based cyberattack detection for Industrial Control Systems (SCADA/DCS).
Trained on the ICSSIM dataset. IEC 62443 compliant.
"""

__version__ = "1.1.0"
__author__ = "Sadhana Devarajan"

try:
    from src.ics_feature_engineer import ICSFeatureEngineer
    from src.models.ensemble_detector import EnsembleICSDetector
    from src.detection.attack_patterns import ICSAttackPatternLibrary
    from src.iec62443_reporter import IEC62443ComplianceReporter
except ImportError:
    pass

__all__ = [
    'ICSFeatureEngineer',
    'EnsembleICSDetector',
    'ICSAttackPatternLibrary',
    'IEC62443ComplianceReporter',
]
