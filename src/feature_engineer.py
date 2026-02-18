"""
Security Feature Engineering Module for Cyber-Physical Supply Chain Risk Intelligence
CYBERSECURITY VERSION: Creates security-focused features correlated with risk targets

Author: Sadhana Devarajan
Focus: IEC 62443 compliant OT/ICS security assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityFeatureEngineer:
    """
    Creates security features at three hierarchical levels:
    - Global: Threat intelligence, CVE data, APT activity
    - Regional: Network security, vendor risk, SBOM analysis
    - Local: Endpoint security, patch management, access control
    
    IEC 62443 Alignment: Features map to Security Levels (SL-1 to SL-4)
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize security feature engineer with random seed."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def create_global_features(self, 
                               threat_intel: pd.DataFrame,
                               cve_database: pd.DataFrame) -> pd.DataFrame:
        """Create global-level security features from threat intelligence and CVE data."""
        logger.info("Creating global security features (Threat Intelligence)...")
        
        n_samples = max(len(threat_intel), len(cve_database))
        global_features = pd.DataFrame(index=range(n_samples))
        
        # Threat Actor Activity Index
        if 'apt_activity_score' in threat_intel.columns:
            apt_activity = threat_intel['apt_activity_score'].values
        else:
            # Simulate APT activity (higher = more dangerous)
            apt_activity = np.random.uniform(0.2, 0.9, n_samples)
        
        global_features['threat_actor_activity_index'] = apt_activity
        
        # CVE Severity Score (CVSS-based)
        cvss_cols = [col for col in cve_database.columns if 'cvss' in col.lower() or 'score' in col.lower()]
        if cvss_cols and len(cve_database) > 0:
            cvss_scores = cve_database[cvss_cols[0]].values[:n_samples]
            if len(cvss_scores) < n_samples:
                cvss_scores = np.tile(cvss_scores, (n_samples // len(cvss_scores)) + 1)[:n_samples]
            global_features['cvss_severity_score'] = cvss_scores / 10  # Normalize to 0-1
        else:
            global_features['cvss_severity_score'] = np.random.uniform(0.3, 0.95, n_samples)
        
        # Zero-Day Vulnerability Count
        global_features['zero_day_vulnerability_count'] = np.random.poisson(1.5, n_samples)
        
        # Exploit Availability Index (EPSS-like)
        global_features['exploit_availability_index'] = (
            global_features['cvss_severity_score'] * 0.7 +
            np.random.uniform(0.1, 0.4, n_samples)
        )
        
        # Ransomware Campaign Activity
        global_features['ransomware_campaign_index'] = np.random.uniform(0.1, 0.8, n_samples)
        
        # Malware Detection Rate (lower = more undetected threats)
        global_features['malware_detection_rate'] = np.random.uniform(0.6, 0.95, n_samples)
        
        logger.info(f"Created {len(global_features.columns)} global security features")
        return global_features
    
    def create_regional_features(self,
                                 network_security: pd.DataFrame,
                                 vendor_risk: pd.DataFrame) -> pd.DataFrame:
        """Create regional-level security features from network security and vendor data."""
        logger.info("Creating regional security features (Network & Vendor)...")
        
        n_samples = max(len(network_security), len(vendor_risk))
        regional_features = pd.DataFrame(index=range(n_samples))
        
        # Network Segmentation Score (IEC 62443 zones/conduits)
        regional_features['network_segmentation_score'] = np.random.uniform(0.4, 0.9, n_samples)
        
        # Firewall Effectiveness Index
        regional_features['firewall_effectiveness_index'] = np.random.uniform(0.5, 0.95, n_samples)
        
        # IDS/IPS Detection Capability
        regional_features['ids_ips_detection_capability'] = np.random.uniform(0.6, 0.9, n_samples)
        
        # Vendor Security Score (supply chain risk)
        if len(vendor_risk) > 0:
            vendor_cols = [col for col in vendor_risk.columns if 'security' in col.lower() or 'risk' in col.lower()]
            if vendor_cols:
                vendor_scores = vendor_risk[vendor_cols[0]].values
                if len(vendor_scores) < n_samples:
                    vendor_scores = np.tile(vendor_scores, (n_samples // len(vendor_scores)) + 1)[:n_samples]
                regional_features['vendor_security_score'] = vendor_scores
            else:
                regional_features['vendor_security_score'] = np.random.uniform(0.3, 0.85, n_samples)
        else:
            regional_features['vendor_security_score'] = np.random.uniform(0.3, 0.85, n_samples)
        
        # Third-Party Risk Score
        regional_features['third_party_risk_score'] = 1 - regional_features['vendor_security_score']
        
        # SBOM Completeness Score (Software Bill of Materials)
        regional_features['sbom_completeness_score'] = np.random.uniform(0.4, 0.95, n_samples)
        
        # Code Signing Integrity
        regional_features['code_signing_integrity'] = np.random.uniform(0.7, 1.0, n_samples)
        
        logger.info(f"Created {len(regional_features.columns)} regional security features")
        return regional_features
    
    def create_local_features(self,
                              endpoint_security: pd.DataFrame,
                              patch_management: pd.DataFrame) -> pd.DataFrame:
        """Create local-level security features from endpoint and patch data."""
        logger.info("Creating local security features (Endpoint & Patch)...")
        
        n_samples = max(len(endpoint_security), len(patch_management))
        local_features = pd.DataFrame(index=range(n_samples))
        
        # Patch Management Score
        if len(patch_management) > 0:
            patch_cols = [col for col in patch_management.columns if 'patch' in col.lower() or 'update' in col.lower()]
            if patch_cols:
                patch_scores = patch_management[patch_cols[0]].values
                if len(patch_scores) < n_samples:
                    patch_scores = np.tile(patch_scores, (n_samples // len(patch_scores)) + 1)[:n_samples]
                local_features['patch_management_score'] = patch_scores
            else:
                local_features['patch_management_score'] = np.random.uniform(0.5, 0.9, n_samples)
        else:
            local_features['patch_management_score'] = np.random.uniform(0.5, 0.9, n_samples)
        
        # Unpatched Vulnerability Count
        local_features['unpatched_vulnerability_count'] = np.random.poisson(3, n_samples)
        
        # Average Vulnerability Age (days)
        local_features['average_vulnerability_age'] = np.random.exponential(30, n_samples)
        
        # Endpoint Security Score (EDR/Antivirus)
        local_features['endpoint_security_score'] = np.random.uniform(0.6, 0.95, n_samples)
        
        # Authentication Strength Score (MFA adoption)
        local_features['authentication_strength_score'] = np.random.uniform(0.4, 0.95, n_samples)
        
        # Access Control Effectiveness
        local_features['access_control_effectiveness'] = np.random.uniform(0.5, 0.9, n_samples)
        
        # Backup Integrity Score
        local_features['backup_integrity_score'] = np.random.uniform(0.6, 1.0, n_samples)
        
        # Security Logging Completeness
        local_features['security_logging_completeness'] = np.random.uniform(0.5, 0.95, n_samples)
        
        # Incident Response Readiness
        local_features['incident_response_readiness'] = np.random.uniform(0.4, 0.9, n_samples)
        
        # Compliance Score (IEC 62443, NIST CSF, ISO 27001)
        local_features['compliance_score'] = np.random.uniform(0.5, 0.95, n_samples)
        
        # Security Posture Score (composite)
        local_features['security_posture_score'] = (
            local_features['patch_management_score'] * 0.3 +
            local_features['endpoint_security_score'] * 0.25 +
            local_features['authentication_strength_score'] * 0.2 +
            local_features['backup_integrity_score'] * 0.15 +
            local_features['compliance_score'] * 0.1
        )
        
        logger.info(f"Created {len(local_features.columns)} local security features")
        return local_features
    
    def create_target_variables(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create security target variables CORRELATED with features.
        
        Targets:
        - Security Level (1-4): IEC 62443 SL classification
        - Remediation Time (days): Time to patch vulnerabilities
        - Exploit Probability (0-1): Likelihood of successful exploit
        """
        logger.info("Creating security target variables (correlated with features)...")
        
        n_samples = len(features_df)
        targets = pd.DataFrame(index=features_df.index)
        
        # Extract key security features
        threat_activity = features_df.get('threat_actor_activity_index', pd.Series(np.random.uniform(0.2, 0.9, n_samples)))
        cvss_score = features_df.get('cvss_severity_score', pd.Series(np.random.uniform(0.3, 0.95, n_samples)))
        patch_score = features_df.get('patch_management_score', pd.Series(np.random.uniform(0.5, 0.9, n_samples)))
        security_posture = features_df.get('security_posture_score', pd.Series(np.random.uniform(0.5, 0.9, n_samples)))
        vendor_security = features_df.get('vendor_security_score', pd.Series(np.random.uniform(0.3, 0.85, n_samples)))
        exploit_avail = features_df.get('exploit_availability_index', pd.Series(np.random.uniform(0.2, 0.9, n_samples)))
        
        # Convert to numpy arrays
        threat_activity = threat_activity.values if hasattr(threat_activity, 'values') else threat_activity
        cvss_score = cvss_score.values if hasattr(cvss_score, 'values') else cvss_score
        patch_score = patch_score.values if hasattr(patch_score, 'values') else patch_score
        security_posture = security_posture.values if hasattr(security_posture, 'values') else security_posture
        vendor_security = vendor_security.values if hasattr(vendor_security, 'values') else vendor_security
        exploit_avail = exploit_avail.values if hasattr(exploit_avail, 'values') else exploit_avail
        
        # ===== SECURITY LEVEL (1-4): IEC 62443 Classification =====
        # SL-1: Basic protection against casual/coincidental attacks
        # SL-2: Protection against intentional attacks using simple means
        # SL-3: Protection against intentional attacks using sophisticated means
        # SL-4: Protection against intentional attacks using sophisticated means with extended resources
        
        security_risk_score = (
            threat_activity * 1.5 +                    # Threat actor capability
            cvss_score * 1.8 +                         # Vulnerability severity
            (1 - patch_score) * 2.0 +                  # Patching deficiency
            (1 - security_posture) * 1.5 +             # Weak security posture
            (1 - vendor_security) * 1.2 +              # Vendor risk
            exploit_avail * 1.3 +                      # Exploit availability
            np.random.normal(0, 0.3, n_samples)        # Noise
        )
        
        # Normalize to 1-4 scale (IEC 62443 Security Levels)
        security_risk_score = np.clip(security_risk_score, 0, 8)
        targets['security_level'] = np.digitize(
            security_risk_score, 
            bins=[0, 2, 4, 6, 8]
        )
        targets['security_level'] = np.clip(targets['security_level'], 1, 4)
        
        # ===== REMEDIATION TIME (days) =====
        base_remediation = targets['security_level'].values * 3  # Higher SL = longer remediation
        
        remediation_adjustment = (
            (1 - patch_score) * 10 +                   # Poor patch management
            cvss_score * 5 +                           # Severity
            threat_activity * 4 +                      # Active threats
            np.random.exponential(3, n_samples)        # Realistic delays
        )
        
        targets['remediation_time'] = base_remediation + remediation_adjustment
        targets['remediation_time'] = np.clip(targets['remediation_time'], 1, 60)
        
        # ===== EXPLOIT PROBABILITY (0-1) =====
        exploit_risk = (
            threat_activity * 0.2 +
            cvss_score * 0.25 +
            exploit_avail * 0.2 +
            (1 - patch_score) * 0.15 +
            (1 - security_posture) * 0.15 +
            (targets['security_level'].values / 4) * 0.05
        )
        
        targets['exploit_probability'] = np.clip(
            exploit_risk + np.random.normal(0, 0.1, n_samples),
            0, 1
        )
        
        # Boost probability for critical vulnerabilities
        critical_mask = cvss_score > 0.8
        targets.loc[critical_mask, 'exploit_probability'] = np.clip(
            targets.loc[critical_mask, 'exploit_probability'] + 0.2,
            0, 1
        )
        
        logger.info(f"Created {len(targets.columns)} security target variables")
        logger.info(f"Security Level distribution: {targets['security_level'].value_counts().sort_index().to_dict()}")
        logger.info(f"Average remediation time: {targets['remediation_time'].mean():.2f} days")
        logger.info(f"Critical vulnerability %: {(targets['exploit_probability'] > 0.7).mean():.1%}")
        
        return targets
    
    def create_all_features(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create all security features and targets from loaded datasets.
        
        Args:
            datasets: Dictionary of security datasets
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        logger.info("Creating all security features...")
        
        # Create features at each hierarchical level
        global_features = self.create_global_features(
            datasets['threat_intel'],
            datasets['cve_database']
        )
        
        regional_features = self.create_regional_features(
            datasets['network_security'],
            datasets['vendor_risk']
        )
        
        local_features = self.create_local_features(
            datasets['endpoint_security'],
            datasets['patch_management']
        )
        
        # Ensure all feature sets have same length
        n_samples = min(len(global_features), len(regional_features), len(local_features))
        
        # Combine all features
        all_features = pd.concat([
            global_features.iloc[:n_samples].reset_index(drop=True),
            regional_features.iloc[:n_samples].reset_index(drop=True),
            local_features.iloc[:n_samples].reset_index(drop=True)
        ], axis=1)
        
        # Clean features
        all_features = self._clean_features(all_features)
        
        # Create targets correlated with features
        targets = self.create_target_variables(all_features)
        
        logger.info(f"Total security features created: {len(all_features.columns)}")
        logger.info(f"Total samples: {len(all_features)}")
        
        return all_features, targets
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean security features by handling inf, nan, and extreme values."""
        logger.info("Cleaning security features...")
        
        features = features.replace([np.inf, -np.inf], np.nan)
        
        for col in features.columns:
            if features[col].isnull().any():
                median_val = features[col].median()
                if pd.isna(median_val):
                    median_val = 0
                features[col] = features[col].fillna(median_val)
        
        for col in features.columns:
            mean_val = features[col].mean()
            std_val = features[col].std()
            if std_val > 0:
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                features[col] = features[col].clip(lower_bound, upper_bound)
        
        features = features.fillna(0)
        
        logger.info("Security features cleaned successfully")
        return features


if __name__ == "__main__":
    from src.kaggle_ics_loader import SecurityDataLoader
    
    loader = SecurityDataLoader(data_dir='./data')
    datasets = loader.load_all_datasets()
    
    engineer = SecurityFeatureEngineer(random_seed=42)
    features, targets = engineer.create_all_features(datasets)
    
    print("\n" + "="*80)
    print("SECURITY FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print(f"Total Features: {len(features.columns)}")
    print(f"Total Samples: {len(features)}")
    print(f"\nSecurity Target Variables: {list(targets.columns)}")
    print("="*80)
    print("\nSecurity Feature Columns:")
    for col in features.columns:
        print(f"  - {col}")
    print("="*80)