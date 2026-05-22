"""
NVD CVE Risk Mapper
Queries the NIST National Vulnerability Database (NVD) API to enrich
ICS anomaly detections with CVE severity data and IEC 62443 risk mapping.

Author: Sadhana Devarajan
API: https://services.nvd.nist.gov/rest/json/cves/2.0
"""

import requests
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ICS-relevant CPE vendor strings for filtering CVEs
ICS_VENDORS = {
    'schneider_electric': ['modicon', 'ecostruxure', 'wonderware', 'clearscada'],
    'yokogawa':           ['centum', 'exaquantum', 'stardom'],
    'siemens':            ['s7-300', 's7-400', 's7-1200', 's7-1500', 'simatic'],
    'rockwell':           ['logix', 'allen-bradley', 'factorytalk'],
    'abb':                ['ac800m', 'symphony'],
}

# CVSS → IEC 62443 Security Level mapping
CVSS_TO_IEC_SL = {
    (9.0, 10.0): 'SL-4',
    (7.0,  9.0): 'SL-3',
    (4.0,  7.0): 'SL-2',
    (0.0,  4.0): 'SL-1',
}


class NVDCVEMapper:
    """
    Queries the NVD REST API v2 for ICS-related CVEs and maps
    CVSS scores to IEC 62443 Security Levels.
    """

    BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "./data/nvd_cache"):
        """
        Args:
            api_key:   NVD API key (optional; increases rate limit from 5 to 50 req/30s).
                       Get one free at https://nvd.nist.gov/developers/request-an-api-key
            cache_dir: Directory to cache API responses so repeated runs don't re-query.
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"apiKey": api_key})

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    def fetch_cves(self, keyword: str, max_results: int = 20) -> List[Dict]:
        """
        Fetch CVEs from NVD matching a keyword (e.g. 'Modbus', 'Schneider').

        Returns a list of dicts with id, description, cvss_score, severity.
        Results are cached locally (1-hour TTL) so identical queries skip the network.
        Empty results are NOT cached so a later run can retry against the live API.
        """
        import time as _time
        cache_file = self.cache_dir / f"{keyword.replace(' ', '_')}.json"
        cache_ttl  = 3600  # 1 hour in seconds

        if cache_file.exists():
            age = _time.time() - cache_file.stat().st_mtime
            if age < cache_ttl:
                logger.debug(f"Cache hit: {keyword}")
                with open(cache_file) as f:
                    cached = json.load(f)
                if cached:          # only trust non-empty cache entries
                    return cached
                logger.debug(f"Cache has empty result for '{keyword}' — re-querying NVD")

        params = {
            "keywordSearch": keyword,
            "resultsPerPage": min(max_results, 2000),
        }

        try:
            logger.info(f"Querying NVD for: {keyword}")
            resp = self.session.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            cves = self._parse_response(data)

            if cves:   # only cache non-empty responses to avoid poisoning the cache
                with open(cache_file, 'w') as f:
                    json.dump(cves, f, indent=2)

            # NVD rate limit: 5 requests/30s without key, 50/30s with key
            _time.sleep(0.6 if self.api_key else 6.0)
            return cves

        except requests.RequestException as e:
            logger.error(f"NVD API error for '{keyword}': {e}")
            return []

    def _parse_response(self, data: Dict) -> List[Dict]:
        """Parse NVD API v2 response into flat dicts."""
        results = []
        for item in data.get("vulnerabilities", []):
            cve = item.get("cve", {})
            cve_id = cve.get("id", "unknown")

            # Description (English preferred)
            descs = cve.get("descriptions", [])
            description = next(
                (d["value"] for d in descs if d.get("lang") == "en"),
                "No description available"
            )

            # CVSS score — prefer v3.1, fall back to v3.0 then v2
            metrics = cve.get("metrics", {})
            cvss_score = None
            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                entries = metrics.get(key, [])
                if entries:
                    cvss_score = entries[0].get("cvssData", {}).get("baseScore")
                    break

            results.append({
                "cve_id":      cve_id,
                "description": description[:300],
                "cvss_score":  cvss_score,
                "severity":    self._score_to_severity(cvss_score),
                "iec_sl":      self.map_cvss_to_iec_sl(cvss_score),
                "published":   cve.get("published", ""),
            })
        return results

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_severity(score: Optional[float]) -> str:
        if score is None:
            return "UNKNOWN"
        if score >= 9.0:
            return "CRITICAL"
        if score >= 7.0:
            return "HIGH"
        if score >= 4.0:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def map_cvss_to_iec_sl(score: Optional[float]) -> str:
        """Map a CVSS base score to an IEC 62443 Security Level."""
        if score is None:
            return "SL-1"
        for (low, high), sl in CVSS_TO_IEC_SL.items():
            if low <= score <= high:
                return sl
        return "SL-1"

    # ------------------------------------------------------------------
    # Batch enrichment
    # ------------------------------------------------------------------

    def enrich_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Enrich anomaly detection results with CVE data.

        Looks up CVEs for each detected attack pattern (e.g. 'Modbus flooding')
        and appends relevant CVE IDs and the required IEC 62443 SL to counter them.

        Args:
            detections: List of dicts from ICSAttackPatternLibrary.detect_all_patterns()

        Returns:
            Same list with 'cve_enrichment' key added to each detection.
        """
        enriched = []
        for detection in detections:
            pattern = detection.get('pattern', '')
            keyword = self._pattern_to_keyword(pattern)
            cves = self.fetch_cves(keyword, max_results=5) if keyword else []

            detection['cve_enrichment'] = {
                'keyword_searched': keyword,
                'cves_found': len(cves),
                'top_cve': cves[0] if cves else None,
                'required_iec_sl': cves[0]['iec_sl'] if cves else 'SL-2',
            }
            enriched.append(detection)
        return enriched

    @staticmethod
    def _pattern_to_keyword(pattern: str) -> str:
        # Keywords tuned to match NVD's index — overly specific phrases return 0 results.
        # Verified against NVD API v2: broader ICS/protocol terms consistently return CVEs.
        mapping = {
            'modbus_flooding':     'Modbus ICS',
            'plc_scanning':        'Siemens S7 SCADA',
            'unauthorized_writes': 'Modbus unauthorized',
            'protocol_fuzzing':    'industrial protocol',
            'man_in_the_middle':   'SCADA MITM',
            'command_injection':   'ICS remote code execution',
            'replay_attack':       'SCADA replay',
        }
        return mapping.get(pattern, '')

    # ------------------------------------------------------------------
    # ICS vendor CVE summary (resume-friendly demo)
    # ------------------------------------------------------------------

    def fetch_ics_vendor_summary(self) -> Dict:
        """
        Fetch recent CVE counts for major ICS vendors.
        Useful for the IEC 62443 compliance report and SOC dashboard.
        """
        summary = {}
        for vendor in ['Schneider Electric Modicon', 'Yokogawa CENTUM',
                       'Siemens S7', 'Rockwell Allen-Bradley']:
            cves = self.fetch_cves(vendor, max_results=10)
            critical = sum(1 for c in cves if c['severity'] == 'CRITICAL')
            high     = sum(1 for c in cves if c['severity'] == 'HIGH')
            summary[vendor] = {
                'total_cves':    len(cves),
                'critical':      critical,
                'high':          high,
                'max_iec_sl':    max((c['iec_sl'] for c in cves), default='SL-1'),
            }
            logger.info(f"  {vendor}: {len(cves)} CVEs ({critical} critical)")
        return summary

    def print_summary(self, summary: Dict):
        """Print vendor CVE summary in human-readable form."""
        print("\n" + "="*70)
        print("ICS VENDOR CVE SUMMARY (NVD API)")
        print("="*70)
        for vendor, stats in summary.items():
            print(f"\n{vendor}")
            print(f"  Total CVEs:   {stats['total_cves']}")
            print(f"  Critical:     {stats['critical']}")
            print(f"  High:         {stats['high']}")
            print(f"  Required IEC 62443 SL: {stats['max_iec_sl']}")
        print("="*70 + "\n")

    def clear_empty_cache(self):
        """
        Delete any cache files that contain an empty list [].
        Run this once after upgrading to fix results poisoned by previous
        runs that cached empty NVD responses.
        """
        removed = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list) and len(data) == 0:
                    f.unlink()
                    removed += 1
                    logger.info(f"Removed empty cache: {f.name}")
            except Exception:
                pass
        logger.info(f"clear_empty_cache: removed {removed} empty cache file(s)")


if __name__ == "__main__":
    print("="*70)
    print("NVD CVE MAPPER - ICS RISK ENRICHMENT")
    print("="*70)

    mapper = NVDCVEMapper()   # No API key = 5 req/30s limit, still works

    # Clear any empty cache files from previous failed/empty NVD responses
    mapper.clear_empty_cache()

    # Fetch ICS vendor summary
    summary = mapper.fetch_ics_vendor_summary()
    mapper.print_summary(summary)

    # Example: look up Modbus-specific CVEs
    print("Searching for Modbus CVEs...")
    cves = mapper.fetch_cves("Modbus ICS", max_results=5)
    for cve in cves:
        print(f"  {cve['cve_id']} | CVSS {cve['cvss_score']} | "
              f"{cve['severity']} | {cve['iec_sl']}")
