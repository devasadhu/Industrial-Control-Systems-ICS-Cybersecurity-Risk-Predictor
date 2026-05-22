"""
Download Sample PCAP Files for Testing
Gets real network captures from public sources

Author: Sadhana Devarajan
"""

import urllib.request
import ssl
from pathlib import Path

# Disable SSL verification for downloads
ssl._create_default_https_context = ssl._create_unverified_context

def download_sample_pcaps():
    """Download sample PCAP files."""
    
    print("="*70)
    print("DOWNLOADING SAMPLE PCAP FILES")
    print("="*70)
    
    # Create data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample PCAPs from Wireshark wiki
    samples = {
        'sample.pcap': 'https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/http.cap',
        'modbus.pcap': 'https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/modbus.pcap',
        'dns.pcap': 'https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/dns.cap'
    }
    
    for filename, url in samples.items():
        output_path = data_dir / filename
        
        if output_path.exists():
            print(f"✅ {filename} already exists - skipping")
            continue
        
        try:
            print(f"\n📥 Downloading {filename}...")
            print(f"   URL: {url}")
            
            urllib.request.urlretrieve(url, output_path)
            
            # Check file size
            size_kb = output_path.stat().st_size / 1024
            print(f"   ✅ Downloaded {filename} ({size_kb:.1f} KB)")
            
        except Exception as e:
            print(f"   ❌ Failed to download {filename}: {e}")
    
    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE")
    print("="*70)
    print("\nAvailable PCAP files:")
    for pcap in data_dir.glob("*.pcap"):
        size_kb = pcap.stat().st_size / 1024
        print(f"  • {pcap.name} ({size_kb:.1f} KB)")
    
    print("\nNow run:")
    print("  python src/pcap/pcap_processor.py")
    print("="*70)


if __name__ == "__main__":
    download_sample_pcaps()