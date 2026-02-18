"""
Simple ICS Data Loader - For Manually Downloaded Datasets
Just extracts and loads CSV files from data/raw/kaggle/

Author: Sadhana Devarajan
"""

from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ManualICSLoader:
    """Loads manually downloaded ICS datasets."""
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw' / 'kaggle'
        logger.info(f"Data directory: {self.data_dir.absolute()}")
    
    def list_available_datasets(self):
        """List what datasets are available."""
        if not self.raw_dir.exists():
            logger.warning(f"Directory not found: {self.raw_dir}")
            logger.info("Create directory and extract datasets there")
            return []
        
        datasets = [d for d in self.raw_dir.iterdir() if d.is_dir()]
        
        if not datasets:
            logger.warning("No datasets found")
            logger.info(f"Extract datasets to: {self.raw_dir}")
            return []
        
        logger.info(f"\n✅ Found {len(datasets)} dataset(s):")
        for dataset_dir in datasets:
            csv_files = list(dataset_dir.glob('*.csv'))
            logger.info(f"   • {dataset_dir.name}: {len(csv_files)} CSV files")
        
        return [d.name for d in datasets]
    
    def load_dataset(self, dataset_name: str, max_files: int = 5):
        """
        Load a dataset from manually extracted files.
        
        Args:
            dataset_name: Folder name (e.g., 'icssim', 'wind_scada')
            max_files: Maximum number of CSV files to load (to save memory)
        """
        dataset_dir = self.raw_dir / dataset_name
        
        if not dataset_dir.exists():
            logger.error(f"Dataset not found: {dataset_name}")
            logger.info(f"Expected location: {dataset_dir}")
            logger.info("\nAvailable datasets:")
            self.list_available_datasets()
            return None
        
        csv_files = list(dataset_dir.glob('*.csv'))
        
        if not csv_files:
            logger.error(f"No CSV files in {dataset_name}")
            return None
        
        logger.info(f"\n📂 Loading: {dataset_name}")
        logger.info(f"   Found {len(csv_files)} CSV files")
        
        # Load first N files
        dfs = []
        for csv_file in csv_files[:max_files]:
            try:
                logger.info(f"   Loading: {csv_file.name}...")
                df = pd.read_csv(csv_file, low_memory=False)
                dfs.append(df)
                logger.info(f"      ✅ {len(df):,} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.warning(f"      ⚠️  Failed: {e}")
        
        if not dfs:
            logger.error("No files could be loaded")
            return None
        
        # Combine
        if len(dfs) == 1:
            combined = dfs[0]
        else:
            logger.info(f"\n   Combining {len(dfs)} files...")
            combined = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"\n✅ LOADED SUCCESSFULLY")
        logger.info(f"   Total rows: {len(combined):,}")
        logger.info(f"   Total columns: {len(combined.columns)}")
        logger.info(f"   Memory: {combined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Show column info
        logger.info(f"\n📊 Column Info:")
        logger.info(f"   Columns: {list(combined.columns)[:10]}")
        if len(combined.columns) > 10:
            logger.info(f"   ... and {len(combined.columns) - 10} more")
        
        return combined
    
    def save_sample(self, df: pd.DataFrame, output_name: str = 'ics_sample.csv', n_rows: int = 10000):
        """Save a sample of the data for quick testing."""
        sample = df.head(n_rows)
        output_path = self.data_dir / 'processed' / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sample.to_csv(output_path, index=False)
        logger.info(f"\n✅ Saved sample: {output_path}")
        logger.info(f"   Sample size: {len(sample):,} rows")
        
        return output_path


if __name__ == "__main__":
    loader = ManualICSLoader(data_dir='./data')
    
    print("\n" + "="*80)
    print("MANUAL ICS DATA LOADER")
    print("="*80)
    print("\nThis loads datasets you've manually downloaded and extracted.")
    print("\nExpected structure:")
    print("   data/raw/kaggle/icssim/*.csv")
    print("   data/raw/kaggle/wind_scada/*.csv")
    print("="*80 + "\n")
    
    # List available
    available = loader.list_available_datasets()
    
    if not available:
        print("\n⚠️  No datasets found!")
        print("\nTo add datasets:")
        print("   1. Download from Kaggle (as ZIP)")
        print("   2. Extract to: data/raw/kaggle/[dataset_name]/")
        print("   3. Run this script again")
    else:
        print("\n" + "="*80)
        dataset_name = input(f"\nWhich dataset to load? ({', '.join(available)}): ").strip()
        
        if dataset_name in available:
            df = loader.load_dataset(dataset_name)
            
            if df is not None:
                print("\n" + "="*80)
                print("DATASET LOADED!")
                print("="*80)
                print("\nSample data:")
                print(df.head().to_string())
                print("\n" + "="*80)
                
                # Save sample
                save = input("\nSave a sample (10K rows) for quick testing? (y/n): ")
                if save.lower() == 'y':
                    loader.save_sample(df)
        else:
            print(f"\n❌ Invalid dataset name: {dataset_name}")