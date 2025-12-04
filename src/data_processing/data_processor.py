"""
Data Processor for Irrigation Predictor
Merges JSON files, cleans data, and prepares for feature engineering
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Process raw JSON weather data into clean CSV format
    """
    
    def __init__(self, location_id: str = "karmana"):
        """
        Initialize data processor
        
        Args:
            location_id: Location identifier
        """
        self.location_id = location_id
        self.raw_dir = Path(f"data/raw/historical/{location_id}")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_json_files(self) -> List[Dict]:
        """
        Load all JSON files from raw directory
        
        Returns:
            List of data dictionaries
        """
        print(f"📂 Loading JSON files from: {self.raw_dir}")
        
        json_files = sorted(self.raw_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.raw_dir}")
        
        data_list = []
        for file_path in json_files:
            print(f"  ✓ Loading: {file_path.name}")
            with open(file_path, 'r') as f:
                data = json.load(f)
                data_list.append(data)
        
        print(f"✅ Loaded {len(data_list)} files")
        return data_list
    
    def merge_to_dataframe(self, data_list: List[Dict]) -> pd.DataFrame:
        """
        Merge JSON data into single DataFrame
        
        Args:
            data_list: List of JSON data dictionaries
            
        Returns:
            Merged DataFrame
        """
        print("\n🔄 Merging data into DataFrame...")
        
        all_dfs = []
        
        for data in data_list:
            # Extract hourly data
            hourly = data.get('hourly', {})
            
            # Create DataFrame
            df = pd.DataFrame(hourly)
            
            # Add metadata
            df['latitude'] = data.get('latitude')
            df['longitude'] = data.get('longitude')
            df['elevation'] = data.get('elevation')
            
            all_dfs.append(df)
        
        # Concatenate all DataFrames
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        print(f"✅ Merged shape: {merged_df.shape}")
        return merged_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("\n🧹 Cleaning data...")
        
        initial_rows = len(df)
        
        # 1. Convert timestamp
        df['time'] = pd.to_datetime(df['time'])
        df = df.rename(columns={'time': 'timestamp'})
        
        # 2. Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 3. Remove duplicates
        duplicates = df.duplicated(subset=['timestamp'], keep='first').sum()
        if duplicates > 0:
            print(f"  ⚠️  Removing {duplicates} duplicate timestamps")
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # 4. Handle missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"  ⚠️  Found {missing_count} missing values")
            
            # Interpolate weather parameters (continuous variables)
            weather_cols = ['temperature_2m', 'wind_speed_10m', 
                          'shortwave_radiation', 'soil_temperature_0_to_7cm']
            
            for col in weather_cols:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear', limit=6)
            
            # Precipitation: forward fill (rain continues)
            if 'precipitation' in df.columns:
                df['precipitation'] = df['precipitation'].fillna(0)  # No data = no rain
        
        # 5. Validate value ranges
        df = self._validate_ranges(df)
        
        # 6. Create data quality flags
        df['data_quality'] = 'good'
        df.loc[df.isnull().any(axis=1), 'data_quality'] = 'missing'
        
        final_rows = len(df)
        print(f"✅ Cleaned: {initial_rows} → {final_rows} rows")
        
        return df
    
    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and flag outliers
        """
        print("  🔍 Validating value ranges...")
        
        # Define valid ranges for Karmana, Uzbekistan
        ranges = {
            'temperature_2m': (-30, 50),          # °C (winter can be cold, summer hot)
            'precipitation': (0, 100),            # mm/hour (max reasonable)
            'wind_speed_10m': (0, 150),          # km/h
            'shortwave_radiation': (0, 1500),    # W/m²
            'soil_temperature_0_to_7cm': (-20, 60)  # °C
        }
        
        outlier_count = 0
        
        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                outliers = (df[col] < min_val) | (df[col] > max_val)
                if outliers.sum() > 0:
                    print(f"    ⚠️  {col}: {outliers.sum()} outliers")
                    # Cap outliers to valid range
                    df.loc[df[col] < min_val, col] = min_val
                    df.loc[df[col] > max_val, col] = max_val
                    outlier_count += outliers.sum()
        
        if outlier_count > 0:
            print(f"  ✓ Fixed {outlier_count} outliers")
        
        return df
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic time-based features
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with basic features
        """
        print("\n⚙️  Adding basic features...")
        
        # Time features
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Season (Uzbekistan)
        # Winter: Dec-Feb (12,1,2), Spring: Mar-May (3,4,5), 
        # Summer: Jun-Aug (6,7,8), Fall: Sep-Nov (9,10,11)
        df['season'] = df['month'].map({
            12: 1, 1: 1, 2: 1,      # Winter
            3: 2, 4: 2, 5: 2,        # Spring
            6: 3, 7: 3, 8: 3,        # Summer
            9: 4, 10: 4, 11: 4       # Fall
        })
        
        # Is daytime (6am - 8pm)
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 20)).astype(int)
        
        # Cotton growing season (April-September)
        df['is_growing_season'] = df['month'].isin([4, 5, 6, 7, 8, 9]).astype(int)
        
        print(f"✅ Added {8} basic features")
        
        return df
    
    def process_all(self) -> pd.DataFrame:
        """
        Run complete processing pipeline
        
        Returns:
            Processed DataFrame
        """
        print("="*80)
        print("🚀 STARTING DATA PROCESSING PIPELINE")
        print("="*80)
        
        # Load JSON files
        data_list = self.load_json_files()
        
        # Merge to DataFrame
        df = self.merge_to_dataframe(data_list)
        
        # Clean data
        df = self.clean_data(df)
        
        # Add basic features
        df = self.add_basic_features(df)
        
        # Save intermediate result
        output_file = self.processed_dir / f"{self.location_id}_cleaned.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved cleaned data: {output_file}")
        
        # Print summary
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print data summary"""
        print("\n" + "="*80)
        print("📊 DATA SUMMARY")
        print("="*80)
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
        print(f"Columns: {len(df.columns)}")
        print(f"\nMissing values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  ✅ No missing values!")
        print("="*80)


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor("karmana")
    df = processor.process_all()
    print("\n✅ Processing complete!")
