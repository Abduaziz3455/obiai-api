"""
Feature Engineering for Irrigation Predictor
Creates derived features from weather data
"""

import pandas as pd
import numpy as np
from pathlib import Path


class FeatureEngineer:
    """
    Engineer features for irrigation prediction model
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        pass
    
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window aggregations
        
        Args:
            df: DataFrame with timestamp and weather data
            
        Returns:
            DataFrame with rolling features
        """
        print("\n📈 Creating rolling aggregations...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Temperature features
        print("  • Temperature aggregations...")
        df['temp_24h_mean'] = df['temperature_2m'].rolling(window=24, min_periods=1).mean()
        df['temp_24h_max'] = df['temperature_2m'].rolling(window=24, min_periods=1).max()
        df['temp_24h_min'] = df['temperature_2m'].rolling(window=24, min_periods=1).min()
        df['temp_7d_mean'] = df['temperature_2m'].rolling(window=24*7, min_periods=1).mean()
        df['temp_14d_mean'] = df['temperature_2m'].rolling(window=24*14, min_periods=1).mean()
        df['temp_7d_max'] = df['temperature_2m'].rolling(window=24*7, min_periods=1).max()
        df['temp_7d_min'] = df['temperature_2m'].rolling(window=24*7, min_periods=1).min()
        
        # Precipitation features
        print("  • Precipitation aggregations...")
        df['precip_24h_sum'] = df['precipitation'].rolling(window=24, min_periods=1).sum()
        df['precip_7d_sum'] = df['precipitation'].rolling(window=24*7, min_periods=1).sum()
        df['precip_14d_sum'] = df['precipitation'].rolling(window=24*14, min_periods=1).sum()
        df['precip_30d_sum'] = df['precipitation'].rolling(window=24*30, min_periods=1).sum()
        
        # Radiation features
        print("  • Radiation aggregations...")
        df['radiation_24h_mean'] = df['shortwave_radiation'].rolling(window=24, min_periods=1).mean()
        df['radiation_7d_mean'] = df['shortwave_radiation'].rolling(window=24*7, min_periods=1).mean()
        df['radiation_24h_sum'] = df['shortwave_radiation'].rolling(window=24, min_periods=1).sum()
        
        # Wind features
        print("  • Wind aggregations...")
        df['wind_24h_mean'] = df['wind_speed_10m'].rolling(window=24, min_periods=1).mean()
        df['wind_7d_mean'] = df['wind_speed_10m'].rolling(window=24*7, min_periods=1).mean()
        
        # Soil temperature features
        if 'soil_temperature_0_to_7cm' in df.columns:
            print("  • Soil temperature aggregations...")
            df['soil_temp_24h_mean'] = df['soil_temperature_0_to_7cm'].rolling(window=24, min_periods=1).mean()
            df['soil_temp_7d_mean'] = df['soil_temperature_0_to_7cm'].rolling(window=24*7, min_periods=1).mean()
        
        print(f"✅ Added {26} rolling features")
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived/calculated features
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with derived features
        """
        print("\n🧮 Creating derived features...")
        
        # Temperature differences
        df['temp_day_night_diff'] = df['temp_24h_max'] - df['temp_24h_min']
        df['temp_change_24h'] = df['temperature_2m'] - df['temperature_2m'].shift(24)
        
        # Days since significant rain (>5mm)
        df['significant_rain'] = (df['precipitation'] > 5).astype(int)
        df['days_since_rain'] = 0.0
        
        rain_indices = df[df['significant_rain'] == 1].index
        for i in range(len(df)):
            if df.loc[i, 'significant_rain'] == 1:
                df.loc[i, 'days_since_rain'] = 0
            elif i > 0:
                df.loc[i, 'days_since_rain'] = df.loc[i-1, 'days_since_rain'] + (1/24)  # Hours to days
        
        # Consecutive dry days (no rain in 24h)
        df['is_dry_day'] = (df['precip_24h_sum'] < 1).astype(int)
        df['consecutive_dry_days'] = df['is_dry_day'].groupby(
            (df['is_dry_day'] != df['is_dry_day'].shift()).cumsum()
        ).cumsum() / 24  # Convert hours to days
        
        # Hot dry days count (temp > 30°C and no rain in last 7 days)
        df['is_hot_dry'] = ((df['temp_24h_mean'] > 30) & (df['precip_7d_sum'] < 5)).astype(int)
        df['hot_dry_days_7d'] = df['is_hot_dry'].rolling(window=24*7, min_periods=1).sum() / 24
        
        # Evapotranspiration estimate (simplified)
        # Based on temperature and radiation
        df['et0_estimate'] = 1.5 * 0.0023 * (df['temp_24h_mean'] + 17.8) * \
                            np.sqrt(df['temp_day_night_diff']) * \
                            (df['radiation_24h_mean'] / 41.6)
        
        # Vapor pressure deficit estimate (affects plant water stress)
        # Higher VPD = more water loss
        df['vpd_estimate'] = 0.611 * np.exp(
            (17.27 * df['temperature_2m']) / (df['temperature_2m'] + 237.3)
        ) * 0.5  # Simplified, assuming 50% humidity
        
        # Growing Degree Days (GDD) for cotton (base 15°C)
        df['gdd_daily'] = np.maximum(df['temp_24h_mean'] - 15, 0)
        df['gdd_cumulative'] = 0.0
        
        print(f"✅ Added {11} derived features")
        
        return df
    
    def add_cotton_features(self, df: pd.DataFrame, planting_date: str = "04-15") -> pd.DataFrame:
        """
        Add cotton-specific features
        
        Args:
            df: DataFrame with timestamp
            planting_date: Cotton planting date (MM-DD format)
            
        Returns:
            DataFrame with cotton features
        """
        print("\n🌱 Creating cotton-specific features...")
        
        # For each year, calculate days since planting
        df['days_since_planting'] = -1
        df['crop_stage'] = 0
        df['crop_kc'] = 0.0
        
        for year in df['year'].unique():
            # Planting date for this year
            plant_date = pd.to_datetime(f"{year}-{planting_date}")
            harvest_date = plant_date + pd.Timedelta(days=160)
            
            # Filter data for this growing season
            season_mask = (df['timestamp'] >= plant_date) & (df['timestamp'] <= harvest_date)
            
            if season_mask.sum() > 0:
                # Calculate days since planting
                df.loc[season_mask, 'days_since_planting'] = (
                    (df.loc[season_mask, 'timestamp'] - plant_date).dt.total_seconds() / 86400
                ).astype(int)
                
                # Determine crop stage
                # Stage 1: Germination-Flowering (0-60 days)
                # Stage 2: Flowering-Boll (60-120 days)  
                # Stage 3: Maturation (120-160 days)
                stage1_mask = season_mask & (df['days_since_planting'] < 60)
                stage2_mask = season_mask & (df['days_since_planting'] >= 60) & (df['days_since_planting'] < 120)
                stage3_mask = season_mask & (df['days_since_planting'] >= 120)
                
                df.loc[stage1_mask, 'crop_stage'] = 1
                df.loc[stage2_mask, 'crop_stage'] = 2
                df.loc[stage3_mask, 'crop_stage'] = 3
                
                # Calculate crop coefficient (Kc) - linear interpolation within stages
                # Stage 1: 0.35 → 0.80
                df.loc[stage1_mask, 'crop_kc'] = 0.35 + (df.loc[stage1_mask, 'days_since_planting'] / 60) * (0.80 - 0.35)
                
                # Stage 2: 0.80 → 1.15
                df.loc[stage2_mask, 'crop_kc'] = 0.80 + ((df.loc[stage2_mask, 'days_since_planting'] - 60) / 60) * (1.15 - 0.80)
                
                # Stage 3: 1.15 → 0.50
                df.loc[stage3_mask, 'crop_kc'] = 1.15 - ((df.loc[stage3_mask, 'days_since_planting'] - 120) / 40) * (1.15 - 0.50)
        
        # Critical water period flag (flowering to boll formation)
        df['is_critical_period'] = (df['crop_stage'] == 2).astype(int)
        
        # Calculate GDD cumulative during growing season
        for year in df['year'].unique():
            plant_date = pd.to_datetime(f"{year}-{planting_date}")
            season_mask = df['timestamp'] >= plant_date
            
            if season_mask.sum() > 0:
                df.loc[season_mask, 'gdd_cumulative'] = df.loc[season_mask, 'gdd_daily'].cumsum()
        
        print(f"✅ Added {5} cotton features")
        
        return df
    
    def engineer_all(self, df: pd.DataFrame, planting_date: str = "04-15") -> pd.DataFrame:
        """
        Run complete feature engineering pipeline
        
        Args:
            df: Cleaned DataFrame
            planting_date: Cotton planting date (MM-DD)
            
        Returns:
            DataFrame with all engineered features
        """
        print("\n" + "="*80)
        print("⚙️  FEATURE ENGINEERING PIPELINE")
        print("="*80)
        
        initial_cols = len(df.columns)
        
        # Add rolling features
        df = self.add_rolling_features(df)
        
        # Add derived features
        df = self.add_derived_features(df)
        
        # Add cotton-specific features
        df = self.add_cotton_features(df, planting_date)
        
        final_cols = len(df.columns)
        new_features = final_cols - initial_cols
        
        print("\n" + "="*80)
        print(f"✅ FEATURE ENGINEERING COMPLETE")
        print(f"   Added {new_features} new features")
        print(f"   Total columns: {final_cols}")
        print("="*80)
        
        return df


if __name__ == "__main__":
    # Example usage
    from data_processor import DataProcessor
    
    # Load cleaned data
    df = pd.read_csv("data/processed/karmana_cleaned.csv", parse_dates=['timestamp'])
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_all(df, planting_date="04-15")
    
    # Save
    df_features.to_csv("data/processed/karmana_features.csv", index=False)
    print("\n💾 Saved: data/processed/karmana_features.csv")
