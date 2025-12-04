"""
FIXED: Irrigation Label Generator for Karmana, Uzbekistan
Combines soil moisture tracking and irrigation decisions in ONE loop
"""

import pandas as pd
import numpy as np
from pathlib import Path


class IrrigationLabelGenerator:
    """
    Generate irrigation labels for cotton in Karmana district
    Based on FAO-56 + Uzbekistan research (70-70-60% rule)
    """
    
    def __init__(self):
        """Initialize with Karmana-specific parameters"""
        # Soil parameters
        self.field_capacity = 0.30
        self.wilting_point = 0.12
        self.root_depth = {1: 0.5, 2: 0.8, 3: 0.7}
        
        # Irrigation thresholds (70-70-60% rule)
        self.irrigation_thresholds = {
            1: 0.90,
            2: 0.90,
            3: 0.85
        }  
        # System parameters
        self.irrigation_efficiency = 0.60
        self.flow_rate = 0.40
        self.field_area_ha = 1.0
        
        # Limits
        self.min_irrigation_mm = 30
        self.max_irrigation_mm = 100

        # Weather thresholds
        self.rain_threshold_mm = 30    # Very high (only block for heavy rain)
        self.temp_threshold_c = 5      # Very low (almost never block)
        self.max_irrigation_percent = 100
    
    def calculate_etc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate crop evapotranspiration"""
        print("  🌡️  Calculating crop evapotranspiration (ETc)...")
        df['etc_mm_day'] = df['et0_estimate'] * df['crop_kc']
        df['etc_mm_hour'] = df['etc_mm_day'] / 24
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIXED: Combined soil moisture tracking and irrigation decisions
        Processes in ONE loop to avoid irrigation not affecting next hour
        """
        print("\n" + "="*80)
        print("🏷️  GENERATING IRRIGATION LABELS (FIXED VERSION)")
        print("   Method: FAO-56 + Uzbekistan Research (70-70-60% rule)")
        print("="*80)
        
        # Calculate ETc
        df = self.calculate_etc(df)
        
        # Initialize columns
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['soil_moisture'] = self.field_capacity
        df['irrigation_applied_mm'] = 0.0
        df['irrigation_needed'] = 0
        df['recommended_water_mm'] = 0.0
        df['recommended_water_percent'] = 0.0
        df['irrigation_time_min'] = 0.0
        
        print("  💧 Tracking soil moisture and generating decisions...")
        
        # Track for each year separately
        for year in df['year'].unique():
            year_mask = df['year'] == year
            
            if not year_mask.any():
                continue
            
            # Start of growing season
            start_idx = df[year_mask & (df['crop_stage'] > 0)].index.min()
            
            if pd.isna(start_idx):
                continue
            
            # Initialize at field capacity
            df.loc[start_idx, 'soil_moisture'] = self.field_capacity
            
            # Get all indices for this growing season
            season_indices = df[year_mask & (df['crop_stage'] > 0)].index.tolist()
            
            # Track moisture hour by hour
            for i in range(len(season_indices)):
                idx = season_indices[i]
                
                if i == 0:
                    continue  # Already initialized
                
                prev_idx = season_indices[i-1]
                
                # Get current state
                stage = int(df.loc[idx, 'crop_stage'])
                root_depth_m = self.root_depth.get(stage, 0.7)
                prev_moisture = df.loc[prev_idx, 'soil_moisture']
                
                # Water inputs from previous hour
                rain_mm = df.loc[idx, 'precipitation']
                rain_input = rain_mm / (root_depth_m * 1000)
                
                # CRITICAL: Get irrigation from PREVIOUS hour (not current)
                prev_irrigation_mm = df.loc[prev_idx, 'irrigation_applied_mm']
                irrigation_input = prev_irrigation_mm / (root_depth_m * 1000)
                
                # Water loss (ETc from current hour)
                etc_mm = df.loc[idx, 'etc_mm_hour']
                etc_loss = etc_mm / (root_depth_m * 1000)
                
                # Calculate new moisture
                new_moisture = prev_moisture + rain_input + irrigation_input - etc_loss
                
                # Cap at field capacity
                if new_moisture > self.field_capacity:
                    new_moisture = self.field_capacity
                
                # Can't go below wilting point
                if new_moisture < self.wilting_point:
                    new_moisture = self.wilting_point
                
                # Update moisture BEFORE making irrigation decision
                df.loc[idx, 'soil_moisture'] = new_moisture
                
                # NOW make irrigation decision based on CURRENT moisture
                threshold_pct = self.irrigation_thresholds.get(stage, 0.70)
                threshold_moisture = self.field_capacity * threshold_pct
                
                if new_moisture < threshold_moisture:
                    # Check blocking conditions
                    recent_rain = df.loc[idx, 'precip_24h_sum']
                    temp = df.loc[idx, 'temp_24h_mean']
                    hour = df.loc[idx, 'hour']
                    
                    # Don't irrigate if:
                    if recent_rain > self.rain_threshold_mm:
                        continue
                    if temp < self.temp_threshold_c:
                        continue
                    if hour < 4 or hour > 18:
                        continue
                    
                    # Calculate water needed
                    moisture_deficit = self.field_capacity - new_moisture
                    water_deficit_mm = moisture_deficit * root_depth_m * 1000
                    water_to_apply_mm = water_deficit_mm / self.irrigation_efficiency
                    
                    # Apply limits
                    if water_to_apply_mm < self.min_irrigation_mm:
                        continue
                    
                    water_to_apply_mm = min(water_to_apply_mm, self.max_irrigation_mm)
                    
                    # Convert to percentage and time
                    water_percent = (water_to_apply_mm / 100) * 100
                    water_percent = min(water_percent, self.max_irrigation_percent)
                    
                    irrigation_time = water_to_apply_mm * 2.5
                    irrigation_time = max(60, min(irrigation_time, 300))
                    
                    # Record irrigation for THIS hour
                    # (will be applied in NEXT hour's moisture calculation)
                    df.loc[idx, 'irrigation_needed'] = 1
                    df.loc[idx, 'recommended_water_mm'] = round(water_to_apply_mm, 1)
                    df.loc[idx, 'recommended_water_percent'] = round(water_percent, 1)
                    df.loc[idx, 'irrigation_time_min'] = round(irrigation_time, 0)
                    df.loc[idx, 'irrigation_applied_mm'] = water_to_apply_mm
        
        # Print statistics
        self._print_statistics(df)
        
        print("\n✅ LABEL GENERATION COMPLETE")
        
        return df
    
    def _print_statistics(self, df: pd.DataFrame):
        """Print label statistics"""
        growing_season = df[df['crop_stage'] > 0]
        irrigation_events = df[df['irrigation_needed'] == 1]
        
        print("\n" + "="*80)
        print("📊 IRRIGATION LABEL STATISTICS")
        print("="*80)
        
        print(f"Total records: {len(df):,}")
        print(f"Growing season records: {len(growing_season):,}")
        print(f"Irrigation events: {len(irrigation_events):,}")
        print(f"Irrigation frequency: {len(irrigation_events) / max(len(growing_season), 1) * 100:.2f}% of growing season")
        
        if len(irrigation_events) > 0:
            print(f"\nIrrigation amounts:")
            print(f"  Mean: {irrigation_events['recommended_water_mm'].mean():.1f} mm")
            print(f"  Median: {irrigation_events['recommended_water_mm'].median():.1f} mm")
            print(f"  Min: {irrigation_events['recommended_water_mm'].min():.1f} mm")
            print(f"  Max: {irrigation_events['recommended_water_mm'].max():.1f} mm")
            
            print(f"\nIrrigation duration:")
            print(f"  Mean: {irrigation_events['irrigation_time_min'].mean():.0f} minutes")
            print(f"  Median: {irrigation_events['irrigation_time_min'].median():.0f} minutes")
            
            print(f"\nBy crop stage:")
            for stage in [1, 2, 3]:
                stage_irrig = irrigation_events[irrigation_events['crop_stage'] == stage]
                if len(stage_irrig) > 0:
                    print(f"  Stage {stage}: {len(stage_irrig)} events, "
                          f"avg {stage_irrig['recommended_water_mm'].mean():.1f} mm")
        
        print("="*80)


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/processed/karmana_features.csv", parse_dates=['timestamp'])
    
    generator = IrrigationLabelGenerator()
    df_labeled = generator.generate_labels(df)
    
    # Save
    df_labeled.to_csv("data/processed/karmana_labeled.csv", index=False)
    print("\n💾 Saved: data/processed/karmana_labeled.csv")