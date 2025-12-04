import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
pred = pd.read_csv('predictions_hybrid.csv')
df = pd.read_csv('data/processed/karmana_ml_ready.csv')

# Get irrigation cases
irrigation = pred[pred['irrigation_needed'] == 1].copy()
irrigation_idx = irrigation.index

# Add context from original data
irrigation['crop_stage'] = df.loc[irrigation_idx, 'crop_stage'].values
irrigation['month'] = df.loc[irrigation_idx, 'month'].values if 'month' in df.columns else pd.to_datetime(df.loc[irrigation_idx, 'timestamp']).dt.month
irrigation['temp_7d'] = df.loc[irrigation_idx, 'temp_7d_mean'].values
irrigation['days_since_rain'] = df.loc[irrigation_idx, 'days_since_rain'].values

print("="*70)
print("🌾 IRRIGATION PREDICTION VALIDATION")
print("="*70)

print(f"\n📊 OVERALL STATISTICS:")
print(f"Total predictions: {len(pred):,}")
print(f"Irrigation events: {len(irrigation):,} ({len(irrigation)/len(pred)*100:.2f}%)")
print(f"No irrigation: {(pred['irrigation_needed']==0).sum():,} ({(pred['irrigation_needed']==0).sum()/len(pred)*100:.2f}%)")

print(f"\n💧 WATER AMOUNTS:")
print(f"Mean: {irrigation['recommended_water_percent'].mean():.1f}%")
print(f"Std: {irrigation['recommended_water_percent'].std():.1f}%")
print(f"Min: {irrigation['recommended_water_percent'].min():.1f}%")
print(f"Max: {irrigation['recommended_water_percent'].max():.1f}%")

print(f"\n⏱️  DURATIONS:")
print(f"Mean: {irrigation['irrigation_time_min'].mean():.0f} minutes")
print(f"Std: {irrigation['irrigation_time_min'].std():.0f} minutes")
print(f"Min: {irrigation['irrigation_time_min'].min():.0f} minutes")
print(f"Max: {irrigation['irrigation_time_min'].max():.0f} minutes")

print(f"\n🌱 BY CROP STAGE:")
for stage in [1, 2, 3]:
    stage_irrig = irrigation[irrigation['crop_stage'] == stage]
    if len(stage_irrig) > 0:
        print(f"Stage {stage}: {len(stage_irrig)} events, avg {stage_irrig['recommended_water_percent'].mean():.1f}%")

print(f"\n📅 BY MONTH:")
for month in sorted(irrigation['month'].unique()):
    month_irrig = irrigation[irrigation['month'] == month]
    print(f"Month {month}: {len(month_irrig)} events, avg {month_irrig['recommended_water_percent'].mean():.1f}%")

print(f"\n🌡️  BY TEMPERATURE:")
hot = irrigation[irrigation['temp_7d'] > 30]
warm = irrigation[(irrigation['temp_7d'] >= 25) & (irrigation['temp_7d'] <= 30)]
cool = irrigation[irrigation['temp_7d'] < 25]
print(f"Hot (>30°C): {len(hot)} events, avg {hot['recommended_water_percent'].mean():.1f}%" if len(hot) > 0 else "Hot: 0 events")
print(f"Warm (25-30°C): {len(warm)} events, avg {warm['recommended_water_percent'].mean():.1f}%" if len(warm) > 0 else "Warm: 0 events")
print(f"Cool (<25°C): {len(cool)} events, avg {cool['recommended_water_percent'].mean():.1f}%" if len(cool) > 0 else "Cool: 0 events")

# Plot distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Water % distribution
axes[0, 0].hist(irrigation['recommended_water_percent'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Water %')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Irrigation Amounts')
axes[0, 0].axvline(irrigation['recommended_water_percent'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].legend()

# Duration distribution
axes[0, 1].hist(irrigation['irrigation_time_min'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Duration (minutes)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Irrigation Duration')
axes[0, 1].axvline(irrigation['irrigation_time_min'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 1].legend()

# By month
month_counts = irrigation.groupby('month').size()
axes[1, 0].bar(month_counts.index, month_counts.values, edgecolor='black')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Number of Irrigation Events')
axes[1, 0].set_title('Irrigation Events by Month')
axes[1, 0].set_xticks(range(1, 13))

# By crop stage
stage_counts = irrigation.groupby('crop_stage').size()
stage_amounts = irrigation.groupby('crop_stage')['recommended_water_percent'].mean()
ax2 = axes[1, 1]
ax2.bar(stage_counts.index, stage_counts.values, alpha=0.7, label='Count', edgecolor='black')
ax2.set_xlabel('Crop Stage')
ax2.set_ylabel('Number of Events', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_title('Irrigation by Crop Stage')
ax2.legend(loc='upper left')

ax2_twin = ax2.twinx()
ax2_twin.plot(stage_amounts.index, stage_amounts.values, 'r-o', linewidth=2, markersize=8, label='Avg Water %')
ax2_twin.set_ylabel('Average Water %', color='red')
ax2_twin.tick_params(axis='y', labelcolor='red')
ax2_twin.legend(loc='upper right')

plt.tight_layout()
plt.savefig('irrigation_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot: irrigation_analysis.png")
plt.show()

print("\n" + "="*70)