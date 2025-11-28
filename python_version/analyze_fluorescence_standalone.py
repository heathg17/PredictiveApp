"""
Fluorescence Analysis - Standalone

Since OPTI sample names don't match Concentrations.csv, this analyzes:
1. Area under reflectance curve above R=1 (fluorescence indicator)
2. Comparison of raw vs PP-background-subtracted
3. Creates sample mapping guide for user to add concentration data

User can manually map OPTI samples to formulations in Concentrations.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz
import os

# Configuration
OPTISPECTRA_PATH = '/Users/GoergeH/Documents/OptiSpectra.csv'

# PP Background reflectance
PP_BACKGROUND = np.array([
    6.32230798e-1, 8.24865798e-1, 9.64723766e-1, 9.74486450e-1, 9.75640754e-1,
    9.78552401e-1, 9.75615660e-1, 9.77584740e-1, 9.89334762e-1, 9.98248855e-1,
    1.00193385e+0, 1.00288971e+0, 1.00431999e+0, 1.00225039e+0, 1.00409655e+0,
    1.00195293e+0, 1.00040476e+0, 1.00163217e+0, 1.00373669e+0, 1.00462874e+0,
    1.00526512e+0, 1.00307862e+0, 9.99236067e-1, 9.90266760e-1, 9.87492502e-1,
    9.90339776e-1, 9.88663038e-1, 9.86457149e-1, 9.83821054e-1, 9.81393794e-1,
    9.78041152e-1
])

WAVELENGTHS = np.arange(400, 710, 10)

print("=" * 80)
print("FLUORESCENCE ANALYSIS - STANDALONE")
print("=" * 80)
print("\nReading OptiSpectra.csv...")

# Read CSV
df = pd.read_csv(OPTISPECTRA_PATH, sep=';', skiprows=3)

results = []

for idx, row in df.iterrows():
    user_color_name = str(row['User Color Name']).strip()

    # Skip PP background
    if 'SUB' in user_color_name or 'PP' in user_color_name:
        continue

    # Infer thickness
    if user_color_name.endswith('H'):
        thickness = 12.0
        base_name = user_color_name[:-1]
    else:
        thickness = 8.0
        base_name = user_color_name

    # Extract reflectance
    reflectance_cols = [col for col in df.columns if col.startswith('R') and 'nm' in col]
    reflectance = np.array([float(row[col]) for col in reflectance_cols])

    # Background-subtracted
    reflectance_sub = reflectance - PP_BACKGROUND

    # Calculate area above R=1 (RAW)
    above_one_raw = reflectance - 1.0
    above_one_raw[above_one_raw < 0] = 0
    area_raw = trapz(above_one_raw, WAVELENGTHS)

    # Calculate area above R=0 (BACKGROUND-SUBTRACTED)
    above_zero_sub = reflectance_sub.copy()
    above_zero_sub[above_zero_sub < 0] = 0
    area_sub = trapz(above_zero_sub, WAVELENGTHS)

    # Max reflectance
    max_R_raw = np.max(reflectance)
    max_R_sub = np.max(reflectance_sub)

    # Peak wavelength (where R > 1)
    if area_raw > 0:
        peak_idx_raw = np.argmax(reflectance)
        peak_wl_raw = WAVELENGTHS[peak_idx_raw]
    else:
        peak_wl_raw = 0

    if area_sub > 0:
        peak_idx_sub = np.argmax(reflectance_sub)
        peak_wl_sub = WAVELENGTHS[peak_idx_sub]
    else:
        peak_wl_sub = 0

    # CIELAB
    L = float(row['L']) if pd.notna(row['L']) else np.nan
    a = float(row['a']) if pd.notna(row['a']) else np.nan
    b = float(row['b']) if pd.notna(row['b']) else np.nan

    results.append({
        'Sample': user_color_name,
        'Base_Name': base_name,
        'Thickness_um': thickness,
        'L': L,
        'a': a,
        'b': b,
        'Area_Above_R1_Raw': area_raw,
        'Area_Above_R0_Sub': area_sub,
        'Max_R_Raw': max_R_raw,
        'Max_R_Sub': max_R_sub,
        'Peak_WL_Raw_nm': peak_wl_raw,
        'Peak_WL_Sub_nm': peak_wl_sub,
        'Reflectance': reflectance.tolist(),
        'Reflectance_Sub': reflectance_sub.tolist()
    })

results_df = pd.DataFrame(results)

print(f"✓ Analyzed {len(results_df)} samples")

# Statistics
print("\n" + "=" * 80)
print("FLUORESCENCE STATISTICS")
print("=" * 80)

print("\n### RAW REFLECTANCE (Area above R=1) ###")
print(f"Mean: {results_df['Area_Above_R1_Raw'].mean():.4f}")
print(f"Std:  {results_df['Area_Above_R1_Raw'].std():.4f}")
print(f"Min:  {results_df['Area_Above_R1_Raw'].min():.4f}")
print(f"Max:  {results_df['Area_Above_R1_Raw'].max():.4f}")

print("\n### BACKGROUND-SUBTRACTED (Area above R=0) ###")
print(f"Mean: {results_df['Area_Above_R0_Sub'].mean():.4f}")
print(f"Std:  {results_df['Area_Above_R0_Sub'].std():.4f}")
print(f"Min:  {results_df['Area_Above_R0_Sub'].min():.4f}")
print(f"Max:  {results_df['Area_Above_R0_Sub'].max():.4f}")

# Comparison
print("\n### COMPARISON: Raw vs Background-Subtracted ###")
ratio = results_df['Area_Above_R0_Sub'].mean() / results_df['Area_Above_R1_Raw'].mean()
print(f"Area ratio (Sub/Raw): {ratio:.3f}")

if ratio > 1.5:
    print("✓ Background subtraction INCREASES fluorescence signal significantly (+{:.0f}%)".format((ratio-1)*100))
elif ratio > 1.1:
    print("✓ Background subtraction slightly increases fluorescence signal (+{:.0f}%)".format((ratio-1)*100))
else:
    print("→ Background subtraction has minimal effect")

# Sort by fluorescence intensity
results_df_sorted = results_df.sort_values('Area_Above_R1_Raw', ascending=False)

print("\n### TOP 10 SAMPLES BY FLUORESCENCE (Raw) ###")
print(results_df_sorted[['Sample', 'Thickness_um', 'Area_Above_R1_Raw', 'Max_R_Raw', 'Peak_WL_Raw_nm']].head(10).to_string(index=False))

# Visualization
os.makedirs('results/fluorescence', exist_ok=True)

# Plot 1: Comparison of raw vs subtracted
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Area comparison
axes[0, 0].scatter(results_df['Area_Above_R1_Raw'], results_df['Area_Above_R0_Sub'],
                   c=results_df['Thickness_um'], cmap='viridis', s=100, alpha=0.7, edgecolors='black')
axes[0, 0].plot([0, results_df['Area_Above_R1_Raw'].max()],
                [0, results_df['Area_Above_R1_Raw'].max()], 'r--', label='y=x')
axes[0, 0].set_xlabel('Area Above R=1 (Raw)', fontsize=11)
axes[0, 0].set_ylabel('Area Above R=0 (Background-Sub)', fontsize=11)
axes[0, 0].set_title('Fluorescence Area: Raw vs Background-Subtracted', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Histogram of raw areas
axes[0, 1].hist(results_df['Area_Above_R1_Raw'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Area Above R=1 (Raw)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Distribution of Fluorescence (Raw)', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Thickness effect
thick_8 = results_df[results_df['Thickness_um'] == 8.0]
thick_12 = results_df[results_df['Thickness_um'] == 12.0]

axes[1, 0].boxplot([thick_8['Area_Above_R1_Raw'], thick_12['Area_Above_R1_Raw']],
                    labels=['8 μm', '12 μm'])
axes[1, 0].set_ylabel('Area Above R=1 (Raw)', fontsize=11)
axes[1, 0].set_xlabel('Thickness', fontsize=11)
axes[1, 0].set_title('Fluorescence vs Thickness', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Peak wavelength distribution
axes[1, 1].hist(results_df[results_df['Peak_WL_Raw_nm'] > 0]['Peak_WL_Raw_nm'],
                bins=15, color='coral', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Peak Wavelength (nm)', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Fluorescence Peak Wavelength Distribution', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/fluorescence/fluorescence_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: results/fluorescence/fluorescence_analysis.png")

# Plot 2: Sample curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw curves
n_samples = min(10, len(results_df))
for i in range(n_samples):
    row = results_df_sorted.iloc[i]
    refl = np.array(row['Reflectance'])
    axes[0].plot(WAVELENGTHS, refl, marker='o', markersize=4,
                 label=f"{row['Sample']} (A={row['Area_Above_R1_Raw']:.2f})", alpha=0.7)

axes[0].plot(WAVELENGTHS, PP_BACKGROUND, 'k--', linewidth=2, label='PP Background', alpha=0.8)
axes[0].axhline(y=1.0, color='red', linestyle=':', linewidth=2, label='R = 1.0', alpha=0.6)
axes[0].set_xlabel('Wavelength (nm)', fontsize=11)
axes[0].set_ylabel('Reflectance', fontsize=11)
axes[0].set_title('Raw Reflectance - Top 10 by Fluorescence', fontsize=12, fontweight='bold')
axes[0].legend(loc='best', fontsize=8)
axes[0].grid(True, alpha=0.3)

# Background-subtracted curves
for i in range(n_samples):
    row = results_df_sorted.iloc[i]
    refl_sub = np.array(row['Reflectance_Sub'])
    axes[1].plot(WAVELENGTHS, refl_sub, marker='o', markersize=4,
                 label=f"{row['Sample']} (A={row['Area_Above_R0_Sub']:.2f})", alpha=0.7)

axes[1].axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
axes[1].set_xlabel('Wavelength (nm)', fontsize=11)
axes[1].set_ylabel('Reflectance - PP Background', fontsize=11)
axes[1].set_title('Background-Subtracted Reflectance', fontsize=12, fontweight='bold')
axes[1].legend(loc='best', fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/fluorescence/reflectance_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/fluorescence/reflectance_curves.png")

# Save results
output_df = results_df[['Sample', 'Base_Name', 'Thickness_um', 'L', 'a', 'b',
                         'Area_Above_R1_Raw', 'Area_Above_R0_Sub',
                         'Max_R_Raw', 'Max_R_Sub', 'Peak_WL_Raw_nm', 'Peak_WL_Sub_nm']]
output_df.to_csv('results/fluorescence/fluorescence_results.csv', index=False)
print("✓ Saved: results/fluorescence/fluorescence_results.csv")

# Create mapping template
mapping_df = results_df[['Sample', 'Base_Name', 'Thickness_um']].copy()
mapping_df['GXT_percent'] = ''
mapping_df['BiVaO4_percent'] = ''
mapping_df['PG_percent'] = ''
mapping_df['PearlB_percent'] = ''
mapping_df.to_csv('results/fluorescence/sample_mapping_template.csv', index=False)
print("✓ Saved: results/fluorescence/sample_mapping_template.csv (fill in concentrations)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n✓ Analyzed {len(results_df)} samples")
print(f"  8μm samples: {len(results_df[results_df['Thickness_um']==8])}")
print(f"  12μm samples: {len(results_df[results_df['Thickness_um']==12])}")
print(f"\n✓ Background subtraction {'significantly increases' if ratio > 1.5 else 'slightly increases' if ratio > 1.1 else 'minimally affects'} fluorescence signal")
print("\nNext steps:")
print("  1. Fill in sample_mapping_template.csv with concentration data")
print("  2. Run correlation analysis with concentrations")
print()

plt.close('all')
