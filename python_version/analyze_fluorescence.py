"""
Fluorescence Data Analysis

This script:
1. Parses OptiSpectra.csv fluorescence measurements
2. Merges with concentration data from Concentrations.csv
3. Infers thickness from Band Vol column in sample names
4. Calculates area under reflectance curve above y=1 (fluorescence indicator)
5. Tests correlation between reflectance>1 area and mean fluorescence intensities
6. Compares raw vs PP-background-subtracted reflectance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import trapz
import os

# Configuration
OPTISPECTRA_PATH = '/Users/GoergeH/Documents/OptiSpectra.csv'
CONCENTRATIONS_PATH = '/Users/GoergeH/Documents/github/PredictiveApp/public/Concentrations.csv'
SPECTRA_PATH = '/Users/GoergeH/Documents/github/PredictiveApp/public/Spectra.csv'

# PP Background reflectance (from user-provided data)
PP_BACKGROUND = np.array([
    6.32230798e-1, 8.24865798e-1, 9.64723766e-1, 9.74486450e-1, 9.75640754e-1,
    9.78552401e-1, 9.75615660e-1, 9.77584740e-1, 9.89334762e-1, 9.98248855e-1,
    1.00193385e+0, 1.00288971e+0, 1.00431999e+0, 1.00225039e+0, 1.00409655e+0,
    1.00195293e+0, 1.00040476e+0, 1.00163217e+0, 1.00373669e+0, 1.00462874e+0,
    1.00526512e+0, 1.00307862e+0, 9.99236067e-1, 9.90266760e-1, 9.87492502e-1,
    9.90339776e-1, 9.88663038e-1, 9.86457149e-1, 9.83821054e-1, 9.81393794e-1,
    9.78041152e-1
])

WAVELENGTHS = np.arange(400, 710, 10)  # 400-700nm in 10nm intervals

def parse_optispectra(filepath):
    """
    Parse OptiSpectra.csv file

    Returns:
        DataFrame with columns: sample_name, thickness, reflectance_data, CIELAB values
    """
    print(f"Reading OptiSpectra data from: {filepath}")

    # Read CSV (semicolon delimited)
    df = pd.read_csv(filepath, sep=';', skiprows=3)

    samples = []

    for idx, row in df.iterrows():
        try:
            # Extract sample name
            user_color_name = str(row['User Color Name']).strip()

            # Skip PP SUB background
            if 'SUB' in user_color_name or 'PP' in user_color_name:
                continue

            # Infer thickness from name (H suffix = heavy = 12μm, no suffix = 8μm)
            if user_color_name.endswith('H'):
                thickness = 12.0
                base_name = user_color_name[:-1]  # Remove 'H'
            else:
                thickness = 8.0
                base_name = user_color_name

            # Extract reflectance data (columns R400 nm through R700 nm)
            reflectance_cols = [col for col in df.columns if col.startswith('R') and 'nm' in col]
            reflectance = []
            for col in reflectance_cols:
                try:
                    val = float(row[col])
                    reflectance.append(val)
                except:
                    reflectance.append(np.nan)

            reflectance = np.array(reflectance)

            # Extract CIELAB values
            L = float(row['L']) if 'L' in row and pd.notna(row['L']) else np.nan
            a = float(row['a']) if 'a' in row and pd.notna(row['a']) else np.nan
            b = float(row['b']) if 'b' in row and pd.notna(row['b']) else np.nan

            samples.append({
                'sample_name': user_color_name,
                'base_name': base_name,
                'thickness': thickness,
                'reflectance': reflectance,
                'L': L,
                'a': a,
                'b': b,
                'index': idx
            })

        except Exception as e:
            print(f"  Warning: Could not parse row {idx}: {e}")
            continue

    df_samples = pd.DataFrame(samples)
    print(f"✓ Parsed {len(df_samples)} fluorescence samples")

    return df_samples


def parse_concentrations(filepath):
    """Parse Concentrations.csv"""
    print(f"\nReading concentrations from: {filepath}")

    df = pd.read_csv(filepath, sep=',')  # Comma-separated

    # Clean sample IDs
    df['Sample ID'] = df['Sample ID'].str.strip()

    # Clean percentage values (remove % signs)
    for col in ['GXT-10 (%)', 'BiVaO4 (%)', 'PG (%)', 'PearlB (%)']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '').astype(float)

    print(f"✓ Loaded {len(df)} concentration records")

    return df


def calculate_area_above_one(reflectance, wavelengths):
    """
    Calculate area under reflectance curve where R > 1.0

    This represents excess reflectance due to fluorescence
    """
    # Only consider values above 1.0
    above_one = reflectance - 1.0
    above_one[above_one < 0] = 0

    # Calculate area using trapezoidal integration
    area = trapz(above_one, wavelengths)

    return area


def calculate_area_above_zero(reflectance, wavelengths):
    """
    Calculate area under reflectance curve where R > 0.0

    This is used for background-subtracted data where the baseline is normalized to 0
    """
    # Only consider positive values
    above_zero = reflectance.copy()
    above_zero[above_zero < 0] = 0

    # Calculate area using trapezoidal integration
    area = trapz(above_zero, wavelengths)

    return area


def calculate_mean_fluorescence(reflectance, wavelengths, threshold=1.0):
    """
    Calculate mean intensity of fluorescence (reflectance > threshold)
    """
    fluor_values = reflectance[reflectance > threshold]

    if len(fluor_values) == 0:
        return 0.0

    return np.mean(fluor_values - threshold)


def merge_with_concentrations(fluor_df, conc_df):
    """
    Merge fluorescence data with concentration data

    Matches based on sample name patterns (e.g., OPTI6 -> OPTI6, OPTI6H -> OPTI6)
    """
    print("\nMerging fluorescence with concentration data...")

    merged_data = []

    for idx, row in fluor_df.iterrows():
        sample_name = row['sample_name']
        base_name = row['base_name']

        # Try to find matching concentration record
        # Match patterns: OPTI6 -> "OPTI 6", OPTI10H -> "OPTI 10", etc.
        # Handle both "OPTI6" and "OPTI 6" formats
        search_pattern = base_name.replace('OPTI', 'OPTI ')  # Add space
        matches = conc_df[conc_df['Sample ID'].str.contains(search_pattern, case=False, na=False, regex=False)]

        # If no match with space, try exact match without space
        if len(matches) == 0:
            matches = conc_df[conc_df['Sample ID'].str.contains(base_name, case=False, na=False, regex=False)]

        if len(matches) > 0:
            match = matches.iloc[0]

            merged_data.append({
                'sample_name': sample_name,
                'base_name': base_name,
                'thickness': row['thickness'],
                'GXT': float(match['GXT-10 (%)']) if pd.notna(match['GXT-10 (%)']) else 0.0,
                'BiVaO4': float(match['BiVaO4 (%)']) if pd.notna(match['BiVaO4 (%)']) else 0.0,
                'PG': float(match['PG (%)']) if pd.notna(match['PG (%)']) else 0.0,
                'PearlB': float(match['PearlB (%)']) if pd.notna(match['PearlB (%)']) else 0.0,
                'reflectance': row['reflectance'],
                'L': row['L'],
                'a': row['a'],
                'b': row['b']
            })
        else:
            print(f"  Warning: No concentration match for {sample_name}")

    merged_df = pd.DataFrame(merged_data)
    print(f"✓ Merged {len(merged_df)} samples with concentration data")

    return merged_df


def analyze_fluorescence_correlation(merged_df):
    """
    Analyze correlation between area above R=1 and fluorescence indicators

    Tests both raw and background-subtracted reflectance
    """
    print("\n" + "=" * 80)
    print("FLUORESCENCE CORRELATION ANALYSIS")
    print("=" * 80)

    results = []

    for idx, row in merged_df.iterrows():
        reflectance_raw = row['reflectance']

        # Background-subtracted reflectance
        reflectance_sub = reflectance_raw - PP_BACKGROUND

        # Calculate metrics for RAW reflectance
        area_raw = calculate_area_above_one(reflectance_raw, WAVELENGTHS)
        mean_fluor_raw = calculate_mean_fluorescence(reflectance_raw, WAVELENGTHS, threshold=1.0)
        max_raw = np.max(reflectance_raw)

        # Calculate metrics for BACKGROUND-SUBTRACTED reflectance
        # After background subtraction, baseline is normalized to 0, so we calculate area above 0
        area_sub = calculate_area_above_zero(reflectance_sub, WAVELENGTHS)
        mean_fluor_sub = calculate_mean_fluorescence(reflectance_sub, WAVELENGTHS, threshold=0.0)
        max_sub = np.max(reflectance_sub)

        results.append({
            'sample_name': row['sample_name'],
            'thickness': row['thickness'],
            'GXT': row['GXT'],
            'BiVaO4': row['BiVaO4'],
            'PG': row['PG'],
            'PearlB': row['PearlB'],
            'L': row['L'],
            'a': row['a'],
            'b': row['b'],
            # Raw metrics
            'area_above_1_raw': area_raw,
            'mean_fluor_raw': mean_fluor_raw,
            'max_R_raw': max_raw,
            # Background-subtracted metrics
            'area_above_0_sub': area_sub,
            'mean_fluor_sub': mean_fluor_sub,
            'max_R_sub': max_sub
        })

    results_df = pd.DataFrame(results)

    # Statistical analysis
    print("\n### RAW REFLECTANCE (R > 1.0) ###")
    print(f"\nArea above R=1 statistics:")
    print(f"  Mean: {results_df['area_above_1_raw'].mean():.4f}")
    print(f"  Std: {results_df['area_above_1_raw'].std():.4f}")
    print(f"  Min: {results_df['area_above_1_raw'].min():.4f}")
    print(f"  Max: {results_df['area_above_1_raw'].max():.4f}")

    # Correlations with GXT (fluorescent pigment)
    if results_df['GXT'].std() > 0:
        corr_gxt_raw, p_gxt_raw = stats.pearsonr(results_df['GXT'], results_df['area_above_1_raw'])
        print(f"\nCorrelation with GXT concentration:")
        print(f"  Pearson r = {corr_gxt_raw:.4f}, p = {p_gxt_raw:.4e}")

    # Correlation with CIELAB b* (yellowness - fluorescence shifts toward yellow/green)
    if results_df['b'].std() > 0:
        corr_b_raw, p_b_raw = stats.pearsonr(results_df['b'], results_df['area_above_1_raw'])
        print(f"\nCorrelation with CIELAB b* (yellowness):")
        print(f"  Pearson r = {corr_b_raw:.4f}, p = {p_b_raw:.4e}")

    print("\n### BACKGROUND-SUBTRACTED REFLECTANCE (R - R_PP) ###")
    print(f"\nArea above R=0 (subtracted) statistics:")
    print(f"  Mean: {results_df['area_above_0_sub'].mean():.4f}")
    print(f"  Std: {results_df['area_above_0_sub'].std():.4f}")
    print(f"  Min: {results_df['area_above_0_sub'].min():.4f}")
    print(f"  Max: {results_df['area_above_0_sub'].max():.4f}")

    if results_df['GXT'].std() > 0:
        corr_gxt_sub, p_gxt_sub = stats.pearsonr(results_df['GXT'], results_df['area_above_0_sub'])
        print(f"\nCorrelation with GXT concentration:")
        print(f"  Pearson r = {corr_gxt_sub:.4f}, p = {p_gxt_sub:.4e}")

    if results_df['b'].std() > 0:
        corr_b_sub, p_b_sub = stats.pearsonr(results_df['b'], results_df['area_above_0_sub'])
        print(f"\nCorrelation with CIELAB b* (yellowness):")
        print(f"  Pearson r = {corr_b_sub:.4f}, p = {p_b_sub:.4e}")

    # Comparison
    print("\n### COMPARISON: RAW vs BACKGROUND-SUBTRACTED ###")
    if results_df['GXT'].std() > 0:
        print(f"\nGXT correlation:")
        print(f"  Raw (R>1):        r = {corr_gxt_raw:.4f}")
        print(f"  Background-sub:   r = {corr_gxt_sub:.4f}")
        print(f"  Difference:       Δr = {abs(corr_gxt_sub - corr_gxt_raw):.4f}")

        if abs(corr_gxt_sub) > abs(corr_gxt_raw):
            print(f"  ✓ Background subtraction IMPROVES correlation by {((abs(corr_gxt_sub) - abs(corr_gxt_raw))/abs(corr_gxt_raw)*100):.1f}%")
        else:
            print(f"  ✗ Background subtraction WEAKENS correlation by {((abs(corr_gxt_raw) - abs(corr_gxt_sub))/abs(corr_gxt_raw)*100):.1f}%")

    # CRITICAL: Test how well area predicts actual fluorescence intensity
    print("\n### PREDICTIVE POWER: AREA vs ACTUAL FLUORESCENCE INTENSITY ###")

    # For raw method: correlate area_above_1 with mean_fluor_raw
    if results_df['mean_fluor_raw'].std() > 0:
        corr_area_fluor_raw, p_area_fluor_raw = stats.pearsonr(
            results_df['area_above_1_raw'],
            results_df['mean_fluor_raw']
        )
        print(f"\nRaw Method (Area above R=1 vs Mean Fluorescence Intensity):")
        print(f"  Pearson r = {corr_area_fluor_raw:.4f}, p = {p_area_fluor_raw:.4e}")
        print(f"  R² = {corr_area_fluor_raw**2:.4f} ({corr_area_fluor_raw**2*100:.1f}% variance explained)")

    # For background-subtracted: correlate area_above_0_sub with mean_fluor_sub
    if results_df['mean_fluor_sub'].std() > 0:
        corr_area_fluor_sub, p_area_fluor_sub = stats.pearsonr(
            results_df['area_above_0_sub'],
            results_df['mean_fluor_sub']
        )
        print(f"\nBackground-Subtracted Method (Area above R=0 vs Mean Fluorescence Intensity):")
        print(f"  Pearson r = {corr_area_fluor_sub:.4f}, p = {p_area_fluor_sub:.4e}")
        print(f"  R² = {corr_area_fluor_sub**2:.4f} ({corr_area_fluor_sub**2*100:.1f}% variance explained)")

    # Compare which method is better predictor
    if results_df['mean_fluor_raw'].std() > 0 and results_df['mean_fluor_sub'].std() > 0:
        print(f"\nComparison:")
        print(f"  Raw method R²:              {corr_area_fluor_raw**2:.4f}")
        print(f"  Background-subtracted R²:   {corr_area_fluor_sub**2:.4f}")
        if corr_area_fluor_sub**2 > corr_area_fluor_raw**2:
            improvement = ((corr_area_fluor_sub**2 - corr_area_fluor_raw**2) / corr_area_fluor_raw**2 * 100)
            print(f"  ✓ Background subtraction IMPROVES predictive power by {improvement:.1f}%")
            print(f"  ✓ Recommended: Use background-subtracted method for scalability")
        else:
            print(f"  → Both methods have similar predictive power")

    return results_df


def visualize_fluorescence_analysis(results_df, merged_df):
    """Create visualization plots"""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    os.makedirs('results/fluorescence', exist_ok=True)

    # Figure 1: Raw vs Background-subtracted correlation with GXT
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Raw reflectance
    axes[0].scatter(results_df['GXT'], results_df['area_above_1_raw'],
                    c=results_df['thickness'], cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    axes[0].set_xlabel('GXT Concentration (%)', fontsize=12)
    axes[0].set_ylabel('Area Above R=1 (Raw)', fontsize=12)
    axes[0].set_title('Raw Reflectance > 1.0 vs GXT', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    if results_df['GXT'].std() > 0:
        z = np.polyfit(results_df['GXT'], results_df['area_above_1_raw'], 1)
        p = np.poly1d(z)
        axes[0].plot(results_df['GXT'], p(results_df['GXT']), "r--", alpha=0.8, linewidth=2)
        corr, pval = stats.pearsonr(results_df['GXT'], results_df['area_above_1_raw'])
        axes[0].text(0.05, 0.95, f'r = {corr:.3f}\np = {pval:.2e}',
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Background-subtracted
    axes[1].scatter(results_df['GXT'], results_df['area_above_0_sub'],
                    c=results_df['thickness'], cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    axes[1].set_xlabel('GXT Concentration (%)', fontsize=12)
    axes[1].set_ylabel('Area Above R=0 (Background-Subtracted)', fontsize=12)
    axes[1].set_title('Background-Subtracted Reflectance vs GXT', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    if results_df['GXT'].std() > 0:
        z = np.polyfit(results_df['GXT'], results_df['area_above_0_sub'], 1)
        p = np.poly1d(z)
        axes[1].plot(results_df['GXT'], p(results_df['GXT']), "r--", alpha=0.8, linewidth=2)
        corr, pval = stats.pearsonr(results_df['GXT'], results_df['area_above_0_sub'])
        axes[1].text(0.05, 0.95, f'r = {corr:.3f}\np = {pval:.2e}',
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=8, vmax=12))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', pad=0.1, aspect=30)
    cbar.set_label('Thickness (μm)', fontsize=11)

    plt.tight_layout()
    plt.savefig('results/fluorescence/gxt_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/fluorescence/gxt_correlation.png")

    # Figure 2: Sample reflectance curves
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot a few example curves
    for idx in range(min(10, len(merged_df))):
        row = merged_df.iloc[idx]
        reflectance = row['reflectance']
        label = f"{row['sample_name']} (GXT={row['GXT']:.1f}%)"
        ax.plot(WAVELENGTHS, reflectance, marker='o', markersize=4, label=label, alpha=0.7)

    # Add PP background
    ax.plot(WAVELENGTHS, PP_BACKGROUND, 'k--', linewidth=2, label='PP Background', alpha=0.8)

    # Add R=1 line
    ax.axhline(y=1.0, color='red', linestyle=':', linewidth=2, label='R = 1.0', alpha=0.6)

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title('Sample Reflectance Curves with PP Background', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/fluorescence/reflectance_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/fluorescence/reflectance_curves.png")

    # Figure 3: Area vs Actual Fluorescence Intensity (Predictive Power)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Raw method: Area above R=1 vs Mean Fluorescence Intensity
    axes[0].scatter(results_df['area_above_1_raw'], results_df['mean_fluor_raw'],
                    c=results_df['thickness'], cmap='plasma', s=100, alpha=0.7, edgecolors='black')
    axes[0].set_xlabel('Area Above R=1 (Raw)', fontsize=12)
    axes[0].set_ylabel('Mean Fluorescence Intensity', fontsize=12)
    axes[0].set_title('Raw Method: Area vs Fluorescence Intensity', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    if results_df['mean_fluor_raw'].std() > 0:
        z = np.polyfit(results_df['area_above_1_raw'], results_df['mean_fluor_raw'], 1)
        p = np.poly1d(z)
        axes[0].plot(results_df['area_above_1_raw'], p(results_df['area_above_1_raw']),
                     "r--", alpha=0.8, linewidth=2)
        corr, pval = stats.pearsonr(results_df['area_above_1_raw'], results_df['mean_fluor_raw'])
        axes[0].text(0.05, 0.95, f'r = {corr:.4f}\nR² = {corr**2:.4f}\np = {pval:.2e}',
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Background-subtracted method: Area above R=0 vs Mean Fluorescence Intensity
    axes[1].scatter(results_df['area_above_0_sub'], results_df['mean_fluor_sub'],
                    c=results_df['thickness'], cmap='plasma', s=100, alpha=0.7, edgecolors='black')
    axes[1].set_xlabel('Area Above R=0 (Background-Subtracted)', fontsize=12)
    axes[1].set_ylabel('Mean Fluorescence Intensity', fontsize=12)
    axes[1].set_title('Background-Subtracted: Area vs Fluorescence', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    if results_df['mean_fluor_sub'].std() > 0:
        z = np.polyfit(results_df['area_above_0_sub'], results_df['mean_fluor_sub'], 1)
        p = np.poly1d(z)
        axes[1].plot(results_df['area_above_0_sub'], p(results_df['area_above_0_sub']),
                     "r--", alpha=0.8, linewidth=2)
        corr, pval = stats.pearsonr(results_df['area_above_0_sub'], results_df['mean_fluor_sub'])
        axes[1].text(0.05, 0.95, f'r = {corr:.4f}\nR² = {corr**2:.4f}\np = {pval:.2e}',
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=8, vmax=12))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', pad=0.1, aspect=30)
    cbar.set_label('Thickness (μm)', fontsize=11)

    plt.tight_layout()
    plt.savefig('results/fluorescence/area_vs_fluorescence.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/fluorescence/area_vs_fluorescence.png")

    plt.close('all')


def save_results(results_df):
    """Save results to CSV"""
    output_path = 'results/fluorescence/fluorescence_analysis.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved results to: {output_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FLUORESCENCE DATA ANALYSIS")
    print("=" * 80)

    # Parse fluorescence data
    fluor_df = parse_optispectra(OPTISPECTRA_PATH)

    # Parse concentrations
    conc_df = parse_concentrations(CONCENTRATIONS_PATH)

    # Merge data
    merged_df = merge_with_concentrations(fluor_df, conc_df)

    # Analyze correlations
    results_df = analyze_fluorescence_correlation(merged_df)

    # Create visualizations
    visualize_fluorescence_analysis(results_df, merged_df)

    # Save results
    save_results(results_df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"  Total samples analyzed: {len(results_df)}")
    print(f"  Samples with GXT: {len(results_df[results_df['GXT'] > 0])}")
    print(f"  8μm samples: {len(results_df[results_df['thickness'] == 8])}")
    print(f"  12μm samples: {len(results_df[results_df['thickness'] == 12])}")
    print("\nOutput files:")
    print("  - results/fluorescence/gxt_correlation.png")
    print("  - results/fluorescence/reflectance_curves.png")
    print("  - results/fluorescence/fluorescence_analysis.csv")
    print()
