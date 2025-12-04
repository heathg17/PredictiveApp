"""
Prepare training data for fluorescence NN comparison
Matches data from Concentrations.csv, Spectra.csv, and Fluorescence_Master.csv
"""
import pandas as pd
import numpy as np
import pickle
import os
from scipy.integrate import trapezoid


def safe_parse_pct(val):
    """Safely parse percentage values"""
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        return float(val.rstrip('%'))
    return float(val)


def calculate_integrated_area(reflectance, wavelengths=None):
    """
    Calculate integrated area under reflectance curve
    This is a key feature for fluorescence prediction

    Args:
        reflectance: array of 31 reflectance values
        wavelengths: optional wavelengths (default: 400-700nm, 10nm steps)

    Returns:
        Integrated area
    """
    if wavelengths is None:
        wavelengths = np.arange(400, 710, 10)  # 400-700nm in 10nm steps

    return trapezoid(reflectance, wavelengths)


def load_and_match_fluorescence_data():
    """
    Load and match data from all three CSV files
    Returns matched samples with concentrations, spectra, CIELAB, and fluorescence
    """

    print("=" * 80)
    print("LOADING FLUORESCENCE COMPARISON DATA")
    print("=" * 80)

    # Load fluorescence (new format)
    print("\nLoading Fluorescence_Master.csv...")
    fluor_df = pd.read_csv('../public/Fluorescence_Master.csv')
    print(f"✓ Loaded {len(fluor_df)} fluorescence measurements")
    print(f"  ct/s range: {fluor_df['Fluorescence [ct/s]'].min():.0f} - {fluor_df['Fluorescence [ct/s]'].max():.0f}")

    # Load spectra
    print("\nLoading Spectra.csv...")
    spectra_df = pd.read_csv('../public/Spectra.csv', sep=';', skiprows=3)

    reflectance_cols = [col for col in spectra_df.columns if 'R' in col and 'nm' in col]
    reflectance_cols = sorted(reflectance_cols, key=lambda x: int(x.split('R')[1].split('nm')[0]))[:31]

    print(f"✓ Loaded {len(spectra_df)} spectra")
    print(f"  Found {len(reflectance_cols)} wavelength columns")

    # Load concentrations
    print("\nLoading Concentrations.csv...")
    conc_df = pd.read_csv('../public/Concentrations.csv')

    # Parse concentrations
    conc_df['GXT'] = conc_df['GXT-10 (%)'].apply(safe_parse_pct)
    conc_df['BiVaO4'] = conc_df['BiVaO4 (%)'].apply(safe_parse_pct)
    conc_df['PG'] = conc_df['PG (%)'].apply(safe_parse_pct)
    conc_df['PearlB'] = conc_df['PearlB (%)'].apply(safe_parse_pct)

    print(f"✓ Loaded {len(conc_df)} concentration records")

    training_samples = []

    # Match all samples
    print("\nMatching samples across all datasets...")
    matched_count = 0
    skipped_count = 0

    for _, fluor_row in fluor_df.iterrows():
        sample_id = fluor_row['Sample ID']
        fluorescence_cts = fluor_row['Fluorescence [ct/s]']

        # Determine thickness from sample name
        if sample_id.endswith('H'):
            thickness = 12.0
            base_name = sample_id[:-1]  # Remove 'H' suffix
        else:
            thickness = 8.0
            base_name = sample_id

        # Match spectrum
        spec_matches = spectra_df[spectra_df['User Color Name'] == sample_id]

        if len(spec_matches) == 0:
            print(f"  Warning: No spectrum for {sample_id}")
            skipped_count += 1
            continue

        spectrum_row = spec_matches.iloc[0]
        reflectance = spectrum_row[reflectance_cols].values.astype(float)

        # Match concentrations
        # Try exact match first
        conc_matches = conc_df[conc_df['Sample ID'] == base_name]

        # Try with space for OPTI samples (OPTI1 -> OPTI 1)
        if len(conc_matches) == 0 and base_name.startswith('OPTI'):
            opti_num = base_name.replace('OPTI', '')
            with_space = f"OPTI {opti_num}"
            conc_matches = conc_df[conc_df['Sample ID'] == with_space]

        # For GXT samples, create synthetic concentration
        if len(conc_matches) == 0 and base_name.startswith('GXT'):
            # Extract GXT concentration from name (e.g., GXT10 -> 10%)
            gxt_conc = float(base_name.replace('GXT', ''))
            concentrations = {
                'GXT': gxt_conc,
                'BiVaO4': 0.0,
                'PG': 0.0,
                'PearlB': 0.0
            }
        elif len(conc_matches) > 0:
            conc_row = conc_matches.iloc[0]
            concentrations = {
                'GXT': conc_row['GXT'],
                'BiVaO4': conc_row['BiVaO4'],
                'PG': conc_row['PG'],
                'PearlB': conc_row['PearlB']
            }
        else:
            print(f"  Warning: No concentration data for {sample_id}")
            skipped_count += 1
            continue

        # Calculate integrated area
        integrated_area = calculate_integrated_area(reflectance)

        # Create sample
        training_samples.append({
            'id': sample_id,
            'GXT': concentrations['GXT'],
            'BiVaO4': concentrations['BiVaO4'],
            'PG': concentrations['PG'],
            'PearlB': concentrations['PearlB'],
            'thickness': thickness,
            'reflectance': reflectance,
            'integrated_area': integrated_area,
            'L': float(spectrum_row['L']),
            'a': float(spectrum_row['a']),
            'b': float(spectrum_row['b']),
            'c': float(spectrum_row['c']),
            'h': float(spectrum_row['h']),
            'fluorescence_cts': fluorescence_cts
        })
        matched_count += 1

    print(f"\n✓ Matched {matched_count} samples")
    print(f"  Skipped {skipped_count} samples (missing data)")

    # Analyze matched data
    print("\n" + "=" * 80)
    print("MATCHED DATA STATISTICS")
    print("=" * 80)

    gxt_vals = [s['GXT'] for s in training_samples]
    fluor_vals = [s['fluorescence_cts'] for s in training_samples]
    area_vals = [s['integrated_area'] for s in training_samples]

    print(f"\nConcentrations:")
    print(f"  GXT: {min(gxt_vals):.1f}% - {max(gxt_vals):.1f}%")
    print(f"  BiVaO4: {min(s['BiVaO4'] for s in training_samples):.1f}% - {max(s['BiVaO4'] for s in training_samples):.1f}%")
    print(f"  PG: {min(s['PG'] for s in training_samples):.1f}% - {max(s['PG'] for s in training_samples):.1f}%")
    print(f"  PearlB: {min(s['PearlB'] for s in training_samples):.1f}% - {max(s['PearlB'] for s in training_samples):.1f}%")

    print(f"\nFluorescence:")
    print(f"  Min: {min(fluor_vals):.0f} ct/s")
    print(f"  Max: {max(fluor_vals):.0f} ct/s")
    print(f"  Mean: {np.mean(fluor_vals):.0f} ct/s")
    print(f"  Std: {np.std(fluor_vals):.0f} ct/s")

    print(f"\nIntegrated Area:")
    print(f"  Min: {min(area_vals):.1f}")
    print(f"  Max: {max(area_vals):.1f}")
    print(f"  Mean: {np.mean(area_vals):.1f}")

    print(f"\nThickness distribution:")
    print(f"  8μm samples: {sum(1 for s in training_samples if s['thickness'] == 8.0)}")
    print(f"  12μm samples: {sum(1 for s in training_samples if s['thickness'] == 12.0)}")

    return training_samples


def prepare_option1_data(samples):
    """
    Prepare data for Option 1: Separate Fluorescence-Only NN

    Inputs (6 features):
        - GXT concentration (%)
        - BiVaO4 concentration (%)
        - PG concentration (%)
        - PearlB concentration (%)
        - Thickness (μm)
        - Integrated area under reflectance curve

    Output (1 feature):
        - Fluorescence (ct/s)
    """
    X = []
    Y = []

    for sample in samples:
        X.append([
            sample['GXT'],
            sample['BiVaO4'],
            sample['PG'],
            sample['PearlB'],
            sample['thickness'],
            sample['integrated_area']
        ])

        Y.append([sample['fluorescence_cts']])

    return np.array(X), np.array(Y)


def prepare_option3_data(samples):
    """
    Prepare data for Option 3: Multi-Task Learning with 3 Outputs

    Inputs (5 features):
        - GXT concentration (%)
        - BiVaO4 concentration (%)
        - PG concentration (%)
        - PearlB concentration (%)
        - Thickness (μm)

    Outputs (3 heads):
        - Y_spectral (31 wavelengths)
        - Y_cielab (5 values: L, a, b, c, h)
        - Y_fluorescence (1 value: ct/s)
    """
    X = []
    Y_spectral = []
    Y_cielab = []
    Y_fluorescence = []

    for sample in samples:
        X.append([
            sample['GXT'],
            sample['BiVaO4'],
            sample['PG'],
            sample['PearlB'],
            sample['thickness']
        ])

        Y_spectral.append(sample['reflectance'])

        Y_cielab.append([
            sample['L'],
            sample['a'],
            sample['b'],
            sample['c'],
            sample['h']
        ])

        Y_fluorescence.append([sample['fluorescence_cts']])

    return np.array(X), np.array(Y_spectral), np.array(Y_cielab), np.array(Y_fluorescence)


def main():
    """Prepare and save training data for both options"""

    # Load matched samples
    samples = load_and_match_fluorescence_data()

    if len(samples) == 0:
        raise ValueError("No matching samples found!")

    # Prepare Option 1 data
    print("\n" + "=" * 80)
    print("PREPARING OPTION 1 DATA (Separate Fluorescence NN)")
    print("=" * 80)

    X_opt1, Y_opt1 = prepare_option1_data(samples)
    print(f"X shape: {X_opt1.shape} (6 input features)")
    print(f"Y shape: {Y_opt1.shape} (1 output: fluorescence ct/s)")

    # Prepare Option 3 data
    print("\n" + "=" * 80)
    print("PREPARING OPTION 3 DATA (Multi-Task Learning)")
    print("=" * 80)

    X_opt3, Y_spectral, Y_cielab, Y_fluor = prepare_option3_data(samples)
    print(f"X shape: {X_opt3.shape} (5 input features)")
    print(f"Y_spectral shape: {Y_spectral.shape} (31 wavelengths)")
    print(f"Y_cielab shape: {Y_cielab.shape} (5 CIELAB values)")
    print(f"Y_fluorescence shape: {Y_fluor.shape} (1 ct/s value)")

    # Save data
    os.makedirs('training_data', exist_ok=True)

    # Save Option 1 data
    option1_data = {
        'X': X_opt1,
        'Y': Y_opt1,
        'samples': samples
    }

    with open('training_data/option1_fluorescence_data.pkl', 'wb') as f:
        pickle.dump(option1_data, f)

    print("\n✓ Saved Option 1 data to: training_data/option1_fluorescence_data.pkl")

    # Save Option 3 data
    option3_data = {
        'X': X_opt3,
        'Y_spectral': Y_spectral,
        'Y_cielab': Y_cielab,
        'Y_fluorescence': Y_fluor,
        'samples': samples
    }

    with open('training_data/option3_multitask_data.pkl', 'wb') as f:
        pickle.dump(option3_data, f)

    print("✓ Saved Option 3 data to: training_data/option3_multitask_data.pkl")

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print("\nReady to train both models for performance comparison!")


if __name__ == "__main__":
    main()
