"""
Prepare training data with fluorescence predictions

This script:
1. Loads existing spectral/CIELAB training data
2. Merges with fluorescence measurements from OptiSpectra
3. Calculates background-subtracted fluorescence area for each sample
4. Creates enhanced training dataset with fluorescence targets
"""

import numpy as np
import pandas as pd
from scipy.integrate import trapz
import pickle

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


def calculate_fluorescence_area(reflectance, background=PP_BACKGROUND):
    """
    Calculate background-subtracted fluorescence area

    Args:
        reflectance: Array of reflectance values (31 wavelengths)
        background: Background reflectance to subtract

    Returns:
        Fluorescence area (background-subtracted, area above R=0)
    """
    # Subtract background
    reflectance_sub = reflectance - background

    # Only consider positive values (above normalized baseline)
    reflectance_sub[reflectance_sub < 0] = 0

    # Calculate area using trapezoidal integration
    area = trapz(reflectance_sub, WAVELENGTHS)

    return area


def load_optispectra_fluorescence():
    """Load fluorescence data from OptiSpectra.csv"""
    print("Loading OptiSpectra fluorescence data...")

    filepath = '/Users/GoergeH/Documents/OptiSpectra.csv'
    df = pd.read_csv(filepath, sep=';', skiprows=3)

    fluorescence_data = {}

    for idx, row in df.iterrows():
        try:
            sample_name = str(row['User Color Name']).strip()

            # Skip PP SUB background
            if 'SUB' in sample_name or 'PP' in sample_name:
                continue

            # Parse base name (remove H suffix if present)
            if sample_name.endswith('H'):
                base_name = sample_name[:-1]
                thickness = 12.0
            else:
                base_name = sample_name
                thickness = 8.0

            # Extract reflectance data
            reflectance_cols = [col for col in df.columns if col.startswith('R') and 'nm' in col]
            reflectance = np.array([float(row[col]) for col in reflectance_cols])

            # Calculate fluorescence area (background-subtracted)
            fluor_area = calculate_fluorescence_area(reflectance)

            # Store with both naming conventions
            for name_variant in [base_name, f"OPTI {base_name.replace('OPTI', '').strip()}"]:
                key = (name_variant.strip(), thickness)
                fluorescence_data[key] = fluor_area

        except Exception as e:
            continue

    print(f"✓ Loaded fluorescence data for {len(fluorescence_data)} sample/thickness combinations")
    return fluorescence_data


def merge_with_training_data():
    """Merge fluorescence data with existing training data"""
    print("\nLoading existing training data...")

    # Load concentrations
    conc_df = pd.read_csv('../public/Concentrations.csv', sep=',')
    conc_df['Sample ID'] = conc_df['Sample ID'].str.strip()

    # Clean percentage values
    for col in ['GXT-10 (%)', 'BiVaO4 (%)', 'PG (%)', 'PearlB (%)']:
        if col in conc_df.columns:
            conc_df[col] = conc_df[col].astype(str).str.replace('%', '').astype(float)

    # Load spectra
    spectra_df = pd.read_csv('../public/Spectra.csv', sep=';', skiprows=3)

    # Load fluorescence data
    fluor_data = load_optispectra_fluorescence()

    print("\nMerging datasets...")

    merged_samples = []

    for idx, conc_row in conc_df.iterrows():
        sample_id = conc_row['Sample ID'].strip()

        # Extract concentrations
        gxt = float(conc_row['GXT-10 (%)']) if pd.notna(conc_row['GXT-10 (%)']) else 0.0
        bivao4 = float(conc_row['BiVaO4 (%)']) if pd.notna(conc_row['BiVaO4 (%)']) else 0.0
        pg = float(conc_row['PG (%)']) if pd.notna(conc_row['PG (%)']) else 0.0
        pearlb = float(conc_row['PearlB (%)']) if pd.notna(conc_row['PearlB (%)']) else 0.0

        # Find matching spectra rows (8μm and 12μm)
        for thickness in [8.0, 12.0]:
            # Try to find matching spectrum based on User Color Name
            # 12μm samples end with 'H', 8μm samples don't
            if thickness == 12.0:
                # Look for samples ending with H
                spectrum_matches = spectra_df[
                    spectra_df['User Color Name'].str.contains(sample_id, case=False, na=False, regex=False) &
                    spectra_df['User Color Name'].str.endswith('H', na=False)
                ]
            else:
                # Look for samples NOT ending with H
                spectrum_matches = spectra_df[
                    spectra_df['User Color Name'].str.contains(sample_id, case=False, na=False, regex=False) &
                    ~spectra_df['User Color Name'].str.endswith('H', na=False)
                ]

            if len(spectrum_matches) == 0:
                continue

            spectrum_row = spectrum_matches.iloc[0]

            # Extract reflectance and CIELAB
            reflectance_cols = [col for col in spectra_df.columns if col.startswith('R') and 'nm' in col]
            reflectance = np.array([float(spectrum_row[col]) for col in reflectance_cols])

            L = float(spectrum_row['L']) if 'L' in spectrum_row and pd.notna(spectrum_row['L']) else np.nan
            a = float(spectrum_row['a']) if 'a' in spectrum_row and pd.notna(spectrum_row['a']) else np.nan
            b = float(spectrum_row['b']) if 'b' in spectrum_row and pd.notna(spectrum_row['b']) else np.nan
            c = float(spectrum_row['c']) if 'c' in spectrum_row and pd.notna(spectrum_row['c']) else np.nan
            h = float(spectrum_row['h']) if 'h' in spectrum_row and pd.notna(spectrum_row['h']) else np.nan

            # Look up fluorescence area
            fluor_area = None
            for key_variant in [(sample_id, thickness), (f"OPTI {sample_id.replace('OPTI', '').strip()}", thickness)]:
                if key_variant in fluor_data:
                    fluor_area = fluor_data[key_variant]
                    break

            # If no fluorescence data, calculate from reflectance
            if fluor_area is None:
                fluor_area = calculate_fluorescence_area(reflectance)

            merged_samples.append({
                'sample_id': sample_id,
                'thickness': thickness,
                'GXT': gxt,
                'BiVaO4': bivao4,
                'PG': pg,
                'PearlB': pearlb,
                'reflectance': reflectance,
                'L': L,
                'a': a,
                'b': b,
                'c': c,
                'h': h,
                'fluorescence_area': fluor_area
            })

    print(f"✓ Merged {len(merged_samples)} samples with fluorescence data")

    return merged_samples


def prepare_training_arrays(samples):
    """
    Prepare input/output arrays for neural network training

    Inputs (5):  GXT, BiVaO4, PG, PearlB, Thickness
    Outputs (37): 31 spectral + 5 CIELAB + 1 fluorescence
    """
    print("\nPreparing training arrays...")

    X = []  # Inputs
    Y = []  # Outputs

    for sample in samples:
        # Input: 5 features (concentrations + thickness)
        x = np.array([
            sample['GXT'],
            sample['BiVaO4'],
            sample['PG'],
            sample['PearlB'],
            sample['thickness']
        ])

        # Output: 31 spectral + 5 CIELAB + 1 fluorescence = 37 values
        y = np.concatenate([
            sample['reflectance'],  # 31 wavelengths
            [sample['L'], sample['a'], sample['b'], sample['c'], sample['h']],  # 5 CIELAB
            [sample['fluorescence_area']]  # 1 fluorescence area
        ])

        # Skip samples with NaN values
        if not np.any(np.isnan(x)) and not np.any(np.isnan(y)):
            X.append(x)
            Y.append(y)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(f"✓ Prepared training data:")
    print(f"  Input shape:  {X.shape} (samples × features)")
    print(f"  Output shape: {Y.shape} (samples × predictions)")
    print(f"  Features: [GXT, BiVaO4, PG, PearlB, Thickness]")
    print(f"  Outputs: [31 Spectral + 5 CIELAB + 1 Fluorescence] = 37 total")

    return X, Y, samples


def save_training_data(X, Y, samples):
    """Save prepared training data"""
    output = {
        'X': X,
        'Y': Y,
        'samples': samples,
        'input_features': ['GXT', 'BiVaO4', 'PG', 'PearlB', 'Thickness'],
        'output_features': (
            [f'R{w}nm' for w in WAVELENGTHS] +  # 31 spectral
            ['L', 'a', 'b', 'c', 'h'] +          # 5 CIELAB
            ['fluorescence_area']                 # 1 fluorescence
        ),
        'pp_background': PP_BACKGROUND,
        'wavelengths': WAVELENGTHS
    }

    output_path = 'training_data/fluorescence_training_data.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"\n✓ Saved training data to: {output_path}")
    print(f"  Total samples: {len(X)}")
    print(f"  Input dimensions: {X.shape[1]}")
    print(f"  Output dimensions: {Y.shape[1]}")

    return output_path


if __name__ == "__main__":
    print("=" * 80)
    print("PREPARING FLUORESCENCE TRAINING DATA")
    print("=" * 80)

    # Merge all data sources
    samples = merge_with_training_data()

    # Prepare training arrays
    X, Y, samples = prepare_training_arrays(samples)

    # Save for training
    save_training_data(X, Y, samples)

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Train enhanced neural network with fluorescence output")
    print("2. Validate fluorescence predictions")
    print("3. Update API to serve fluorescence predictions")
