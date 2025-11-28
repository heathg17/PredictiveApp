"""
Prepare Fluorescence Training Data with Actual ct/s Values

This script creates training data that links:
1. Concentrations (from Concentrations.csv)
2. Predicted reflectance spectra (from neural network)
3. Actual fluorescence ct/s measurements (from Fluorescence.csv)

The goal is to train a model that predicts ACTUAL ct/s values (2000-8000 range)
instead of normalized mean fluorescence (0.01-0.09 range).
"""

import pandas as pd
import numpy as np
import pickle
from scipy.integrate import trapezoid

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

WAVELENGTHS = np.arange(400, 710, 10)  # 400-700nm in 10nm intervals


def calculate_fluorescence_area(reflectance):
    """Calculate background-subtracted fluorescence area"""
    reflectance_sub = np.array(reflectance) - PP_BACKGROUND
    reflectance_sub[reflectance_sub < 0] = 0
    area = trapezoid(reflectance_sub, WAVELENGTHS)
    return area


def parse_concentration(conc_str):
    """Parse concentration string like '5%', '10%' to float"""
    if pd.isna(conc_str) or conc_str == '':
        return 0.0
    return float(str(conc_str).replace('%', ''))


def load_fluorescence_csv():
    """Load Fluorescence.csv with actual ct/s measurements"""
    print("Loading Fluorescence.csv...")
    df = pd.read_csv('../public/Fluorescence.csv')

    # Use 'New Fl Mean ct/s' if available, otherwise 'Fluro ct/s'
    if 'New Fl Mean ct/s' in df.columns:
        df['fluorescence_cts'] = pd.to_numeric(df['New Fl Mean ct/s'], errors='coerce')
        df['fluorescence_cts'] = df['fluorescence_cts'].fillna(pd.to_numeric(df['Fluro ct/s'], errors='coerce'))
    else:
        df['fluorescence_cts'] = pd.to_numeric(df['Fluro ct/s'], errors='coerce')

    # Parse concentrations
    df['GXT'] = df['Conc, wt/wt'].apply(parse_concentration)
    df['thickness'] = pd.to_numeric(df['Band Vol cm3/m2'], errors='coerce')
    df['sample_name'] = df['Material'].astype(str)

    # Filter valid rows
    df = df.dropna(subset=['fluorescence_cts', 'thickness'])
    df = df[df['fluorescence_cts'] > 0]  # Only positive fluorescence

    print(f"✓ Loaded {len(df)} fluorescence measurements")
    print(f"  ct/s range: {df['fluorescence_cts'].min():.0f} - {df['fluorescence_cts'].max():.0f}")

    return df


def load_spectra_data():
    """Load Spectra.csv to get reflectance data"""
    print("\nLoading Spectra.csv...")
    # Spectra.csv format:
    # Line 1: sep=;
    # Line 2: metadata
    # Line 3: column headers
    # Line 4+: data rows
    df = pd.read_csv('../public/Spectra.csv', sep=';', skiprows=2)

    # Get reflectance columns (400-700nm)
    reflectance_cols = [col for col in df.columns if 'R' in col and 'nm' in col]
    reflectance_cols = sorted(reflectance_cols, key=lambda x: int(x.split('R')[1].split('nm')[0]))[:31]

    print(f"✓ Loaded {len(df)} spectra")

    return df, reflectance_cols


def load_concentrations():
    """Load Concentrations.csv"""
    print("\nLoading Concentrations.csv...")
    df = pd.read_csv('../public/Concentrations.csv')

    print(f"✓ Loaded {len(df)} concentration records")

    return df


def match_samples():
    """Match fluorescence measurements with concentrations and spectra"""
    fluor_df = load_fluorescence_csv()
    spectra_df, reflectance_cols = load_spectra_data()
    conc_df = load_concentrations()

    training_data = []

    print("\nMatching samples...")

    # For GXT samples (pure GXT)
    for _, fluor_row in fluor_df[fluor_df['sample_name'] == 'GXT'].iterrows():
        gxt_conc = fluor_row['GXT']
        thickness = fluor_row['thickness']
        fluorescence_cts = fluor_row['fluorescence_cts']

        # Find matching spectrum
        # GXT samples at different thicknesses
        if thickness == 12.0:
            # Look for samples with 'H' suffix (12μm)
            matches = spectra_df[
                (spectra_df['User Color Name'].str.contains('GXT', case=False, na=False)) &
                (spectra_df['User Color Name'].str.contains(f'{int(gxt_conc)}%', case=False, na=False)) &
                (spectra_df['User Color Name'].str.endswith('H', na=False))
            ]
        else:
            # 8μm samples
            matches = spectra_df[
                (spectra_df['User Color Name'].str.contains('GXT', case=False, na=False)) &
                (spectra_df['User Color Name'].str.contains(f'{int(gxt_conc)}%', case=False, na=False)) &
                (~spectra_df['User Color Name'].str.endswith('H', na=False))
            ]

        if len(matches) > 0:
            spectrum_row = matches.iloc[0]
            reflectance = spectrum_row[reflectance_cols].values.astype(float)
            fluor_area = calculate_fluorescence_area(reflectance)

            training_data.append({
                'sample_name': f"GXT_{int(gxt_conc)}%_{int(thickness)}um",
                'GXT': gxt_conc,
                'BiVaO4': 0.0,
                'PG': 0.0,
                'PearlB': 0.0,
                'thickness': thickness,
                'fluorescence_area': fluor_area,
                'fluorescence_cts': fluorescence_cts
            })

    # For OPTI samples (mixtures)
    for _, fluor_row in fluor_df[fluor_df['sample_name'].str.startswith('OPT', na=False)].iterrows():
        sample_name = fluor_row['sample_name']
        thickness = fluor_row['thickness']
        fluorescence_cts = fluor_row['fluorescence_cts']

        # Find matching concentration data
        base_name = sample_name.replace('I', ' ')  # OPTI11 -> OPT 11
        conc_matches = conc_df[conc_df['Sample ID'].str.contains(base_name, case=False, na=False)]

        if len(conc_matches) == 0:
            continue

        conc_row = conc_matches.iloc[0]

        # Get concentrations
        gxt = float(conc_row['GXT-10 (%)']) if pd.notna(conc_row['GXT-10 (%)']) else 0.0
        bivao4 = float(conc_row['BiVaO4 (%)']) if pd.notna(conc_row['BiVaO4 (%)']) else 0.0
        pg = float(conc_row['PG (%)']) if pd.notna(conc_row['PG (%)']) else 0.0
        pearlb = float(conc_row['PearlB (%)']) if pd.notna(conc_row['PearlB (%)']) else 0.0

        # Find matching spectrum
        if thickness == 12.0:
            spectrum_matches = spectra_df[
                (spectra_df['User Color Name'].str.contains(sample_name, case=False, na=False)) &
                (spectra_df['User Color Name'].str.endswith('H', na=False))
            ]
        else:
            spectrum_matches = spectra_df[
                (spectra_df['User Color Name'].str.contains(sample_name, case=False, na=False)) &
                (~spectra_df['User Color Name'].str.endswith('H', na=False))
            ]

        if len(spectrum_matches) > 0:
            spectrum_row = spectrum_matches.iloc[0]
            reflectance = spectrum_row[reflectance_cols].values.astype(float)
            fluor_area = calculate_fluorescence_area(reflectance)

            training_data.append({
                'sample_name': f"{sample_name}_{int(thickness)}um",
                'GXT': gxt,
                'BiVaO4': bivao4,
                'PG': pg,
                'PearlB': pearlb,
                'thickness': thickness,
                'fluorescence_area': fluor_area,
                'fluorescence_cts': fluorescence_cts
            })

    print(f"✓ Matched {len(training_data)} samples")

    return training_data


def prepare_training_data():
    """Prepare final training arrays"""
    data = match_samples()

    if len(data) == 0:
        raise ValueError("No matching samples found!")

    df = pd.DataFrame(data)

    # Features: [GXT%, BiVaO4%, PG%, PearlB%, fluorescence_area]
    X = df[['GXT', 'BiVaO4', 'PG', 'PearlB', 'fluorescence_area']].values

    # Target: actual ct/s values
    y = df['fluorescence_cts'].values

    print(f"\n{'='*80}")
    print("TRAINING DATA SUMMARY")
    print(f"{'='*80}")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]} (GXT, BiVaO4, PG, PearlB, fluor_area)")
    print(f"\nTarget Statistics (ct/s):")
    print(f"  Mean:   {y.mean():.1f}")
    print(f"  Median: {np.median(y):.1f}")
    print(f"  Min:    {y.min():.1f}")
    print(f"  Max:    {y.max():.1f}")
    print(f"  Std:    {y.std():.1f}")

    # Save training data
    output = {
        'X': X,
        'y': y,
        'samples': data,
        'feature_names': ['GXT', 'BiVaO4', 'PG', 'PearlB', 'fluorescence_area']
    }

    with open('training_data/fluorescence_cts_training_data.pkl', 'wb') as f:
        pickle.dump(output, f)

    print(f"\n✓ Saved to: training_data/fluorescence_cts_training_data.pkl")
    print(f"{'='*80}\n")

    return output


if __name__ == "__main__":
    prepare_training_data()
