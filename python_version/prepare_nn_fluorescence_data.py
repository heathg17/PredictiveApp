"""
Prepare training data for multi-output neural network with fluorescence
Combines spectral, CIELAB, and fluorescence data for NN training
"""
import pandas as pd
import numpy as np
import pickle

# Reuse the proven matching logic from prepare_fluorescence_cts_data.py

def parse_concentration(conc_str):
    """Parse concentration string like '5%', '10%' to float"""
    if pd.isna(conc_str) or conc_str == '':
        return 0.0
    return float(str(conc_str).replace('%', ''))


def safe_parse_pct(val):
    """Safely parse percentage values"""
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        return float(val.rstrip('%'))
    return float(val)


def load_and_match_data():
    """Load and match data from all three CSV files"""

    # Load fluorescence
    print("Loading Fluorescence.csv...")
    fluor_df = pd.read_csv('../public/Fluorescence.csv')

    # Parse fluorescence values
    if 'New Fl Mean ct/s' in fluor_df.columns:
        fluor_df['fluorescence_cts'] = pd.to_numeric(fluor_df['New Fl Mean ct/s'], errors='coerce')
        fluor_df['fluorescence_cts'] = fluor_df['fluorescence_cts'].fillna(
            pd.to_numeric(fluor_df['Fluro ct/s'], errors='coerce')
        )
    else:
        fluor_df['fluorescence_cts'] = pd.to_numeric(fluor_df['Fluro ct/s'], errors='coerce')

    fluor_df['GXT'] = fluor_df['Conc, wt/wt'].apply(parse_concentration)
    fluor_df['thickness'] = pd.to_numeric(fluor_df['Band Vol cm3/m2'], errors='coerce')
    fluor_df['sample_name'] = fluor_df['Material'].astype(str)

    fluor_df = fluor_df.dropna(subset=['fluorescence_cts', 'thickness'])
    fluor_df = fluor_df[fluor_df['fluorescence_cts'] >= 0]

    print(f"✓ Loaded {len(fluor_df)} fluorescence measurements")
    print(f"  ct/s range: {fluor_df['fluorescence_cts'].min():.0f} - {fluor_df['fluorescence_cts'].max():.0f}")

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

    print(f"✓ Loaded {len(conc_df)} concentration records")

    training_samples = []

    # Match GXT samples
    print("\nMatching GXT samples...")
    for _, fluor_row in fluor_df[fluor_df['sample_name'] == 'GXT'].iterrows():
        gxt_conc = fluor_row['GXT']
        thickness = fluor_row['thickness']
        fluorescence_cts = fluor_row['fluorescence_cts']

        # Match spectrum (format: "GXT5", "GXT10H", etc.)
        search_name = f"GXT{int(gxt_conc)}"
        if thickness == 12.0:
            search_name_with_thickness = search_name + "H"
            matches = spectra_df[spectra_df['User Color Name'] == search_name_with_thickness]
        else:
            matches = spectra_df[spectra_df['User Color Name'] == search_name]

        if len(matches) > 0:
            spectrum_row = matches.iloc[0]
            reflectance = spectrum_row[reflectance_cols].values.astype(float)

            training_samples.append({
                'id': f"GXT_{int(gxt_conc)}%_T{int(thickness)}",
                'GXT': gxt_conc,
                'BiVaO4': 0.0,
                'PG': 0.0,
                'PearlB': 0.0,
                'thickness': thickness,
                'reflectance': reflectance,
                'L': float(spectrum_row['L']),
                'a': float(spectrum_row['a']),
                'b': float(spectrum_row['b']),
                'c': float(spectrum_row['c']),
                'h': float(spectrum_row['h']),
                'fluorescence_cts': fluorescence_cts
            })

    print(f"  Matched {len(training_samples)} GXT samples")

    # Match OPTI samples
    print("\nMatching OPTI samples...")
    opti_count = 0
    for _, fluor_row in fluor_df[fluor_df['sample_name'].str.startswith('OPT', na=False)].iterrows():
        sample_name = fluor_row['sample_name']
        thickness = fluor_row['thickness']
        fluorescence_cts = fluor_row['fluorescence_cts']

        # Match concentration
        base_name = sample_name.replace('I', ' ')  # OPTI11 -> OPT 11
        conc_matches = conc_df[conc_df['Sample ID'].str.contains(base_name, case=False, na=False)]

        if len(conc_matches) == 0:
            continue

        conc_row = conc_matches.iloc[0]

        gxt = safe_parse_pct(conc_row['GXT-10 (%)'])
        bivao4 = safe_parse_pct(conc_row['BiVaO4 (%)'])
        pg = safe_parse_pct(conc_row['PG (%)'])
        pearlb = safe_parse_pct(conc_row['PearlB (%)'])

        # Match spectrum
        if thickness == 12.0:
            spec_matches = spectra_df[
                (spectra_df['User Color Name'].str.contains(sample_name, case=False, na=False)) &
                (spectra_df['User Color Name'].str.endswith('H', na=False))
            ]
        else:
            spec_matches = spectra_df[
                (spectra_df['User Color Name'].str.contains(sample_name, case=False, na=False)) &
                (~spectra_df['User Color Name'].str.endswith('H', na=False))
            ]

        if len(spec_matches) > 0:
            spectrum_row = spec_matches.iloc[0]
            reflectance = spectrum_row[reflectance_cols].values.astype(float)

            training_samples.append({
                'id': f"{sample_name}_T{int(thickness)}",
                'GXT': gxt,
                'BiVaO4': bivao4,
                'PG': pg,
                'PearlB': pearlb,
                'thickness': thickness,
                'reflectance': reflectance,
                'L': float(spectrum_row['L']),
                'a': float(spectrum_row['a']),
                'b': float(spectrum_row['b']),
                'c': float(spectrum_row['c']),
                'h': float(spectrum_row['h']),
                'fluorescence_cts': fluorescence_cts
            })
            opti_count += 1

    print(f"  Matched {opti_count} OPTI samples")

    return training_samples


def prepare_nn_training_data():
    """Prepare X and Y arrays for neural network training"""

    samples = load_and_match_data()

    if len(samples) == 0:
        raise ValueError("No matching samples found!")

    print(f"\n{'='*80}")
    print(f"MATCHED {len(samples)} TOTAL SAMPLES")
    print(f"{'='*80}")

    # Prepare input/output arrays
    X = []
    Y_spectral = []
    Y_cielab = []
    Y_fluorescence = []

    for sample in samples:
        # Input: [GXT%, BiVaO4%, PG%, PearlB%, thickness]
        X.append([
            sample['GXT'],
            sample['BiVaO4'],
            sample['PG'],
            sample['PearlB'],
            sample['thickness']
        ])

        # Output 1: Spectral (31 wavelengths)
        Y_spectral.append(sample['reflectance'])

        # Output 2: CIELAB (5 values)
        Y_cielab.append([
            sample['L'],
            sample['a'],
            sample['b'],
            sample['c'],
            sample['h']
        ])

        # Output 3: Fluorescence (1 value)
        Y_fluorescence.append([sample['fluorescence_cts']])

    X = np.array(X)
    Y_spectral = np.array(Y_spectral)
    Y_cielab = np.array(Y_cielab)
    Y_fluorescence = np.array(Y_fluorescence)

    print(f"\nData shapes:")
    print(f"  X: {X.shape} (concentrations + thickness)")
    print(f"  Y_spectral: {Y_spectral.shape} (31 wavelengths)")
    print(f"  Y_cielab: {Y_cielab.shape} (L, a, b, c, h)")
    print(f"  Y_fluorescence: {Y_fluorescence.shape} (ct/s)")

    print(f"\nFluorescence statistics:")
    print(f"  Min: {Y_fluorescence.min():.0f} ct/s")
    print(f"  Max: {Y_fluorescence.max():.0f} ct/s")
    print(f"  Mean: {Y_fluorescence.mean():.0f} ct/s")
    print(f"  Std: {Y_fluorescence.std():.0f} ct/s")

    # Save training data
    output = {
        'X': X,
        'Y_spectral': Y_spectral,
        'Y_cielab': Y_cielab,
        'Y_fluorescence': Y_fluorescence,
        'samples': samples
    }

    with open('training_data/nn_fluorescence_training_data.pkl', 'wb') as f:
        pickle.dump(output, f)

    print(f"\n✓ Saved to: training_data/nn_fluorescence_training_data.pkl")
    print(f"{'='*80}\n")

    return output


if __name__ == "__main__":
    prepare_nn_training_data()
