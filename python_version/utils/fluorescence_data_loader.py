"""
Enhanced data loader that includes fluorescence ct/s values
Combines spectral, CIELAB, and fluorescence data for multi-output NN training
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class FluorescenceSampleData:
    """Sample with spectral, CIELAB, and fluorescence data"""
    id: str
    name: str
    concentrations: Dict[str, float]  # GXT, BiVaO4, PG, PearlB
    thickness: float  # 8 or 12 μm
    spectrum: np.ndarray  # 31 wavelengths (400-700nm, 10nm intervals)
    L: float  # CIELab Lightness
    a: float  # CIELab a* (green-red)
    b: float  # CIELab b* (blue-yellow)
    c: float  # Chroma
    h: float  # Hue angle
    fluorescence_cts: float  # Fluorescence intensity (ct/s)


def load_fluorescence_dataset(
    concentrations_path: str,
    spectra_path: str,
    fluorescence_path: str
) -> List[FluorescenceSampleData]:
    """
    Load dataset with fluorescence data

    Args:
        concentrations_path: Path to Concentrations.csv
        spectra_path: Path to Spectra.csv
        fluorescence_path: Path to Fluorescence.csv

    Returns:
        List of FluorescenceSampleData with matched samples
    """

    # Load concentrations
    conc_df = pd.read_csv(concentrations_path)
    print(f"Loaded {len(conc_df)} formulations from {concentrations_path}")

    # Load spectra (semicolon-separated with special header)
    # Format: Line 1: sep=;, Line 2: metadata (commas!), Line 3: "Exported From; Saved Library", Line 4: headers
    spectra_df = pd.read_csv(spectra_path, sep=';', skiprows=3)  # Skip first 3 lines
    print(f"Loaded {len(spectra_df)} spectral measurements from {spectra_path}")

    # Load fluorescence
    fluor_df = pd.read_csv(fluorescence_path)
    print(f"Loaded {len(fluor_df)} fluorescence measurements from {fluorescence_path}")

    # Parse concentrations
    def safe_parse_pct(val):
        """Safely parse percentage values"""
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            return float(val.rstrip('%'))
        return float(val)

    conc_df['GXT'] = conc_df['GXT-10 (%)'].apply(safe_parse_pct)
    conc_df['BiVaO4'] = conc_df['BiVaO4 (%)'].apply(safe_parse_pct)
    conc_df['PG'] = conc_df['PG (%)'].apply(safe_parse_pct)
    conc_df['PearlB'] = conc_df['PearlB (%)'].apply(safe_parse_pct)

    # Parse fluorescence data
    def parse_concentration(conc_str):
        """Parse concentration string like '5%', '10%' to float"""
        if pd.isna(conc_str) or conc_str == '':
            return 0.0
        return float(str(conc_str).replace('%', ''))

    fluor_df['GXT_conc'] = fluor_df['Conc, wt/wt'].apply(parse_concentration)
    fluor_df['thickness_val'] = pd.to_numeric(fluor_df['Band Vol cm3/m2'], errors='coerce')
    fluor_df['material_name'] = fluor_df['Material'].astype(str)

    # Use 'New Fl Mean ct/s' if available, otherwise 'Fluro ct/s'
    if 'New Fl Mean ct/s' in fluor_df.columns:
        fluor_df['fluorescence_cts'] = pd.to_numeric(fluor_df['New Fl Mean ct/s'], errors='coerce')
        fluor_df['fluorescence_cts'] = fluor_df['fluorescence_cts'].fillna(
            pd.to_numeric(fluor_df['Fluro ct/s'], errors='coerce')
        )
    else:
        fluor_df['fluorescence_cts'] = pd.to_numeric(fluor_df['Fluro ct/s'], errors='coerce')

    # Filter valid fluorescence rows
    fluor_df = fluor_df.dropna(subset=['fluorescence_cts', 'thickness_val'])
    fluor_df = fluor_df[fluor_df['fluorescence_cts'] >= 0]  # Allow 0 for non-fluorescent samples

    print(f"  Valid fluorescence data: {len(fluor_df)} samples")
    print(f"  ct/s range: {fluor_df['fluorescence_cts'].min():.0f} - {fluor_df['fluorescence_cts'].max():.0f}")

    # Extract wavelength columns
    wavelength_cols = [col for col in spectra_df.columns if 'R' in col and 'nm' in col]
    wavelength_cols = sorted(wavelength_cols, key=lambda x: int(x.split('R')[1].split('nm')[0]))[:31]
    print(f"Found {len(wavelength_cols)} wavelength columns")

    samples = []

    # Match samples across all three datasets
    for _, fluor_row in fluor_df.iterrows():
        material = fluor_row['material_name']
        gxt_conc = fluor_row['GXT_conc']
        thickness = fluor_row['thickness_val']
        fluorescence_cts = fluor_row['fluorescence_cts']

        # Handle GXT samples (pure GXT)
        if material == 'GXT':
            # Match spectrum
            if thickness == 12.0:
                spec_matches = spectra_df[
                    (spectra_df['User Color Name'].str.contains('GXT', case=False, na=False)) &
                    (spectra_df['User Color Name'].str.contains(f'{int(gxt_conc)}%', case=False, na=False)) &
                    (spectra_df['User Color Name'].str.endswith('H', na=False))
                ]
            else:
                spec_matches = spectra_df[
                    (spectra_df['User Color Name'].str.contains('GXT', case=False, na=False)) &
                    (spectra_df['User Color Name'].str.contains(f'{int(gxt_conc)}%', case=False, na=False)) &
                    (~spectra_df['User Color Name'].str.endswith('H', na=False))
                ]

            if len(spec_matches) == 0:
                continue

            spec_row = spec_matches.iloc[0]

            # Create sample
            sample = FluorescenceSampleData(
                id=f"GXT_{int(gxt_conc)}%_T{int(thickness)}",
                name=f"GXT {int(gxt_conc)}%",
                concentrations={'GXT': gxt_conc, 'BiVaO4': 0.0, 'PG': 0.0, 'PearlB': 0.0},
                thickness=thickness,
                spectrum=np.array([float(spec_row[col]) for col in wavelength_cols]),
                L=float(spec_row['L']),
                a=float(spec_row['a']),
                b=float(spec_row['b']),
                c=float(spec_row['c']),
                h=float(spec_row['h']),
                fluorescence_cts=fluorescence_cts
            )
            samples.append(sample)

        # Handle OPTI samples (mixtures)
        elif material.startswith('OPT'):
            sample_name = material

            # Match concentration data
            base_name = sample_name.replace('I', ' ')  # OPTI11 -> OPT 11
            conc_matches = conc_df[conc_df['Sample ID'].str.contains(base_name, case=False, na=False)]

            if len(conc_matches) == 0:
                continue

            conc_row = conc_matches.iloc[0]

            # Get concentrations
            concentrations = {
                'GXT': safe_parse_pct(conc_row['GXT-10 (%)']),
                'BiVaO4': safe_parse_pct(conc_row['BiVaO4 (%)']),
                'PG': safe_parse_pct(conc_row['PG (%)']),
                'PearlB': safe_parse_pct(conc_row['PearlB (%)'])
            }

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

            if len(spec_matches) == 0:
                continue

            spec_row = spec_matches.iloc[0]

            # Create sample
            sample = FluorescenceSampleData(
                id=f"{sample_name}_T{int(thickness)}",
                name=sample_name,
                concentrations=concentrations,
                thickness=thickness,
                spectrum=np.array([float(spec_row[col]) for col in wavelength_cols]),
                L=float(spec_row['L']),
                a=float(spec_row['a']),
                b=float(spec_row['b']),
                c=float(spec_row['c']),
                h=float(spec_row['h']),
                fluorescence_cts=fluorescence_cts
            )
            samples.append(sample)

    print(f"\n✓ Matched {len(samples)} samples with complete data")
    print(f"  8μm samples: {sum(1 for s in samples if s.thickness == 8.0)}")
    print(f"  12μm samples: {sum(1 for s in samples if s.thickness == 12.0)}")

    # Show fluorescence statistics
    fluor_vals = [s.fluorescence_cts for s in samples]
    print(f"\n  Fluorescence statistics:")
    print(f"    Min: {min(fluor_vals):.0f} ct/s")
    print(f"    Max: {max(fluor_vals):.0f} ct/s")
    print(f"    Mean: {np.mean(fluor_vals):.0f} ct/s")
    print(f"    Std: {np.std(fluor_vals):.0f} ct/s")

    return samples


def prepare_fluorescence_training_data(samples: List[FluorescenceSampleData]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare X (inputs) and Y outputs (spectral, CIELAB, fluorescence) for NN training

    Inputs (5 features):
        - GXT concentration (%)
        - BiVaO4 concentration (%)
        - PG concentration (%)
        - PearlB concentration (%)
        - Thickness (8 or 12 μm)

    Outputs:
        Y_spectral (31): wavelength reflectances (400-700nm)
        Y_cielab (5): L, a, b, c, h
        Y_fluorescence (1): fluorescence ct/s
    """

    X = []
    Y_spectral = []
    Y_cielab = []
    Y_fluorescence = []

    for sample in samples:
        # Input: [GXT, BiVaO4, PG, PearlB, Thickness]
        x = [
            sample.concentrations['GXT'],
            sample.concentrations['BiVaO4'],
            sample.concentrations['PG'],
            sample.concentrations['PearlB'],
            sample.thickness
        ]

        # Spectral output (31 wavelengths)
        y_spectral = sample.spectrum

        # CIELAB output (5 values)
        y_cielab = np.array([sample.L, sample.a, sample.b, sample.c, sample.h])

        # Fluorescence output (1 value)
        y_fluorescence = np.array([sample.fluorescence_cts])

        X.append(x)
        Y_spectral.append(y_spectral)
        Y_cielab.append(y_cielab)
        Y_fluorescence.append(y_fluorescence)

    return np.array(X), np.array(Y_spectral), np.array(Y_cielab), np.array(Y_fluorescence)


if __name__ == "__main__":
    # Test the fluorescence data loader
    samples = load_fluorescence_dataset(
        '../public/Concentrations.csv',
        '../public/Spectra.csv',
        '../public/Fluorescence.csv'
    )

    print("\n" + "="*80)
    print("SAMPLE DATA SHAPES")
    print("="*80)

    X, Y_spec, Y_cielab, Y_fluor = prepare_fluorescence_training_data(samples)
    print(f"X shape: {X.shape} (inputs: 5 features)")
    print(f"Y_spectral shape: {Y_spec.shape} (31 wavelengths)")
    print(f"Y_cielab shape: {Y_cielab.shape} (5 CIELAB values)")
    print(f"Y_fluorescence shape: {Y_fluor.shape} (1 ct/s value)")
