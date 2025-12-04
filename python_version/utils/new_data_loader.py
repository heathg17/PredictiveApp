"""
Data loader for new PP substrate dataset
- 4 reagents: GXT, BiVaO4, PG, PearlB (PBlue)
- Dual thicknesses: 8μm (standard) and 12μm (denoted with 'H' suffix)
- Expanded outputs: 31 wavelengths + CIELAB (L, a, b, c, h)
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class EnhancedSampleData:
    """Enhanced sample with CIELAB color coordinates"""
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


def load_new_dataset(
    concentrations_path: str,
    spectra_path: str
) -> Tuple[List[EnhancedSampleData], pd.DataFrame, pd.DataFrame]:
    """
    Load the new PP substrate dataset

    Args:
        concentrations_path: Path to Concentrations.csv
        spectra_path: Path to Spectra.csv

    Returns:
        List of EnhancedSampleData, concentrations DataFrame, spectra DataFrame
    """

    # Load concentrations
    conc_df = pd.read_csv(concentrations_path)
    print(f"Loaded {len(conc_df)} formulations from {concentrations_path}")

    # Load spectra (semicolon-separated with special header)
    spectra_df = pd.read_csv(spectra_path, sep=';', skiprows=3)  # Skip metadata rows
    print(f"Loaded {len(spectra_df)} spectral measurements from {spectra_path}")

    # Parse concentrations - remove % signs and convert to float
    conc_df['GXT'] = conc_df['GXT-10 (%)'].str.rstrip('%').astype(float)
    conc_df['BiVaO4'] = conc_df['BiVaO4 (%)'].str.rstrip('%').astype(float)
    conc_df['PG'] = conc_df['PG (%)'].str.rstrip('%').astype(float)
    conc_df['PearlB'] = conc_df['PearlB (%)'].str.rstrip('%').astype(float)

    # Extract wavelength columns (R400 nm through R700 nm)
    wavelength_cols = [col for col in spectra_df.columns if col.startswith('R') and 'nm' in col]
    print(f"Found {len(wavelength_cols)} wavelength columns")

    # Create samples by matching concentrations with spectra
    samples = []

    for idx, spec_row in spectra_df.iterrows():
        # Get sample name from 'User Color Name' column
        sample_name = spec_row.get('User Color Name', f'Sample_{idx}')

        # Determine thickness: 'H' suffix means 12μm, otherwise 8μm
        thickness = 12.0 if sample_name.endswith('H') else 8.0

        # Match with concentration data
        # Remove 'H' suffix to match with concentration table
        base_name = sample_name.rstrip('H') if sample_name.endswith('H') else sample_name

        # Handle different naming conventions:
        # 1. "PBLUE_1" in spectra -> "PBLUE1" in concentrations
        # 2. "PG_0.5" in spectra -> "PG0.5" in concentrations
        # 3. "OPTI6" in spectra -> "OPTI 6" in concentrations
        # 4. "T22" matches directly
        # 5. "52C" in spectra -> "F052C" in concentrations
        # 6. "F052F H" in spectra -> "F052FH" in concentrations (remove space before H)

        normalized_base = base_name.replace('_', '').replace(' ', '')  # Remove underscores and spaces

        # Try exact match first
        conc_match = conc_df[conc_df['Sample ID'] == base_name]

        # Try without underscores/spaces
        if conc_match.empty:
            conc_match = conc_df[conc_df['Sample ID'] == normalized_base]

        # Try with space for OPTI samples (OPTI6 -> OPTI 6)
        if conc_match.empty and normalized_base.startswith('OPTI'):
            # Extract number from OPTI6 -> 6
            opti_num = normalized_base.replace('OPTI', '')
            with_space = f"OPTI {opti_num}"
            conc_match = conc_df[conc_df['Sample ID'] == with_space]

        # Try adding F0 prefix for F052 samples (52C -> F052C)
        if conc_match.empty and normalized_base.startswith('52'):
            with_prefix = f"F0{normalized_base}"
            conc_match = conc_df[conc_df['Sample ID'] == with_prefix]

        if conc_match.empty:
            print(f"Warning: No concentration match for {sample_name} (tried: {base_name}, {normalized_base})")
            continue

        conc_row = conc_match.iloc[0]

        # Extract concentrations
        concentrations = {
            'GXT': conc_row['GXT'],
            'BiVaO4': conc_row['BiVaO4'],
            'PG': conc_row['PG'],
            'PearlB': conc_row['PearlB']
        }

        # Extract spectrum (31 wavelengths)
        spectrum = np.array([float(spec_row[col]) for col in wavelength_cols])

        # Extract CIELAB values
        L = float(spec_row['L'])
        a = float(spec_row['a'])
        b = float(spec_row['b'])
        c = float(spec_row['c'])
        h = float(spec_row['h'])

        sample = EnhancedSampleData(
            id=f"{base_name}_T{int(thickness)}",
            name=sample_name,
            concentrations=concentrations,
            thickness=thickness,
            spectrum=spectrum,
            L=L,
            a=a,
            b=b,
            c=c,
            h=h
        )

        samples.append(sample)

    print(f"✓ Created {len(samples)} samples")
    print(f"  8μm samples: {sum(1 for s in samples if s.thickness == 8.0)}")
    print(f"  12μm samples: {sum(1 for s in samples if s.thickness == 12.0)}")

    return samples, conc_df, spectra_df


def analyze_dataset(samples: List[EnhancedSampleData]) -> Dict:
    """Analyze dataset statistics"""

    stats = {
        'total_samples': len(samples),
        '8um_samples': sum(1 for s in samples if s.thickness == 8.0),
        '12um_samples': sum(1 for s in samples if s.thickness == 12.0),
        'reagents': ['GXT', 'BiVaO4', 'PG', 'PearlB'],
        'wavelength_count': 31,
        'outputs': ['spectrum (31)', 'L', 'a', 'b', 'c', 'h'],
        'total_outputs': 31 + 5  # 31 wavelengths + 5 CIELAB values
    }

    # Concentration ranges
    for reagent in stats['reagents']:
        concs = [s.concentrations[reagent] for s in samples]
        stats[f'{reagent}_min'] = min(concs)
        stats[f'{reagent}_max'] = max(concs)
        stats[f'{reagent}_mean'] = np.mean(concs)

    # CIELAB ranges
    stats['L_range'] = (min(s.L for s in samples), max(s.L for s in samples))
    stats['a_range'] = (min(s.a for s in samples), max(s.a for s in samples))
    stats['b_range'] = (min(s.b for s in samples), max(s.b for s in samples))

    # Spectrum ranges
    all_spectra = np.array([s.spectrum for s in samples])
    stats['spectrum_min'] = np.min(all_spectra)
    stats['spectrum_max'] = np.max(all_spectra)
    stats['spectrum_mean'] = np.mean(all_spectra)

    return stats


def prepare_training_data(samples: List[EnhancedSampleData]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare X (inputs) and Y (outputs) for neural network training

    Inputs (5 features):
        - GXT concentration (%)
        - BiVaO4 concentration (%)
        - PG concentration (%)
        - PearlB concentration (%)
        - Thickness (8 or 12 μm)

    Outputs (36 features):
        - 31 wavelength reflectances (400-700nm)
        - L (CIELab Lightness)
        - a (CIELab a*)
        - b (CIELab b*)
        - c (Chroma)
        - h (Hue angle)
    """

    X = []
    Y = []

    for sample in samples:
        # Input: [GXT, BiVaO4, PG, PearlB, Thickness]
        x = [
            sample.concentrations['GXT'] / 100.0,  # Normalize to 0-1
            sample.concentrations['BiVaO4'] / 100.0,
            sample.concentrations['PG'] / 100.0,
            sample.concentrations['PearlB'] / 100.0,
            sample.thickness / 12.0  # Normalize thickness (8→0.667, 12→1.0)
        ]

        # Output: [31 wavelengths, L, a, b, c, h]
        y = np.concatenate([
            sample.spectrum,  # 31 wavelengths
            [sample.L / 100.0],  # L normalized to 0-1
            [(sample.a + 128) / 256.0],  # a normalized (range: -128 to 128)
            [(sample.b + 128) / 256.0],  # b normalized (range: -128 to 128)
            [sample.c / 150.0],  # c normalized (typical max ~150)
            [sample.h / 360.0]   # h normalized (0-360 degrees)
        ])

        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)


if __name__ == "__main__":
    # Test the data loader
    samples, conc_df, spec_df = load_new_dataset(
        '../public/Concentrations.csv',
        '../public/Spectra.csv'
    )

    print("\n" + "="*80)
    print("DATASET ANALYSIS")
    print("="*80)

    stats = analyze_dataset(samples)
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n" + "="*80)
    print("SAMPLE DATA SHAPES")
    print("="*80)

    X, Y = prepare_training_data(samples)
    print(f"X shape: {X.shape} (inputs: 5 features)")
    print(f"Y shape: {Y.shape} (outputs: 36 features)")
    print(f"  - 31 wavelengths (400-700nm)")
    print(f"  - 5 CIELAB values (L, a, b, c, h)")
