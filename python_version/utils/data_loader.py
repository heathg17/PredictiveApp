"""
Data loading utilities for spectral data from CSV files
"""
import csv
import numpy as np
from typing import List, Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from types_constants import SampleData, WAVELENGTHS, REAGENTS_LIST


def parse_samples_from_csv(file_path: str) -> List[SampleData]:
    """
    Parse sample data from CSV file

    Expected format can be either:
    1. Concentration + Spectrum in one file
    2. Just concentrations or just spectra

    Args:
        file_path: Path to CSV file

    Returns:
        List of SampleData objects
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    if len(lines) < 2:
        return []

    header = [h.strip() for h in lines[0]]
    rows = lines[1:]

    # Create column index mapping
    col_idx = {h: i for i, h in enumerate(header)}

    # Detect spectrum columns (wavelengths)
    spectrum_indices = []
    for i, h in enumerate(header):
        h_lower = h.lower()
        # Check if column header looks like wavelength (e.g., "400", "410nm", "r_400")
        if (h.isdigit() and 400 <= int(h) <= 700) or \
           h_lower.replace('nm', '').isdigit() or \
           h_lower.startswith('r_') or \
           'reflect' in h_lower or \
           'spec' in h_lower:
            spectrum_indices.append(i)

    # Parse samples
    samples = []
    for row_idx, row in enumerate(rows):
        # Get sample metadata
        sample_id = row[col_idx.get('id', col_idx.get('ID', col_idx.get('Sample', 0)))] if len(row) > 0 else f"row-{row_idx}"
        sample_name = row[col_idx.get('name', col_idx.get('Name', 0))] if 'name' in col_idx or 'Name' in col_idx else sample_id
        substrate = row[col_idx.get('substrate', col_idx.get('Substrate', -1))] if 'substrate' in col_idx or 'Substrate' in col_idx else 'Unknown'

        # Get thickness
        thickness = 4.0
        for thickness_key in ['thickness', 'Thickness', 'X', 'x', 'Thickness (um)']:
            if thickness_key in col_idx and len(row) > col_idx[thickness_key]:
                try:
                    thickness = float(row[col_idx[thickness_key]])
                    break
                except ValueError:
                    pass

        # Extract spectrum
        spectrum = []
        if spectrum_indices:
            for idx in spectrum_indices:
                if idx < len(row):
                    try:
                        spectrum.append(float(row[idx]))
                    except ValueError:
                        spectrum.append(0.0)
                else:
                    spectrum.append(0.0)

        # Pad or trim to 31 wavelengths
        if len(spectrum) > len(WAVELENGTHS):
            spectrum = spectrum[:len(WAVELENGTHS)]
        while len(spectrum) < len(WAVELENGTHS):
            spectrum.append(0.0)

        spectrum = np.array(spectrum)

        # Extract concentrations
        concentrations = {}
        for reagent in REAGENTS_LIST:
            if reagent in col_idx and col_idx[reagent] < len(row):
                try:
                    # Remove " (%)" suffix if present in header
                    conc_val = float(row[col_idx[reagent]])
                    concentrations[reagent] = conc_val
                except ValueError:
                    concentrations[reagent] = 0.0

        # Also check for concentration columns with " (%)" suffix
        for h in header:
            reagent_name = h.replace(' (%)', '').strip()
            if reagent_name and reagent_name not in concentrations and h in col_idx:
                if col_idx[h] < len(row):
                    try:
                        concentrations[reagent_name] = float(row[col_idx[h]])
                    except ValueError:
                        pass

        samples.append(SampleData(
            id=sample_id,
            name=sample_name,
            substrate=substrate,
            thickness=thickness,
            spectrum=spectrum,
            concentrations=concentrations
        ))

    return samples


def load_master_data(conc_path: str, spec_path: str) -> List[SampleData]:
    """
    Load and merge master concentration and spectral data

    Args:
        conc_path: Path to concentration CSV file
        spec_path: Path to spectral CSV file

    Returns:
        List of merged SampleData objects
    """
    # Load concentration data
    with open(conc_path, 'r') as f:
        reader = csv.reader(f)
        conc_lines = list(reader)

    conc_header = [h.strip() for h in conc_lines[0]]
    conc_rows = conc_lines[1:]

    # Build concentration map
    conc_map = {}
    for row in conc_rows:
        if len(row) < 3:
            continue

        sample_name = row[0].strip()
        substrate = row[1].strip() if len(row) > 1 else 'Unknown'
        thickness = float(row[2]) if len(row) > 2 and row[2] else 4.0

        concentrations = {}
        # Parse concentration columns (starting from index 3)
        for i in range(3, len(conc_header)):
            reagent_name = conc_header[i].replace(' (%)', '').strip()
            if i < len(row):
                try:
                    concentrations[reagent_name] = float(row[i])
                except ValueError:
                    concentrations[reagent_name] = 0.0

        conc_map[sample_name] = {
            'substrate': substrate,
            'thickness': thickness,
            'concentrations': concentrations
        }

    # Load spectral data (no header row)
    with open(spec_path, 'r') as f:
        reader = csv.reader(f)
        spec_lines = list(reader)

    samples = []
    seen_samples = {}

    for line in spec_lines:
        if len(line) < 39:
            continue

        sample_name = line[0].strip()
        if not sample_name:
            continue

        # Extract spectrum (columns 8-38 should be the 31 reflectance values)
        spectrum = []
        for i in range(8, 39):
            if i < len(line):
                try:
                    spectrum.append(float(line[i]))
                except ValueError:
                    spectrum.append(0.0)
            else:
                spectrum.append(0.0)

        # Ensure exactly 31 values
        while len(spectrum) < 31:
            spectrum.append(0.0)
        if len(spectrum) > 31:
            spectrum = spectrum[:31]

        spectrum = np.array(spectrum)

        # Check if spectrum is valid
        valid_values = np.sum(spectrum > 0)
        is_valid = valid_values >= 15

        # Get concentration data
        conc_data = conc_map.get(sample_name, {
            'substrate': 'Unknown',
            'thickness': 4.0,
            'concentrations': {}
        })

        new_sample = SampleData(
            id=sample_name,
            name=sample_name,
            substrate=conc_data['substrate'],
            thickness=conc_data['thickness'],
            spectrum=spectrum,
            concentrations=conc_data['concentrations']
        )

        # Handle duplicates - keep the one with more valid spectrum values
        if sample_name in seen_samples:
            existing_idx = seen_samples[sample_name]
            existing_valid = np.sum(samples[existing_idx].spectrum > 0)

            if is_valid and (not existing_valid >= 15 or valid_values > existing_valid):
                samples[existing_idx] = new_sample
        else:
            seen_samples[sample_name] = len(samples)
            samples.append(new_sample)

    print(f"Loaded {len(samples)} samples from master CSVs")
    return samples
