"""
OptiMix - Spectral Formulation Engine (Python/PyTorch Version)

Main application for training and using spectral prediction models
Compares Kubelka-Munk physics-based models with neural network models
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import argparse
import os

from types_constants import SampleData, WAVELENGTHS, REAGENTS_LIST, INITIAL_SAMPLES
from services.km_service import train_model, predict_reflectance
from utils.data_loader import load_master_data, parse_samples_from_csv


def plot_spectral_comparison(
    wavelengths: np.ndarray,
    reference: np.ndarray,
    single_prediction: np.ndarray,
    neural_prediction: np.ndarray,
    title: str = "Spectral Comparison"
):
    """
    Plot spectral curves comparing reference and predictions

    Args:
        wavelengths: Wavelength array
        reference: Reference spectrum
        single_prediction: Single-layer K-M prediction
        neural_prediction: Neural network prediction
        title: Plot title
    """
    plt.figure(figsize=(12, 6))

    plt.plot(wavelengths, reference, 'b-', linewidth=2, label='Reference', alpha=0.7)
    plt.plot(wavelengths, single_prediction, 'c--', linewidth=2, label='K-M Single Layer')
    plt.plot(wavelengths, neural_prediction, 'm-.', linewidth=2, label='Neural Network')

    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Reflectance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(400, 700)
    plt.ylim(-0.1, 1.5)

    plt.tight_layout()
    plt.show()


def create_initial_samples() -> List[SampleData]:
    """Create SampleData objects from INITIAL_SAMPLES"""
    return [
        SampleData(
            id=s['id'],
            name=s['name'],
            substrate=s['substrate'],
            thickness=s['thickness'],
            spectrum=s['spectrum'],
            concentrations=s['concentrations']
        )
        for s in INITIAL_SAMPLES
    ]


def main():
    parser = argparse.ArgumentParser(description='OptiMix - Spectral Formulation Engine')
    parser.add_argument('--data-dir', type=str, default='../public',
                       help='Directory containing master CSV files')
    parser.add_argument('--conc-file', type=str, default='Master conc.csv',
                       help='Concentration CSV file name')
    parser.add_argument('--spec-file', type=str, default='Master spec - master_sample_library.csv',
                       help='Spectral CSV file name')
    parser.add_argument('--use-initial', action='store_true',
                       help='Use initial sample data instead of loading from files')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting (for headless environments)')

    args = parser.parse_args()

    print("=" * 80)
    print("OptiMix - Spectral Formulation Engine (Python/PyTorch)")
    print("=" * 80)
    print()

    # Load data
    if args.use_initial:
        print("Using initial sample data...")
        samples = create_initial_samples()
    else:
        conc_path = os.path.join(args.data_dir, args.conc_file)
        spec_path = os.path.join(args.data_dir, args.spec_file)

        if os.path.exists(conc_path) and os.path.exists(spec_path):
            print(f"Loading master data from {args.data_dir}...")
            samples = load_master_data(conc_path, spec_path)
        else:
            print(f"Master CSV files not found in {args.data_dir}")
            print("Using initial sample data instead...")
            samples = create_initial_samples()

    if len(samples) == 0:
        print("ERROR: No samples loaded!")
        return

    print(f"Loaded {len(samples)} samples")
    print()

    # Train models
    print("-" * 80)
    print("Training Single-Layer Kubelka-Munk Model...")
    print("-" * 80)
    single_model = train_model(samples, 'single')
    print()

    print("-" * 80)
    print("Training Neural Network Model...")
    print("-" * 80)
    neural_model = train_model(samples, 'neural-net')
    print()

    # Test prediction on first sample
    print("-" * 80)
    print("Testing Predictions")
    print("-" * 80)

    test_sample = samples[0]
    print(f"Test Sample: {test_sample.name}")
    print(f"Concentrations: {test_sample.concentrations}")
    print(f"Thickness: {test_sample.thickness}")
    print()

    # Make predictions
    single_pred = predict_reflectance(
        test_sample.concentrations,
        single_model,
        test_sample.thickness
    )

    neural_pred = predict_reflectance(
        test_sample.concentrations,
        neural_model,
        test_sample.thickness
    )

    # Calculate errors
    single_error = np.mean(np.abs(single_pred - test_sample.spectrum))
    neural_error = np.mean(np.abs(neural_pred - test_sample.spectrum))

    print(f"Single-Layer K-M Mean Absolute Error: {single_error:.6f}")
    print(f"Neural Network Mean Absolute Error: {neural_error:.6f}")
    print()

    # Custom formulation example
    print("-" * 80)
    print("Custom Formulation Example")
    print("-" * 80)

    custom_concentrations = {
        'BiVaO4': 8.0,
        'PG': 3.0,
        'PB': 0.5,
        'LY': 2.0
    }
    custom_thickness = 4.0

    print(f"Formulation: {custom_concentrations}")
    print(f"Thickness: {custom_thickness}")
    print()

    single_custom = predict_reflectance(custom_concentrations, single_model, custom_thickness)
    neural_custom = predict_reflectance(custom_concentrations, neural_model, custom_thickness)

    print("Predicted Spectra (first 5 wavelengths):")
    print(f"Wavelength (nm): {WAVELENGTHS[:5]}")
    print(f"K-M Single:      {single_custom[:5]}")
    print(f"Neural Network:  {neural_custom[:5]}")
    print()

    # Plot if enabled
    if not args.no_plot:
        print("Generating plots...")

        # Plot test sample comparison
        plot_spectral_comparison(
            WAVELENGTHS,
            test_sample.spectrum,
            single_pred,
            neural_pred,
            title=f"Spectral Prediction Comparison - {test_sample.name}"
        )

        # Plot custom formulation
        plt.figure(figsize=(12, 6))
        plt.plot(WAVELENGTHS, single_custom, 'c-', linewidth=2, label='K-M Single Layer')
        plt.plot(WAVELENGTHS, neural_custom, 'm-', linewidth=2, label='Neural Network')
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Reflectance', fontsize=12)
        plt.title('Custom Formulation Prediction', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(400, 700)
        plt.ylim(-0.1, 1.5)
        plt.tight_layout()
        plt.show()

    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
