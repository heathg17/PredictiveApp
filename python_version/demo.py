"""
Detailed demonstration of the PyTorch neural network model
"""
import numpy as np
import sys

from types_constants import SampleData, WAVELENGTHS, INITIAL_SAMPLES
from services.km_service import train_model, predict_reflectance

print("=" * 80)
print("OptiMix PyTorch Neural Network Demonstration")
print("=" * 80)
print()

# Create sample data
print("Loading sample data...")
samples = [
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
print(f"✓ Loaded {len(samples)} samples")
print()

# Show sample information
print("Sample Overview:")
print("-" * 80)
for i, s in enumerate(samples, 1):
    has_fluor = np.any(s.spectrum > 1.0)
    fluor_marker = "🌟 FLUORESCENT" if has_fluor else ""
    print(f"{i}. {s.name:12} | Substrate: {s.substrate:8} | "
          f"Thickness: {s.thickness}μm {fluor_marker}")
    print(f"   Reagents: {', '.join([f'{k}:{v:.1f}%' for k, v in s.concentrations.items() if v > 0])}")
print()

# Train both models
print("=" * 80)
print("Training Models")
print("=" * 80)
print()

print("1. Kubelka-Munk Single Layer Model (Physics-Based)")
print("-" * 80)
km_model = train_model(samples, 'single')
print(f"✓ Model trained with {len(km_model.alpha)} reagents")
print()

print("2. PyTorch Neural Network Model (Data-Driven)")
print("-" * 80)
nn_model = train_model(samples, 'neural-net')
print(f"✓ Neural network trained")
print(f"  Architecture: {nn_model.neural_net['model'].fc1.in_features} inputs → "
      f"{nn_model.neural_net['model'].fc1.out_features} hidden → "
      f"{nn_model.neural_net['model'].fc2.out_features} outputs")
print()

# Test on each sample
print("=" * 80)
print("Model Performance on Training Samples")
print("=" * 80)
print()

km_errors = []
nn_errors = []

for sample in samples:
    print(f"Sample: {sample.name}")
    print(f"  Formulation: {', '.join([f'{k}:{v:.1f}%' for k, v in sample.concentrations.items() if v > 0])}")

    # Predict with both models
    km_pred = predict_reflectance(sample.concentrations, km_model, sample.thickness)
    nn_pred = predict_reflectance(sample.concentrations, nn_model, sample.thickness)

    # Calculate errors
    km_error = np.mean(np.abs(km_pred - sample.spectrum))
    nn_error = np.mean(np.abs(nn_pred - sample.spectrum))

    km_errors.append(km_error)
    nn_errors.append(nn_error)

    print(f"  K-M Error:  {km_error:.6f}")
    print(f"  NN Error:   {nn_error:.6f}")

    # Show key wavelengths
    idx_550 = 15  # 550nm
    print(f"  Reflectance @ 550nm:")
    print(f"    Reference: {sample.spectrum[idx_550]:.4f}")
    print(f"    K-M:       {km_pred[idx_550]:.4f}")
    print(f"    Neural:    {nn_pred[idx_550]:.4f}")
    print()

print("=" * 80)
print("Overall Performance Summary")
print("=" * 80)
print(f"Kubelka-Munk Model:")
print(f"  Average MAE: {np.mean(km_errors):.6f}")
print(f"  Std Dev:     {np.std(km_errors):.6f}")
print()
print(f"Neural Network Model:")
print(f"  Average MAE: {np.mean(nn_errors):.6f}")
print(f"  Std Dev:     {np.std(nn_errors):.6f}")
print()

# Test custom formulations
print("=" * 80)
print("Custom Formulation Predictions")
print("=" * 80)
print()

custom_formulations = [
    {'name': 'Yellow Blend', 'conc': {'BiVaO4': 10.0, 'LY': 5.0}, 'thickness': 4.0},
    {'name': 'Green Mix', 'conc': {'PG': 8.0, 'LY': 3.0}, 'thickness': 4.0},
    {'name': 'Blue Tint', 'conc': {'PB': 2.0, 'TiO2': 1.0}, 'thickness': 4.0},
]

for formulation in custom_formulations:
    print(f"Formulation: {formulation['name']}")
    print(f"  Composition: {', '.join([f'{k}:{v:.1f}%' for k, v in formulation['conc'].items()])}")
    print(f"  Thickness: {formulation['thickness']}μm")

    km_custom = predict_reflectance(formulation['conc'], km_model, formulation['thickness'])
    nn_custom = predict_reflectance(formulation['conc'], nn_model, formulation['thickness'])

    # Show spectrum at key wavelengths
    print(f"  Predicted Reflectance:")
    for wl_idx, wl_name in [(0, '400nm'), (10, '500nm'), (15, '550nm'), (20, '600nm'), (30, '700nm')]:
        print(f"    {wl_name:6} - K-M: {km_custom[wl_idx]:.4f}, NN: {nn_custom[wl_idx]:.4f}")
    print()

# Check for fluorescence predictions
print("=" * 80)
print("Fluorescence Detection")
print("=" * 80)
print()

# Test high fluorescent pigment concentration
fluor_test = {'GXT': 25.0}
km_fluor = predict_reflectance(fluor_test, km_model, 4.0)
nn_fluor = predict_reflectance(fluor_test, nn_model, 4.0)

print("High concentration fluorescent pigment (GXT: 25%)")
print(f"  K-M max reflectance: {np.max(km_fluor):.4f} {'(clamped at 1.0)' if np.max(km_fluor) <= 1.0 else ''}")
print(f"  NN max reflectance:  {np.max(nn_fluor):.4f} {'(fluorescence detected!)' if np.max(nn_fluor) > 1.0 else ''}")
print()

print("=" * 80)
print("PyTorch Model Analysis Complete")
print("=" * 80)
print()
print("Key Observations:")
print("  • Neural network learns non-linear pigment interactions")
print("  • Can predict fluorescence effects (R > 1.0)")
print("  • Better interpolation for complex mixtures")
print("  • K-M model is more interpretable and physically grounded")
print()
print("Both models complement each other:")
print("  • Use K-M for physical insight and simple formulations")
print("  • Use NN for accurate predictions with complex pigments")
