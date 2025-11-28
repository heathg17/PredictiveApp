"""
Train PyTorch neural network on FULL master CSV dataset
"""
import numpy as np
from utils.data_loader import load_master_data
from services.km_service import train_model, predict_reflectance
from types_constants import WAVELENGTHS

print("=" * 80)
print("TRAINING ON FULL MASTER DATASET")
print("=" * 80)
print()

# Load full master dataset
print("Loading master CSV files...")
samples = load_master_data('../public/Master conc.csv', '../public/Master spec - master_sample_library.csv')
print(f"✓ Loaded {len(samples)} samples from master CSVs")
print()

# Filter samples with valid concentration data
samples_with_conc = [s for s in samples if len(s.concentrations) > 0 and any(c > 0 for c in s.concentrations.values())]
print(f"Samples with concentration data: {len(samples_with_conc)}")
print()

# Show sample distribution
print("Sample Statistics:")
print(f"  Total samples: {len(samples)}")
print(f"  With concentrations: {len(samples_with_conc)}")
print(f"  Without concentrations: {len(samples) - len(samples_with_conc)}")
print()

# Check fluorescence
fluorescent_count = sum(1 for s in samples_with_conc if np.any(s.spectrum > 1.0))
print(f"  Fluorescent samples (R > 1.0): {fluorescent_count}")
print()

# Get unique reagents
all_reagents = set()
for s in samples_with_conc:
    all_reagents.update(s.concentrations.keys())
print(f"  Unique reagents found: {len(all_reagents)}")
print(f"  Reagents: {', '.join(sorted(list(all_reagents))[:10])}...")
print()

# Train models
print("=" * 80)
print("TRAINING KUBELKA-MUNK MODEL")
print("=" * 80)
km_model = train_model(samples_with_conc, 'single')
print()

print("=" * 80)
print("TRAINING PYTORCH NEURAL NETWORK (with improvements)")
print("=" * 80)
print("  • Mix-up augmentation: ON")
print("  • Dropout regularization: ON")
print("  • Z-score normalization: ON")
print()
nn_model = train_model(samples_with_conc, 'neural-net')
print()

# Test on sample data
print("=" * 80)
print("TESTING PREDICTIONS")
print("=" * 80)
print()

# Test on a few samples
test_indices = [0, 50, 100, 200, 300]
errors_km = []
errors_nn = []

for idx in test_indices:
    if idx < len(samples_with_conc):
        sample = samples_with_conc[idx]

        km_pred = predict_reflectance(sample.concentrations, km_model, sample.thickness)
        nn_pred = predict_reflectance(sample.concentrations, nn_model, sample.thickness)

        km_error = np.mean(np.abs(km_pred - sample.spectrum))
        nn_error = np.mean(np.abs(nn_pred - sample.spectrum))

        errors_km.append(km_error)
        errors_nn.append(nn_error)

        print(f"Sample: {sample.name:20} | K-M MAE: {km_error:.6f} | NN MAE: {nn_error:.6f}")

print()
print(f"Average K-M MAE:  {np.mean(errors_km):.6f}")
print(f"Average NN MAE:   {np.mean(errors_nn):.6f}")
print()

# Test custom formulation
print("=" * 80)
print("CUSTOM FORMULATION TEST")
print("=" * 80)
print()

test_formulations = [
    {'name': 'Yellow Mix', 'conc': {'BiVaO4': 10.0, 'LY': 5.0}, 'thickness': 4.0},
    {'name': 'Green Blend', 'conc': {'PG': 8.0, 'LY': 3.0}, 'thickness': 4.0},
]

for test in test_formulations:
    km_pred = predict_reflectance(test['conc'], km_model, test['thickness'])
    nn_pred = predict_reflectance(test['conc'], nn_model, test['thickness'])

    print(f"{test['name']}:")
    print(f"  Formulation: {test['conc']}")
    print(f"  K-M Peak: {np.max(km_pred):.4f} @ {WAVELENGTHS[np.argmax(km_pred)]}nm")
    print(f"  NN Peak:  {np.max(nn_pred):.4f} @ {WAVELENGTHS[np.argmax(nn_pred)]}nm")
    print()

print("=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print()
print(f"Models trained on {len(samples_with_conc)} samples")
print(f"With mix-up augmentation: ~{len(samples_with_conc) * 5} effective samples")
print()
print("Models are ready for use!")
