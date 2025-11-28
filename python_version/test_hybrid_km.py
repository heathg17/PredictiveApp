"""
Test Hybrid Kubelka-Munk + Fluorescence NN Model

This tests a physics-informed hybrid approach:
1. K-M theory for non-fluorescent pigments (BiVaO4, PG, PearlB)
2. Neural network learns only the fluorescence component (GXT)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time
import pickle

from utils.new_data_loader import load_new_dataset, prepare_training_data
from models.km_hybrid_model import train_hybrid_model, predict_hybrid
from models.enhanced_neural_network import train_enhanced_neural_network, predict_enhanced_neural_network

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 100)
print("HYBRID K-M + FLUORESCENCE NN TEST")
print("=" * 100)
print()

# Load data
print("Loading dataset...")
samples, _, _ = load_new_dataset(
    '../public/Concentrations.csv',
    '../public/Spectra.csv'
)

X, Y = prepare_training_data(samples)
print(f"✓ Dataset loaded: {X.shape[0]} samples")
print(f"  Inputs: {X.shape[1]} features (GXT, BiVaO4, PG, PearlB, Thickness)")
print(f"  Outputs: {Y.shape[1]} features (31 spectral + 5 CIELAB)")
print()

# Train/Val/Test split
print("Splitting dataset (60/20/20)...")
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_SEED
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.25, random_state=RANDOM_SEED
)

print(f"  Training:   {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test:       {len(X_test)} samples")
print()

# ============================================================================
# Train Baseline Model (for comparison)
# ============================================================================
print("\n" + "=" * 100)
print("BASELINE: Pure Neural Network (Current Approach)")
print("=" * 100)

start_time = time.time()
baseline_model = train_enhanced_neural_network(
    X_train, Y_train,
    X_val, Y_val,
    hidden_size=64,
    learning_rate=0.001,
    epochs=2000,
    batch_size=16,
    l2_lambda=0.001,
    dropout_rate=0.2,
    verbose=True
)
baseline_time = time.time() - start_time

# ============================================================================
# Train Hybrid K-M Model
# ============================================================================
print("\n" + "=" * 100)
print("HYBRID: Kubelka-Munk + Fluorescence NN")
print("=" * 100)

start_time = time.time()
hybrid_model = train_hybrid_model(
    X_train, Y_train,
    X_val, Y_val,
    hidden_size=32,  # Smaller network since only learning fluorescence
    learning_rate=0.001,
    epochs=2000,
    batch_size=16,
    l2_lambda=0.001,
    dropout_rate=0.2,
    verbose=True
)
hybrid_time = time.time() - start_time

# ============================================================================
# Evaluate Both Models
# ============================================================================
print("\n" + "=" * 100)
print("EVALUATION ON TEST SET")
print("=" * 100)

def evaluate_model(model_data, X_test, Y_test, model_type='baseline'):
    """Evaluate model and return metrics"""
    Y_pred_list = []

    for x in X_test:
        if model_type == 'baseline':
            spectrum, cielab = predict_enhanced_neural_network(
                x,
                model_data['model'],
                model_data['input_mean'],
                model_data['input_std'],
                model_data['output_mean'],
                model_data['output_std']
            )
        else:  # hybrid
            GXT, BiVaO4, PG, PearlB, thickness = x[0]*100, x[1]*100, x[2]*100, x[3]*100, x[4]*12
            spectrum, cielab = predict_hybrid(
                GXT, BiVaO4, PG, PearlB, thickness,
                model_data['model'],
                model_data['input_mean'],
                model_data['input_std'],
                model_data['output_mean'],
                model_data['output_std']
            )

        # Reconstruct full prediction
        y_pred = np.concatenate([
            spectrum,
            [cielab['L'] / 100.0,
             (cielab['a'] + 128) / 256.0,
             (cielab['b'] + 128) / 256.0,
             cielab['c'] / 150.0,
             cielab['h'] / 360.0]
        ])
        Y_pred_list.append(y_pred)

    Y_pred = np.array(Y_pred_list)

    # Calculate metrics
    # Spectral
    Y_test_spectral = Y_test[:, :31]
    Y_pred_spectral = Y_pred[:, :31]

    spectral_mae = mean_absolute_error(Y_test_spectral.flatten(), Y_pred_spectral.flatten())
    spectral_rmse = np.sqrt(mean_squared_error(Y_test_spectral.flatten(), Y_pred_spectral.flatten()))
    spectral_r2 = r2_score(Y_test_spectral.flatten(), Y_pred_spectral.flatten())

    # CIELAB
    Y_test_cielab = Y_test[:, 31:]
    Y_pred_cielab = Y_pred[:, 31:]

    cielab_mae = mean_absolute_error(Y_test_cielab.flatten(), Y_pred_cielab.flatten())
    cielab_rmse = np.sqrt(mean_squared_error(Y_test_cielab.flatten(), Y_pred_cielab.flatten()))
    cielab_r2 = r2_score(Y_test_cielab.flatten(), Y_pred_cielab.flatten())

    return {
        'spectral_mae': spectral_mae,
        'spectral_rmse': spectral_rmse,
        'spectral_r2': spectral_r2,
        'cielab_mae': cielab_mae,
        'cielab_rmse': cielab_rmse,
        'cielab_r2': cielab_r2,
        'Y_pred': Y_pred
    }

print("\nEvaluating Baseline...")
baseline_results = evaluate_model(baseline_model, X_test, Y_test, 'baseline')

print("\nEvaluating Hybrid K-M...")
hybrid_results = evaluate_model(hybrid_model, X_test, Y_test, 'hybrid')

# ============================================================================
# Results Comparison
# ============================================================================
print("\n" + "=" * 100)
print("RESULTS COMPARISON")
print("=" * 100)

print("\n{:<25} {:>15} {:>15} {:>15}".format(
    "Metric", "Baseline NN", "Hybrid K-M", "Improvement"
))
print("-" * 100)

metrics = [
    ('Spectral MAE', baseline_results['spectral_mae'], hybrid_results['spectral_mae']),
    ('Spectral RMSE', baseline_results['spectral_rmse'], hybrid_results['spectral_rmse']),
    ('Spectral R²', baseline_results['spectral_r2'], hybrid_results['spectral_r2']),
    ('CIELAB MAE', baseline_results['cielab_mae'], hybrid_results['cielab_mae']),
    ('CIELAB RMSE', baseline_results['cielab_rmse'], hybrid_results['cielab_rmse']),
    ('CIELAB R²', baseline_results['cielab_r2'], hybrid_results['cielab_r2']),
]

for name, baseline_val, hybrid_val in metrics:
    if 'R²' in name:
        improvement = ((hybrid_val - baseline_val) / abs(baseline_val)) * 100
        print("{:<25} {:>15.6f} {:>15.6f} {:>14.1f}%".format(
            name, baseline_val, hybrid_val, improvement
        ))
    else:
        improvement = ((baseline_val - hybrid_val) / baseline_val) * 100
        print("{:<25} {:>15.6f} {:>15.6f} {:>14.1f}%".format(
            name, baseline_val, hybrid_val, improvement
        ))

print("\n{:<25} {:>15} {:>15}".format("", "Baseline", "Hybrid"))
print("-" * 100)
print("{:<25} {:>15.1f}s {:>15.1f}s".format("Training Time", baseline_time, hybrid_time))
print("{:<25} {:>15} {:>15}".format(
    "Parameters",
    baseline_model['model'].count_parameters(),
    hybrid_model['model'].count_parameters()
))
print("{:<25} {:>15.6f} {:>15.6f}".format(
    "Best Val Loss",
    baseline_model['best_val_loss'],
    hybrid_model['best_val_loss']
))

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "=" * 100)
print("GENERATING PLOTS")
print("=" * 100)

os.makedirs('results/hybrid_km', exist_ok=True)

# Plot 1: Comparison of predictions for a sample
sample_idx = 0
x_sample = X_test[sample_idx]
y_true = Y_test[sample_idx]

GXT, BiVaO4, PG, PearlB, thickness = x_sample[0]*100, x_sample[1]*100, x_sample[2]*100, x_sample[3]*100, x_sample[4]*12

# Get predictions
baseline_spec, baseline_cielab = predict_enhanced_neural_network(
    x_sample,
    baseline_model['model'],
    baseline_model['input_mean'],
    baseline_model['input_std'],
    baseline_model['output_mean'],
    baseline_model['output_std']
)

hybrid_spec, hybrid_cielab = predict_hybrid(
    GXT, BiVaO4, PG, PearlB, thickness,
    hybrid_model['model'],
    hybrid_model['input_mean'],
    hybrid_model['input_std'],
    hybrid_model['output_mean'],
    hybrid_model['output_std']
)

wavelengths = np.arange(400, 710, 10)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Spectral comparison
axes[0].plot(wavelengths, y_true[:31], 'k-', linewidth=2, label='Ground Truth')
axes[0].plot(wavelengths, baseline_spec, 'b--', linewidth=1.5, label='Baseline NN')
axes[0].plot(wavelengths, hybrid_spec, 'r--', linewidth=1.5, label='Hybrid K-M')
axes[0].set_xlabel('Wavelength (nm)')
axes[0].set_ylabel('Reflectance')
axes[0].set_title(f'Spectral Prediction\nGXT={GXT:.1f}%, BiVaO4={BiVaO4:.1f}%, PG={PG:.1f}%, PearlB={PearlB:.1f}%, {thickness}μm')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Error comparison
baseline_error = np.abs(baseline_spec - y_true[:31])
hybrid_error = np.abs(hybrid_spec - y_true[:31])

axes[1].plot(wavelengths, baseline_error, 'b-', linewidth=1.5, label=f'Baseline (MAE={np.mean(baseline_error):.4f})')
axes[1].plot(wavelengths, hybrid_error, 'r-', linewidth=1.5, label=f'Hybrid K-M (MAE={np.mean(hybrid_error):.4f})')
axes[1].set_xlabel('Wavelength (nm)')
axes[1].set_ylabel('Absolute Error')
axes[1].set_title('Prediction Error by Wavelength')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/hybrid_km/sample_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/hybrid_km/sample_comparison.png")

# Plot 2: Training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(baseline_model['history']['val_loss'], label='Baseline NN', color='blue', alpha=0.7)
axes[0].plot(hybrid_model['history']['val_loss'], label='Hybrid K-M', color='red', alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Loss')
axes[0].set_title('Training Convergence')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bar chart of final metrics
metrics_names = ['Spectral\nMAE', 'Spectral\nRMSE', 'CIELAB\nMAE', 'CIELAB\nRMSE']
baseline_vals = [
    baseline_results['spectral_mae'],
    baseline_results['spectral_rmse'],
    baseline_results['cielab_mae'],
    baseline_results['cielab_rmse']
]
hybrid_vals = [
    hybrid_results['spectral_mae'],
    hybrid_results['spectral_rmse'],
    hybrid_results['cielab_mae'],
    hybrid_results['cielab_rmse']
]

x = np.arange(len(metrics_names))
width = 0.35

axes[1].bar(x - width/2, baseline_vals, width, label='Baseline NN', color='blue', alpha=0.7)
axes[1].bar(x + width/2, hybrid_vals, width, label='Hybrid K-M', color='red', alpha=0.7)
axes[1].set_ylabel('Error')
axes[1].set_title('Test Set Performance')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics_names)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/hybrid_km/training_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/hybrid_km/training_comparison.png")

# ============================================================================
# Save Best Model
# ============================================================================
print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)

if hybrid_results['spectral_mae'] < baseline_results['spectral_mae']:
    improvement = (baseline_results['spectral_mae'] - hybrid_results['spectral_mae']) / baseline_results['spectral_mae'] * 100
    print(f"\n✓ Hybrid K-M model is BETTER by {improvement:.1f}%!")
    print(f"  Saving to: trained_models/hybrid_km_model.pkl")

    os.makedirs('trained_models', exist_ok=True)
    with open('trained_models/hybrid_km_model.pkl', 'wb') as f:
        pickle.dump(hybrid_model, f)

    print(f"\n  Key advantages:")
    print(f"    - More physically interpretable (K-M base + fluorescence)")
    print(f"    - Smaller network ({hybrid_model['model'].count_parameters()} vs {baseline_model['model'].count_parameters()} params)")
    print(f"    - {improvement:.1f}% better spectral prediction accuracy")
else:
    print(f"\n⚠ Baseline model still performs better")
    print(f"  This suggests K-M parameters may need tuning")
    print(f"  Or that the pigment interactions are more complex than assumed")

print()
