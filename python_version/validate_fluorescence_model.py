"""
Validate Fluorescence Model Predictions

This script tests the trained fluorescence model and visualizes results
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from train_fluorescence_model import FluorescenceNN

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("FLUORESCENCE MODEL VALIDATION")
print("=" * 80)
print()

# Load model
print("Loading trained model...")
with open('trained_models/fluorescence_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
model.eval()

print(f"✓ Model loaded")
print(f"  Parameters: {model_data['parameters']}")
print(f"  Architecture: {model_data['config']['hidden_layers']}")
print()

# Load data
print("Loading test data...")
with open('training_data/fluorescence_training_data.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y']

# Split (same as training)
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_SEED
)

print(f"✓ Test set: {len(X_test)} samples")
print()

# Make predictions
X_test_t = torch.FloatTensor(X_test)
with torch.no_grad():
    predictions = model(X_test_t).numpy()

# Split predictions
fluor_pred = predictions[:, 36:37].flatten()
fluor_true = Y_test[:, 36:37].flatten()

# Calculate metrics
mae = mean_absolute_error(fluor_true, fluor_pred)
r2 = r2_score(fluor_true, fluor_pred)

print("=" * 80)
print("FLUORESCENCE PREDICTION PERFORMANCE")
print("=" * 80)
print(f"\nTest Set Metrics:")
print(f"  MAE: {mae:.4f}")
print(f"  R²:  {r2:.4f} ({r2*100:.1f}% variance explained)")
print(f"  Mean True: {fluor_true.mean():.4f} ± {fluor_true.std():.4f}")
print(f"  Mean Pred: {fluor_pred.mean():.4f} ± {fluor_pred.std():.4f}")
print()

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Predicted vs Actual
axes[0].scatter(fluor_true, fluor_pred, s=100, alpha=0.7, edgecolors='black')
axes[0].plot([fluor_true.min(), fluor_true.max()],
             [fluor_true.min(), fluor_true.max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('True Fluorescence Area', fontsize=12)
axes[0].set_ylabel('Predicted Fluorescence Area', fontsize=12)
axes[0].set_title(f'Fluorescence Prediction\nR² = {r2:.3f}, MAE = {mae:.3f}',
                  fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = fluor_pred - fluor_true
axes[1].scatter(fluor_true, residuals, s=100, alpha=0.7, edgecolors='black', color='orange')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('True Fluorescence Area', fontsize=12)
axes[1].set_ylabel('Residual (Predicted - True)', fontsize=12)
axes[1].set_title('Prediction Residuals', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/fluorescence/model_validation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/fluorescence/model_validation.png")

# Test on full dataset to see training performance
print("\n" + "=" * 80)
print("FULL DATASET PERFORMANCE (Train + Val + Test)")
print("=" * 80)

X_full_t = torch.FloatTensor(X)
with torch.no_grad():
    full_predictions = model(X_full_t).numpy()

fluor_pred_full = full_predictions[:, 36:37].flatten()
fluor_true_full = Y[:, 36:37].flatten()

mae_full = mean_absolute_error(fluor_true_full, fluor_pred_full)
r2_full = r2_score(fluor_true_full, fluor_pred_full)

print(f"\nFull Dataset Metrics ({len(X)} samples):")
print(f"  MAE: {mae_full:.4f}")
print(f"  R²:  {r2_full:.4f} ({r2_full*100:.1f}% variance explained)")
print()

# Sample predictions
print("=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)
print()

samples_data = data['samples']

# Show first 5 test samples
print("Test Sample Predictions:")
print(f"{'Sample':<12} {'GXT':>6} {'BiVaO4':>6} {'Thick':>6} | {'True':>8} {'Pred':>8} {'Error':>8}")
print("-" * 75)

test_indices = np.arange(len(X))
_, test_idx = train_test_split(test_indices, test_size=0.2, random_state=RANDOM_SEED)

for idx in test_idx[:5]:
    sample = samples_data[idx]
    x = X[idx:idx+1]
    y_true = Y[idx, 36]

    x_t = torch.FloatTensor(x)
    with torch.no_grad():
        y_pred = model(x_t).numpy()[0, 36]

    error = y_pred - y_true

    print(f"{sample['sample_id']:<12} {sample['GXT']:>6.1f} {sample['BiVaO4']:>6.1f} "
          f"{sample['thickness']:>6.0f} | {y_true:>8.3f} {y_pred:>8.3f} {error:>8.3f}")

print()
print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
print(f"\nKey Finding:")
print(f"  The neural network can predict fluorescence area with R² = {r2:.3f}")
print(f"  This means {r2*100:.1f}% of fluorescence variance is explained by:")
print(f"    - Reagent concentrations (GXT, BiVaO4, PG, PearlB)")
print(f"    - Film thickness")
print(f"\n  ✓ Model is ready for API integration")
