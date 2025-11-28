"""
Train and evaluate enhanced neural network on PP substrate dataset
Includes train/validation/test split and comprehensive evaluation
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

from utils.new_data_loader import load_new_dataset, prepare_training_data
from models.enhanced_neural_network import (
    train_enhanced_neural_network,
    predict_enhanced_neural_network
)

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("ENHANCED NEURAL NETWORK TRAINING - PP SUBSTRATE DATASET")
print("="*80)
print()

# Load data
print("Loading dataset...")
samples, _, _ = load_new_dataset(
    '../public/Concentrations.csv',
    '../public/Spectra.csv'
)

X, Y = prepare_training_data(samples)
print(f"✓ Dataset prepared: {X.shape[0]} samples")
print(f"  Inputs: {X.shape[1]} features (GXT, BiVaO4, PG, PearlB, Thickness)")
print(f"  Outputs: {Y.shape[1]} features (31 wavelengths + 5 CIELAB)")
print()

# Train/Val/Test split: 60/20/20
print("Splitting dataset...")
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_SEED
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.25, random_state=RANDOM_SEED  # 0.25 of 80% = 20% of total
)

print(f"  Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print()

# Train model
print("="*80)
print("TRAINING MODEL")
print("="*80)

model_data = train_enhanced_neural_network(
    X_train, Y_train,
    X_val, Y_val,
    hidden_size=64,
    learning_rate=0.001,
    epochs=3000,
    batch_size=16,
    l2_lambda=0.001,
    dropout_rate=0.2,
    verbose=True
)

print()
print("="*80)
print("EVALUATION ON TEST SET")
print("="*80)

# Test set predictions
model = model_data['model']
Y_test_pred_full = []

for x_test in X_test:
    spectrum, cielab = predict_enhanced_neural_network(
        x_test,
        model,
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
    Y_test_pred_full.append(y_pred)

Y_test_pred = np.array(Y_test_pred_full)

# Calculate metrics
print("\nOverall Metrics:")
print("-" * 80)

# Spectral metrics (first 31 outputs)
Y_test_spectral = Y_test[:, :31]
Y_pred_spectral = Y_test_pred[:, :31]

spectral_mae = mean_absolute_error(Y_test_spectral.flatten(), Y_pred_spectral.flatten())
spectral_rmse = np.sqrt(mean_squared_error(Y_test_spectral.flatten(), Y_pred_spectral.flatten()))
spectral_r2 = r2_score(Y_test_spectral.flatten(), Y_pred_spectral.flatten())

print(f"Spectral Reflectance (31 wavelengths):")
print(f"  MAE:  {spectral_mae:.6f}")
print(f"  RMSE: {spectral_rmse:.6f}")
print(f"  R²:   {spectral_r2:.6f}")

# CIELAB metrics (last 5 outputs)
Y_test_cielab = Y_test[:, 31:]
Y_pred_cielab = Y_test_pred[:, 31:]

cielab_mae = mean_absolute_error(Y_test_cielab.flatten(), Y_pred_cielab.flatten())
cielab_rmse = np.sqrt(mean_squared_error(Y_test_cielab.flatten(), Y_pred_cielab.flatten()))
cielab_r2 = r2_score(Y_test_cielab.flatten(), Y_pred_cielab.flatten())

print(f"\nCIELAB Values (L, a, b, c, h):")
print(f"  MAE:  {cielab_mae:.6f}")
print(f"  RMSE: {cielab_rmse:.6f}")
print(f"  R²:   {cielab_r2:.6f}")

# Individual CIELAB component metrics
cielab_names = ['L', 'a', 'b', 'c', 'h']
print(f"\nIndividual CIELAB Components:")
for i, name in enumerate(cielab_names):
    mae = mean_absolute_error(Y_test_cielab[:, i], Y_pred_cielab[:, i])
    print(f"  {name}: MAE = {mae:.6f}")

print()
print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Create output directory
os.makedirs('results', exist_ok=True)

# Plot 1: Training History
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Overall loss
axes[0, 0].plot(model_data['history']['train_loss'], label='Train Loss', alpha=0.7)
axes[0, 0].plot(model_data['history']['val_loss'], label='Val Loss', alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Overall Training Progress')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Spectral loss
axes[0, 1].plot(model_data['history']['train_spectral_loss'], label='Train Spectral', alpha=0.7)
axes[0, 1].plot(model_data['history']['val_spectral_loss'], label='Val Spectral', alpha=0.7)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Spectral Prediction Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# CIELAB loss
axes[1, 0].plot(model_data['history']['train_cielab_loss'], label='Train CIELAB', alpha=0.7)
axes[1, 0].plot(model_data['history']['val_cielab_loss'], label='Val CIELAB', alpha=0.7)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('CIELAB Prediction Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Loss comparison
final_epoch = len(model_data['history']['train_loss'])
axes[1, 1].bar(['Train\nOverall', 'Val\nOverall', 'Train\nSpectral', 'Val\nSpectral', 'Train\nCIELAB', 'Val\nCIELAB'],
               [model_data['history']['train_loss'][-1],
                model_data['history']['val_loss'][-1],
                model_data['history']['train_spectral_loss'][-1],
                model_data['history']['val_spectral_loss'][-1],
                model_data['history']['train_cielab_loss'][-1],
                model_data['history']['val_cielab_loss'][-1]],
               color=['blue', 'orange', 'blue', 'orange', 'blue', 'orange'],
               alpha=0.7)
axes[1, 1].set_ylabel('Final Loss')
axes[1, 1].set_title(f'Final Loss Comparison (Epoch {final_epoch})')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/training_history.png")

# Plot 2: Test Set Performance - Spectral
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Scatter plot: Predicted vs Actual (all wavelengths)
axes[0, 0].scatter(Y_test_spectral.flatten(), Y_pred_spectral.flatten(), alpha=0.3, s=10)
axes[0, 0].plot([Y_test_spectral.min(), Y_test_spectral.max()],
                [Y_test_spectral.min(), Y_test_spectral.max()],
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Reflectance')
axes[0, 0].set_ylabel('Predicted Reflectance')
axes[0, 0].set_title(f'Spectral Prediction Accuracy (R² = {spectral_r2:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Example spectra (first 5 test samples)
wavelengths = np.arange(400, 701, 10)
for i in range(min(5, len(Y_test))):
    axes[0, 1].plot(wavelengths, Y_test_spectral[i], 'o-', label=f'Sample {i+1} (Actual)', alpha=0.6)
    axes[0, 1].plot(wavelengths, Y_pred_spectral[i], 's--', label=f'Sample {i+1} (Pred)', alpha=0.6)

axes[0, 1].set_xlabel('Wavelength (nm)')
axes[0, 1].set_ylabel('Reflectance')
axes[0, 1].set_title('Example Spectral Predictions')
axes[0, 1].legend(fontsize=8, ncol=2)
axes[0, 1].grid(True, alpha=0.3)

# Residual histogram
residuals_spectral = Y_pred_spectral.flatten() - Y_test_spectral.flatten()
axes[1, 0].hist(residuals_spectral, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Prediction Error')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'Spectral Residuals Distribution (Mean: {residuals_spectral.mean():.6f})')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Per-wavelength MAE
mae_per_wavelength = [mean_absolute_error(Y_test_spectral[:, i], Y_pred_spectral[:, i])
                      for i in range(31)]
axes[1, 1].plot(wavelengths, mae_per_wavelength, 'o-', linewidth=2)
axes[1, 1].set_xlabel('Wavelength (nm)')
axes[1, 1].set_ylabel('MAE')
axes[1, 1].set_title('Prediction Error by Wavelength')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/spectral_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/spectral_performance.png")

# Plot 3: CIELAB Performance
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, name in enumerate(cielab_names):
    row = i // 3
    col = i % 3

    # Scatter plot for each component
    axes[row, col].scatter(Y_test_cielab[:, i], Y_pred_cielab[:, i], alpha=0.6)
    axes[row, col].plot([Y_test_cielab[:, i].min(), Y_test_cielab[:, i].max()],
                        [Y_test_cielab[:, i].min(), Y_test_cielab[:, i].max()],
                        'r--', linewidth=2)
    axes[row, col].set_xlabel(f'Actual {name}')
    axes[row, col].set_ylabel(f'Predicted {name}')

    r2 = r2_score(Y_test_cielab[:, i], Y_pred_cielab[:, i])
    mae = mean_absolute_error(Y_test_cielab[:, i], Y_pred_cielab[:, i])
    axes[row, col].set_title(f'{name} (R²={r2:.4f}, MAE={mae:.4f})')
    axes[row, col].grid(True, alpha=0.3)

# Overall CIELAB metrics bar chart
axes[1, 2].bar(cielab_names,
               [mean_absolute_error(Y_test_cielab[:, i], Y_pred_cielab[:, i])
                for i in range(5)],
               color=['blue', 'green', 'orange', 'red', 'purple'],
               alpha=0.7)
axes[1, 2].set_ylabel('MAE')
axes[1, 2].set_title('CIELAB Component MAE')
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/cielab_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/cielab_performance.png")

# Plot 4: Summary Statistics
fig, ax = plt.subplots(figsize=(12, 8))

summary_data = {
    'Dataset Split': [f'Train: {len(X_train)}', f'Val: {len(X_val)}', f'Test: {len(X_test)}'],
    'Spectral Performance': [f'MAE: {spectral_mae:.6f}', f'RMSE: {spectral_rmse:.6f}', f'R²: {spectral_r2:.6f}'],
    'CIELAB Performance': [f'MAE: {cielab_mae:.6f}', f'RMSE: {cielab_rmse:.6f}', f'R²: {cielab_r2:.6f}'],
    'Model Architecture': [f'Hidden Size: {model_data["hidden_size"]}',
                          f'Parameters: {model.count_parameters():,}',
                          f'Best Val Loss: {model_data["best_val_loss"]:.6f}']
}

ax.axis('off')
table_data = []
for key, values in summary_data.items():
    for value in values:
        table_data.append([key, value])
    table_data.append(['', ''])  # Spacer

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.4, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

for i in range(0, len(table_data), 4):
    for j in range(2):
        table[(i, j)].set_facecolor('#E0E0E0')
        table[(i, j)].set_text_props(weight='bold')

plt.title('Enhanced Neural Network - Summary Statistics', fontsize=16, weight='bold', pad=20)
plt.savefig('results/summary_statistics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/summary_statistics.png")

print()
print("="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nResults saved to: results/")
print(f"  - training_history.png")
print(f"  - spectral_performance.png")
print(f"  - cielab_performance.png")
print(f"  - summary_statistics.png")

# Save model
import pickle
os.makedirs('trained_models', exist_ok=True)
model_path = 'trained_models/enhanced_pp_model.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to: {model_path}")
print()
