"""
Test and compare improved neural network vs baseline

Compares:
1. Baseline (current enhanced model)
2. + Physics features only
3. + Data augmentation only
4. + Physics features + Augmentation (full improvement)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time

from utils.new_data_loader import load_new_dataset, prepare_training_data
from models.enhanced_neural_network import train_enhanced_neural_network, predict_enhanced_neural_network
from models.improved_neural_network import train_improved_neural_network, predict_improved_neural_network

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 100)
print("NEURAL NETWORK IMPROVEMENT COMPARISON TEST")
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
print(f"  Inputs: {X.shape[1]} features")
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

# Common hyperparameters
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 2000  # Reduced for faster comparison
BATCH_SIZE = 16
L2_LAMBDA = 0.001
DROPOUT = 0.2

# Store results
results = {}

def evaluate_model(model_data, X_test, Y_test, model_name, use_improved=False):
    """Evaluate model and return metrics"""
    print(f"\nEvaluating {model_name}...")

    Y_test_pred_list = []

    for x in X_test:
        if use_improved:
            spectrum, cielab = predict_improved_neural_network(
                x, model_data,
                use_physics_features=model_data['use_physics_features']
            )
        else:
            spectrum, cielab = predict_enhanced_neural_network(
                x,
                model_data['model'],
                model_data['input_mean'],
                model_data['input_std'],
                model_data['output_mean'],
                model_data['output_std']
            )

        # Reconstruct prediction
        y_pred = np.concatenate([
            spectrum,
            [cielab['L'] / 100.0,
             (cielab['a'] + 128) / 256.0,
             (cielab['b'] + 128) / 256.0,
             cielab['c'] / 150.0,
             cielab['h'] / 360.0]
        ])
        Y_test_pred_list.append(y_pred)

    Y_pred = np.array(Y_test_pred_list)

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

    print(f"  Spectral - MAE: {spectral_mae:.6f}, RMSE: {spectral_rmse:.6f}, R²: {spectral_r2:.6f}")
    print(f"  CIELAB   - MAE: {cielab_mae:.6f}, RMSE: {cielab_rmse:.6f}, R²: {cielab_r2:.6f}")

    return {
        'spectral_mae': spectral_mae,
        'spectral_rmse': spectral_rmse,
        'spectral_r2': spectral_r2,
        'cielab_mae': cielab_mae,
        'cielab_rmse': cielab_rmse,
        'cielab_r2': cielab_r2,
        'best_val_loss': model_data['best_val_loss'],
        'Y_pred': Y_pred
    }


# ============================================================================
# Model 1: Baseline (Current Enhanced Model)
# ============================================================================
print("\n" + "=" * 100)
print("MODEL 1: BASELINE (Current Enhanced Model)")
print("=" * 100)

start_time = time.time()
baseline_model = train_enhanced_neural_network(
    X_train, Y_train,
    X_val, Y_val,
    hidden_size=HIDDEN_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    l2_lambda=L2_LAMBDA,
    dropout_rate=DROPOUT,
    verbose=True
)
baseline_time = time.time() - start_time

results['baseline'] = evaluate_model(baseline_model, X_test, Y_test, "Baseline", use_improved=False)
results['baseline']['training_time'] = baseline_time

# ============================================================================
# Model 2: Physics Features Only
# ============================================================================
print("\n" + "=" * 100)
print("MODEL 2: PHYSICS FEATURES ONLY")
print("=" * 100)

start_time = time.time()
physics_model = train_improved_neural_network(
    X_train, Y_train,
    X_val, Y_val,
    use_physics_features=True,
    use_augmentation=False,  # No augmentation
    hidden_size=HIDDEN_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    l2_lambda=L2_LAMBDA,
    dropout_rate=DROPOUT,
    verbose=True
)
physics_time = time.time() - start_time

results['physics'] = evaluate_model(physics_model, X_test, Y_test, "Physics Features", use_improved=True)
results['physics']['training_time'] = physics_time

# ============================================================================
# Model 3: Data Augmentation Only
# ============================================================================
print("\n" + "=" * 100)
print("MODEL 3: DATA AUGMENTATION ONLY")
print("=" * 100)

start_time = time.time()
augmentation_model = train_improved_neural_network(
    X_train, Y_train,
    X_val, Y_val,
    use_physics_features=False,  # No physics features
    use_augmentation=True,
    augmentation_factor=2,
    hidden_size=HIDDEN_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    l2_lambda=L2_LAMBDA,
    dropout_rate=DROPOUT,
    verbose=True
)
augmentation_time = time.time() - start_time

results['augmentation'] = evaluate_model(augmentation_model, X_test, Y_test, "Data Augmentation", use_improved=True)
results['augmentation']['training_time'] = augmentation_time

# ============================================================================
# Model 4: Full Improvements (Physics + Augmentation)
# ============================================================================
print("\n" + "=" * 100)
print("MODEL 4: FULL IMPROVEMENTS (Physics + Augmentation)")
print("=" * 100)

start_time = time.time()
full_model = train_improved_neural_network(
    X_train, Y_train,
    X_val, Y_val,
    use_physics_features=True,
    use_augmentation=True,
    augmentation_factor=2,
    hidden_size=HIDDEN_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    l2_lambda=L2_LAMBDA,
    dropout_rate=DROPOUT,
    verbose=True
)
full_time = time.time() - start_time

results['full'] = evaluate_model(full_model, X_test, Y_test, "Full Improvements", use_improved=True)
results['full']['training_time'] = full_time

# ============================================================================
# Comparison Summary
# ============================================================================
print("\n" + "=" * 100)
print("COMPARISON SUMMARY")
print("=" * 100)

print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
    "Model", "Spectral MAE", "CIELAB MAE", "Val Loss", "Time (s)"
))
print("-" * 100)

for name, display_name in [
    ('baseline', 'Baseline'),
    ('physics', 'Physics Features'),
    ('augmentation', 'Data Augmentation'),
    ('full', 'Full Improvements')
]:
    r = results[name]
    print("{:<25} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.1f}".format(
        display_name,
        r['spectral_mae'],
        r['cielab_mae'],
        r['best_val_loss'],
        r['training_time']
    ))

print("\n" + "=" * 100)
print("IMPROVEMENT PERCENTAGES (vs Baseline)")
print("=" * 100)

baseline_spectral = results['baseline']['spectral_mae']
baseline_cielab = results['baseline']['cielab_mae']

print("\n{:<25} {:>15} {:>15}".format("Model", "Spectral MAE", "CIELAB MAE"))
print("-" * 100)

for name, display_name in [
    ('physics', 'Physics Features'),
    ('augmentation', 'Data Augmentation'),
    ('full', 'Full Improvements')
]:
    spectral_improvement = (baseline_spectral - results[name]['spectral_mae']) / baseline_spectral * 100
    cielab_improvement = (baseline_cielab - results[name]['cielab_mae']) / baseline_cielab * 100

    print("{:<25} {:>14.1f}% {:>14.1f}%".format(
        display_name,
        spectral_improvement,
        cielab_improvement
    ))

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "=" * 100)
print("GENERATING COMPARISON PLOTS")
print("=" * 100)

os.makedirs('results/comparison', exist_ok=True)

# Plot 1: MAE Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

models = ['Baseline', 'Physics\nFeatures', 'Data\nAugmentation', 'Full\nImprovements']
spectral_maes = [results[k]['spectral_mae'] for k in ['baseline', 'physics', 'augmentation', 'full']]
cielab_maes = [results[k]['cielab_mae'] for k in ['baseline', 'physics', 'augmentation', 'full']]

x = np.arange(len(models))
width = 0.35

ax1.bar(x, spectral_maes, width, color=['gray', 'blue', 'green', 'red'], alpha=0.7)
ax1.set_ylabel('MAE')
ax1.set_title('Spectral Reflectance MAE')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.grid(True, alpha=0.3, axis='y')

# Add improvement percentages
for i, mae in enumerate(spectral_maes):
    if i > 0:
        improvement = (spectral_maes[0] - mae) / spectral_maes[0] * 100
        ax1.text(i, mae + 0.001, f'{improvement:+.1f}%', ha='center', va='bottom', fontweight='bold')

ax2.bar(x, cielab_maes, width, color=['gray', 'blue', 'green', 'red'], alpha=0.7)
ax2.set_ylabel('MAE')
ax2.set_title('CIELAB MAE')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.grid(True, alpha=0.3, axis='y')

# Add improvement percentages
for i, mae in enumerate(cielab_maes):
    if i > 0:
        improvement = (cielab_maes[0] - mae) / cielab_maes[0] * 100
        ax2.text(i, mae + 0.0005, f'{improvement:+.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('results/comparison/mae_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/comparison/mae_comparison.png")

# Plot 2: Training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

model_data = [
    (baseline_model, 'Baseline', 'gray'),
    (physics_model, 'Physics Features', 'blue'),
    (augmentation_model, 'Data Augmentation', 'green'),
    (full_model, 'Full Improvements', 'red')
]

for model, name, color in model_data:
    axes[0, 0].plot(model['history']['val_loss'], label=name, color=color, alpha=0.7)
    axes[0, 1].plot(model['history']['val_spectral_loss'], label=name, color=color, alpha=0.7)
    axes[1, 0].plot(model['history']['val_cielab_loss'], label=name, color=color, alpha=0.7)

axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Overall Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Spectral Validation Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('CIELAB Validation Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Final metrics comparison
metrics = ['Spectral\nMAE', 'Spectral\nRMSE', 'CIELAB\nMAE', 'CIELAB\nRMSE']
baseline_vals = [results['baseline']['spectral_mae'], results['baseline']['spectral_rmse'],
                 results['baseline']['cielab_mae'], results['baseline']['cielab_rmse']]
full_vals = [results['full']['spectral_mae'], results['full']['spectral_rmse'],
             results['full']['cielab_mae'], results['full']['cielab_rmse']]

x = np.arange(len(metrics))
width = 0.35

axes[1, 1].bar(x - width/2, baseline_vals, width, label='Baseline', color='gray', alpha=0.7)
axes[1, 1].bar(x + width/2, full_vals, width, label='Full Improvements', color='red', alpha=0.7)
axes[1, 1].set_ylabel('Error')
axes[1, 1].set_title('Baseline vs Full Improvements')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/comparison/training_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/comparison/training_comparison.png")

print("\n" + "=" * 100)
print("COMPARISON TEST COMPLETE!")
print("=" * 100)
print(f"\nResults saved to: results/comparison/")
print(f"  - mae_comparison.png")
print(f"  - training_comparison.png")

# Save best model
if results['full']['spectral_mae'] < results['baseline']['spectral_mae']:
    print(f"\n✓ Full improvements model is better! Saving...")
    import pickle
    os.makedirs('trained_models', exist_ok=True)

    with open('trained_models/improved_pp_model.pkl', 'wb') as f:
        pickle.dump(full_model, f)

    print(f"  Saved to: trained_models/improved_pp_model.pkl")
    print(f"\n  Improvement summary:")
    print(f"    Spectral MAE: {baseline_spectral:.6f} → {results['full']['spectral_mae']:.6f} "
          f"({(baseline_spectral - results['full']['spectral_mae']) / baseline_spectral * 100:.1f}% better)")
    print(f"    CIELAB MAE:   {baseline_cielab:.6f} → {results['full']['cielab_mae']:.6f} "
          f"({(baseline_cielab - results['full']['cielab_mae']) / baseline_cielab * 100:.1f}% better)")
else:
    print(f"\n⚠ Baseline model performed better - keeping original model")

print()
