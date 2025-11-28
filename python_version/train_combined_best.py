"""
Train and save the combined best model:
- Baseline NN for spectral reflectance
- Hybrid K-M for CIELAB predictions
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from utils.new_data_loader import load_new_dataset, prepare_training_data
from models.combined_best_model import train_combined_models, save_combined_models, predict_combined

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 100)
print("TRAINING COMBINED BEST MODEL")
print("Baseline NN (Spectral) + Hybrid K-M (CIELAB)")
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
print()

# Split data
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_SEED
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.25, random_state=RANDOM_SEED
)

print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")
print()

# Train both models
baseline_model, hybrid_model = train_combined_models(
    X_train, Y_train,
    X_val, Y_val,
    baseline_epochs=2000,
    hybrid_epochs=2000,
    verbose=True
)

# Evaluate on test set
print("\n" + "=" * 100)
print("EVALUATING COMBINED MODEL ON TEST SET")
print("=" * 100)

Y_pred_spectral = []
Y_pred_cielab = []

for x in X_test:
    GXT, BiVaO4, PG, PearlB, thickness = x[0]*100, x[1]*100, x[2]*100, x[3]*100, x[4]*12

    reflectance, cielab = predict_combined(
        GXT, BiVaO4, PG, PearlB, thickness,
        baseline_model, hybrid_model
    )

    Y_pred_spectral.append(reflectance)
    Y_pred_cielab.append([
        cielab['L'] / 100.0,
        (cielab['a'] + 128) / 256.0,
        (cielab['b'] + 128) / 256.0,
        cielab['c'] / 150.0,
        cielab['h'] / 360.0
    ])

Y_pred_spectral = np.array(Y_pred_spectral)
Y_pred_cielab = np.array(Y_pred_cielab)

# Calculate metrics
Y_test_spectral = Y_test[:, :31]
Y_test_cielab = Y_test[:, 31:]

spectral_mae = mean_absolute_error(Y_test_spectral.flatten(), Y_pred_spectral.flatten())
spectral_r2 = r2_score(Y_test_spectral.flatten(), Y_pred_spectral.flatten())

cielab_mae = mean_absolute_error(Y_test_cielab.flatten(), Y_pred_cielab.flatten())
cielab_r2 = r2_score(Y_test_cielab.flatten(), Y_pred_cielab.flatten())

print(f"\nTest Set Performance:")
print(f"  Spectral MAE: {spectral_mae:.6f}")
print(f"  Spectral R²:  {spectral_r2:.6f}")
print(f"  CIELAB MAE:   {cielab_mae:.6f}")
print(f"  CIELAB R²:    {cielab_r2:.6f}")

# Save models
print("\n" + "=" * 100)
print("SAVING COMBINED MODEL")
print("=" * 100)

save_combined_models(baseline_model, hybrid_model, 'trained_models/combined_best_model.pkl')

print("\n✓ Training complete!")
print(f"\nModel summary:")
print(f"  Spectral prediction: Baseline NN ({baseline_model['model'].count_parameters()} params)")
print(f"  CIELAB prediction:   Hybrid K-M ({hybrid_model['model'].count_parameters()} params)")
print(f"  Total parameters:    {baseline_model['model'].count_parameters() + hybrid_model['model'].count_parameters()}")
print()
