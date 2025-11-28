"""
Train the best configuration from hyperparameter optimization
Configuration: [64, 128, 64] with Leaky ReLU, lr=0.002, batch=16
"""
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from utils.new_data_loader import load_new_dataset, prepare_training_data
from optimize_hyperparameters import FlexibleNN, train_model, evaluate_model

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 100)
print("TRAINING BEST MODEL FROM OPTIMIZATION")
print("=" * 100)
print()

# Load data
print("Loading dataset...")
samples, _, _ = load_new_dataset(
    '../public/Concentrations.csv',
    '../public/Spectra.csv'
)

X, Y = prepare_training_data(samples)
print(f"✓ Dataset loaded: {X.shape[0]} samples\n")

# Split data
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_SEED
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.2, random_state=RANDOM_SEED
)

print(f"Training:   {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples")
print(f"Test:       {len(X_test)} samples")
print()

# Best configuration from optimization
best_config = {
    'hidden_layers': [64, 128, 64],
    'learning_rate': 0.002,
    'batch_size': 16,
    'dropout': 0.0,
    'l2_lambda': 0.001,
    'activation': 'leaky_relu',
    'use_batchnorm': False,
    'spectral_weight': 1.0,
    'cielab_weight': 1.0,
    'use_lr_scheduler': False,
    'gradient_clip': 0,
    'epochs': 2000,
    'patience': 100
}

print("Best Configuration:")
print(f"  Architecture: {best_config['hidden_layers']}")
print(f"  Learning Rate: {best_config['learning_rate']}")
print(f"  Batch Size: {best_config['batch_size']}")
print(f"  Dropout: {best_config['dropout']}")
print(f"  L2: {best_config['l2_lambda']}")
print(f"  Activation: {best_config['activation']}")
print(f"  Batch Normalization: {best_config['use_batchnorm']}")
print()

# Train model
print("Training...")
model_data = train_model(X_train, Y_train, X_val, Y_val, best_config, verbose=True)

# Evaluate
print("\n" + "=" * 100)
print("EVALUATION ON TEST SET")
print("=" * 100)

metrics = evaluate_model(model_data, X_test, Y_test)

print(f"\nTest Set Performance:")
print(f"  Spectral MAE:  {metrics['spectral_mae']:.6f}")
print(f"  Spectral RMSE: {metrics['spectral_rmse']:.6f}")
print(f"  Spectral R²:   {metrics['spectral_r2']:.6f}")
print(f"  CIELAB MAE:    {metrics['cielab_mae']:.6f}")
print(f"  CIELAB RMSE:   {metrics['cielab_rmse']:.6f}")
print(f"  CIELAB R²:     {metrics['cielab_r2']:.6f}")
print(f"  Combined Score: {metrics['combined_score']:.6f}")

# Save model
print("\n" + "=" * 100)
print("SAVING MODEL")
print("=" * 100)

with open('trained_models/optimized_best_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to: trained_models/optimized_best_model.pkl")
print(f"  Parameters: {model_data['parameters']}")
print(f"  Best validation loss: {model_data['best_val_loss']:.6f}")
print()
