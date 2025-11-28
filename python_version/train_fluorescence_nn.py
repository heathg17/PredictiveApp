"""
Train Multi-Output Neural Network with Fluorescence Predictions

This NN predicts 3 outputs from pigment concentrations:
1. Spectral reflectance (31 wavelengths)
2. CIELAB color (5 values)
3. Fluorescence intensity (1 value in ct/s)

The model learns complex interactions between pigments and how
non-fluorescent pigments reduce fluorescence through absorption/scattering.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

from optimize_hyperparameters import FlexibleNN

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_training_data():
    """Load prepared training data with fluorescence"""
    print("Loading training data...")

    with open('training_data/nn_fluorescence_training_data.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    Y_spectral = data['Y_spectral']
    Y_cielab = data['Y_cielab']
    Y_fluorescence = data['Y_fluorescence']

    print(f"✓ Loaded {len(X)} samples")
    print(f"  X shape: {X.shape}")
    print(f"  Y_spectral shape: {Y_spectral.shape}")
    print(f"  Y_cielab shape: {Y_cielab.shape}")
    print(f"  Y_fluorescence shape: {Y_fluorescence.shape}")

    return X, Y_spectral, Y_cielab, Y_fluorescence


def train_fluorescence_model(
    X_train, Y_spectral_train, Y_cielab_train, Y_fluor_train,
    X_val, Y_spectral_val, Y_cielab_val, Y_fluor_val,
    epochs=2000,
    learning_rate=0.001,
    hidden_layers=[128, 128, 64],
    batch_size=8,
    verbose=True
):
    """
    Train multi-output NN with weighted multi-task loss

    Loss = w1 * MSE(spectral) + w2 * MSE(cielab) + w3 * MSE(fluorescence)
    """

    # Normalize inputs (z-score normalization)
    input_mean = np.mean(X_train, axis=0)
    input_std = np.std(X_train, axis=0) + 1e-8

    X_train_norm = (X_train - input_mean) / input_std
    X_val_norm = (X_val - input_mean) / input_std

    # Normalize outputs (z-score normalization)
    spectral_mean = np.mean(Y_spectral_train, axis=0)
    spectral_std = np.std(Y_spectral_train, axis=0) + 1e-8

    cielab_mean = np.mean(Y_cielab_train, axis=0)
    cielab_std = np.std(Y_cielab_train, axis=0) + 1e-8

    fluor_mean = np.mean(Y_fluor_train, axis=0)
    fluor_std = np.std(Y_fluor_train, axis=0) + 1e-8

    Y_spectral_train_norm = (Y_spectral_train - spectral_mean) / spectral_std
    Y_spectral_val_norm = (Y_spectral_val - spectral_mean) / spectral_std

    Y_cielab_train_norm = (Y_cielab_train - cielab_mean) / cielab_std
    Y_cielab_val_norm = (Y_cielab_val - cielab_mean) / cielab_std

    Y_fluor_train_norm = (Y_fluor_train - fluor_mean) / fluor_std
    Y_fluor_val_norm = (Y_fluor_val - fluor_mean) / fluor_std

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_norm)
    Y_spectral_train_t = torch.FloatTensor(Y_spectral_train_norm)
    Y_cielab_train_t = torch.FloatTensor(Y_cielab_train_norm)
    Y_fluor_train_t = torch.FloatTensor(Y_fluor_train_norm)

    X_val_t = torch.FloatTensor(X_val_norm)
    Y_spectral_val_t = torch.FloatTensor(Y_spectral_val_norm)
    Y_cielab_val_t = torch.FloatTensor(Y_cielab_val_norm)
    Y_fluor_val_t = torch.FloatTensor(Y_fluor_val_norm)

    # Create model
    model = FlexibleNN(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        activation='relu',
        dropout_rate=0.1,
        use_batchnorm=True
    )

    print(f"\n{'='*80}")
    print(f"MODEL ARCHITECTURE")
    print(f"{'='*80}")
    print(f"Input size: {X_train.shape[1]} (GXT, BiVaO4, PG, PearlB, thickness)")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Output heads:")
    print(f"  - Spectral: 31 wavelengths")
    print(f"  - CIELAB: 5 values (L, a, b, c, h)")
    print(f"  - Fluorescence: 1 value (ct/s)")
    print(f"Total parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )

    # Loss weights (balance the three tasks)
    # Spectral and CIELAB have more outputs, so weight fluorescence higher
    w_spectral = 1.0
    w_cielab = 1.0
    w_fluorescence = 3.0  # Weight fluorescence higher since it's single value

    print(f"\nLoss weights:")
    print(f"  Spectral: {w_spectral}")
    print(f"  CIELAB: {w_cielab}")
    print(f"  Fluorescence: {w_fluorescence}")

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience = 300
    patience_counter = 0

    train_losses = []
    val_losses = []

    print(f"\n{'='*80}")
    print(f"TRAINING")
    print(f"{'='*80}")

    for epoch in range(epochs):
        model.train()

        # Forward pass
        spectral_pred, cielab_pred, fluor_pred = model(X_train_t)

        # Multi-task loss
        loss_spectral = nn.MSELoss()(spectral_pred, Y_spectral_train_t)
        loss_cielab = nn.MSELoss()(cielab_pred, Y_cielab_train_t)
        loss_fluor = nn.MSELoss()(fluor_pred, Y_fluor_train_t)

        loss = w_spectral * loss_spectral + w_cielab * loss_cielab + w_fluorescence * loss_fluor

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            spectral_val_pred, cielab_val_pred, fluor_val_pred = model(X_val_t)

            val_loss_spectral = nn.MSELoss()(spectral_val_pred, Y_spectral_val_t)
            val_loss_cielab = nn.MSELoss()(cielab_val_pred, Y_cielab_val_t)
            val_loss_fluor = nn.MSELoss()(fluor_val_pred, Y_fluor_val_t)

            val_loss = w_spectral * val_loss_spectral + w_cielab * val_loss_cielab + w_fluorescence * val_loss_fluor

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        # Print progress
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {loss.item():.6f} (S: {loss_spectral.item():.4f}, C: {loss_cielab.item():.4f}, F: {loss_fluor.item():.4f})")
            print(f"  Val Loss:   {val_loss.item():.6f} (S: {val_loss_spectral.item():.4f}, C: {val_loss_cielab.item():.4f}, F: {val_loss_fluor.item():.4f})")
            print(f"  Best Val:   {best_val_loss:.6f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        spectral_pred, cielab_pred, fluor_pred = model(X_val_t)

        # Denormalize predictions
        spectral_pred_denorm = spectral_pred.numpy() * spectral_std + spectral_mean
        cielab_pred_denorm = cielab_pred.numpy() * cielab_std + cielab_mean
        fluor_pred_denorm = fluor_pred.numpy() * fluor_std + fluor_mean

    # Calculate metrics
    print(f"\n{'='*80}")
    print(f"VALIDATION METRICS")
    print(f"{'='*80}")

    # Spectral metrics
    spectral_mae = mean_absolute_error(Y_spectral_val, spectral_pred_denorm)
    spectral_r2 = r2_score(Y_spectral_val.flatten(), spectral_pred_denorm.flatten())
    print(f"Spectral:")
    print(f"  MAE: {spectral_mae:.4f}")
    print(f"  R²:  {spectral_r2:.4f}")

    # CIELAB metrics
    cielab_mae = mean_absolute_error(Y_cielab_val, cielab_pred_denorm)
    cielab_r2 = r2_score(Y_cielab_val.flatten(), cielab_pred_denorm.flatten())
    print(f"\nCIELAB:")
    print(f"  MAE: {cielab_mae:.4f}")
    print(f"  R²:  {cielab_r2:.4f}")

    # Fluorescence metrics
    fluor_mae = mean_absolute_error(Y_fluor_val, fluor_pred_denorm)
    fluor_r2 = r2_score(Y_fluor_val.flatten(), fluor_pred_denorm.flatten())
    print(f"\nFluorescence:")
    print(f"  MAE: {fluor_mae:.1f} ct/s")
    print(f"  R²:  {fluor_r2:.4f}")
    print(f"  Range: {Y_fluor_val.min():.0f} - {Y_fluor_val.max():.0f} ct/s")

    # Save model
    model_data = {
        'model_state': best_model_state,
        'input_mean': input_mean,
        'input_std': input_std,
        'spectral_mean': spectral_mean,
        'spectral_std': spectral_std,
        'cielab_mean': cielab_mean,
        'cielab_std': cielab_std,
        'fluorescence_mean': fluor_mean,
        'fluorescence_std': fluor_std,
        'hidden_layers': hidden_layers,
        'input_size': X_train.shape[1],
        'metrics': {
            'spectral_mae': spectral_mae,
            'spectral_r2': spectral_r2,
            'cielab_mae': cielab_mae,
            'cielab_r2': cielab_r2,
            'fluorescence_mae': fluor_mae,
            'fluorescence_r2': fluor_r2
        },
        'best_val_loss': best_val_loss
    }

    return model, model_data


def main():
    """Main training pipeline"""

    print("="*80)
    print("MULTI-OUTPUT NEURAL NETWORK TRAINING")
    print("Spectral + CIELAB + Fluorescence Predictions")
    print("="*80)

    # Load data
    X, Y_spectral, Y_cielab, Y_fluorescence = load_training_data()

    # Split data (80/20 train/val)
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)

    X_train, X_val = X[train_idx], X[val_idx]
    Y_spectral_train, Y_spectral_val = Y_spectral[train_idx], Y_spectral[val_idx]
    Y_cielab_train, Y_cielab_val = Y_cielab[train_idx], Y_cielab[val_idx]
    Y_fluor_train, Y_fluor_val = Y_fluorescence[train_idx], Y_fluorescence[val_idx]

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")

    # Train model
    model, model_data = train_fluorescence_model(
        X_train, Y_spectral_train, Y_cielab_train, Y_fluor_train,
        X_val, Y_spectral_val, Y_cielab_val, Y_fluor_val,
        epochs=2000,
        learning_rate=0.001,
        hidden_layers=[128, 128, 64],
        verbose=True
    )

    # Save model
    os.makedirs('trained_models', exist_ok=True)

    save_path = 'trained_models/fluorescence_nn_model.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n{'='*80}")
    print(f"MODEL SAVED")
    print(f"{'='*80}")
    print(f"Saved to: {save_path}")
    print(f"Model includes:")
    print(f"  - Trained weights")
    print(f"  - Normalization statistics")
    print(f"  - Performance metrics")
    print(f"\nThis model can now predict:")
    print(f"  1. Spectral reflectance from concentrations")
    print(f"  2. CIELAB color from concentrations")
    print(f"  3. Fluorescence (ct/s) from concentrations")
    print(f"\nThe fluorescence predictions capture complex interactions")
    print(f"between all pigments, not just a simple GXT-only model!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
