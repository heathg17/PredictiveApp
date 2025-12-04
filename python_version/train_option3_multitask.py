"""
Option 3: Train Multi-Task Neural Network with 3 Outputs

This NN predicts:
1. Spectral reflectance (31 wavelengths)
2. CIELAB color (5 values: L, a, b, c, h)
3. Fluorescence (1 value: ct/s)

All from pigment concentrations + thickness.

Advantages:
- Single unified model for all predictions
- Shared representations across tasks
- May capture cross-task correlations
- Saves as separate file (won't affect optimized_best_model.pkl)

Risks:
- Need to ensure spectral/CIELAB performance doesn't degrade
- More complex training (multi-task loss balancing)
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle
import os


# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class MultiTaskNN(nn.Module):
    """
    Multi-task NN with 3 output heads

    Inputs: 5 features (GXT, BiVaO4, PG, PearlB, thickness)
    Outputs:
        - Spectral (31 wavelengths)
        - CIELAB (5 values)
        - Fluorescence (1 value)
    """
    def __init__(self, hidden_layers=[128, 128, 64], dropout_rate=0.1):
        super(MultiTaskNN, self).__init__()

        # Shared layers
        shared_layers = []
        input_size = 5

        for hidden_size in hidden_layers:
            shared_layers.append(nn.Linear(input_size, hidden_size))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.BatchNorm1d(hidden_size))
            shared_layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        self.shared = nn.Sequential(*shared_layers)

        final_hidden = hidden_layers[-1] if hidden_layers else 5

        # Spectral head (31 wavelengths)
        self.spectral_head = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            nn.ReLU(),
            nn.Linear(final_hidden // 2, 31)
        )

        # CIELAB head (5 values: L, a, b, c, h)
        self.cielab_head = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            nn.ReLU(),
            nn.Linear(final_hidden // 2, 5)
        )

        # Fluorescence head (1 value: ct/s)
        self.fluorescence_head = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            nn.ReLU(),
            nn.Linear(final_hidden // 2, 1)
        )

    def forward(self, x):
        shared_features = self.shared(x)
        spectral = self.spectral_head(shared_features)
        cielab = self.cielab_head(shared_features)
        fluorescence = self.fluorescence_head(shared_features)
        return spectral, cielab, fluorescence

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_training_data():
    """Load prepared training data for Option 3"""
    print("Loading Option 3 training data...")

    with open('training_data/option3_multitask_data.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    Y_spectral = data['Y_spectral']
    Y_cielab = data['Y_cielab']
    Y_fluorescence = data['Y_fluorescence']

    print(f"✓ Loaded {len(X)} samples")
    print(f"  X shape: {X.shape} (5 features)")
    print(f"  Y_spectral shape: {Y_spectral.shape}")
    print(f"  Y_cielab shape: {Y_cielab.shape}")
    print(f"  Y_fluorescence shape: {Y_fluorescence.shape}")

    return X, Y_spectral, Y_cielab, Y_fluorescence, data['samples']


def train_multitask_model(
    X_train, Y_spectral_train, Y_cielab_train, Y_fluor_train,
    X_val, Y_spectral_val, Y_cielab_val, Y_fluor_val,
    epochs=2000,
    learning_rate=0.001,
    hidden_layers=[128, 128, 64],
    batch_size=8,
    verbose=True
):
    """Train multi-task NN with 3 outputs"""

    # Normalize inputs
    input_mean = np.mean(X_train, axis=0)
    input_std = np.std(X_train, axis=0) + 1e-8

    X_train_norm = (X_train - input_mean) / input_std
    X_val_norm = (X_val - input_mean) / input_std

    # Normalize outputs
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
    model = MultiTaskNN(hidden_layers=hidden_layers, dropout_rate=0.1)

    print(f"\n{'='*80}")
    print(f"OPTION 3: MULTI-TASK NN ARCHITECTURE")
    print(f"{'='*80}")
    print(f"Input size: 5 features")
    print(f"  - GXT concentration (%)")
    print(f"  - BiVaO4 concentration (%)")
    print(f"  - PG concentration (%)")
    print(f"  - PearlB concentration (%)")
    print(f"  - Thickness (μm)")
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
    print(f"OPTION 3: VALIDATION METRICS")
    print(f"{'='*80}")

    # Spectral metrics
    spectral_mae = mean_absolute_error(Y_spectral_val, spectral_pred_denorm)
    spectral_rmse = np.sqrt(mean_squared_error(Y_spectral_val, spectral_pred_denorm))
    spectral_r2 = r2_score(Y_spectral_val.flatten(), spectral_pred_denorm.flatten())
    print(f"Spectral Reflectance:")
    print(f"  MAE:  {spectral_mae:.4f}")
    print(f"  RMSE: {spectral_rmse:.4f}")
    print(f"  R²:   {spectral_r2:.4f}")

    # CIELAB metrics
    cielab_mae = mean_absolute_error(Y_cielab_val, cielab_pred_denorm)
    cielab_rmse = np.sqrt(mean_squared_error(Y_cielab_val, cielab_pred_denorm))
    cielab_r2 = r2_score(Y_cielab_val.flatten(), cielab_pred_denorm.flatten())
    print(f"\nCIELAB Color:")
    print(f"  MAE:  {cielab_mae:.4f}")
    print(f"  RMSE: {cielab_rmse:.4f}")
    print(f"  R²:   {cielab_r2:.4f}")

    # Fluorescence metrics
    fluor_mae = mean_absolute_error(Y_fluor_val, fluor_pred_denorm)
    fluor_rmse = np.sqrt(mean_squared_error(Y_fluor_val, fluor_pred_denorm))
    fluor_r2 = r2_score(Y_fluor_val.flatten(), fluor_pred_denorm.flatten())
    print(f"\nFluorescence:")
    print(f"  MAE:  {fluor_mae:.1f} ct/s")
    print(f"  RMSE: {fluor_rmse:.1f} ct/s")
    print(f"  R²:   {fluor_r2:.4f}")
    print(f"  Range: {Y_fluor_val.min():.0f} - {Y_fluor_val.max():.0f} ct/s")

    # Calculate percentage error
    mean_fluor = Y_fluor_val.mean()
    fluor_pct_error = (fluor_mae / mean_fluor) * 100
    print(f"  Mean Percentage Error: {fluor_pct_error:.1f}%")

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
        'metrics': {
            'spectral_mae': spectral_mae,
            'spectral_rmse': spectral_rmse,
            'spectral_r2': spectral_r2,
            'cielab_mae': cielab_mae,
            'cielab_rmse': cielab_rmse,
            'cielab_r2': cielab_r2,
            'fluorescence_mae': fluor_mae,
            'fluorescence_rmse': fluor_rmse,
            'fluorescence_r2': fluor_r2,
            'fluorescence_pct_error': fluor_pct_error
        },
        'best_val_loss': best_val_loss
    }

    return model, model_data


def test_zero_gxt_constraint(model, input_mean, input_std, fluor_mean, fluor_std):
    """
    Test that 0% GXT returns ~0 ct/s (physics constraint)
    """
    print(f"\n{'='*80}")
    print(f"TESTING PHYSICS CONSTRAINT: 0% GXT → 0 ct/s")
    print(f"{'='*80}")

    model.eval()

    test_cases = [
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0, "thickness": 8.0},
        {"GXT": 0.0, "BiVaO4": 10.0, "PG": 0.0, "PearlB": 0.0, "thickness": 8.0},
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 2.0, "PearlB": 0.0, "thickness": 8.0},
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 5.0, "thickness": 8.0},
    ]

    for case in test_cases:
        x = np.array([[
            case["GXT"],
            case["BiVaO4"],
            case["PG"],
            case["PearlB"],
            case["thickness"]
        ]])

        # Normalize
        x_norm = (x - input_mean) / input_std

        # Predict
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_norm)
            _, _, fluor_pred_norm = model(x_tensor)
            fluor_pred = fluor_pred_norm.numpy() * fluor_std + fluor_mean

        desc = f"GXT:{case['GXT']:.0f}% BiVaO4:{case['BiVaO4']:.0f}% PG:{case['PG']:.0f}% PearlB:{case['PearlB']:.0f}%"
        print(f"{desc:50s} → {fluor_pred[0, 0]:.0f} ct/s")

    print("\nExpected: All predictions should be close to 0 ct/s")


def main():
    """Main training pipeline for Option 3"""

    print("="*80)
    print("OPTION 3: MULTI-TASK NEURAL NETWORK (3 OUTPUTS)")
    print("="*80)

    # Load data
    X, Y_spectral, Y_cielab, Y_fluorescence, samples = load_training_data()

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
    model, model_data = train_multitask_model(
        X_train, Y_spectral_train, Y_cielab_train, Y_fluor_train,
        X_val, Y_spectral_val, Y_cielab_val, Y_fluor_val,
        epochs=2000,
        learning_rate=0.001,
        hidden_layers=[128, 128, 64],
        verbose=True
    )

    # Test physics constraint
    test_zero_gxt_constraint(
        model,
        model_data['input_mean'],
        model_data['input_std'],
        model_data['fluorescence_mean'],
        model_data['fluorescence_std']
    )

    # Save model
    os.makedirs('trained_models', exist_ok=True)

    save_path = 'trained_models/option3_multitask_nn.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n{'='*80}")
    print(f"OPTION 3 MODEL SAVED")
    print(f"{'='*80}")
    print(f"Saved to: {save_path}")
    print(f"\nPerformance Summary:")
    print(f"  Spectral R²:      {model_data['metrics']['spectral_r2']:.4f}")
    print(f"  CIELAB R²:        {model_data['metrics']['cielab_r2']:.4f}")
    print(f"  Fluorescence R²:  {model_data['metrics']['fluorescence_r2']:.4f}")
    print(f"  Fluorescence MAE: {model_data['metrics']['fluorescence_mae']:.1f} ct/s")
    print(f"  Fluorescence Error: {model_data['metrics']['fluorescence_pct_error']:.1f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
