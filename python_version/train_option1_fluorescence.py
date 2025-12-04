"""
Option 1: Train Separate Fluorescence-Only Neural Network

This NN predicts ONLY fluorescence (ct/s) from:
- Pigment concentrations (GXT, BiVaO4, PG, PearlB)
- Thickness
- Integrated area under reflectance curve

Advantages:
- Zero risk to existing spectral/CIELAB predictions
- Can use integrated area as a feature (physics-based)
- Simple, focused architecture
- Easy to integrate as separate predictor
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


class FluorescenceOnlyNN(nn.Module):
    """
    Focused NN for fluorescence prediction only

    Inputs: 6 features (GXT, BiVaO4, PG, PearlB, thickness, integrated_area)
    Output: 1 value (fluorescence ct/s)
    """
    def __init__(self, hidden_layers=[64, 32], dropout_rate=0.1):
        super(FluorescenceOnlyNN, self).__init__()

        layers = []
        input_size = 6

        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        # Output layer (1 value: fluorescence ct/s)
        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_training_data():
    """Load prepared training data for Option 1"""
    print("Loading Option 1 training data...")

    with open('training_data/option1_fluorescence_data.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    Y = data['Y']

    print(f"✓ Loaded {len(X)} samples")
    print(f"  X shape: {X.shape} (6 features)")
    print(f"  Y shape: {Y.shape} (1 output: fluorescence ct/s)")

    return X, Y, data['samples']


def train_fluorescence_model(
    X_train, Y_train,
    X_val, Y_val,
    epochs=2000,
    learning_rate=0.001,
    hidden_layers=[64, 32],
    batch_size=8,
    verbose=True
):
    """Train fluorescence-only NN"""

    # Normalize inputs (z-score normalization)
    input_mean = np.mean(X_train, axis=0)
    input_std = np.std(X_train, axis=0) + 1e-8

    X_train_norm = (X_train - input_mean) / input_std
    X_val_norm = (X_val - input_mean) / input_std

    # Normalize output
    output_mean = np.mean(Y_train, axis=0)
    output_std = np.std(Y_train, axis=0) + 1e-8

    Y_train_norm = (Y_train - output_mean) / output_std
    Y_val_norm = (Y_val - output_mean) / output_std

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_norm)
    Y_train_t = torch.FloatTensor(Y_train_norm)
    X_val_t = torch.FloatTensor(X_val_norm)
    Y_val_t = torch.FloatTensor(Y_val_norm)

    # Create model
    model = FluorescenceOnlyNN(hidden_layers=hidden_layers, dropout_rate=0.1)

    print(f"\n{'='*80}")
    print(f"OPTION 1: FLUORESCENCE-ONLY NN ARCHITECTURE")
    print(f"{'='*80}")
    print(f"Input size: 6 features")
    print(f"  - GXT concentration (%)")
    print(f"  - BiVaO4 concentration (%)")
    print(f"  - PG concentration (%)")
    print(f"  - PearlB concentration (%)")
    print(f"  - Thickness (μm)")
    print(f"  - Integrated area")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Output: 1 value (fluorescence ct/s)")
    print(f"Total parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )

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
        pred = model(X_train_t)
        loss = nn.MSELoss()(pred, Y_train_t)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = nn.MSELoss()(val_pred, Y_val_t)

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
            print(f"  Train Loss: {loss.item():.6f}")
            print(f"  Val Loss:   {val_loss.item():.6f}")
            print(f"  Best Val:   {best_val_loss:.6f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        pred = model(X_val_t)

        # Denormalize predictions
        pred_denorm = pred.numpy() * output_std + output_mean

    # Calculate metrics
    print(f"\n{'='*80}")
    print(f"OPTION 1: VALIDATION METRICS")
    print(f"{'='*80}")

    mae = mean_absolute_error(Y_val, pred_denorm)
    rmse = np.sqrt(mean_squared_error(Y_val, pred_denorm))
    r2 = r2_score(Y_val, pred_denorm)

    print(f"Fluorescence Predictions:")
    print(f"  MAE:  {mae:.1f} ct/s")
    print(f"  RMSE: {rmse:.1f} ct/s")
    print(f"  R²:   {r2:.4f}")
    print(f"  Range: {Y_val.min():.0f} - {Y_val.max():.0f} ct/s")

    # Calculate percentage error
    mean_fluor = Y_val.mean()
    pct_error = (mae / mean_fluor) * 100
    print(f"  Mean Percentage Error: {pct_error:.1f}%")

    # Save model
    model_data = {
        'model_state': best_model_state,
        'input_mean': input_mean,
        'input_std': input_std,
        'output_mean': output_mean,
        'output_std': output_std,
        'hidden_layers': hidden_layers,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pct_error': pct_error
        },
        'best_val_loss': best_val_loss
    }

    return model, model_data


def test_zero_gxt_constraint(model, input_mean, input_std, output_mean, output_std):
    """
    Test that 0% GXT returns ~0 ct/s (physics constraint)
    """
    print(f"\n{'='*80}")
    print(f"TESTING PHYSICS CONSTRAINT: 0% GXT → 0 ct/s")
    print(f"{'='*80}")

    model.eval()

    test_cases = [
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0, "thickness": 8.0, "area": 240.0},
        {"GXT": 0.0, "BiVaO4": 10.0, "PG": 0.0, "PearlB": 0.0, "thickness": 8.0, "area": 240.0},
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 2.0, "PearlB": 0.0, "thickness": 8.0, "area": 240.0},
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 5.0, "thickness": 8.0, "area": 240.0},
    ]

    for case in test_cases:
        x = np.array([[
            case["GXT"],
            case["BiVaO4"],
            case["PG"],
            case["PearlB"],
            case["thickness"],
            case["area"]
        ]])

        # Normalize
        x_norm = (x - input_mean) / input_std

        # Predict
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_norm)
            pred_norm = model(x_tensor)
            pred = pred_norm.numpy() * output_std + output_mean

        desc = f"GXT:{case['GXT']:.0f}% BiVaO4:{case['BiVaO4']:.0f}% PG:{case['PG']:.0f}% PearlB:{case['PearlB']:.0f}%"
        print(f"{desc:50s} → {pred[0, 0]:.0f} ct/s")

    print("\nExpected: All predictions should be close to 0 ct/s")


def main():
    """Main training pipeline for Option 1"""

    print("="*80)
    print("OPTION 1: SEPARATE FLUORESCENCE-ONLY NEURAL NETWORK")
    print("="*80)

    # Load data
    X, Y, samples = load_training_data()

    # Split data (80/20 train/val)
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)

    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")

    # Train model
    model, model_data = train_fluorescence_model(
        X_train, Y_train,
        X_val, Y_val,
        epochs=2000,
        learning_rate=0.001,
        hidden_layers=[64, 32],
        verbose=True
    )

    # Test physics constraint
    test_zero_gxt_constraint(
        model,
        model_data['input_mean'],
        model_data['input_std'],
        model_data['output_mean'],
        model_data['output_std']
    )

    # Save model
    os.makedirs('trained_models', exist_ok=True)

    save_path = 'trained_models/option1_fluorescence_nn.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n{'='*80}")
    print(f"OPTION 1 MODEL SAVED")
    print(f"{'='*80}")
    print(f"Saved to: {save_path}")
    print(f"\nPerformance Summary:")
    print(f"  R²:   {model_data['metrics']['r2']:.4f}")
    print(f"  MAE:  {model_data['metrics']['mae']:.1f} ct/s")
    print(f"  RMSE: {model_data['metrics']['rmse']:.1f} ct/s")
    print(f"  Error: {model_data['metrics']['pct_error']:.1f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
