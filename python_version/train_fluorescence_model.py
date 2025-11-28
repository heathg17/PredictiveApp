"""
Train Enhanced Neural Network with Fluorescence Prediction

Architecture:
  Input (5) → [64, 128, 64] → Three output heads:
    - Spectral Head: 31 wavelengths
    - CIELAB Head: 5 values (L, a, b, c, h)
    - Fluorescence Head: 1 value (background-subtracted area)

Based on optimized hyperparameters:
  - Layers: [64, 128, 64]
  - Activation: Leaky ReLU
  - Learning Rate: 0.002
  - Batch Size: 16
  - L2: 0.001
  - No Dropout
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class FluorescenceNN(nn.Module):
    """
    Enhanced Neural Network with Fluorescence Prediction

    Input: 5 features (GXT, BiVaO4, PG, PearlB, Thickness)
    Output: 37 values (31 Spectral + 5 CIELAB + 1 Fluorescence)
    """

    def __init__(self, hidden_layers=[64, 128, 64], activation='leaky_relu'):
        super(FluorescenceNN, self).__init__()

        self.hidden_layers = hidden_layers
        self.activation_name = activation

        # Shared backbone
        layers = []
        input_size = 5  # 5 input features

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))

            if activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.01))
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())

            input_size = hidden_size

        self.backbone = nn.Sequential(*layers)

        # Output heads
        last_hidden = hidden_layers[-1]

        self.spectral_head = nn.Linear(last_hidden, 31)  # 31 wavelengths
        self.cielab_head = nn.Linear(last_hidden, 5)     # L, a, b, c, h
        self.fluorescence_head = nn.Linear(last_hidden, 1)  # Fluorescence area

    def forward(self, x):
        # Shared features
        features = self.backbone(x)

        # Three outputs
        spectral = self.spectral_head(features)
        cielab = self.cielab_head(features)
        fluorescence = self.fluorescence_head(features)

        # Concatenate all outputs: [31 spectral, 5 cielab, 1 fluorescence]
        output = torch.cat([spectral, cielab, fluorescence], dim=1)

        return output

    def predict_components(self, x):
        """
        Separate prediction method that returns individual components
        """
        features = self.backbone(x)

        spectral = self.spectral_head(features)
        cielab = self.cielab_head(features)
        fluorescence = self.fluorescence_head(features)

        return {
            'spectral': spectral,
            'cielab': cielab,
            'fluorescence': fluorescence
        }


def weighted_loss(output, target, spectral_weight=1.0, cielab_weight=1.0, fluorescence_weight=1.0):
    """
    Weighted MSE loss for multi-task learning

    Output structure: [31 spectral, 5 cielab, 1 fluorescence]
    """
    # Split outputs
    spectral_pred = output[:, :31]
    cielab_pred = output[:, 31:36]
    fluorescence_pred = output[:, 36:37]

    spectral_target = target[:, :31]
    cielab_target = target[:, 31:36]
    fluorescence_target = target[:, 36:37]

    # Calculate individual losses
    spectral_loss = nn.functional.mse_loss(spectral_pred, spectral_target)
    cielab_loss = nn.functional.mse_loss(cielab_pred, cielab_target)
    fluorescence_loss = nn.functional.mse_loss(fluorescence_pred, fluorescence_target)

    # Weighted combination
    total_loss = (
        spectral_weight * spectral_loss +
        cielab_weight * cielab_loss +
        fluorescence_weight * fluorescence_loss
    )

    return total_loss, spectral_loss, cielab_loss, fluorescence_loss


def train_model(X_train, Y_train, X_val, Y_val, config, verbose=True):
    """Train the fluorescence neural network"""

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)

    # Create model
    model = FluorescenceNN(
        hidden_layers=config['hidden_layers'],
        activation=config['activation']
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['l2_lambda']
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 100)

    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        model.train()

        # Mini-batch training
        batch_size = config['batch_size']
        indices = torch.randperm(len(X_train_t))

        epoch_train_loss = 0
        num_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train_t[batch_indices]
            Y_batch = Y_train_t[batch_indices]

            optimizer.zero_grad()

            output = model(X_batch)
            loss, spec_loss, cielab_loss, fluor_loss = weighted_loss(
                output, Y_batch,
                config.get('spectral_weight', 1.0),
                config.get('cielab_weight', 1.0),
                config.get('fluorescence_weight', 1.0)
            )

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        epoch_train_loss /= num_batches
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss, val_spec, val_cielab, val_fluor = weighted_loss(
                val_output, Y_val_t,
                config.get('spectral_weight', 1.0),
                config.get('cielab_weight', 1.0),
                config.get('fluorescence_weight', 1.0)
            )
            val_losses.append(val_loss.item())

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:4d} | Train: {epoch_train_loss:.6f} | "
                  f"Val: {val_loss:.6f} (Spec:{val_spec:.4f} CIELAB:{val_cielab:.4f} Fluor:{val_fluor:.4f}) | "
                  f"Best: {best_val_loss:.6f}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_model_state)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())

    return {
        'model': model,
        'model_state': best_model_state,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'parameters': num_params,
        'config': config,
        'input_features': ['GXT', 'BiVaO4', 'PG', 'PearlB', 'Thickness'],
        'output_features': 37  # 31 spectral + 5 cielab + 1 fluorescence
    }


def evaluate_model(model_data, X_test, Y_test):
    """Evaluate model on test set"""

    model = model_data['model']
    model.eval()

    X_test_t = torch.FloatTensor(X_test)
    Y_test_t = torch.FloatTensor(Y_test)

    with torch.no_grad():
        predictions = model(X_test_t).numpy()

    Y_test_np = Y_test_t.numpy()

    # Split predictions by type
    spectral_pred = predictions[:, :31]
    cielab_pred = predictions[:, 31:36]
    fluorescence_pred = predictions[:, 36:37]

    spectral_true = Y_test_np[:, :31]
    cielab_true = Y_test_np[:, 31:36]
    fluorescence_true = Y_test_np[:, 36:37]

    # Calculate metrics
    metrics = {
        'spectral_mae': mean_absolute_error(spectral_true, spectral_pred),
        'spectral_r2': r2_score(spectral_true, spectral_pred),
        'cielab_mae': mean_absolute_error(cielab_true, cielab_pred),
        'cielab_r2': r2_score(cielab_true, cielab_pred),
        'fluorescence_mae': mean_absolute_error(fluorescence_true, fluorescence_pred),
        'fluorescence_r2': r2_score(fluorescence_true, fluorescence_pred),
    }

    return metrics


if __name__ == "__main__":
    print("=" * 80)
    print("TRAINING ENHANCED MODEL WITH FLUORESCENCE PREDICTION")
    print("=" * 80)
    print()

    # Load prepared data
    print("Loading training data...")
    with open('training_data/fluorescence_training_data.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    Y = data['Y']

    print(f"✓ Loaded {len(X)} samples")
    print(f"  Input features: {X.shape[1]}")
    print(f"  Output features: {Y.shape[1]} (31 spectral + 5 CIELAB + 1 fluorescence)")
    print()

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

    # Use optimized configuration
    config = {
        'hidden_layers': [64, 128, 64],
        'learning_rate': 0.002,
        'batch_size': 16,
        'dropout': 0.0,
        'l2_lambda': 0.001,
        'activation': 'leaky_relu',
        'spectral_weight': 1.0,
        'cielab_weight': 1.0,
        'fluorescence_weight': 1.0,  # Equal weight for fluorescence
        'epochs': 2000,
        'patience': 100
    }

    print("Configuration:")
    print(f"  Architecture: {config['hidden_layers']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Activation: {config['activation']}")
    print(f"  Weights: Spectral={config['spectral_weight']}, "
          f"CIELAB={config['cielab_weight']}, Fluorescence={config['fluorescence_weight']}")
    print()

    # Train model
    print("Training...")
    model_data = train_model(X_train, Y_train, X_val, Y_val, config, verbose=True)

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)

    metrics = evaluate_model(model_data, X_test, Y_test)

    print(f"\nTest Set Performance:")
    print(f"  Spectral MAE:       {metrics['spectral_mae']:.6f}")
    print(f"  Spectral R²:        {metrics['spectral_r2']:.6f}")
    print(f"  CIELAB MAE:         {metrics['cielab_mae']:.6f}")
    print(f"  CIELAB R²:          {metrics['cielab_r2']:.6f}")
    print(f"  Fluorescence MAE:   {metrics['fluorescence_mae']:.6f}")
    print(f"  Fluorescence R²:    {metrics['fluorescence_r2']:.6f}")

    # Save model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model_data['metrics'] = metrics

    # Add normalization statistics for API server
    model_data['input_mean'] = X.mean(axis=0)
    model_data['input_std'] = X.std(axis=0)
    model_data['output_mean'] = Y.mean(axis=0)
    model_data['output_std'] = Y.std(axis=0)

    with open('trained_models/fluorescence_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n✓ Model saved to: trained_models/fluorescence_model.pkl")
    print(f"  Parameters: {model_data['parameters']}")
    print(f"  Best validation loss: {model_data['best_val_loss']:.6f}")
    print(f"  Saved normalization stats for API deployment")
    print()
