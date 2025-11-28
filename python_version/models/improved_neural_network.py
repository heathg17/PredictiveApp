"""
Improved Neural Network with Physics-Informed Features

Key improvements:
1. Physics-informed input features (Kubelka-Munk calculations)
2. Separate normalization for spectral and CIELAB outputs
3. Data augmentation
4. Enhanced architecture
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Optional, Dict
import numpy as np

from models.enhanced_neural_network import EnhancedSpectralNN
from utils.physics_features import add_physics_informed_features, add_data_augmentation


def train_improved_neural_network(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    use_physics_features: bool = True,
    use_augmentation: bool = True,
    augmentation_factor: int = 2,
    hidden_size: int = 64,
    learning_rate: float = 0.001,
    epochs: int = 3000,
    batch_size: int = 16,
    l2_lambda: float = 0.001,
    dropout_rate: float = 0.2,
    device: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Train improved neural network with physics-informed features

    Args:
        X_train: Training inputs [n_train, 5] (GXT, BiVaO4, PG, PearlB, Thickness)
        Y_train: Training outputs [n_train, 36] (31 spectral + 5 CIELAB)
        X_val: Validation inputs
        Y_val: Validation outputs
        use_physics_features: Add K-M and physics features
        use_augmentation: Apply data augmentation
        augmentation_factor: Number of augmented samples per original
        ... (other params same as enhanced_neural_network)

    Returns:
        Dict with model, normalization params, and training history
    """

    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if verbose:
        print("=" * 80)
        print("IMPROVED PHYSICS-INFORMED NEURAL NETWORK")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Physics Features: {'Enabled' if use_physics_features else 'Disabled'}")
        print(f"Data Augmentation: {'Enabled' if use_augmentation else 'Disabled'}")

    # 1. Add physics-informed features
    feature_names = None
    if use_physics_features:
        if verbose:
            print("\nAdding physics-informed features...")
        X_train_enhanced, feature_names = add_physics_informed_features(X_train, include_km_features=True)
        X_val_enhanced, _ = add_physics_informed_features(X_val, include_km_features=True)

        if verbose:
            print(f"  Original features: {X_train.shape[1]}")
            print(f"  Enhanced features: {X_train_enhanced.shape[1]}")
            print(f"  Feature names: {', '.join(feature_names[:10])}...")
    else:
        X_train_enhanced = X_train
        X_val_enhanced = X_val

    # 2. Data augmentation (training set only)
    if use_augmentation:
        if verbose:
            print(f"\nApplying data augmentation (factor={augmentation_factor})...")
            print(f"  Original training samples: {len(X_train_enhanced)}")

        X_train_aug, Y_train_aug = add_data_augmentation(
            X_train_enhanced, Y_train,
            noise_level=0.015,
            n_augmented=augmentation_factor
        )

        if verbose:
            print(f"  Augmented training samples: {len(X_train_aug)}")
    else:
        X_train_aug = X_train_enhanced
        Y_train_aug = Y_train

    # 3. Separate normalization for inputs
    if verbose:
        print("\nNormalizing inputs (z-score)...")

    X_mean = np.mean(X_train_aug, axis=0)
    X_std = np.std(X_train_aug, axis=0)
    X_std[X_std < 1e-8] = 1.0  # Avoid division by zero

    X_train_norm = (X_train_aug - X_mean) / X_std
    X_val_norm = (X_val_enhanced - X_mean) / X_std

    # 4. IMPROVED: Separate normalization for spectral and CIELAB outputs
    if verbose:
        print("Normalizing outputs (separate for spectral and CIELAB)...")

    # Spectral outputs (indices 0-30)
    Y_spectral_mean = np.mean(Y_train_aug[:, :31], axis=0)
    Y_spectral_std = np.std(Y_train_aug[:, :31], axis=0)
    Y_spectral_std[Y_spectral_std < 1e-8] = 1.0

    # CIELAB outputs (indices 31-35)
    Y_cielab_mean = np.mean(Y_train_aug[:, 31:], axis=0)
    Y_cielab_std = np.std(Y_train_aug[:, 31:], axis=0)
    Y_cielab_std[Y_cielab_std < 1e-8] = 1.0

    # Normalize separately
    Y_train_spectral_norm = (Y_train_aug[:, :31] - Y_spectral_mean) / Y_spectral_std
    Y_train_cielab_norm = (Y_train_aug[:, 31:] - Y_cielab_mean) / Y_cielab_std

    Y_val_spectral_norm = (Y_val[:, :31] - Y_spectral_mean) / Y_spectral_std
    Y_val_cielab_norm = (Y_val[:, 31:] - Y_cielab_mean) / Y_cielab_std

    if verbose:
        print(f"  Spectral mean: {Y_spectral_mean.mean():.4f}, std: {Y_spectral_std.mean():.4f}")
        print(f"  CIELAB mean: {Y_cielab_mean.mean():.4f}, std: {Y_cielab_std.mean():.4f}")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
    Y_train_spectral_tensor = torch.FloatTensor(Y_train_spectral_norm).to(device)
    Y_train_cielab_tensor = torch.FloatTensor(Y_train_cielab_norm).to(device)

    X_val_tensor = torch.FloatTensor(X_val_norm).to(device)
    Y_val_spectral_tensor = torch.FloatTensor(Y_val_spectral_norm).to(device)
    Y_val_cielab_tensor = torch.FloatTensor(Y_val_cielab_norm).to(device)

    # 5. Create model (input size adjusted for enhanced features)
    input_size = X_train_norm.shape[1]

    if verbose:
        print(f"\nCreating neural network...")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Architecture: {input_size} → {hidden_size}×4 → [31 + 5]")

    model = EnhancedSpectralNN(
        input_size=input_size,
        hidden_size=hidden_size,
        spectral_output_size=31,
        cielab_output_size=5,
        dropout_rate=dropout_rate
    ).to(device)

    if verbose:
        print(f"  Total parameters: {model.count_parameters():,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_spectral_loss': [],
        'train_cielab_loss': [],
        'val_spectral_loss': [],
        'val_cielab_loss': []
    }

    # Training loop
    n_train = X_train_tensor.shape[0]
    best_val_loss = float('inf')
    patience = 300
    patience_counter = 0

    if verbose:
        print(f"\nTraining for up to {epochs} epochs...")
        print(f"  Training samples: {n_train}")
        print(f"  Validation samples: {len(X_val_tensor)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Early stopping patience: {patience}")
        print("=" * 80)

    for epoch in range(epochs):
        model.train()

        # Shuffle training data
        indices = torch.randperm(n_train)
        total_loss = 0.0
        total_spectral_loss = 0.0
        total_cielab_loss = 0.0

        # Mini-batch training
        for batch_start in range(0, n_train, batch_size):
            batch_end = min(batch_start + batch_size, n_train)
            batch_indices = indices[batch_start:batch_end]

            batch_X = X_train_tensor[batch_indices]
            batch_Y_spectral = Y_train_spectral_tensor[batch_indices]
            batch_Y_cielab = Y_train_cielab_tensor[batch_indices]

            # Forward pass
            pred_spectral, pred_cielab = model(batch_X)

            # Separate losses
            loss_spectral = criterion(pred_spectral, batch_Y_spectral)
            loss_cielab = criterion(pred_cielab, batch_Y_cielab)

            # Combined loss (weighted)
            loss = loss_spectral + 0.5 * loss_cielab  # CIELAB has lower weight

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * len(batch_indices)
            total_spectral_loss += loss_spectral.item() * len(batch_indices)
            total_cielab_loss += loss_cielab.item() * len(batch_indices)

        avg_train_loss = total_loss / n_train
        avg_train_spectral = total_spectral_loss / n_train
        avg_train_cielab = total_cielab_loss / n_train

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred_spectral, val_pred_cielab = model(X_val_tensor)
            val_loss_spectral = criterion(val_pred_spectral, Y_val_spectral_tensor).item()
            val_loss_cielab = criterion(val_pred_cielab, Y_val_cielab_tensor).item()
            val_loss = val_loss_spectral + 0.5 * val_loss_cielab

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_spectral_loss'].append(avg_train_spectral)
        history['train_cielab_loss'].append(avg_train_cielab)
        history['val_spectral_loss'].append(val_loss_spectral)
        history['val_cielab_loss'].append(val_loss_cielab)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.6f}")
            break

        # Log progress
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Spectral: {val_loss_spectral:.6f} | CIELAB: {val_loss_cielab:.6f} | LR: {lr:.6f}")

    if verbose:
        print("=" * 80)
        print("Training complete!")
        print(f"Final validation loss: {val_loss:.6f}")
        print(f"Best validation loss: {best_val_loss:.6f}")

    # Move to CPU for storage
    model.to('cpu')

    return {
        'model': model,
        'input_mean': X_mean,
        'input_std': X_std,
        'spectral_mean': Y_spectral_mean,
        'spectral_std': Y_spectral_std,
        'cielab_mean': Y_cielab_mean,
        'cielab_std': Y_cielab_std,
        'history': history,
        'best_val_loss': best_val_loss,
        'hidden_size': hidden_size,
        'use_physics_features': use_physics_features,
        'feature_names': feature_names,
        'input_size': input_size
    }


def predict_improved_neural_network(
    input_data: np.ndarray,
    model_data: Dict,
    use_physics_features: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Make predictions with improved neural network

    Args:
        input_data: Input array [5] or [n, 5] - original concentrations + thickness
        model_data: Dictionary from train_improved_neural_network
        use_physics_features: Must match training setting

    Returns:
        spectrum: 31 wavelengths
        cielab: Dict with L, a, b, c, h values
    """
    model = model_data['model']
    model.eval()

    # Handle single sample
    single_sample = False
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
        single_sample = True

    # Add physics features if used during training
    if use_physics_features:
        input_enhanced, _ = add_physics_informed_features(input_data, include_km_features=True)
    else:
        input_enhanced = input_data

    # Normalize input
    input_norm = (input_enhanced - model_data['input_mean']) / model_data['input_std']
    input_tensor = torch.FloatTensor(input_norm)

    # Predict
    with torch.no_grad():
        pred_spectral_norm, pred_cielab_norm = model(input_tensor)

    # Denormalize using separate parameters
    spectral = pred_spectral_norm.numpy() * model_data['spectral_std'] + model_data['spectral_mean']
    cielab_values = pred_cielab_norm.numpy() * model_data['cielab_std'] + model_data['cielab_mean']

    # Clip spectrum to reasonable range (allow fluorescence >1)
    spectral = np.clip(spectral, -0.1, 1.5)

    if single_sample:
        cielab = {
            'L': float(cielab_values[0, 0]),
            'a': float(cielab_values[0, 1]),
            'b': float(cielab_values[0, 2]),
            'c': float(cielab_values[0, 3]),
            'h': float(cielab_values[0, 4])
        }
        return spectral[0], cielab
    else:
        cielab_batch = []
        for i in range(len(cielab_values)):
            cielab_batch.append({
                'L': float(cielab_values[i, 0]),
                'a': float(cielab_values[i, 1]),
                'b': float(cielab_values[i, 2]),
                'c': float(cielab_values[i, 3]),
                'h': float(cielab_values[i, 4])
            })
        return spectral, cielab_batch
