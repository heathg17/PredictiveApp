"""
Enhanced PyTorch Neural Network for PP Substrate Spectral + CIELAB Prediction

Architecture: Multi-output deep neural network
  Inputs (5): GXT, BiVaO4, PG, PearlB concentrations + Thickness
  Outputs (36): 31 wavelengths + 5 CIELAB values (L, a, b, c, h)

Improvements:
  - 4×64 deep architecture (optimized for 77 samples)
  - Dropout regularization
  - Batch normalization
  - Separate heads for spectral and CIELAB predictions
  - Adam optimizer with learning rate scheduling
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt


class EnhancedSpectralNN(nn.Module):
    """
    Multi-output deep neural network for spectral and CIELAB prediction

    Architecture:
        Input (5) → Shared Layers (4×64) → Split:
            → Spectral Head (31 outputs)
            → CIELAB Head (5 outputs)
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        spectral_output_size: int = 31,
        cielab_output_size: int = 5,
        dropout_rate: float = 0.2
    ):
        super(EnhancedSpectralNN, self).__init__()

        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 2
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 3
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 4
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        # Spectral output head (31 wavelengths)
        self.spectral_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, spectral_output_size)
        )

        # CIELAB output head (L, a, b, c, h)
        self.cielab_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, cielab_output_size)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, 5]

        Returns:
            spectral: Spectral predictions [batch_size, 31]
            cielab: CIELAB predictions [batch_size, 5]
        """
        # Shared features
        features = self.shared_layers(x)

        # Separate predictions
        spectral = self.spectral_head(features)
        cielab = self.cielab_head(features)

        return spectral, cielab

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_enhanced_neural_network(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
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
    Train enhanced neural network with train/val split

    Args:
        X_train: Training inputs [n_train, 5]
        Y_train: Training outputs [n_train, 36] (31 spectral + 5 CIELAB)
        X_val: Validation inputs [n_val, 5]
        Y_val: Validation outputs [n_val, 36]
        hidden_size: Hidden layer size (default: 64)
        learning_rate: Initial learning rate
        epochs: Maximum training epochs
        batch_size: Mini-batch size
        l2_lambda: L2 regularization
        dropout_rate: Dropout rate
        device: 'cuda', 'cpu', or None
        verbose: Print training progress

    Returns:
        Dict with model, normalization params, and training history
    """

    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if verbose:
        print(f"Training Enhanced Neural Network (4×{hidden_size}) on {device}")
        print(f"Architecture: 5 → {hidden_size} → {hidden_size} → {hidden_size} → {hidden_size} → [31 + 5] outputs")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Normalize inputs
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std

    # Normalize outputs
    Y_mean = np.mean(Y_train, axis=0)
    Y_std = np.std(Y_train, axis=0)
    Y_std[Y_std < 1e-8] = 1.0
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_val_norm = (Y_val - Y_mean) / Y_std

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
    Y_train_tensor = torch.FloatTensor(Y_train_norm).to(device)
    X_val_tensor = torch.FloatTensor(X_val_norm).to(device)
    Y_val_tensor = torch.FloatTensor(Y_val_norm).to(device)

    # Separate spectral and CIELAB outputs
    Y_train_spectral = Y_train_tensor[:, :31]
    Y_train_cielab = Y_train_tensor[:, 31:]
    Y_val_spectral = Y_val_tensor[:, :31]
    Y_val_cielab = Y_val_tensor[:, 31:]

    # Create model
    model = EnhancedSpectralNN(
        input_size=5,
        hidden_size=hidden_size,
        spectral_output_size=31,
        cielab_output_size=5,
        dropout_rate=dropout_rate
    ).to(device)

    if verbose:
        print(f"Total parameters: {model.count_parameters():,}")

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
            batch_Y_spectral = Y_train_spectral[batch_indices]
            batch_Y_cielab = Y_train_cielab[batch_indices]

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
            val_loss_spectral = criterion(val_pred_spectral, Y_val_spectral).item()
            val_loss_cielab = criterion(val_pred_cielab, Y_val_cielab).item()
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
                print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
            break

        # Log progress
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Spectral: {val_loss_spectral:.6f} | CIELAB: {val_loss_cielab:.6f}")

    if verbose:
        print("Training complete!")

    # Move to CPU for storage
    model.to('cpu')

    return {
        'model': model,
        'input_mean': X_mean,
        'input_std': X_std,
        'output_mean': Y_mean,
        'output_std': Y_std,
        'history': history,
        'best_val_loss': best_val_loss,
        'hidden_size': hidden_size
    }


def predict_enhanced_neural_network(
    input_data: np.ndarray,
    model: EnhancedSpectralNN,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Make predictions with enhanced neural network

    Returns:
        spectrum: 31 wavelengths
        cielab: Dict with L, a, b, c, h values
    """
    model.eval()

    # Handle single sample
    single_sample = False
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
        single_sample = True

    # Normalize input
    input_norm = (input_data - input_mean) / input_std
    input_tensor = torch.FloatTensor(input_norm)

    # Predict
    with torch.no_grad():
        pred_spectral, pred_cielab = model(input_tensor)

    # Denormalize
    spectral = pred_spectral.numpy() * output_std[:31] + output_mean[:31]
    cielab_values = pred_cielab.numpy() * output_std[31:] + output_mean[31:]

    # Clip spectrum to reasonable range
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
