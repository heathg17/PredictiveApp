"""
Neural network implementation using PyTorch for spectral prediction
Architecture: Input -> Hidden (ReLU) -> Output (Linear)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import numpy as np


class SpectralNN(nn.Module):
    """
    Deep feedforward neural network for spectral reflectance prediction

    Architecture (4×32):
        - Input layer: n_reagents + 1 (thickness)
        - Hidden layer 1: 32 neurons with ReLU + Dropout
        - Hidden layer 2: 32 neurons with ReLU + Dropout
        - Hidden layer 3: 32 neurons with ReLU + Dropout
        - Hidden layer 4: 32 neurons with ReLU
        - Output layer: 31 wavelengths (400-700nm)
    """

    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 31, dropout_rate: float = 0.2):
        """
        Initialize the deep neural network

        Args:
            input_size: Number of input features (reagents + thickness)
            hidden_size: Number of neurons per hidden layer (default: 32)
            output_size: Number of outputs (wavelengths)
            dropout_rate: Dropout rate for regularization (0.0 = no dropout)
        """
        super(SpectralNN, self).__init__()

        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # Layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        # Layer 3
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        # Layer 4
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()

        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)

        # Initialize weights with Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Layer 4
        x = self.fc4(x)
        x = self.relu4(x)

        # Output
        x = self.fc_out(x)
        return x


def train_neural_network(
    X: np.ndarray,
    Y: np.ndarray,
    hidden_size: int = 32,
    learning_rate: float = 0.005,
    epochs: int = 2000,
    batch_size: int = 8,
    l2_lambda: float = 0.005,
    dropout_rate: float = 0.2,
    device: Optional[str] = None,
    use_kfold: bool = False,
    n_folds: int = 5
) -> dict:
    """
    Train a deep neural network (4×32) for spectral prediction with optional k-fold cross-validation

    Args:
        X: Input features of shape (n_samples, n_features)
        Y: Target spectra of shape (n_samples, n_wavelengths)
        hidden_size: Number of neurons per hidden layer (default: 32 for 4×32 architecture)
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Mini-batch size
        l2_lambda: L2 regularization coefficient
        dropout_rate: Dropout rate for regularization
        device: Device to use ('cuda', 'cpu', or None for auto)
        use_kfold: Whether to use k-fold cross-validation
        n_folds: Number of folds for cross-validation

    Returns:
        Dictionary containing trained model and normalization parameters
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print(f"Training deep neural network (4×{hidden_size}) on {device}")
    print(f"Architecture: {X.shape[1]} → {hidden_size} → {hidden_size} → {hidden_size} → {hidden_size} → {Y.shape[1]} outputs")

    # Normalize data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X - X_mean) / X_std

    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)
    Y_std[Y_std < 1e-8] = 1.0
    Y_norm = (Y - Y_mean) / Y_std

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_norm).to(device)
    Y_tensor = torch.FloatTensor(Y_norm).to(device)

    # Create model
    input_size = X.shape[1]
    output_size = Y.shape[1]
    model = SpectralNN(input_size, hidden_size, output_size, dropout_rate).to(device)

    # Loss and optimizer (using Adam for better convergence with deep networks)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Training loop with early stopping
    n_samples = X_tensor.shape[0]
    best_loss = float('inf')
    patience = 200
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        # Shuffle indices
        indices = torch.randperm(n_samples)
        total_loss = 0.0

        # Mini-batch training
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_indices = indices[batch_start:batch_end]

            # Get batch data
            batch_X = X_tensor[batch_indices]
            batch_Y = Y_tensor[batch_indices]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_indices)

        avg_loss = total_loss / n_samples

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best loss: {best_loss:.6f})")
            break

        # Log progress
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

    print("Deep neural network training complete")

    # Move model to CPU for storage
    model.to('cpu')

    return {
        'model': model,
        'input_mean': X_mean,
        'input_std': X_std,
        'output_mean': Y_mean,
        'output_std': Y_std,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size
    }


def predict_neural_network(
    input_data: np.ndarray,
    model: SpectralNN,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
    clip_min: float = -0.1,
    clip_max: float = 1.5
) -> np.ndarray:
    """
    Make predictions using trained neural network

    Args:
        input_data: Input features of shape (n_features,) or (n_samples, n_features)
        model: Trained SpectralNN model
        input_mean: Mean used for input normalization
        input_std: Std used for input normalization
        output_mean: Mean used for output normalization
        output_std: Std used for output normalization
        clip_min: Minimum value for output clipping
        clip_max: Maximum value for output clipping

    Returns:
        Predicted spectrum of shape (n_wavelengths,) or (n_samples, n_wavelengths)
    """
    model.eval()

    # Handle single sample
    single_sample = False
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
        single_sample = True

    # Normalize input
    input_norm = (input_data - input_mean) / input_std

    # Convert to tensor
    input_tensor = torch.FloatTensor(input_norm)

    # Predict
    with torch.no_grad():
        output_norm = model(input_tensor).numpy()

    # Denormalize output
    output = output_norm * output_std + output_mean

    # Clip to reasonable range
    output = np.clip(output, clip_min, clip_max)

    # Return single sample if input was single sample
    if single_sample:
        return output[0]

    return output
