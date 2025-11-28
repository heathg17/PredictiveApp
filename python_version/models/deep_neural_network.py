"""
Deep neural network with multiple layers for spectral prediction
Comparing shallow (1x128) vs deep (4x32) architectures
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional
import numpy as np


class DeepSpectralNN(nn.Module):
    """
    Deep feedforward neural network for spectral reflectance prediction

    Architecture:
        - Input layer: n_reagents + 1 (thickness)
        - Multiple hidden layers with configurable sizes
        - Output layer: 31 wavelengths (400-700nm)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [32, 32, 32, 32],
        output_size: int = 31,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the deep neural network

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes (e.g., [32, 32, 32, 32])
            output_size: Number of outputs (wavelengths)
            dropout_rate: Dropout rate for regularization
        """
        super(DeepSpectralNN, self).__init__()

        self.hidden_sizes = hidden_sizes
        layers = []

        # Build hidden layers
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            # ReLU activation
            layers.append(nn.ReLU())
            # Dropout (except last hidden layer for stability)
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Build sequential model
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer (no activation)
        self.output_layer = nn.Linear(prev_size, output_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_deep_neural_network(
    X: np.ndarray,
    Y: np.ndarray,
    hidden_sizes: List[int] = [32, 32, 32, 32],
    learning_rate: float = 0.005,
    epochs: int = 2000,
    batch_size: int = 8,
    l2_lambda: float = 0.005,
    dropout_rate: float = 0.2,
    device: Optional[str] = None
) -> dict:
    """
    Train a deep neural network for spectral prediction

    Args:
        X: Input features of shape (n_samples, n_features)
        Y: Target spectra of shape (n_samples, n_wavelengths)
        hidden_sizes: List of hidden layer sizes
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Mini-batch size
        l2_lambda: L2 regularization coefficient
        dropout_rate: Dropout rate for regularization
        device: Device to use ('cuda', 'cpu', or None for auto)

    Returns:
        Dictionary containing trained model and normalization parameters
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    arch_str = "→".join([str(X.shape[1])] + [str(h) for h in hidden_sizes] + [str(Y.shape[1])])
    print(f"Training deep neural network on {device}")
    print(f"Architecture: {arch_str}")

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
    model = DeepSpectralNN(input_size, hidden_sizes, output_size, dropout_rate).to(device)

    # Print parameter count
    param_count = model.count_parameters()
    print(f"Total parameters: {param_count:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Training loop
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
            print(f"Early stopping at epoch {epoch}")
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
        'hidden_sizes': hidden_sizes,
        'output_size': output_size,
        'param_count': param_count
    }


def predict_deep_neural_network(
    input_data: np.ndarray,
    model: DeepSpectralNN,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
    clip_min: float = -0.1,
    clip_max: float = 1.5
) -> np.ndarray:
    """Make predictions using trained deep neural network"""
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
