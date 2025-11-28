"""
Kubelka-Munk Hybrid Neural Network
===================================

Separates the prediction into two components:
1. K-M model for non-fluorescent pigments (BiVaO4, PG, PearlB) - physics-based
2. Neural network for fluorescent pigment (GXT) - data-driven

This is more interpretable and physically grounded than a pure NN approach.
"""
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Kubelka-Munk Theory for Non-Fluorescent Pigments
# ============================================================================

def kubelka_munk_reflectance(K, S, thickness_um=8.0):
    """
    Calculate reflectance using Kubelka-Munk theory for non-fluorescent pigments

    Args:
        K: Absorption coefficient (dimensionless)
        S: Scattering coefficient (dimensionless)
        thickness_um: Coating thickness in micrometers

    Returns:
        R: Reflectance (0-1)

    Reference:
        Kubelka, P., & Munk, F. (1931). "An article on optics of paint layers"
        Zeit. Für Tekn. Physik, 12, 593-601.
    """
    # Avoid division by zero
    if S < 1e-8:
        return 1.0  # No absorption or scattering = full reflectance

    # K/S ratio (fundamental K-M parameter)
    a = 1 + K / (S + 1e-8)

    # b parameter
    b = np.sqrt(a**2 - 1)

    # Thickness parameter (normalized by scattering)
    bSX = b * S * (thickness_um / 10.0)  # Normalize thickness

    # K-M reflectance equation (assumes infinite background reflectance = 1)
    sinh_term = np.sinh(bSX)
    cosh_term = np.cosh(bSX)

    numerator = 1
    denominator = a * sinh_term + cosh_term

    R = numerator / (denominator + 1e-8)

    # Clip to valid range
    R = np.clip(R, 0.0, 1.0)

    return R


def calculate_km_coefficients(concentrations, pigment_properties):
    """
    Calculate K and S coefficients for a mixture of non-fluorescent pigments

    Args:
        concentrations: dict with keys ['BiVaO4', 'PG', 'PearlB'] (percentages)
        pigment_properties: dict with K and S values per unit concentration

    Returns:
        K_total, S_total: Total absorption and scattering coefficients
    """
    K_total = 0.0
    S_total = 0.0

    for pigment in ['BiVaO4', 'PG', 'PearlB']:
        conc = concentrations.get(pigment, 0.0) / 100.0  # Normalize to 0-1

        if pigment in pigment_properties:
            K_total += conc * pigment_properties[pigment]['K']
            S_total += conc * pigment_properties[pigment]['S']

    return K_total, S_total


def predict_base_reflectance(BiVaO4, PG, PearlB, thickness, wavelength_idx=None):
    """
    Predict base reflectance using K-M theory for non-fluorescent pigments

    Args:
        BiVaO4, PG, PearlB: Concentrations (%)
        thickness: Coating thickness (μm)
        wavelength_idx: Optional wavelength index for wavelength-dependent K/S

    Returns:
        R_base: Base reflectance from non-fluorescent pigments (0-1)
    """
    # Pigment optical properties (K = absorption, S = scattering)
    # These are empirical values that would ideally be measured
    # Units: per unit concentration (0-1 scale)

    # BiVaO4 (Bismuth Vanadate - Yellow pigment)
    # - Moderate absorption in blue (400-500nm), low in red
    # - Good scattering across all wavelengths
    K_BiVaO4 = 0.3
    S_BiVaO4 = 0.7

    # PG (Pigment Green)
    # - High absorption in red and blue, low in green
    # - Moderate scattering
    K_PG = 0.8
    S_PG = 0.4

    # PearlB (Pearl Blue - Pearlescent)
    # - Very low absorption (transparent)
    # - Excellent scattering (pearl effect)
    K_PearlB = 0.15
    S_PearlB = 0.9

    # Calculate total K and S (linear mixing assumption)
    concentrations = {
        'BiVaO4': BiVaO4,
        'PG': PG,
        'PearlB': PearlB
    }

    pigment_properties = {
        'BiVaO4': {'K': K_BiVaO4, 'S': S_BiVaO4},
        'PG': {'K': K_PG, 'S': S_PG},
        'PearlB': {'K': K_PearlB, 'S': S_PearlB}
    }

    K_total, S_total = calculate_km_coefficients(concentrations, pigment_properties)

    # Calculate reflectance using K-M theory
    R_base = kubelka_munk_reflectance(K_total, S_total, thickness)

    return R_base


# ============================================================================
# Neural Network for Fluorescent Component (GXT)
# ============================================================================

class FluorescenceNN(nn.Module):
    """
    Neural network that learns the fluorescence contribution (GXT)

    Inputs:
        - GXT concentration
        - Base reflectance (from K-M model)
        - Thickness

    Outputs:
        - Delta reflectance (fluorescence contribution) for 31 wavelengths
        - CIELAB values (L, a, b, c, h)
    """
    def __init__(self, hidden_size=32):
        super().__init__()

        # Input: GXT, R_base (31), thickness = 33 features
        input_size = 1 + 31 + 1

        # Smaller network since we're only learning fluorescence
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2)
        )

        # Spectral output: Delta reflectance (how much fluorescence adds)
        self.spectral_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 31)  # 31 wavelengths
        )

        # CIELAB output
        self.cielab_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 5)  # L, a, b, c, h
        )

    def forward(self, x):
        shared = self.shared(x)
        spectral = self.spectral_head(shared)
        cielab = self.cielab_head(shared)
        return spectral, cielab

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Hybrid Model: K-M + NN
# ============================================================================

def predict_hybrid(GXT, BiVaO4, PG, PearlB, thickness, nn_model,
                   input_mean, input_std, output_mean, output_std):
    """
    Hybrid prediction combining K-M theory and neural network

    Args:
        GXT, BiVaO4, PG, PearlB: Concentrations (%)
        thickness: Coating thickness (μm)
        nn_model: Trained FluorescenceNN model
        input_mean, input_std: Normalization parameters for NN inputs
        output_mean, output_std: Normalization parameters for NN outputs

    Returns:
        R_total: Total reflectance (31 wavelengths)
        cielab: CIELAB color values
    """
    # Step 1: Calculate base reflectance using K-M theory (31 wavelengths)
    R_base = np.array([
        predict_base_reflectance(BiVaO4, PG, PearlB, thickness, wl_idx)
        for wl_idx in range(31)
    ])

    # Step 2: Prepare NN input [GXT, R_base[31], thickness]
    x = np.concatenate([
        [GXT / 100.0],  # Normalize GXT
        R_base,  # Base reflectance (already 0-1)
        [thickness / 12.0]  # Normalize thickness
    ])

    # Step 3: Normalize and predict fluorescence contribution
    x_normalized = (x - input_mean) / (input_std + 1e-8)
    x_tensor = torch.FloatTensor(x_normalized).unsqueeze(0)

    nn_model.eval()
    with torch.no_grad():
        delta_R, cielab_pred = nn_model(x_tensor)
        delta_R = delta_R.numpy().flatten()
        cielab_pred = cielab_pred.numpy().flatten()

    # Step 4: Denormalize outputs
    delta_R = delta_R * (output_std[:31] + 1e-8) + output_mean[:31]
    cielab_norm = cielab_pred * (output_std[31:] + 1e-8) + output_mean[31:]

    # Step 5: Combine base + fluorescence
    R_total = R_base + delta_R
    R_total = np.clip(R_total, 0.0, 1.0)  # Physical constraint

    # Step 6: Denormalize CIELAB
    cielab = {
        'L': float(cielab_norm[0] * 100.0),
        'a': float(cielab_norm[1] * 256.0 - 128.0),
        'b': float(cielab_norm[2] * 256.0 - 128.0),
        'c': float(cielab_norm[3] * 150.0),
        'h': float(cielab_norm[4] * 360.0)
    }

    return R_total, cielab


def train_hybrid_model(X_train, Y_train, X_val, Y_val,
                       hidden_size=32,
                       learning_rate=0.001,
                       epochs=2000,
                       batch_size=16,
                       l2_lambda=0.001,
                       dropout_rate=0.2,
                       verbose=True):
    """
    Train the hybrid K-M + NN model

    Args:
        X_train, Y_train: Training data [GXT, BiVaO4, PG, PearlB, Thickness] -> [31 spectral + 5 CIELAB]
        X_val, Y_val: Validation data

    Returns:
        model_data: Dict with trained model and normalization parameters
    """
    print("Training Hybrid K-M + Fluorescence NN Model")
    print("=" * 80)

    # Step 1: Calculate K-M base reflectance for all training samples
    print("Calculating K-M base reflectance for training set...")
    R_base_train = []
    delta_R_train = []

    for i, x in enumerate(X_train):
        GXT, BiVaO4, PG, PearlB, thickness = x[0]*100, x[1]*100, x[2]*100, x[3]*100, x[4]*12

        # K-M base reflectance (non-fluorescent)
        R_base = np.array([
            predict_base_reflectance(BiVaO4, PG, PearlB, thickness, wl_idx)
            for wl_idx in range(31)
        ])
        R_base_train.append(R_base)

        # Target: Delta R = R_total - R_base (fluorescence contribution)
        R_total_target = Y_train[i, :31]
        delta_R = R_total_target - R_base
        delta_R_train.append(delta_R)

    R_base_train = np.array(R_base_train)
    delta_R_train = np.array(delta_R_train)

    # Step 2: Prepare NN inputs [GXT, R_base, thickness]
    X_train_nn = np.concatenate([
        X_train[:, 0:1],  # GXT only
        R_base_train,     # K-M base reflectance (31)
        X_train[:, 4:5]   # Thickness
    ], axis=1)

    # NN outputs: [delta_R (31), CIELAB (5)]
    Y_train_nn = np.concatenate([
        delta_R_train,
        Y_train[:, 31:]  # CIELAB
    ], axis=1)

    # Step 3: Same for validation set
    print("Calculating K-M base reflectance for validation set...")
    R_base_val = []
    delta_R_val = []

    for i, x in enumerate(X_val):
        GXT, BiVaO4, PG, PearlB, thickness = x[0]*100, x[1]*100, x[2]*100, x[3]*100, x[4]*12

        R_base = np.array([
            predict_base_reflectance(BiVaO4, PG, PearlB, thickness, wl_idx)
            for wl_idx in range(31)
        ])
        R_base_val.append(R_base)

        R_total_target = Y_val[i, :31]
        delta_R = R_total_target - R_base
        delta_R_val.append(delta_R)

    R_base_val = np.array(R_base_val)
    delta_R_val = np.array(delta_R_val)

    X_val_nn = np.concatenate([
        X_val[:, 0:1],
        R_base_val,
        X_val[:, 4:5]
    ], axis=1)

    Y_val_nn = np.concatenate([
        delta_R_val,
        Y_val[:, 31:]
    ], axis=1)

    # Step 4: Normalize
    input_mean = np.mean(X_train_nn, axis=0)
    input_std = np.std(X_train_nn, axis=0)

    output_mean = np.mean(Y_train_nn, axis=0)
    output_std = np.std(Y_train_nn, axis=0)

    X_train_norm = (X_train_nn - input_mean) / (input_std + 1e-8)
    X_val_norm = (X_val_nn - input_mean) / (input_std + 1e-8)

    Y_train_norm = (Y_train_nn - output_mean) / (output_std + 1e-8)
    Y_val_norm = (Y_val_nn - output_mean) / (output_std + 1e-8)

    # Step 5: Train fluorescence NN
    print(f"\nTraining fluorescence NN:")
    print(f"  Input size: {X_train_nn.shape[1]} (GXT + 31 R_base + thickness)")
    print(f"  Output size: {Y_train_nn.shape[1]} (31 delta_R + 5 CIELAB)")
    print(f"  Training samples: {len(X_train_nn)}")
    print(f"  Hidden size: {hidden_size}")

    model = FluorescenceNN(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_norm)
    Y_train_tensor = torch.FloatTensor(Y_train_norm)
    X_val_tensor = torch.FloatTensor(X_val_norm)
    Y_val_tensor = torch.FloatTensor(Y_val_norm)

    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_spectral_loss': [],
        'val_cielab_loss': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 100

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        indices = torch.randperm(len(X_train_tensor))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_tensor[batch_idx]
            Y_batch = Y_train_tensor[batch_idx]

            optimizer.zero_grad()

            pred_spectral, pred_cielab = model(X_batch)
            pred = torch.cat([pred_spectral, pred_cielab], dim=1)

            loss = nn.MSELoss()(pred, Y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            pred_spectral_val, pred_cielab_val = model(X_val_tensor)
            pred_val = torch.cat([pred_spectral_val, pred_cielab_val], dim=1)

            val_loss = nn.MSELoss()(pred_val, Y_val_tensor).item()
            val_spectral_loss = nn.MSELoss()(pred_spectral_val, Y_val_tensor[:, :31]).item()
            val_cielab_loss = nn.MSELoss()(pred_cielab_val, Y_val_tensor[:, 31:]).item()

        history['val_loss'].append(val_loss)
        history['val_spectral_loss'].append(val_spectral_loss)
        history['val_cielab_loss'].append(val_cielab_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1:4d}: Val Loss = {val_loss:.6f} "
                  f"(Spectral: {val_spectral_loss:.6f}, CIELAB: {val_cielab_loss:.6f})")

    print(f"\n✓ Training complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Model parameters: {model.count_parameters()}")

    return {
        'model': model,
        'input_mean': input_mean,
        'input_std': input_std,
        'output_mean': output_mean,
        'output_std': output_std,
        'history': history,
        'best_val_loss': best_val_loss,
        'model_type': 'hybrid_km_fluorescence',
        'hidden_size': hidden_size
    }
