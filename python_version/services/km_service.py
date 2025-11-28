"""
Kubelka-Munk model training and prediction service
Supports single-layer, two-layer, and neural network models
"""
import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from types_constants import SampleData, ModelCoefficients, WAVELENGTHS, Rg_DEFAULT
from utils.matrix_ops import pseudo_inverse, solve_least_squares
from models.neural_network import train_neural_network, predict_neural_network


# --- Fluorescence Detection ---

def has_fluorescence(spectrum: np.ndarray) -> bool:
    """
    Detect if a sample contains fluorescent pigments
    Fluorescent samples have R > 1 at some wavelengths

    Args:
        spectrum: Reflectance spectrum

    Returns:
        True if fluorescence detected
    """
    return np.any(spectrum > 1.0)


def extract_fluorescence(spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fluorescence component from spectrum
    For fluorescent samples: F(λ) = max(0, R(λ) - 1)

    Args:
        spectrum: Original reflectance spectrum

    Returns:
        Tuple of (reflectance, fluorescence) components
    """
    reflectance = np.minimum(spectrum, 1.0)
    fluorescence = np.maximum(0, spectrum - 1.0)
    return reflectance, fluorescence


# --- Single Layer K-M Functions ---

def reflectance_to_ks(R: np.ndarray) -> np.ndarray:
    """
    Convert reflectance to K/S ratio using Kubelka-Munk equation

    Args:
        R: Reflectance values

    Returns:
        K/S ratios
    """
    # Clamp to valid range for fluorescent materials
    R_safe = np.clip(R, 0.001, 0.999)
    return (1 - R_safe) ** 2 / (2 * R_safe)


def ks_to_reflectance(KS: np.ndarray) -> np.ndarray:
    """
    Convert K/S ratio to reflectance using inverse K-M equation

    Args:
        KS: K/S ratio values

    Returns:
        Reflectance values
    """
    term = np.sqrt(KS ** 2 + 2 * KS)
    return 1 + KS - term


# --- Two Layer K-M Functions ---

def two_layer_forward(K_mix: float, S_mix: float, Rg: float, X: float) -> float:
    """
    Two-layer Kubelka-Munk forward model

    Args:
        K_mix: Absorption coefficient
        S_mix: Scattering coefficient
        Rg: Background reflectance
        X: Film thickness

    Returns:
        Predicted reflectance
    """
    S = max(1e-6, S_mix)
    K = max(0, K_mix)

    a = 1 + K / S
    b = np.sqrt(a ** 2 - 1)

    # Limit exp power to avoid overflow/underflow
    bSX = np.clip(b * S * X, -50, 50)

    exp_bSX = np.exp(bSX)
    exp_neg = np.exp(-bSX)

    num = (1 - Rg * (a - b)) * exp_bSX * (a + b) - (1 - Rg * (a + b)) * exp_neg * (a - b)
    den = (a + b - Rg * (a - b)) * exp_bSX - (a - b - Rg * (a + b)) * exp_neg

    if den == 0:
        return 0

    R = num / den
    return np.clip(R, 0, 1)


# --- Training Functions ---

def train_model(samples: List[SampleData], model_type: str) -> ModelCoefficients:
    """
    Train spectral prediction model

    Args:
        samples: List of training samples
        model_type: 'single', 'two-layer', or 'neural-net'

    Returns:
        Trained model coefficients
    """
    if len(samples) == 0:
        raise ValueError("No samples to train on")

    # Filter samples with concentration data
    samples_with_conc = [
        s for s in samples
        if len(s.concentrations) > 0 and any(c > 0 for c in s.concentrations.values())
    ]

    if len(samples_with_conc) == 0:
        raise ValueError("No samples with concentration data found")

    print(f"Filtered to {len(samples_with_conc)} samples with concentration data "
          f"(from {len(samples)} total)")

    # Detect fluorescent samples
    fluorescent_samples = [s for s in samples_with_conc if has_fluorescence(s.spectrum)]
    print(f"Found {len(fluorescent_samples)} fluorescent samples (R > 1.0)")

    # Get unique reagents
    reagent_set = set()
    for s in samples_with_conc:
        for reagent, conc in s.concentrations.items():
            if conc > 0:
                reagent_set.add(reagent)
    reagents = sorted(list(reagent_set))

    print(f"Training with {len(samples_with_conc)} samples and {len(reagents)} reagents: "
          f"{', '.join(reagents)}")

    # Build concentration matrix [Samples x Reagents]
    C = np.array([
        [s.concentrations.get(r, 0) / 100.0 for r in reagents]
        for s in samples_with_conc
    ])

    if model_type == 'single':
        return train_single_layer_model(samples_with_conc, reagents, C,
                                        len(fluorescent_samples) > 0)
    elif model_type == 'two-layer':
        return train_two_layer_model(samples_with_conc, reagents, C,
                                      len(fluorescent_samples) > 0)
    elif model_type == 'neural-net':
        return train_neural_net_model(samples_with_conc, reagents, C)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_single_layer_model(samples: List[SampleData], reagents: List[str],
                             C: np.ndarray, has_fluorescent: bool) -> ModelCoefficients:
    """Train single-layer Kubelka-Munk model"""

    # Separate reflectance and fluorescence components
    reflectance_spectra = []
    fluorescence_spectra = []

    for s in samples:
        refl, fluor = extract_fluorescence(s.spectrum)
        reflectance_spectra.append(refl)
        fluorescence_spectra.append(fluor)

    reflectance_spectra = np.array(reflectance_spectra)
    fluorescence_spectra = np.array(fluorescence_spectra)

    # Convert reflectance to K/S
    KS_measured = reflectance_to_ks(reflectance_spectra)

    # Solve for alpha coefficients using pseudo-inverse
    try:
        C_pinv = pseudo_inverse(C)
        Alpha = C_pinv @ KS_measured  # [Reagents x Wavelengths]

        alpha_map = {reagent: Alpha[i, :] for i, reagent in enumerate(reagents)}

        # Train fluorescence model if needed
        fluor_map = None
        if has_fluorescent:
            print("Training fluorescence emission model...")
            Fluor = C_pinv @ fluorescence_spectra
            fluor_map = {reagent: Fluor[i, :] for i, reagent in enumerate(reagents)}
            print("Fluorescence model trained successfully")

        print("Single-layer model trained successfully")

        return ModelCoefficients(
            type='single',
            alpha=alpha_map,
            fluorescence=fluor_map,
            wavelengths=WAVELENGTHS
        )

    except Exception as e:
        print(f"Failed to compute pseudo-inverse: {e}")
        raise ValueError("Failed to train single-layer model. "
                        "Try using fewer samples or check data quality.")


def train_two_layer_model(samples: List[SampleData], reagents: List[str],
                          C: np.ndarray, has_fluorescent: bool) -> ModelCoefficients:
    """Train two-layer Kubelka-Munk model using gradient descent"""

    num_wavelengths = len(WAVELENGTHS)
    num_reagents = len(reagents)

    # Separate reflectance and fluorescence
    processed_samples = []
    for s in samples:
        refl, fluor = extract_fluorescence(s.spectrum)
        processed_samples.append({
            'reflectance': refl,
            'fluorescence': fluor,
            'thickness': s.thickness
        })

    K_map = {r: np.zeros(num_wavelengths) for r in reagents}
    S_map = {r: np.zeros(num_wavelengths) for r in reagents}
    F_map = {r: np.zeros(num_wavelengths) for r in reagents} if has_fluorescent else None

    # Optimize for each wavelength independently
    for w in range(num_wavelengths):
        R_targets = np.array([s['reflectance'][w] for s in processed_samples])
        X_vals = np.array([s['thickness'] for s in processed_samples])

        # Initial guess: K ~ 1.0, S ~ 10.0
        params = np.concatenate([
            np.ones(num_reagents),      # K values
            np.ones(num_reagents) * 10  # S values
        ])

        # Gradient descent with momentum
        initial_lr = 0.001
        iterations = 200
        momentum = 0.9
        velocity = np.zeros(2 * num_reagents)

        for iter in range(iterations):
            grads = np.zeros(2 * num_reagents)

            # Compute gradients
            for s_idx in range(len(processed_samples)):
                concs = C[s_idx]
                X = X_vals[s_idx]
                R_tgt = R_targets[s_idx]

                # Current K_mix, S_mix
                K_mix = np.dot(concs, params[:num_reagents])
                S_mix = np.dot(concs, params[num_reagents:])

                R_pred = two_layer_forward(K_mix, S_mix, Rg_DEFAULT, X)
                error = R_pred - R_tgt

                # Finite difference approximation
                epsilon = 1e-4

                for r in range(num_reagents):
                    # dL/dK_r
                    R_k_plus = two_layer_forward(K_mix + epsilon * concs[r], S_mix, Rg_DEFAULT, X)
                    dR_dK = (R_k_plus - R_pred) / epsilon
                    grads[r] += 2 * error * dR_dK

                    # dL/dS_r
                    R_s_plus = two_layer_forward(K_mix, S_mix + epsilon * concs[r], Rg_DEFAULT, X)
                    dR_dS = (R_s_plus - R_pred) / epsilon
                    grads[r + num_reagents] += 2 * error * dR_dS

            # Normalize gradients
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1:
                grads = grads / grad_norm

            # Update with momentum
            lr = initial_lr / (1 + iter / 100)
            velocity = momentum * velocity - lr * grads
            params += velocity

            # Enforce bounds
            params[:num_reagents] = np.maximum(0.01, params[:num_reagents])  # K >= 0.01
            params[num_reagents:] = np.maximum(0.1, params[num_reagents:])   # S >= 0.1

            # Early stopping
            if grad_norm < 1e-6 and iter > 50:
                break

        # Store optimized values
        for r_idx, reagent in enumerate(reagents):
            K_map[reagent][w] = params[r_idx]
            S_map[reagent][w] = params[r_idx + num_reagents]

        # Log progress
        if w % 10 == 0:
            avg_K = np.mean(params[:num_reagents])
            avg_S = np.mean(params[num_reagents:])
            print(f"λ={WAVELENGTHS[w]}nm: avg K={avg_K:.3f}, avg S={avg_S:.3f}")

        # Train fluorescence separately
        if has_fluorescent:
            F_targets = np.array([s['fluorescence'][w] for s in processed_samples])
            C_pinv = pseudo_inverse(C)
            F_coeffs = (C_pinv @ F_targets.reshape(-1, 1)).flatten()
            for r_idx, reagent in enumerate(reagents):
                F_map[reagent][w] = max(0, F_coeffs[r_idx])

    print("Two-layer model trained successfully")

    return ModelCoefficients(
        type='two-layer',
        K=K_map,
        S=S_map,
        fluorescence=F_map,
        wavelengths=WAVELENGTHS
    )


def train_neural_net_model(samples: List[SampleData], reagents: List[str],
                           C: np.ndarray, use_mixup: bool = True,
                           use_kfold: bool = False) -> ModelCoefficients:
    """Train neural network model with optional augmentation"""

    print("Training neural network (black-box model)...")

    # Apply mix-up augmentation if requested
    if use_mixup and len(samples) >= 2:
        from utils.data_augmentation import create_mixup_augmented_dataset
        augmented_samples = create_mixup_augmented_dataset(
            samples,
            n_synthetic=min(20, len(samples) * 4),  # 4x augmentation
            lambda_min=0.2,
            lambda_max=0.8
        )

        # Rebuild concentration matrix for augmented data
        C_aug = np.array([
            [s.concentrations.get(r, 0) / 100.0 for r in reagents]
            for s in augmented_samples
        ])
        X = np.column_stack([C_aug, np.array([s.thickness for s in augmented_samples])])
        Y = np.array([s.spectrum for s in augmented_samples])
    else:
        X = np.column_stack([C, np.array([s.thickness for s in samples])])
        Y = np.array([s.spectrum for s in samples])

    # Adjust dropout based on dataset size
    dropout_rate = 0.1 if len(samples) < 20 else 0.2

    # Train neural network with improvements (4×32 deep architecture)
    nn_data = train_neural_network(
        X, Y,
        hidden_size=32,
        learning_rate=0.005,
        epochs=2000,
        batch_size=min(8, max(4, len(samples) // 5)),
        l2_lambda=0.005,
        dropout_rate=dropout_rate,
        use_kfold=use_kfold
    )

    return ModelCoefficients(
        type='neural-net',
        neural_net={
            'reagents': reagents,
            'model': nn_data['model'],
            'input_mean': nn_data['input_mean'],
            'input_std': nn_data['input_std'],
            'output_mean': nn_data['output_mean'],
            'output_std': nn_data['output_std']
        },
        wavelengths=WAVELENGTHS
    )


# --- Prediction Functions ---

def predict_reflectance(concentrations: Dict[str, float],
                       model: ModelCoefficients,
                       thickness: float = 4.0) -> np.ndarray:
    """
    Predict reflectance spectrum for given formulation

    Args:
        concentrations: Dictionary of reagent concentrations (%)
        model: Trained model coefficients
        thickness: Film thickness

    Returns:
        Predicted reflectance spectrum
    """
    if model.type == 'neural-net' and model.neural_net:
        # Neural network prediction
        nn = model.neural_net
        reagents = nn['reagents']

        # Prepare input
        input_data = np.array([concentrations.get(r, 0) / 100.0 for r in reagents] + [thickness])

        return predict_neural_network(
            input_data,
            nn['model'],
            nn['input_mean'],
            nn['input_std'],
            nn['output_mean'],
            nn['output_std']
        )

    # K-M based predictions
    num_wavelengths = len(model.wavelengths)
    predicted_R = np.zeros(num_wavelengths)

    for i in range(num_wavelengths):
        if model.type == 'single':
            # Single Layer K-M
            ks_sum = 0.0
            for reagent, percent in concentrations.items():
                if model.alpha and reagent in model.alpha:
                    unit_ks = model.alpha[reagent][i]
                    ks_sum += (percent / 100.0) * unit_ks

            ks_sum = max(0.0001, ks_sum)
            base_reflectance = ks_to_reflectance(np.array([ks_sum]))[0]

        else:  # two-layer
            k_sum = 0.0
            s_sum = 0.0
            for reagent, percent in concentrations.items():
                if model.K and reagent in model.K:
                    unit_k = model.K[reagent][i]
                    k_sum += (percent / 100.0) * unit_k
                if model.S and reagent in model.S:
                    unit_s = model.S[reagent][i]
                    s_sum += (percent / 100.0) * unit_s

            base_reflectance = two_layer_forward(k_sum, s_sum, Rg_DEFAULT, thickness)

        # Add fluorescence contribution
        fluorescence = 0.0
        if model.fluorescence:
            for reagent, percent in concentrations.items():
                if reagent in model.fluorescence:
                    unit_f = model.fluorescence[reagent][i]
                    fluorescence += (percent / 100.0) * unit_f

        predicted_R[i] = base_reflectance + fluorescence

    return predicted_R
