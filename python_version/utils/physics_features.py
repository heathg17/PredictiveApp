"""
Physics-Informed Feature Engineering for Neural Networks

Adds Kubelka-Munk theory and pigment interaction features to help
the neural network learn spectral predictions more efficiently.
"""
import numpy as np


def km_forward(K, S, Rg=0.95):
    """
    Kubelka-Munk forward model (infinite substrate)

    Args:
        K: Absorption coefficient
        S: Scattering coefficient
        Rg: Background reflectance (default: 0.95 for white substrate)

    Returns:
        R: Reflectance
    """
    # Avoid division by zero
    S = np.maximum(S, 1e-8)

    a = 1 + K / S
    b = np.sqrt(a**2 - 1)

    # Saunderson correction for infinite substrate
    num = a - b * (1 - Rg) / (1 + Rg * b)
    den = a + b * (1 - Rg) / (1 + Rg * b)

    R = num / den
    return np.clip(R, 0, 1)


def reflectance_to_ks(R):
    """
    Convert reflectance to K/S ratio using Kubelka-Munk equation

    R = Reflectance (0-1, clamped to avoid division issues)
    Returns: K/S ratio
    """
    R_safe = np.clip(R, 0.001, 0.999)
    return (1 - R_safe)**2 / (2 * R_safe)


def add_physics_informed_features(X, include_km_features=True):
    """
    Add physics-informed features to input concentrations

    Args:
        X: Input array [n_samples, 5]
           Columns: [GXT, BiVaO4, PG, PearlB, Thickness]
        include_km_features: Include K-M derived features for non-fluorescent pigments

    Returns:
        X_enhanced: Extended feature array [n_samples, n_features]
        feature_names: List of feature names for interpretability
    """
    n_samples = X.shape[0]

    # Original features
    GXT = X[:, 0]
    BiVaO4 = X[:, 1]
    PG = X[:, 2]
    PearlB = X[:, 3]
    thickness = X[:, 4]

    features = []
    feature_names = []

    # 1. Original features (5)
    features.append(X)
    feature_names.extend(['GXT', 'BiVaO4', 'PG', 'PearlB', 'Thickness'])

    # 2. Total concentration (1)
    total_conc = (GXT + BiVaO4 + PG + PearlB).reshape(-1, 1)
    features.append(total_conc)
    feature_names.append('Total_Conc')

    # 3. Non-fluorescent concentration (BiVaO4, PG, PearlB) (1)
    non_fluor_conc = (BiVaO4 + PG + PearlB).reshape(-1, 1)
    features.append(non_fluor_conc)
    feature_names.append('NonFluor_Conc')

    # 4. Concentration ratios (4)
    # Avoid division by zero
    total_safe = np.maximum(total_conc, 1e-8)
    features.append((GXT / total_safe.flatten()).reshape(-1, 1))
    features.append((BiVaO4 / total_safe.flatten()).reshape(-1, 1))
    features.append((PG / total_safe.flatten()).reshape(-1, 1))
    features.append((PearlB / total_safe.flatten()).reshape(-1, 1))
    feature_names.extend(['Ratio_GXT', 'Ratio_BiVaO4', 'Ratio_PG', 'Ratio_PearlB'])

    # 5. Pairwise interactions (6 combinations)
    # Important for pigment mixing effects
    features.append((GXT * BiVaO4).reshape(-1, 1))
    features.append((GXT * PG).reshape(-1, 1))
    features.append((GXT * PearlB).reshape(-1, 1))
    features.append((BiVaO4 * PG).reshape(-1, 1))
    features.append((BiVaO4 * PearlB).reshape(-1, 1))
    features.append((PG * PearlB).reshape(-1, 1))
    feature_names.extend(['GXT×BiVaO4', 'GXT×PG', 'GXT×PearlB',
                         'BiVaO4×PG', 'BiVaO4×PearlB', 'PG×PearlB'])

    # 6. Thickness interactions (5)
    # Coating thickness affects opacity differently based on concentration
    features.append((thickness * total_conc.flatten()).reshape(-1, 1))
    features.append((thickness * GXT).reshape(-1, 1))
    features.append((thickness * BiVaO4).reshape(-1, 1))
    features.append((thickness * PG).reshape(-1, 1))
    features.append((thickness * PearlB).reshape(-1, 1))
    feature_names.extend(['Thickness×Total', 'Thickness×GXT', 'Thickness×BiVaO4',
                         'Thickness×PG', 'Thickness×PearlB'])

    # 7. Quadratic terms (5)
    # Non-linear concentration effects
    features.append((GXT ** 2).reshape(-1, 1))
    features.append((BiVaO4 ** 2).reshape(-1, 1))
    features.append((PG ** 2).reshape(-1, 1))
    features.append((PearlB ** 2).reshape(-1, 1))
    features.append((thickness ** 2).reshape(-1, 1))
    feature_names.extend(['GXT²', 'BiVaO4²', 'PG²', 'PearlB²', 'Thickness²'])

    # 8. PHYSICS-INFORMED: Kubelka-Munk derived features
    if include_km_features:
        # For non-fluorescent pigments, estimate K and S coefficients
        # Based on typical values for common pigments

        # Typical K/S ratios for pigments (literature values, approximate)
        # BiVaO4 (Bismuth Vanadate - yellow): Higher scattering
        # PG (Phthalocyanine Green): High absorption, medium scattering
        # PearlB (Pearl Blue): High scattering, low absorption

        # Estimated K coefficients (absorption) - normalized
        K_BiVaO4_est = BiVaO4 * 0.3  # Yellow pigments: moderate absorption
        K_PG_est = PG * 0.8           # Green pigments: high absorption
        K_PearlB_est = PearlB * 0.15  # Pearl pigments: low absorption

        # Estimated S coefficients (scattering) - normalized
        S_BiVaO4_est = BiVaO4 * 0.7   # Good scattering
        S_PG_est = PG * 0.4           # Moderate scattering
        S_PearlB_est = PearlB * 0.9   # Excellent scattering (pearl effect)

        # Total K and S for non-fluorescent mixture
        K_total = K_BiVaO4_est + K_PG_est + K_PearlB_est
        S_total = S_BiVaO4_est + S_PG_est + S_PearlB_est

        features.append(K_total.reshape(-1, 1))
        features.append(S_total.reshape(-1, 1))
        feature_names.extend(['KM_K_total', 'KM_S_total'])

        # K/S ratio (absorption/scattering)
        S_safe = np.maximum(S_total, 1e-8)
        KS_ratio = K_total / S_safe
        features.append(KS_ratio.reshape(-1, 1))
        feature_names.append('KM_KS_ratio')

        # Estimated reflectance from K-M theory (simplified, single wavelength approximation)
        R_est = km_forward(K_total, S_total, Rg=0.95)
        features.append(R_est.reshape(-1, 1))
        feature_names.append('KM_R_estimate')

        # Opacity estimate (Beer-Lambert like)
        # Higher K*thickness = more opaque
        opacity = 1 - np.exp(-K_total * thickness / 10.0)  # Normalized
        features.append(opacity.reshape(-1, 1))
        feature_names.append('KM_Opacity')

        # Hiding power (combination of K, S, and thickness)
        hiding_power = (K_total + S_total) * thickness / 10.0
        features.append(hiding_power.reshape(-1, 1))
        feature_names.append('KM_HidingPower')

    # Concatenate all features
    X_enhanced = np.hstack(features)

    return X_enhanced, feature_names


def add_data_augmentation(X, Y, noise_level=0.015, n_augmented=2):
    """
    Generate augmented training samples with small perturbations

    Simulates measurement noise and slight formulation variations

    Args:
        X: Input samples [n_samples, n_features]
        Y: Output samples [n_samples, n_outputs]
        noise_level: Gaussian noise standard deviation (as fraction of std)
        n_augmented: Number of augmented samples per original sample

    Returns:
        X_aug, Y_aug: Augmented datasets
    """
    X_aug_list = [X]
    Y_aug_list = [Y]

    # Calculate standard deviations for noise scaling
    X_std = np.std(X, axis=0)
    Y_std = np.std(Y, axis=0)

    for _ in range(n_augmented):
        # Add Gaussian noise to inputs (concentrations)
        X_noise = X + np.random.normal(0, noise_level * X_std, X.shape)

        # Concentrations must be non-negative
        # Thickness must stay within reasonable bounds
        X_noise[:, :4] = np.maximum(X_noise[:, :4], 0)  # Concentrations >= 0
        X_noise[:, 4] = np.clip(X_noise[:, 4], 6.0, 15.0)  # Thickness: 6-15 μm

        # Add smaller noise to outputs (spectral measurements have some noise)
        # Less noise on outputs to maintain consistency
        Y_noise = Y + np.random.normal(0, noise_level * 0.3 * Y_std, Y.shape)

        # Reflectance should stay in reasonable bounds (allow slightly >1 for fluorescence)
        Y_noise[:, :31] = np.clip(Y_noise[:, :31], -0.05, 1.5)

        X_aug_list.append(X_noise)
        Y_aug_list.append(Y_noise)

    X_augmented = np.vstack(X_aug_list)
    Y_augmented = np.vstack(Y_aug_list)

    return X_augmented, Y_augmented


if __name__ == "__main__":
    # Test the feature engineering
    print("Testing Physics-Informed Feature Engineering")
    print("=" * 80)

    # Create test sample
    X_test = np.array([
        [10.0, 5.0, 1.5, 0.0, 8.0],   # GXT, BiVaO4, PG, PearlB, Thickness
        [0.0, 8.0, 0.0, 2.0, 12.0],
        [15.0, 0.0, 2.0, 1.0, 8.0]
    ])

    print(f"Original features: {X_test.shape}")
    print(X_test)
    print()

    # Add physics-informed features
    X_enhanced, feature_names = add_physics_informed_features(X_test, include_km_features=True)

    print(f"Enhanced features: {X_enhanced.shape}")
    print(f"Number of features: {len(feature_names)}")
    print()

    print("Feature names:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1:2d}. {name:20s} = {X_enhanced[0, i]:.4f}")

    print()
    print("=" * 80)
    print("Testing Data Augmentation")
    print("=" * 80)

    Y_test = np.random.rand(3, 36)  # Dummy output
    X_aug, Y_aug = add_data_augmentation(X_test, Y_test, noise_level=0.01, n_augmented=2)

    print(f"Original dataset: {X_test.shape[0]} samples")
    print(f"Augmented dataset: {X_aug.shape[0]} samples")
    print(f"Augmentation factor: {X_aug.shape[0] / X_test.shape[0]:.1f}x")
