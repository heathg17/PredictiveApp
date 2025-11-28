"""
Data augmentation utilities for spectral prediction
Includes mix-up augmentation for generating synthetic training samples
"""
import numpy as np
import random
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from types_constants import SampleData


def mixup_samples(sample1: SampleData, sample2: SampleData, lambda_mix: float) -> SampleData:
    """
    Create a synthetic sample by mixing two samples

    Mix-up formula:
        mixed_x = lambda * x1 + (1 - lambda) * x2
        mixed_y = lambda * y1 + (1 - lambda) * y2

    Args:
        sample1: First sample
        sample2: Second sample
        lambda_mix: Mixing ratio (0 to 1)

    Returns:
        Synthetic mixed sample
    """
    # Mix concentrations
    all_reagents = set(sample1.concentrations.keys()) | set(sample2.concentrations.keys())
    mixed_concentrations = {}

    for reagent in all_reagents:
        c1 = sample1.concentrations.get(reagent, 0.0)
        c2 = sample2.concentrations.get(reagent, 0.0)
        mixed_concentrations[reagent] = lambda_mix * c1 + (1 - lambda_mix) * c2

    # Mix spectra
    mixed_spectrum = lambda_mix * sample1.spectrum + (1 - lambda_mix) * sample2.spectrum

    # Mix thickness
    mixed_thickness = lambda_mix * sample1.thickness + (1 - lambda_mix) * sample2.thickness

    # Create ID for mixed sample
    mixed_id = f"mixup_{sample1.id}_{sample2.id}_{lambda_mix:.2f}"
    mixed_name = f"MixUp({sample1.name}+{sample2.name})"

    # Use substrate from dominant sample
    substrate = sample1.substrate if lambda_mix > 0.5 else sample2.substrate

    return SampleData(
        id=mixed_id,
        name=mixed_name,
        substrate=substrate,
        thickness=mixed_thickness,
        spectrum=mixed_spectrum,
        concentrations=mixed_concentrations
    )


def create_mixup_augmented_dataset(
    samples: List[SampleData],
    n_synthetic: int = 20,
    lambda_min: float = 0.2,
    lambda_max: float = 0.8,
    seed: int = None
) -> List[SampleData]:
    """
    Generate augmented dataset using mix-up

    Args:
        samples: Original training samples
        n_synthetic: Number of synthetic samples to generate
        lambda_min: Minimum mixing ratio
        lambda_max: Maximum mixing ratio
        seed: Random seed for reproducibility

    Returns:
        List containing original + synthetic samples
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if len(samples) < 2:
        print("Warning: Need at least 2 samples for mix-up augmentation")
        return samples

    print(f"Generating {n_synthetic} synthetic samples via mix-up augmentation...")
    print(f"  Lambda range: [{lambda_min:.2f}, {lambda_max:.2f}]")

    synthetic_samples = []

    for i in range(n_synthetic):
        # Randomly select two different samples
        s1, s2 = random.sample(samples, 2)

        # Random mixing ratio
        lambda_mix = random.uniform(lambda_min, lambda_max)

        # Create mixed sample
        mixed_sample = mixup_samples(s1, s2, lambda_mix)
        synthetic_samples.append(mixed_sample)

    # Combine original and synthetic
    augmented_dataset = samples + synthetic_samples

    print(f"✓ Dataset augmented: {len(samples)} → {len(augmented_dataset)} samples")
    print(f"  Original: {len(samples)}")
    print(f"  Synthetic: {len(synthetic_samples)}")

    return augmented_dataset


def create_mixup_batch(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.4
) -> tuple:
    """
    Apply mix-up augmentation to a batch during training

    This is an alternative approach that applies mix-up during training
    rather than pre-generating synthetic samples

    Args:
        X: Batch of input features (batch_size, n_features)
        Y: Batch of target outputs (batch_size, n_outputs)
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        Tuple of (mixed_X, mixed_Y)
    """
    batch_size = X.shape[0]

    # Sample mixing ratios from Beta distribution
    if alpha > 0:
        lambda_mix = np.random.beta(alpha, alpha, batch_size)
    else:
        lambda_mix = np.ones(batch_size)

    # Reshape for broadcasting
    lambda_X = lambda_mix.reshape(-1, 1)
    lambda_Y = lambda_mix.reshape(-1, 1)

    # Randomly shuffle indices
    indices = np.random.permutation(batch_size)

    # Mix inputs and outputs
    mixed_X = lambda_X * X + (1 - lambda_X) * X[indices]
    mixed_Y = lambda_Y * Y + (1 - lambda_Y) * Y[indices]

    return mixed_X, mixed_Y


def augment_with_noise(
    samples: List[SampleData],
    noise_level: float = 0.01,
    n_copies: int = 2
) -> List[SampleData]:
    """
    Augment dataset by adding small noise to samples

    This simulates measurement noise and creates slight variations

    Args:
        samples: Original samples
        noise_level: Standard deviation of Gaussian noise
        n_copies: Number of noisy copies per sample

    Returns:
        Augmented dataset
    """
    augmented = list(samples)  # Start with originals

    for sample in samples:
        for i in range(n_copies):
            # Add noise to spectrum
            noise = np.random.normal(0, noise_level, sample.spectrum.shape)
            noisy_spectrum = np.clip(sample.spectrum + noise, 0, 1.5)

            # Add small noise to concentrations (±1%)
            noisy_conc = {}
            for reagent, conc in sample.concentrations.items():
                conc_noise = np.random.normal(0, conc * 0.01)
                noisy_conc[reagent] = max(0, conc + conc_noise)

            # Create noisy sample
            noisy_sample = SampleData(
                id=f"{sample.id}_noise_{i}",
                name=f"{sample.name}_noisy{i}",
                substrate=sample.substrate,
                thickness=sample.thickness,
                spectrum=noisy_spectrum,
                concentrations=noisy_conc
            )
            augmented.append(noisy_sample)

    print(f"Noise augmentation: {len(samples)} → {len(augmented)} samples")
    return augmented


def validate_augmented_samples(samples: List[SampleData]) -> bool:
    """
    Validate that augmented samples are physically reasonable

    Args:
        samples: List of samples to validate

    Returns:
        True if all samples are valid
    """
    for sample in samples:
        # Check spectrum is in reasonable range
        if np.any(sample.spectrum < -0.1) or np.any(sample.spectrum > 1.5):
            print(f"Warning: Sample {sample.id} has spectrum out of range")
            return False

        # Check concentrations are non-negative
        if any(c < 0 for c in sample.concentrations.values()):
            print(f"Warning: Sample {sample.id} has negative concentrations")
            return False

        # Check total concentration isn't too high
        total_conc = sum(sample.concentrations.values())
        if total_conc > 150:  # Allow some headroom above 100%
            print(f"Warning: Sample {sample.id} has total concentration {total_conc:.1f}%")
            return False

        # Check thickness is positive
        if sample.thickness <= 0:
            print(f"Warning: Sample {sample.id} has invalid thickness")
            return False

    return True
