"""
K-fold cross-validation utilities for model evaluation
"""
import numpy as np
from typing import List, Tuple, Callable
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from types_constants import SampleData


def kfold_split(samples: List[SampleData], n_folds: int = 5, shuffle: bool = True, seed: int = None) -> List[Tuple[List[SampleData], List[SampleData]]]:
    """
    Split samples into k folds for cross-validation

    Args:
        samples: List of samples to split
        n_folds: Number of folds
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility

    Returns:
        List of (train_samples, val_samples) tuples for each fold
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(samples)

    if n_folds > n_samples:
        print(f"Warning: n_folds ({n_folds}) > n_samples ({n_samples}). Using {n_samples}-fold (leave-one-out)")
        n_folds = n_samples

    # Shuffle samples if requested
    if shuffle:
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    # Calculate fold sizes
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1

    # Create folds
    folds = []
    current = 0
    for fold_size in fold_sizes:
        fold_indices = indices[current:current + fold_size]
        train_indices = np.concatenate([indices[:current], indices[current + fold_size:]])

        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in fold_indices]

        folds.append((train_samples, val_samples))
        current += fold_size

    return folds


def cross_validate_model(
    samples: List[SampleData],
    train_fn: Callable,
    predict_fn: Callable,
    n_folds: int = 5,
    seed: int = None
) -> dict:
    """
    Perform k-fold cross-validation

    Args:
        samples: All samples
        train_fn: Function to train model, signature: train_fn(train_samples) -> model
        predict_fn: Function to predict, signature: predict_fn(sample, model) -> prediction
        n_folds: Number of folds
        seed: Random seed

    Returns:
        Dictionary with cross-validation results
    """
    print(f"\nRunning {n_folds}-fold cross-validation...")
    print(f"Total samples: {len(samples)}")

    folds = kfold_split(samples, n_folds, shuffle=True, seed=seed)

    fold_errors = []
    fold_mae = []
    fold_rmse = []

    for fold_idx, (train_samples, val_samples) in enumerate(folds):
        print(f"\nFold {fold_idx + 1}/{n_folds}:")
        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val:   {len(val_samples)} samples")

        # Train model on this fold
        model = train_fn(train_samples)

        # Evaluate on validation set
        errors = []
        for val_sample in val_samples:
            prediction = predict_fn(val_sample, model)
            error = np.abs(prediction - val_sample.spectrum)
            errors.append(error)

        # Calculate metrics for this fold
        fold_errors_array = np.array(errors)
        mae = np.mean(fold_errors_array)
        rmse = np.sqrt(np.mean(fold_errors_array ** 2))

        fold_mae.append(mae)
        fold_rmse.append(rmse)

        print(f"  Fold MAE:  {mae:.6f}")
        print(f"  Fold RMSE: {rmse:.6f}")

    # Overall results
    print(f"\n{'='*60}")
    print("Cross-Validation Results:")
    print(f"{'='*60}")
    print(f"Mean MAE:  {np.mean(fold_mae):.6f} ± {np.std(fold_mae):.6f}")
    print(f"Mean RMSE: {np.mean(fold_rmse):.6f} ± {np.std(fold_rmse):.6f}")
    print(f"{'='*60}\n")

    return {
        'fold_mae': fold_mae,
        'fold_rmse': fold_rmse,
        'mean_mae': np.mean(fold_mae),
        'std_mae': np.std(fold_mae),
        'mean_rmse': np.mean(fold_rmse),
        'std_rmse': np.std(fold_rmse),
        'n_folds': n_folds
    }


def stratified_split(
    samples: List[SampleData],
    test_size: float = 0.2,
    seed: int = None
) -> Tuple[List[SampleData], List[SampleData]]:
    """
    Split samples into train and test sets

    Args:
        samples: All samples
        test_size: Fraction for test set (0.0 to 1.0)
        seed: Random seed

    Returns:
        Tuple of (train_samples, test_samples)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(samples)
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]

    print(f"Train/Test split:")
    print(f"  Train: {len(train_samples)} samples ({100*(1-test_size):.0f}%)")
    print(f"  Test:  {len(test_samples)} samples ({100*test_size:.0f}%)")

    return train_samples, test_samples
