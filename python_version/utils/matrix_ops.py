"""
Matrix utilities for Kubelka-Munk calculations
Uses NumPy for efficient matrix operations
"""
import numpy as np


def pseudo_inverse(A: np.ndarray) -> np.ndarray:
    """
    Compute the Moore-Penrose pseudo-inverse of matrix A
    Uses NumPy's built-in pinv for robust computation

    Args:
        A: Input matrix of shape (m, n)

    Returns:
        Pseudo-inverse of shape (n, m)
    """
    return np.linalg.pinv(A)


def solve_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve least squares problem: minimize ||Ax - b||^2

    Args:
        A: Coefficient matrix of shape (m, n)
        b: Target matrix of shape (m, p)

    Returns:
        Solution x of shape (n, p)
    """
    # Use lstsq for numerical stability
    solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return solution


def normalize_data(data: np.ndarray, mean: np.ndarray = None,
                   std: np.ndarray = None) -> tuple:
    """
    Normalize data to zero mean and unit variance

    Args:
        data: Data array of shape (n_samples, n_features)
        mean: Optional pre-computed mean
        std: Optional pre-computed std

    Returns:
        Tuple of (normalized_data, mean, std)
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
        # Prevent division by zero
        std[std < 1e-8] = 1.0

    normalized = (data - mean) / std
    return normalized, mean, std


def denormalize_data(normalized: np.ndarray, mean: np.ndarray,
                     std: np.ndarray) -> np.ndarray:
    """
    Denormalize data from normalized space back to original scale

    Args:
        normalized: Normalized data
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized data
    """
    return normalized * std + mean
