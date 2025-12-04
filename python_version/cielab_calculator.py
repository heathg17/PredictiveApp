"""
Deterministic CIELAB Calculator from Reflectance Spectrum

Converts spectral reflectance to CIELAB color coordinates using CIE standard
illuminants and observer functions. This ensures physically accurate color
predictions without relying on neural network approximations.

Process:
1. Calculate XYZ tristimulus values from reflectance
2. Convert XYZ to CIELAB using standard formulas
3. Calculate chroma (c) and hue angle (h) from a* and b*
"""
import numpy as np
from typing import Dict, Tuple

# CIE D65 Illuminant (daylight, 6500K) - 10nm intervals from 400-700nm
# Normalized so that Y_n = 100
D65_ILLUMINANT = np.array([
    49.98, 54.65, 82.75, 91.49, 93.43, 86.68, 104.87, 117.01, 117.81, 114.86,
    115.92, 108.81, 109.35, 107.80, 104.79, 107.69, 104.41, 104.05, 100.00,
    96.33, 95.79, 88.69, 90.01, 89.60, 87.70, 83.29, 83.70, 80.03, 80.21,
    82.28, 78.28
])

# CIE 1964 10° Standard Observer Color Matching Functions (x̄, ȳ, z̄)
# 10nm intervals from 400-700nm
CMF_X = np.array([
    0.0191, 0.0847, 0.2045, 0.3147, 0.3837, 0.3707, 0.3023, 0.1956, 0.0805,
    0.0162, 0.0038, 0.0375, 0.1177, 0.2365, 0.3768, 0.5298, 0.7052, 0.8787,
    1.0142, 1.1185, 1.1240, 1.0305, 0.8563, 0.6475, 0.4316, 0.2683, 0.1526,
    0.0813, 0.0409, 0.0199, 0.0096
])

CMF_Y = np.array([
    0.0020, 0.0088, 0.0214, 0.0387, 0.0621, 0.0895, 0.1282, 0.1852, 0.2536,
    0.3391, 0.4608, 0.6067, 0.7618, 0.8752, 0.9620, 0.9918, 0.9973, 0.9556,
    0.8689, 0.7774, 0.6583, 0.5280, 0.3981, 0.2835, 0.1798, 0.1076, 0.0603,
    0.0318, 0.0159, 0.0077, 0.0037
])

CMF_Z = np.array([
    0.0860, 0.3894, 0.9725, 1.5535, 1.9673, 1.9948, 1.7454, 1.3176, 0.7721,
    0.4153, 0.2185, 0.1120, 0.0607, 0.0305, 0.0137, 0.0040, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000
])

# White point for D65 illuminant with 10° observer
# Calculated from integrating D65 * CMF with k normalization
X_N = 93.253
Y_N = 100.000
Z_N = 94.247


def reflectance_to_xyz(reflectance: np.ndarray, illuminant: str = 'D65') -> Tuple[float, float, float]:
    """
    Convert spectral reflectance to CIE XYZ tristimulus values

    Args:
        reflectance: Array of 31 reflectance values (400-700nm, 10nm intervals)
        illuminant: Illuminant type (currently only 'D65' supported)

    Returns:
        Tuple of (X, Y, Z) tristimulus values
    """
    if len(reflectance) != 31:
        raise ValueError(f"Expected 31 reflectance values, got {len(reflectance)}")

    if illuminant != 'D65':
        raise ValueError(f"Only D65 illuminant supported, got {illuminant}")

    # Calculate tristimulus values by integrating R(λ) * S(λ) * CMF(λ)
    # where R = reflectance, S = illuminant SPD, CMF = color matching function
    X = np.sum(reflectance * D65_ILLUMINANT * CMF_X)
    Y = np.sum(reflectance * D65_ILLUMINANT * CMF_Y)
    Z = np.sum(reflectance * D65_ILLUMINANT * CMF_Z)

    # Normalize by sum of illuminant * CMF_Y (standard practice)
    k = 100.0 / np.sum(D65_ILLUMINANT * CMF_Y)
    X *= k
    Y *= k
    Z *= k

    return X, Y, Z


def xyz_to_lab(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    """
    Convert XYZ tristimulus values to CIELAB coordinates

    Args:
        X, Y, Z: Tristimulus values

    Returns:
        Tuple of (L*, a*, b*) CIELAB coordinates
    """
    # Normalize by white point
    x = X / X_N
    y = Y / Y_N
    z = Z / Z_N

    # Apply f(t) function
    # f(t) = t^(1/3) if t > (6/29)^3, else (1/3)(29/6)^2 * t + 4/29
    epsilon = (6/29) ** 3  # 0.008856
    kappa = (29/6) ** 2 / 3  # 7.787

    def f(t):
        if t > epsilon:
            return np.cbrt(t)
        else:
            return kappa * t + 4/29

    fx = f(x)
    fy = f(y)
    fz = f(z)

    # Calculate L*, a*, b*
    L_star = 116 * fy - 16
    a_star = 500 * (fx - fy)
    b_star = 200 * (fy - fz)

    return L_star, a_star, b_star


def lab_to_lch(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """
    Convert CIELAB to LCh (Lightness, Chroma, Hue)

    Args:
        L, a, b: CIELAB coordinates

    Returns:
        Tuple of (L*, c*, h°) where h is in degrees [0, 360)
    """
    # Chroma (color intensity)
    c = np.sqrt(a**2 + b**2)

    # Hue angle in degrees
    h = np.arctan2(b, a) * 180 / np.pi
    if h < 0:
        h += 360

    return L, c, h


def reflectance_to_cielab(reflectance: np.ndarray, illuminant: str = 'D65') -> Dict[str, float]:
    """
    Complete conversion from reflectance spectrum to CIELAB color coordinates

    Args:
        reflectance: Array of 31 reflectance values (400-700nm, 10nm intervals)
        illuminant: Illuminant type (default: D65)

    Returns:
        Dictionary with keys: L, a, b, c, h
    """
    # Step 1: Reflectance → XYZ
    X, Y, Z = reflectance_to_xyz(reflectance, illuminant)

    # Step 2: XYZ → LAB
    L, a, b = xyz_to_lab(X, Y, Z)

    # Step 3: LAB → LCh (for c and h)
    L, c, h = lab_to_lch(L, a, b)

    return {
        'L': float(L),
        'a': float(a),
        'b': float(b),
        'c': float(c),
        'h': float(h)
    }


def test_cielab_calculator():
    """Test CIELAB calculator with known reference values"""
    print("="*80)
    print("TESTING DETERMINISTIC CIELAB CALCULATOR")
    print("="*80)

    # Test 1: Perfect white (reflectance = 1.0 everywhere)
    print("\nTest 1: Perfect White Diffuser")
    white_reflectance = np.ones(31)
    white_lab = reflectance_to_cielab(white_reflectance)
    print(f"  L*: {white_lab['L']:.2f} (expected: ~100)")
    print(f"  a*: {white_lab['a']:.2f} (expected: ~0)")
    print(f"  b*: {white_lab['b']:.2f} (expected: ~0)")
    print(f"  c*: {white_lab['c']:.2f} (expected: ~0)")

    # Test 2: Perfect black (reflectance = 0.0 everywhere)
    print("\nTest 2: Perfect Black")
    black_reflectance = np.zeros(31)
    black_lab = reflectance_to_cielab(black_reflectance)
    print(f"  L*: {black_lab['L']:.2f} (expected: ~0)")
    print(f"  a*: {black_lab['a']:.2f} (expected: ~0)")
    print(f"  b*: {black_lab['b']:.2f} (expected: ~0)")

    # Test 3: Yellow pigment (high reflectance at long wavelengths)
    print("\nTest 3: Yellow Pigment")
    yellow_reflectance = np.array([0.1] * 15 + [0.9] * 16)  # Low blue, high green/red
    yellow_lab = reflectance_to_cielab(yellow_reflectance)
    print(f"  L*: {yellow_lab['L']:.2f} (expected: high)")
    print(f"  a*: {yellow_lab['a']:.2f} (expected: slight red)")
    print(f"  b*: {yellow_lab['b']:.2f} (expected: high yellow +)")
    print(f"  c*: {yellow_lab['c']:.2f}")
    print(f"  h°: {yellow_lab['h']:.2f}")

    # Test 4: Blue pigment (high reflectance at short wavelengths)
    print("\nTest 4: Blue Pigment")
    blue_reflectance = np.array([0.9] * 10 + [0.1] * 21)  # High blue, low green/red
    blue_lab = reflectance_to_cielab(blue_reflectance)
    print(f"  L*: {blue_lab['L']:.2f}")
    print(f"  a*: {blue_lab['a']:.2f} (expected: negative green)")
    print(f"  b*: {blue_lab['b']:.2f} (expected: negative blue)")
    print(f"  c*: {blue_lab['c']:.2f}")
    print(f"  h°: {blue_lab['h']:.2f} (expected: ~240°)")

    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_cielab_calculator()
