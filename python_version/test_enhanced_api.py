"""
Test script for enhanced API with CIELAB predictions
"""
import requests
import json

API_URL = "http://localhost:8001"

def test_health():
    """Test health check endpoint"""
    print("=" * 80)
    print("TESTING: Health Check")
    print("=" * 80)
    response = requests.get(f"{API_URL}/")
    print(json.dumps(response.json(), indent=2))
    print()

def test_status():
    """Test status endpoint"""
    print("=" * 80)
    print("TESTING: Model Status")
    print("=" * 80)
    response = requests.get(f"{API_URL}/api/status")
    print(json.dumps(response.json(), indent=2))
    print()

def test_reagents():
    """Test reagents endpoint"""
    print("=" * 80)
    print("TESTING: Available Reagents")
    print("=" * 80)
    response = requests.get(f"{API_URL}/api/reagents")
    print(json.dumps(response.json(), indent=2))
    print()

def test_prediction(name, concentrations, thickness):
    """Test prediction endpoint"""
    print("=" * 80)
    print(f"TESTING: Prediction - {name}")
    print("=" * 80)
    print(f"Formulation:")
    print(f"  GXT: {concentrations.get('GXT', 0)}%")
    print(f"  BiVaO4: {concentrations.get('BiVaO4', 0)}%")
    print(f"  PG: {concentrations.get('PG', 0)}%")
    print(f"  PearlB: {concentrations.get('PearlB', 0)}%")
    print(f"  Thickness: {thickness}μm")
    print()

    payload = {
        "concentrations": concentrations,
        "thickness": thickness
    }

    response = requests.post(f"{API_URL}/api/predict", json=payload)

    if response.status_code == 200:
        result = response.json()

        print("CIELAB Predictions:")
        print(f"  L (Lightness):  {result['cielab']['L']:.2f}")
        print(f"  a (Green-Red):  {result['cielab']['a']:.2f}")
        print(f"  b (Blue-Yellow): {result['cielab']['b']:.2f}")
        print(f"  c (Chroma):     {result['cielab']['c']:.2f}")
        print(f"  h (Hue Angle):  {result['cielab']['h']:.2f}°")
        print()

        # Show first 5 and last 5 wavelengths
        print("Spectral Reflectance (first 5 wavelengths):")
        for i in range(5):
            wl = result['wavelengths'][i]
            refl = result['reflectance'][i]
            print(f"  {wl}nm: {refl:.4f}")
        print("  ...")
        print("Spectral Reflectance (last 5 wavelengths):")
        for i in range(-5, 0):
            wl = result['wavelengths'][i]
            refl = result['reflectance'][i]
            print(f"  {wl}nm: {refl:.4f}")
        print()

        # Check for fluorescence (R > 1)
        max_refl = max(result['reflectance'])
        if max_refl > 1.0:
            print(f"⚠️  Fluorescence detected! Max reflectance: {max_refl:.4f}")
        else:
            print(f"✓ No fluorescence (Max reflectance: {max_refl:.4f})")
        print()

    else:
        print(f"ERROR: {response.status_code}")
        print(response.text)
        print()

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ENHANCED API TEST SUITE - PP SUBSTRATE WITH CIELAB")
    print("=" * 80)
    print()

    # Test basic endpoints
    test_health()
    test_status()
    test_reagents()

    # Test predictions for known formulations
    test_prediction(
        "OPTI 19 (8μm)",
        {"GXT": 15.0, "BiVaO4": 8.0, "PG": 1.0, "PearlB": 3.0},
        8.0
    )

    test_prediction(
        "OPTI 19 (12μm)",
        {"GXT": 15.0, "BiVaO4": 8.0, "PG": 1.0, "PearlB": 3.0},
        12.0
    )

    test_prediction(
        "GXT25 (8μm) - High GXT",
        {"GXT": 25.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0},
        8.0
    )

    test_prediction(
        "PBLUE8 (12μm) - Pearl Blue Only",
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 8.0},
        12.0
    )

    test_prediction(
        "T22 (8μm) - Complex Formulation",
        {"GXT": 20.0, "BiVaO4": 5.0, "PG": 1.5, "PearlB": 0.0},
        8.0
    )

    print("=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
