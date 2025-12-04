"""
Test fluorescence NN predictor with smooth constraints via API
"""
import requests
import json

API_URL = "http://localhost:8001/api/predict"

def test_prediction(name, concentrations, thickness=8.0):
    """Test a single prediction"""
    payload = {
        "concentrations": concentrations,
        "thickness": thickness
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        data = response.json()
        fluor = data['fluorescence']['fluorescence_cts']
        mult = data['fluorescence']['gxt_multiplier']
        print(f"{name:50s} | GXT: {concentrations.get('GXT', 0):5.1f}% | Fluorescence: {fluor:7.0f} ct/s | Multiplier: {mult:.3f}")
        return fluor, mult
    else:
        print(f"ERROR: {name} - {response.status_code}: {response.text}")
        return None, None


print("="*120)
print("FLUORESCENCE NN WITH SMOOTH CONSTRAINTS - API TESTING")
print("="*120)
print()

# Test 1: Zero GXT constraint (should be exactly 0 ct/s)
print("Test 1: Zero GXT Constraint (should be 0 ct/s)")
print("-"*120)
test_prediction("No GXT (control)", {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("Only BiVaO4", {"GXT": 0.0, "BiVaO4": 10.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("Only PG", {"GXT": 0.0, "BiVaO4": 0.0, "PG": 5.0, "PearlB": 0.0}, 8.0)
test_prediction("Only PearlB", {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 10.0}, 8.0)
print()

# Test 2: Low GXT (should show smooth transition)
print("Test 2: Low GXT Samples (smooth S-curve transition)")
print("-"*120)
test_prediction("0.5% GXT pure @ 8μm", {"GXT": 0.5, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("1.0% GXT pure @ 8μm", {"GXT": 1.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("2.0% GXT pure @ 8μm", {"GXT": 2.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("3.0% GXT pure @ 8μm", {"GXT": 3.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
print()

# Test 3: Pure GXT samples (baseline fluorescence)
print("Test 3: Pure GXT Samples (baseline fluorescence)")
print("-"*120)
test_prediction("5% GXT pure @ 8μm", {"GXT": 5.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT pure @ 8μm", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("15% GXT pure @ 8μm", {"GXT": 15.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("20% GXT pure @ 8μm", {"GXT": 20.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("25% GXT pure @ 8μm", {"GXT": 25.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
print()

# Test 4: GXT + BiVaO4 mixtures (should show reduction)
print("Test 4: GXT + BiVaO4 Mixtures (BiVaO4 should reduce fluorescence)")
print("-"*120)
test_prediction("10% GXT + 0% BiVaO4", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 5% BiVaO4", {"GXT": 10.0, "BiVaO4": 5.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 10% BiVaO4", {"GXT": 10.0, "BiVaO4": 10.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 15% BiVaO4", {"GXT": 10.0, "BiVaO4": 15.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
print()

# Test 5: Thickness effect
print("Test 5: Thickness Effect (12μm vs 8μm)")
print("-"*120)
test_prediction("10% GXT @ 8μm", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT @ 12μm", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 12.0)
test_prediction("20% GXT @ 8μm", {"GXT": 20.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("20% GXT @ 12μm", {"GXT": 20.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 12.0)
print()

print("="*120)
print("EXPECTED BEHAVIOR:")
print("1. 0% GXT → 0 ct/s (exact, thanks to smooth constraint)")
print("2. Low GXT (0.5-3%): Smooth S-curve transition with visible multiplier effect")
print("3. Pure GXT (≥5%): Fluorescence increases with concentration, multiplier ~1.0")
print("4. Adding BiVaO4, PG, or PearlB should affect fluorescence via NN predictions")
print("5. Thicker samples (12μm) should show higher fluorescence than 8μm")
print("="*120)
