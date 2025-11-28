"""
Test fluorescence predictions to verify complex pigment interactions
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
        print(f"{name:40s} | GXT: {concentrations.get('GXT', 0):5.1f}% | BiVaO4: {concentrations.get('BiVaO4', 0):5.1f}% | PG: {concentrations.get('PG', 0):5.1f}% | PearlB: {concentrations.get('PearlB', 0):5.1f}% | Fluorescence: {fluor:7.0f} ct/s")
        return fluor
    else:
        print(f"ERROR: {name} - {response.status_code}")
        return None


print("=" * 140)
print("FLUORESCENCE PREDICTION TESTS - VERIFYING PIGMENT INTERACTION EFFECTS")
print("=" * 140)
print()

# Test 1: Zero GXT should give 0 ct/s
print("Test 1: Zero GXT (should be 0 ct/s)")
print("-" * 140)
test_prediction("No GXT (control)", {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("Only BiVaO4", {"GXT": 0.0, "BiVaO4": 10.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("Only PG", {"GXT": 0.0, "BiVaO4": 0.0, "PG": 5.0, "PearlB": 0.0}, 8.0)
test_prediction("Only PearlB", {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 10.0}, 8.0)
print()

# Test 2: Pure GXT samples (baseline fluorescence)
print("Test 2: Pure GXT samples (baseline fluorescence)")
print("-" * 140)
fluor_5pct = test_prediction("5% GXT pure @ 8μm", {"GXT": 5.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
fluor_10pct = test_prediction("10% GXT pure @ 8μm", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
fluor_15pct = test_prediction("15% GXT pure @ 8μm", {"GXT": 15.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
fluor_20pct = test_prediction("20% GXT pure @ 8μm", {"GXT": 20.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
fluor_25pct = test_prediction("25% GXT pure @ 8μm", {"GXT": 25.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
print()

# Test 3: GXT + BiVaO4 mixtures (should show reduction due to absorption/scattering)
print("Test 3: GXT + BiVaO4 mixtures (BiVaO4 should reduce fluorescence)")
print("-" * 140)
test_prediction("10% GXT + 0% BiVaO4", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 5% BiVaO4", {"GXT": 10.0, "BiVaO4": 5.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 10% BiVaO4", {"GXT": 10.0, "BiVaO4": 10.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 15% BiVaO4", {"GXT": 10.0, "BiVaO4": 15.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
print()

# Test 4: GXT + PG mixtures (should show reduction)
print("Test 4: GXT + PG mixtures (PG should reduce fluorescence)")
print("-" * 140)
test_prediction("10% GXT + 0% PG", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 2% PG", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 2.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 5% PG", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 5.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 10% PG", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 10.0, "PearlB": 0.0}, 8.0)
print()

# Test 5: GXT + PearlB mixtures (should show reduction)
print("Test 5: GXT + PearlB mixtures (PearlB should reduce fluorescence)")
print("-" * 140)
test_prediction("10% GXT + 0% PearlB", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 5% PearlB", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 5.0}, 8.0)
test_prediction("10% GXT + 10% PearlB", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 10.0}, 8.0)
test_prediction("10% GXT + 15% PearlB", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 15.0}, 8.0)
print()

# Test 6: Complex mixtures (multiple non-fluorescent pigments)
print("Test 6: Complex mixtures (multiple non-fluorescent pigments)")
print("-" * 140)
test_prediction("10% GXT only", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 5% BiVaO4 + 2% PG", {"GXT": 10.0, "BiVaO4": 5.0, "PG": 2.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 10% BiVaO4 + 5% PG", {"GXT": 10.0, "BiVaO4": 10.0, "PG": 5.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT + 5% BiVaO4 + 2% PG + 5% PearlB", {"GXT": 10.0, "BiVaO4": 5.0, "PG": 2.0, "PearlB": 5.0}, 8.0)
print()

# Test 7: Thickness effect
print("Test 7: Thickness effect (12μm vs 8μm)")
print("-" * 140)
test_prediction("10% GXT @ 8μm", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("10% GXT @ 12μm", {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 12.0)
test_prediction("20% GXT @ 8μm", {"GXT": 20.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0)
test_prediction("20% GXT @ 12μm", {"GXT": 20.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 12.0)
print()

print("=" * 140)
print("EXPECTED BEHAVIOR:")
print("1. 0% GXT → 0 ct/s (no fluorescent pigment)")
print("2. Pure GXT: fluorescence increases with concentration")
print("3. Adding BiVaO4, PG, or PearlB should REDUCE fluorescence at same GXT level")
print("4. More non-fluorescent pigments → more reduction (complex interactions)")
print("5. Thicker samples (12μm) should show higher fluorescence than 8μm at same concentration")
print("=" * 140)
