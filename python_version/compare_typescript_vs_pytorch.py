"""
Compare TypeScript Neural Network vs PyTorch Neural Network
This script runs the Python/PyTorch model and provides instructions for comparing with TypeScript
"""
import numpy as np
import json
from types_constants import SampleData, WAVELENGTHS, INITIAL_SAMPLES
from services.km_service import train_model, predict_reflectance

print("=" * 80)
print("NEURAL NETWORK COMPARISON: TypeScript vs PyTorch")
print("=" * 80)
print()

# Create sample data
samples = [
    SampleData(
        id=s['id'], name=s['name'], substrate=s['substrate'],
        thickness=s['thickness'], spectrum=s['spectrum'],
        concentrations=s['concentrations']
    ) for s in INITIAL_SAMPLES
]

print(f"Training on {len(samples)} samples...")
print()

# Train PyTorch model
print("-" * 80)
print("PYTORCH NEURAL NETWORK (Python)")
print("-" * 80)
pytorch_model = train_model(samples, 'neural-net')
print()

# Test formulations for comparison
test_formulations = [
    {
        'name': 'Test 1: Yellow Dominant',
        'concentrations': {'BiVaO4': 10.0, 'LY': 5.0},
        'thickness': 4.0
    },
    {
        'name': 'Test 2: Green Mix',
        'concentrations': {'PG': 8.0, 'LY': 3.0},
        'thickness': 4.0
    },
    {
        'name': 'Test 3: Blue Tint',
        'concentrations': {'PB': 2.0, 'TiO2': 1.0},
        'thickness': 4.0
    },
    {
        'name': 'Test 4: Fluorescent High',
        'concentrations': {'GXT': 25.0, 'BiVaO4': 5.0},
        'thickness': 4.0
    }
]

# Make predictions with PyTorch
print("=" * 80)
print("PYTORCH PREDICTIONS")
print("=" * 80)
print()

pytorch_results = {}

for test in test_formulations:
    name = test['name']
    conc = test['concentrations']
    thickness = test['thickness']

    prediction = predict_reflectance(conc, pytorch_model, thickness)
    pytorch_results[name] = prediction.tolist()

    print(f"{name}")
    print(f"  Formulation: {', '.join([f'{k}:{v}%' for k,v in conc.items()])}")
    print(f"  Thickness: {thickness}μm")
    print()
    print("  Wavelength (nm) | Reflectance")
    print("  " + "-" * 35)

    # Show every 5th wavelength
    for i in range(0, len(WAVELENGTHS), 5):
        wl = WAVELENGTHS[i]
        refl = prediction[i]
        print(f"  {wl:4d}            | {refl:11.6f}")

    print()
    print(f"  Peak reflectance: {np.max(prediction):.6f} @ {WAVELENGTHS[np.argmax(prediction)]}nm")
    print(f"  Min reflectance:  {np.min(prediction):.6f} @ {WAVELENGTHS[np.argmin(prediction)]}nm")
    print(f"  Fluorescence:     {'Yes (R > 1.0)' if np.max(prediction) > 1.0 else 'No'}")
    print()

# Save PyTorch results to JSON for comparison
output_file = 'pytorch_predictions.json'
with open(output_file, 'w') as f:
    json.dump({
        'test_formulations': test_formulations,
        'pytorch_predictions': pytorch_results,
        'wavelengths': WAVELENGTHS.tolist()
    }, f, indent=2)

print(f"✓ PyTorch predictions saved to: {output_file}")
print()

# Generate instructions for TypeScript comparison
print("=" * 80)
print("COMPARISON INSTRUCTIONS")
print("=" * 80)
print()
print("To compare with TypeScript neural network:")
print()
print("1. Open a new terminal and navigate to the main project directory:")
print("   cd ..")
print()
print("2. Start the TypeScript application:")
print("   npm run dev")
print()
print("3. Open browser at http://localhost:5173")
print()
print("4. For each test formulation, set the sliders to match:")
print()

for test in test_formulations:
    print(f"   {test['name']}:")
    for reagent, percent in test['concentrations'].items():
        print(f"     • {reagent}: {percent}%")
    print()

print("5. Compare the spectral curves shown in the web UI with the")
print("   PyTorch predictions above.")
print()
print("6. Note differences in:")
print("   - Peak positions")
print("   - Fluorescence detection (R > 1.0)")
print("   - Overall curve shapes")
print()

# Create a comparison helper script
print("=" * 80)
print("AUTOMATED COMPARISON")
print("=" * 80)
print()

comparison_script = """
// TypeScript Neural Network Predictions
// Copy this into the browser console while running npm run dev

const testFormulations = """ + json.dumps(test_formulations, indent=2) + """;

console.log("TypeScript Neural Network Predictions:");
console.log("======================================\\n");

// You'll need to manually trigger predictions in the UI
// or extract the prediction logic to run here

testFormulations.forEach(test => {
  console.log(`${test.name}:`);
  console.log(`  Formulation: ${JSON.stringify(test.concentrations)}`);
  console.log(`  Set these values in the UI sliders and observe the purple curve`);
  console.log("");
});

console.log("Compare the purple line (Neural Network) in the UI with PyTorch predictions");
"""

with open('typescript_comparison.js', 'w') as f:
    f.write(comparison_script)

print("Created helper script: typescript_comparison.js")
print("  (Can be pasted into browser console)")
print()

# Create side-by-side comparison table generator
print("=" * 80)
print("VISUAL COMPARISON")
print("=" * 80)
print()

print("PyTorch Model Characteristics:")
print("  ✓ Framework: PyTorch with automatic differentiation")
print("  ✓ Training: 2000 epochs, SGD optimizer")
print("  ✓ Architecture: 8 → 128 → 31 neurons")
print("  ✓ Device: CPU (GPU capable)")
print("  ✓ Regularization: L2 weight decay (λ=0.005)")
print()

print("TypeScript Model Characteristics:")
print("  ✓ Framework: Custom manual backpropagation")
print("  ✓ Training: 2000 epochs, manual gradient descent")
print("  ✓ Architecture: 8 → 128 → 31 neurons")
print("  ✓ Device: Browser JavaScript engine")
print("  ✓ Regularization: L2 weight decay (λ=0.005)")
print()

print("Expected Differences:")
print("  • PyTorch uses more sophisticated optimization")
print("  • Different random weight initialization")
print("  • Slightly different numerical precision")
print("  • May converge to different local minima")
print("  • Overall predictions should be similar but not identical")
print()

# Provide sample comparison
print("=" * 80)
print("EXAMPLE: Training Sample Comparison")
print("=" * 80)
print()

test_sample = samples[0]
pytorch_pred = predict_reflectance(test_sample.concentrations, pytorch_model, test_sample.thickness)

print(f"Sample: {test_sample.name}")
print(f"Formulation: {test_sample.concentrations}")
print()
print("Wavelength | Reference | PyTorch Prediction | Error")
print("-" * 60)

for i in range(0, len(WAVELENGTHS), 5):
    wl = WAVELENGTHS[i]
    ref = test_sample.spectrum[i]
    pred = pytorch_pred[i]
    err = abs(pred - ref)
    print(f"{wl:4d} nm   | {ref:9.6f} | {pred:18.6f} | {err:9.6f}")

print()
print("To see TypeScript predictions for this sample:")
print(f"  1. Run: npm run dev")
print(f"  2. Select '{test_sample.name}' from the dropdown")
print(f"  3. Compare purple line (NN) with reference (blue line)")
print()

print("=" * 80)
print("COMPARISON READY")
print("=" * 80)
print()
print("PyTorch model trained and predictions generated.")
print("Follow the instructions above to compare with TypeScript model.")
print()
print("Files created:")
print(f"  • {output_file} - PyTorch predictions in JSON format")
print(f"  • typescript_comparison.js - Browser console helper")
print()
