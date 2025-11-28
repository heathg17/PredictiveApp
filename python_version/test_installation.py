"""
Quick test script to verify the Python port works correctly
"""
import sys
import numpy as np

print("Testing OptiMix Python Port Installation...")
print("=" * 60)

# Test 1: Import core modules
print("\n1. Testing imports...")
try:
    from types_constants import SampleData, WAVELENGTHS, INITIAL_SAMPLES
    print("   ✓ types_constants imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import types_constants: {e}")
    sys.exit(1)

try:
    from utils.matrix_ops import pseudo_inverse, normalize_data
    print("   ✓ utils.matrix_ops imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import utils.matrix_ops: {e}")
    sys.exit(1)

try:
    from models.neural_network import SpectralNN, train_neural_network
    print("   ✓ models.neural_network imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import models.neural_network: {e}")
    sys.exit(1)

try:
    from services.km_service import train_model, predict_reflectance
    print("   ✓ services.km_service imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import services.km_service: {e}")
    sys.exit(1)

# Test 2: Check data structures
print("\n2. Testing data structures...")
print(f"   Wavelengths: {len(WAVELENGTHS)} values from {WAVELENGTHS[0]} to {WAVELENGTHS[-1]} nm")
print(f"   Initial samples: {len(INITIAL_SAMPLES)} samples")

# Test 3: Create sample data
print("\n3. Testing SampleData creation...")
samples = []
for s in INITIAL_SAMPLES:
    sample = SampleData(
        id=s['id'],
        name=s['name'],
        substrate=s['substrate'],
        thickness=s['thickness'],
        spectrum=s['spectrum'],
        concentrations=s['concentrations']
    )
    samples.append(sample)
print(f"   ✓ Created {len(samples)} SampleData objects")

# Test 4: Test matrix operations
print("\n4. Testing matrix operations...")
A = np.random.rand(5, 3)
A_pinv = pseudo_inverse(A)
print(f"   ✓ Pseudo-inverse computed: {A.shape} -> {A_pinv.shape}")

data = np.random.rand(10, 5)
normalized, mean, std = normalize_data(data)
print(f"   ✓ Data normalization successful")

# Test 5: Test neural network
print("\n5. Testing PyTorch neural network...")
try:
    import torch
    model = SpectralNN(input_size=10, hidden_size=32, output_size=31)
    test_input = torch.randn(2, 10)
    output = model(test_input)
    print(f"   ✓ Neural network forward pass: {test_input.shape} -> {output.shape}")
    print(f"   Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    print(f"   ✗ Neural network test failed: {e}")
    sys.exit(1)

# Test 6: Train a small model
print("\n6. Testing model training (K-M single layer)...")
try:
    single_model = train_model(samples, 'single')
    print(f"   ✓ Single-layer K-M model trained successfully")
    print(f"   Model type: {single_model.type}")
    print(f"   Reagents in model: {len(single_model.alpha) if single_model.alpha else 0}")
except Exception as e:
    print(f"   ✗ Model training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test prediction
print("\n7. Testing prediction...")
try:
    test_concentrations = samples[0].concentrations
    prediction = predict_reflectance(test_concentrations, single_model, 4.0)
    print(f"   ✓ Prediction successful")
    print(f"   Input concentrations: {test_concentrations}")
    print(f"   Output spectrum shape: {prediction.shape}")
    print(f"   Reflectance range: [{prediction.min():.3f}, {prediction.max():.3f}]")
except Exception as e:
    print(f"   ✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nYou can now run the main application:")
print("  python main.py --use-initial")
print("\nOr with plotting disabled:")
print("  python main.py --use-initial --no-plot")
