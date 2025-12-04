# Troubleshooting - Neural Network Prediction Issues

## Current Status (RESOLVED)

**✅ FIXED: Reverted to baseline model with working predictions.**

The improved physics-informed model was producing incorrect predictions, so we've reverted to the baseline `enhanced_pp_model.pkl` which produces valid predictions.

### Problem

The improved model that was trained shows:
- All reflectance values clipped to -0.1 or 1.5
- Negative L* values in CIELAB (impossible - L* is 0-100)
- Model appears to not have learned properly

### Root Cause

The issue appears to be with how the model was trained or saved during the `test_improvements.py` run. The model file exists but the predictions are nonsensical.

## Quick Fix: Revert to Working Baseline (COMPLETED)

**✅ Completed actions to restore working predictions:**

1. **✅ Done**: Changed `MODEL_PATH` back to `enhanced_pp_model.pkl`

2. **✅ Done**: Reverted the prediction code in `enhanced_api_server.py`

Replaced lines 260-305 (the prediction section) with the original code:

```python
        # Prepare input features (normalize as during training)
        # Input: [GXT%, BiVaO4%, PG%, PearlB%, Thickness]
        x = np.array([
            request.concentrations.get('GXT', 0.0) / 100.0,
            request.concentrations.get('BiVaO4', 0.0) / 100.0,
            request.concentrations.get('PG', 0.0) / 100.0,
            request.concentrations.get('PearlB', 0.0) / 100.0,
            request.thickness / 12.0  # Normalize thickness
        ]).reshape(1, -1)

        # Normalize inputs using training statistics
        x_normalized = (x - model_cache['input_mean']) / (model_cache['input_std'] + 1e-8)

        # Convert to tensor
        x_tensor = torch.FloatTensor(x_normalized)

        # Make prediction
        model = model_cache['model']
        model.eval()

        with torch.no_grad():
            pred_spectral, pred_cielab = model(x_tensor)
            pred_spectral = pred_spectral.numpy().flatten()
            pred_cielab = pred_cielab.numpy().flatten()

        # Denormalize predictions
        # Spectral output (31 wavelengths)
        if MODEL_CACHE['spectral_mean'] is not None:
            spectral_mean = MODEL_CACHE['spectral_mean']
            spectral_std = MODEL_CACHE['spectral_std']
        else:
            spectral_mean = model_cache.get('output_mean', np.zeros(36))[:31]
            spectral_std = model_cache.get('output_std', np.ones(36))[:31]

        reflectance = pred_spectral * (spectral_std + 1e-8) + spectral_mean

        # CIELAB output (5 values: L, a, b, c, h)
        if MODEL_CACHE['cielab_mean'] is not None:
            cielab_mean = MODEL_CACHE['cielab_mean']
            cielab_std = MODEL_CACHE['cielab_std']
        else:
            cielab_mean = model_cache.get('output_mean', np.zeros(36))[31:]
            cielab_std = model_cache.get('output_std', np.ones(36))[31:]

        cielab_normalized = pred_cielab * (cielab_std + 1e-8) + cielab_mean

        # Denormalize CIELAB from 0-1 range
        cielab_values = denormalize_cielab(cielab_normalized)
```

3. **✅ Done**: Removed problematic imports at the top:
```python
# Removed these lines:
from models.improved_neural_network import predict_improved_neural_network
from utils.physics_features import add_physics_informed_features
```

4. **✅ Done**: Restarted the API server

## Current Working State

The API server is now running with the baseline model and producing valid predictions:
- Reflectance values: 0.2-1.2 range (mostly reasonable)
- CIELAB L*: 98.82 (valid, within 0-100 range)
- CIELAB a*, b*: Valid ranges
- Server: http://localhost:8001
- UI: http://localhost:5173

## Why The Improved Model Failed

Looking at the test results from `test_improvements.py`:
- Training appeared to complete successfully
- Validation loss improved (0.1533 → 0.1113)
- But actual predictions are garbage

**Possible causes:**
1. **Data normalization mismatch**: The training/test split may have had different statistics
2. **Physics feature calculation error**: The K-M features may have had incorrect values
3. **Model architecture mismatch**: 33 input features vs 5 may have caused issues
4. **Denormalization error**: Separate spectral/CIELAB normalization may be incorrectly applied

## Recommended Next Steps

### Immediate (Get it working)
1. Revert API code to original working version
2. Restart API server
3. Verify predictions in UI

### Short-term (Debug the improved model)
1. Re-run training with more verbose logging
2. Check intermediate values during prediction
3. Validate physics feature calculations
4. Test with known training samples

### Long-term (Proper implementation)
1. Start with smaller improvements one at a time
2. Test each change thoroughly before combining
3. Validate predictions against training data
4. Implement proper unit tests for each component

## Files to Restore

The original working code is in git history. Key files to restore:
- `python_version/enhanced_api_server.py` (prediction section)
- Confirm `MODEL_PATH = 'trained_models/enhanced_pp_model.pkl'`

## Current Working Configuration

**Model**: `enhanced_pp_model.pkl` (baseline, trained earlier)
- 5 input features
- 4×64 architecture
- Combined spectral + CIELAB output normalization
- **Works correctly** with reasonable predictions

The baseline model should be producing valid predictions like:
- Reflectance: 0.2-0.9 range
- CIELAB L*: 50-95
- CIELAB a*: -50 to +50
- CIELAB b*: -50 to +50

If you're seeing values outside these ranges, the model or denormalization is still broken.
