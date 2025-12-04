# Neural Network Comparison Guide
## TypeScript (Browser) vs PyTorch (Python)

This guide helps you compare the two neural network implementations side-by-side.

---

## Quick Start

### Terminal 1: Run PyTorch Model (Python)
```bash
cd python_version
python3 compare_typescript_vs_pytorch.py
```
This generates predictions and saves them to `pytorch_predictions.json`

### Terminal 2: Run TypeScript Model (Browser)
```bash
cd ..
npm run dev
```
Open browser at http://localhost:5173

---

## Test Formulations for Comparison

### Test 1: Yellow Dominant
**Set sliders to:**
- BiVaO4: 10.0%
- LY: 5.0%
- All others: 0%

**PyTorch Results:**
```
Peak reflectance: 0.973 @ 520nm
Fluorescence: No
Reflectance @ 550nm: 0.921
```

**Compare with:** Purple line in browser UI

---

### Test 2: Green Mix
**Set sliders to:**
- PG: 8.0%
- LY: 3.0%
- All others: 0%

**PyTorch Results:**
```
Peak reflectance: 0.920 @ 490nm
Fluorescence: No
Reflectance @ 550nm: 0.812
```

**Compare with:** Purple line in browser UI

---

### Test 3: Blue Tint
**Set sliders to:**
- PB: 2.0%
- TiO2: 1.0%
- All others: 0%

**PyTorch Results:**
```
Peak reflectance: 0.945 @ 520nm
Fluorescence: No
Reflectance @ 550nm: 0.822
```

**Compare with:** Purple line in browser UI

---

### Test 4: Fluorescent High ⭐
**Set sliders to:**
- GXT: 25.0%
- BiVaO4: 5.0%
- All others: 0%

**PyTorch Results:**
```
Peak reflectance: 1.005 @ 530nm
Fluorescence: YES (R > 1.0)
Reflectance @ 550nm: 1.004
```

**Compare with:** Purple line in browser UI
**Look for:** Line going above 1.0 on Y-axis

---

## What to Compare

### 1. **Peak Positions**
- Do both models predict peaks at the same wavelengths?
- Slight differences are expected

### 2. **Fluorescence Detection**
- Test 4 should show R > 1.0
- Both models should detect this
- Check if purple line (TS) goes above 1.0

### 3. **Overall Shape**
- Curves should have similar overall shapes
- May differ in exact values

### 4. **Training Data Fit**
- Select "Eren 1" in browser dropdown
- Purple line = TypeScript NN prediction
- Compare with PyTorch prediction for Eren 1

---

## Expected Differences

Both models have the **same architecture** (8→128→31) but differ in:

| Aspect | TypeScript | PyTorch |
|--------|-----------|---------|
| Training | Manual backprop | Automatic differentiation |
| Optimization | Custom SGD | PyTorch SGD optimizer |
| Initialization | Random | Xavier initialization |
| Precision | JavaScript floats | NumPy float64 |
| Device | Browser | CPU/GPU |

**Result:** Similar but not identical predictions

---

## Side-by-Side Comparison Table

### Test 1: Yellow Dominant @ Key Wavelengths

| λ (nm) | PyTorch | TypeScript | Difference |
|--------|---------|------------|------------|
| 400    | 0.414   | ???        | ???        |
| 500    | 0.926   | ???        | ???        |
| 550    | 0.921   | ???        | ???        |
| 600    | 0.878   | ???        | ???        |
| 700    | 0.879   | ???        | ???        |

**To fill in TypeScript column:**
1. Set sliders to BiVaO4:10%, LY:5%
2. Hover over purple curve at each wavelength
3. Record the reflectance values

---

## Using Browser Console Helper

In the browser console (F12), paste the contents of `typescript_comparison.js`:

```javascript
// This will log the test formulations
// Compare with the purple curve in the UI
```

---

## Visual Comparison Checklist

- [ ] Test 1: Yellow Dominant - curves match shape?
- [ ] Test 2: Green Mix - similar peak positions?
- [ ] Test 3: Blue Tint - comparable values?
- [ ] Test 4: Fluorescent High - both detect R > 1.0?
- [ ] Training sample (Eren 1) - both fit well?

---

## Performance Comparison

| Metric | TypeScript | PyTorch |
|--------|-----------|---------|
| Training Speed | ~30-40s (browser) | ~15s (CPU) |
| GPU Support | No | Yes (CUDA) |
| Batch Processing | Limited | Easy |
| Production Use | Web apps | APIs, pipelines |

---

## Files Generated

After running the comparison script:

```
python_version/
├── pytorch_predictions.json    # All PyTorch predictions
└── typescript_comparison.js    # Browser console helper
```

---

## Troubleshooting

### TypeScript model not showing predictions
- Make sure you moved the sliders
- Check that "Both Models Active" indicator is green
- Reload the page (http://localhost:5173)

### PyTorch predictions different each run
- This is expected due to random initialization
- Use same random seed for reproducibility
- Predictions should be similar in overall shape

### Numbers don't match exactly
- **This is normal!** Different optimizers converge to different solutions
- Focus on:
  - Similar curve shapes ✓
  - Similar peak positions ✓
  - Both detect fluorescence ✓
  - Comparable error rates ✓

---

## Advanced Comparison

### Export TypeScript Predictions

Add this to the browser console:

```javascript
// Get current formulation
const concentrations = {...mixConcentrations};
const thickness = 4.0;

// Get neural network prediction
const nnPred = predictReflectance(concentrations, neuralModel, thickness);

// Log results
console.log("TypeScript NN Prediction:");
console.log(JSON.stringify({
  concentrations,
  thickness,
  prediction: nnPred
}, null, 2));
```

### Compare JSON Outputs

Save both TypeScript and PyTorch predictions as JSON and use a diff tool.

---

## Conclusion

Both neural networks should:
- ✓ Predict similar spectral curves
- ✓ Detect fluorescence (R > 1.0)
- ✓ Fit training data well
- ✓ Make reasonable interpolations

Differences are expected due to:
- Different optimization paths
- Random initialization
- Numerical precision
- Implementation details

**Both are valid implementations** - choose based on your needs:
- **TypeScript:** Web apps, interactive UIs
- **PyTorch:** Production ML, batch processing, GPU acceleration
