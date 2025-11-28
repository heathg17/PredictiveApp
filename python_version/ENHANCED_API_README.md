# Enhanced API Documentation - PP Substrate with CIELAB

## Overview

The Enhanced PyTorch API serves predictions for **PP (Polypropylene) substrate** formulations with **4 reagents** and outputs both **spectral reflectance** and **CIELAB color coordinates**.

---

## API Endpoints

### Base URL
```
http://localhost:8001
```

---

## Endpoints

### 1. Health Check
**GET** `/`

Returns API status and version information.

**Response:**
```json
{
  "status": "ok",
  "message": "OptiMix Enhanced PyTorch API is running",
  "version": "2.0.0",
  "model": "PP Substrate 4-Reagent with CIELAB"
}
```

---

### 2. Model Status
**GET** `/api/status`

Returns detailed model information and capabilities.

**Response:**
```json
{
  "status": "ready",
  "model_type": "Enhanced Multi-Output Neural Network",
  "reagents": ["GXT", "BiVaO4", "PG", "PearlB"],
  "total_samples": 84,
  "architecture": "4×64 (Shared) → [31 Spectral + 5 CIELAB]",
  "parameters": 18724
}
```

---

### 3. Available Reagents
**GET** `/api/reagents`

Returns information about available reagents and outputs.

**Response:**
```json
{
  "reagents": ["GXT", "BiVaO4", "PG", "PearlB"],
  "substrate": "PP (Polypropylene)",
  "thicknesses": [8.0, 12.0],
  "outputs": [
    "spectral_reflectance",
    "cielab_L",
    "cielab_a",
    "cielab_b",
    "cielab_c",
    "cielab_h"
  ]
}
```

---

### 4. Predict Spectral + CIELAB
**POST** `/api/predict`

Predicts spectral reflectance and CIELAB color coordinates for a given formulation.

**Request Body:**
```json
{
  "concentrations": {
    "GXT": 15.0,
    "BiVaO4": 8.0,
    "PG": 1.0,
    "PearlB": 3.0
  },
  "thickness": 8.0
}
```

**Parameters:**
- `concentrations` (object, required): Reagent concentrations as percentages
  - `GXT` (float): GXT-10 concentration (0-25%)
  - `BiVaO4` (float): Bismuth vanadate concentration (0-20%)
  - `PG` (float): Phthalo green concentration (0-2%)
  - `PearlB` (float): Pearl blue concentration (0-8%)
- `thickness` (float, required): Coating thickness in micrometers
  - Must be either `8.0` or `12.0` μm

**Response:**
```json
{
  "wavelengths": [400, 410, 420, ..., 700],
  "reflectance": [0.2367, 0.3204, 0.3236, ..., 0.8308],
  "cielab": {
    "L": 97.64,
    "a": -23.06,
    "b": 59.24,
    "c": 64.56,
    "h": 113.31
  },
  "thickness": 8.0,
  "model_version": "2.0.0-pp-substrate"
}
```

**Response Fields:**
- `wavelengths`: Array of 31 wavelengths (400-700nm, 10nm intervals)
- `reflectance`: Array of 31 reflectance values (can exceed 1.0 for fluorescence)
- `cielab`: CIELAB color coordinates
  - `L`: Lightness (0-100, typically 94-103)
  - `a`: Green (-) to Red (+) axis (typically -30 to 0)
  - `b`: Blue (-) to Yellow (+) axis (typically 0 to 100)
  - `c`: Chroma (color intensity)
  - `h`: Hue angle in degrees (0-360)
- `thickness`: Confirmed thickness used for prediction
- `model_version`: Model version identifier

---

### 5. Batch Predict
**POST** `/api/batch_predict`

Predicts multiple formulations in a single request.

**Request Body:**
```json
[
  {
    "concentrations": {"GXT": 15.0, "BiVaO4": 8.0, "PG": 1.0, "PearlB": 3.0},
    "thickness": 8.0
  },
  {
    "concentrations": {"GXT": 20.0, "BiVaO4": 5.0, "PG": 1.5, "PearlB": 0.0},
    "thickness": 12.0
  }
]
```

**Response:**
```json
[
  {
    "wavelengths": [...],
    "reflectance": [...],
    "cielab": {...},
    "thickness": 8.0,
    "model_version": "2.0.0-pp-substrate"
  },
  {
    "wavelengths": [...],
    "reflectance": [...],
    "cielab": {...},
    "thickness": 12.0,
    "model_version": "2.0.0-pp-substrate"
  }
]
```

---

## Example Usage

### cURL Examples

#### 1. Health Check
```bash
curl http://localhost:8001/
```

#### 2. Get Status
```bash
curl http://localhost:8001/api/status
```

#### 3. Make Prediction
```bash
curl -X POST "http://localhost:8001/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "concentrations": {
      "GXT": 15.0,
      "BiVaO4": 8.0,
      "PG": 1.0,
      "PearlB": 3.0
    },
    "thickness": 8.0
  }'
```

### Python Example

```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:8001/api/predict",
    json={
        "concentrations": {
            "GXT": 15.0,
            "BiVaO4": 8.0,
            "PG": 1.0,
            "PearlB": 3.0
        },
        "thickness": 8.0
    }
)

result = response.json()
print(f"L: {result['cielab']['L']:.2f}")
print(f"a: {result['cielab']['a']:.2f}")
print(f"b: {result['cielab']['b']:.2f}")
```

### JavaScript/TypeScript Example

```typescript
const response = await fetch('http://localhost:8001/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    concentrations: {
      GXT: 15.0,
      BiVaO4: 8.0,
      PG: 1.0,
      PearlB: 3.0
    },
    thickness: 8.0
  })
});

const result = await response.json();
console.log('CIELAB:', result.cielab);
console.log('Reflectance:', result.reflectance);
```

---

## Key Features

### 1. Multi-Output Predictions
The model predicts:
- **31 spectral reflectance values** (400-700nm, 10nm intervals)
- **5 CIELAB color coordinates** (L, a, b, c, h)

### 2. Dual Thickness Support
The model handles both:
- **8μm** - Standard coating thickness
- **12μm** - Heavy coating thickness

### 3. Fluorescence Handling
The model correctly predicts fluorescence effects where reflectance can exceed 1.0 (particularly for GXT-containing formulations around 540nm).

### 4. Four Reagent System
Optimized for PP substrate with:
- **GXT** (GXT-10): Fluorescent pigment
- **BiVaO4**: Bismuth vanadate yellow pigment
- **PG**: Phthalo green pigment
- **PearlB**: Pearl blue pigment

---

## Model Performance

### Spectral Prediction (Excellent ✓)
- **MAE**: 0.059 (5.9% average error)
- **RMSE**: 0.104
- **R²**: 0.864 (explains 86.4% of variance)

### CIELAB Prediction (Mixed)
- **L (Lightness)**: MAE = 0.97 (needs improvement)
- **a (Green-Red)**: MAE = 0.07 (excellent ✓)
- **b (Blue-Yellow)**: MAE = 0.17 (good ✓)
- **c (Chroma)**: MAE = 0.31 (moderate)
- **h (Hue Angle)**: MAE = 0.31 (moderate)

---

## Error Handling

### 400 Bad Request
```json
{
  "detail": "Invalid reagents: {unknown}. Expected: {'GXT', 'BiVaO4', 'PG', 'PearlB'}"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Prediction error: <error message>"
}
```

---

## Running the Server

### Start Server
```bash
cd python_version
python3 enhanced_api_server.py
```

The server will start on `http://0.0.0.0:8001`

### Run Tests
```bash
cd python_version
python3 test_enhanced_api.py
```

---

## Differences from Old API

| Feature | Old API (Port 8000) | Enhanced API (Port 8001) |
|---------|---------------------|--------------------------|
| **Substrate** | Paper | PP (Polypropylene) |
| **Reagents** | 11 (BiVaO4, GXT, LY, PG, etc.) | 4 (GXT, BiVaO4, PG, PearlB) |
| **Thickness** | Single (4μm) | Dual (8μm, 12μm) |
| **Outputs** | 31 wavelengths only | 31 wavelengths + 5 CIELAB |
| **Architecture** | 4×32 | 4×64 with dual heads |
| **Parameters** | 4,607 | 18,724 |
| **Training Data** | 65 samples (45 real + 20 synthetic) | 84 samples (all real) |

---

## Known Formulations (Examples)

### OPTI Series
- **OPTI 19**: GXT=15%, BiVaO4=8%, PG=1%, PearlB=3%
- **OPTI 17**: GXT=15%, BiVaO4=8%, PG=1%, PearlB=0%

### T Series
- **T22**: GXT=20%, BiVaO4=5%, PG=1.5%, PearlB=0%
- **T27**: GXT=20%, BiVaO4=0%, PG=0%, PearlB=3%

### Single Reagent
- **GXT25**: GXT=25%, others=0%
- **PBLUE8**: PearlB=8%, others=0%
- **BiVaO20**: BiVaO4=20%, others=0%
- **PG2**: PG=2%, others=0%

---

## Technical Details

### Model Architecture
```
Input (5 features: GXT, BiVaO4, PG, PearlB, Thickness)
    ↓
Shared Layers (4×64 with BatchNorm + Dropout)
    ├─ Layer 1: Linear(5 → 64) → BatchNorm → ReLU → Dropout(0.2)
    ├─ Layer 2: Linear(64 → 64) → BatchNorm → ReLU → Dropout(0.2)
    ├─ Layer 3: Linear(64 → 64) → BatchNorm → ReLU → Dropout(0.2)
    └─ Layer 4: Linear(64 → 64) → BatchNorm → ReLU
         ↓
    ┌────────────┴────────────┐
    ↓                         ↓
Spectral Head            CIELAB Head
(64 → 32 → 31)           (64 → 32 → 5)
    ↓                         ↓
31 wavelengths           L, a, b, c, h
```

### Training Details
- **Dataset**: 84 samples (41 @ 8μm, 43 @ 12μm)
- **Split**: 60/20/20 (Train/Val/Test)
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience = 300 epochs
- **Training Time**: ~3 minutes
- **Best Epoch**: 734

---

## Support

For issues or questions:
- Check [PP_SUBSTRATE_RESULTS.md](PP_SUBSTRATE_RESULTS.md) for detailed training results
- Run `python3 test_enhanced_api.py` to verify API functionality
- Review training visualizations in `results/` directory

---

## Version History

- **v2.0.0** (Current): PP substrate with CIELAB predictions
- **v1.0.0**: Paper substrate with spectral predictions only
