# Anette - AI-Powered Color Formulation Assistant

**Anette** is an intelligent color prediction system that helps you create precise color formulations for polypropylene (PP) substrates. Using advanced neural networks trained on 96 real-world samples, Anette predicts spectral reflectance, CIELAB color coordinates, and fluorescence intensity for any combination of pigments.

## What Can Anette Do?

### 1. **Forward Prediction**
Enter pigment concentrations and get instant predictions for:
- **Spectral Reflectance** (400-700nm) - See the complete reflectance curve
- **CIELAB Color** (L*, a*, b*, c*, h°) - Industry-standard color coordinates
- **Fluorescence** (ct/s) - Predicted fluorescence intensity

### 2. **Inverse Prediction**
Paste a target spectrum (from NIX Spectro or other tools) and Anette will find the optimal pigment formulation to match it.

## Quick Start

### Step 1: Start the Backend
```bash
cd python_version
python3 enhanced_api_server_v2.py
```
The API server will start on `http://localhost:8001`

### Step 2: Start the Frontend
```bash
npm install  # First time only
npm run dev
```
The web interface will open on `http://localhost:5173`

### Step 3: Make Predictions!
1. **Adjust pigment concentrations** using sliders or manual input
2. **Select film thickness** (8μm or 12μm)
3. **View predictions instantly** - spectral curve, CIELAB values, and fluorescence

## Available Pigments

Anette works with four pigments on PP substrates:

| Pigment | Description | Typical Range |
|---------|-------------|---------------|
| **GXT-10** | Yellow fluorescent | 0-25% |
| **BiVaO4** | Yellow vanadate | 0-20% |
| **PG** | Pigment Green | 0-2% |
| **PearlB** | Pearl Blue | 0-15% |

## Understanding the Results

### Spectral Reflectance Chart
Shows how your formulation reflects light across the visible spectrum (400-700nm). Higher values mean more light is reflected.

### CIELAB Color Values
- **L*** (Lightness): 0 (black) to 100 (white)
- **a***: Green (-) to Red (+)
- **b***: Blue (-) to Yellow (+)
- **c***: Chroma (color intensity)
- **h°**: Hue angle (color direction, 0-360°)

### Fluorescence
Predicted fluorescence intensity in counts per second (ct/s). Higher values indicate stronger fluorescence under UV light.

## Model Performance

Anette's predictions are based on neural networks trained on 96 laboratory measurements:

- **Spectral Accuracy**: R² = 0.9952 (MAE = 0.015)
- **CIELAB Accuracy**: ΔE*ab < 2.5 (industry-excellent)
- **Fluorescence Accuracy**: R² = 0.9734 (MAE = 328 ct/s)

## Tips for Best Results

1. **Stay within training ranges** - Anette is most accurate for concentrations within the ranges shown
2. **Use 8μm for standard applications** - Most training data uses 8μm thickness
3. **Physics-based constraints** - 0% GXT always produces 0 fluorescence (as expected)
4. **Smooth predictions** - Fluorescence uses physics-based constraints for realistic behavior

## System Requirements

- **Python 3.9+** with PyTorch, FastAPI, NumPy, Pandas
- **Node.js 16+** with React, Vite, Recharts
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

## Technical Architecture

```
┌─────────────────┐
│  React Frontend │  Interactive UI with real-time predictions
│  (Port 5173)    │
└────────┬────────┘
         │
         │ HTTP REST API
         │
┌────────▼────────┐
│  FastAPI Server │  Prediction engine and model serving
│  (Port 8001)    │
└────────┬────────┘
         │
         │ PyTorch Models
         │
┌────────▼────────┐
│ Neural Networks │  • Spectral NN (R²=0.9952)
│                 │  • Fluorescence NN (R²=0.9734)
│                 │  • Deterministic CIELAB Calculator
└─────────────────┘
```

## Project Structure

```
PredictiveApp/
├── components/          # React UI components
├── services/            # API communication layer
├── python_version/      # Backend neural network system
│   ├── trained_models/  # Pre-trained PyTorch models
│   ├── utils/          # Data loading and processing
│   └── enhanced_api_server_v2.py  # Main API server
├── public/             # Training data (CSV files)
└── README.md           # This file
```

## Troubleshooting

### Backend won't start
- Check Python version: `python3 --version` (need 3.9+)
- Install dependencies: `pip3 install torch fastapi uvicorn numpy pandas scikit-learn`

### Frontend won't start
- Check Node.js version: `node --version` (need 16+)
- Clear node_modules: `rm -rf node_modules && npm install`

### Predictions seem incorrect
- Verify API server is running on port 8001
- Check browser console for errors (F12 → Console tab)
- Ensure concentrations are within trained ranges

## About the Neural Networks

### Spectral Reflectance Model
- **Architecture**: 5 inputs → [128] hidden layer → 31 spectral outputs
- **Training**: 96 samples with hyperparameter optimization
- **Performance**: R²=0.9952, MAE=0.015

### CIELAB Calculator
- **Method**: Deterministic CIE 1964 10° observer with D65 illuminant
- **Accuracy**: ΔE*ab < 2.5 on validation set
- **White Point**: Corrected (X=93.253, Y=100, Z=94.247)

### Fluorescence Model
- **Architecture**: 6 inputs → [64, 32] hidden layers → 1 fluorescence output
- **Constraint**: Physics-based smooth S-curve (0% GXT → 0 ct/s)
- **Performance**: R²=0.9734, MAE=328 ct/s

## License & Credits

Developed by QuantumBase for color formulation research and development.

**Version**: 2.0.0
**Last Updated**: December 2025
