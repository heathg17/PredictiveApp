# OptiMix - Spectral Formulation Engine (Python/PyTorch)

A Python port of the OptiMix spectral reflectance prediction application, using PyTorch for neural network models and NumPy for scientific computing.

## Overview

This application predicts spectral reflectance curves (400-700nm) for pigment formulations using two complementary approaches:

1. **Kubelka-Munk (K-M) Single Layer Model** - Physics-based approach using light scattering theory
2. **Neural Network Model** - Data-driven approach using PyTorch for learning non-linear relationships

## Features

- Train models on spectral measurement data
- Predict reflectance for custom formulations
- Support for fluorescent pigments (R > 1.0)
- Compare physics-based vs. machine learning predictions
- Visualize spectral curves with matplotlib

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
cd PredictiveApp/python_version
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU acceleration (optional):
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Basic Usage

Run with initial sample data:
```bash
python main.py --use-initial
```

### Using Master CSV Files

If you have master concentration and spectral CSV files:
```bash
python main.py --data-dir ../public --conc-file "Master conc.csv" --spec-file "Master spec - master_sample_library.csv"
```

### Command Line Options

- `--data-dir DIR`: Directory containing CSV files (default: `../public`)
- `--conc-file FILE`: Concentration CSV filename (default: `Master conc.csv`)
- `--spec-file FILE`: Spectral CSV filename (default: `Master spec - master_sample_library.csv`)
- `--use-initial`: Use built-in initial sample data
- `--no-plot`: Disable plotting (useful for headless environments)

### Headless Mode (No Display)

For servers or environments without display:
```bash
python main.py --use-initial --no-plot
```

## Project Structure

```
python_version/
├── main.py                    # Main application entry point
├── types_constants.py         # Data types and constants
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── models/
│   └── neural_network.py      # PyTorch neural network implementation
│
├── services/
│   └── km_service.py          # Kubelka-Munk model training & prediction
│
└── utils/
    ├── matrix_ops.py          # NumPy matrix operations
    └── data_loader.py         # CSV data loading utilities
```

## Data Format

### Concentration CSV Format

```csv
Sample,Substrate,Thickness (um),BiVaO4 (%),PG (%),PB (%),...
Sample_001,Paper,4,10.5,5.2,0,...
```

### Spectral CSV Format (No Header)

```csv
Sample_001,,,L*,a*,b*,,,0.452,0.478,0.501,...,0.823,FilePath,
```

Columns 8-38 contain the 31 reflectance values for wavelengths 400-700nm at 10nm intervals.

## How It Works

### 1. Kubelka-Munk Model

The K-M model uses the fundamental equation:

```
K/S = (1 - R)² / (2R)
```

Where:
- K = absorption coefficient
- S = scattering coefficient
- R = reflectance

The model solves a linear system to find K/S coefficients for each pigment at each wavelength.

**Strengths:**
- Physically interpretable
- Works well for opaque, non-fluorescent samples
- Requires minimal training data

**Limitations:**
- Assumes linearity
- Cannot model fluorescence without extensions
- Poor extrapolation

### 2. Neural Network Model

PyTorch feedforward architecture:

```
Input (n pigments + thickness) → Hidden (128 neurons, ReLU) → Output (31 wavelengths)
```

**Training details:**
- Optimizer: SGD with L2 regularization
- Batch size: Adaptive (4-8 samples)
- Epochs: 2000
- Learning rate: 0.005
- Regularization: λ = 0.005

**Strengths:**
- Captures fluorescence and non-linear effects
- Learns complex pigment interactions
- High accuracy with sufficient data

**Limitations:**
- Requires substantial training data
- Black-box (not interpretable)
- May overfit with insufficient data

## Python vs TypeScript Differences

This Python implementation retains all functionality from the original TypeScript version with these enhancements:

1. **PyTorch instead of manual backpropagation**: More efficient and scalable
2. **NumPy for matrix operations**: Faster and more robust than custom implementations
3. **Command-line interface**: No web UI, but easier to integrate into pipelines
4. **Matplotlib visualization**: Native Python plotting
5. **Better error handling**: More robust numerical stability

## API Usage Example

```python
from types_constants import SampleData, WAVELENGTHS
from services.km_service import train_model, predict_reflectance
import numpy as np

# Create sample data
samples = [...]  # List of SampleData objects

# Train models
single_model = train_model(samples, 'single')
neural_model = train_model(samples, 'neural-net')

# Make prediction
concentrations = {'BiVaO4': 10.0, 'PG': 5.0, 'PB': 1.0}
thickness = 4.0

single_pred = predict_reflectance(concentrations, single_model, thickness)
neural_pred = predict_reflectance(concentrations, neural_model, thickness)

# Plot results
import matplotlib.pyplot as plt
plt.plot(WAVELENGTHS, single_pred, label='K-M')
plt.plot(WAVELENGTHS, neural_pred, label='Neural Net')
plt.legend()
plt.show()
```

## GPU Acceleration

The neural network training automatically uses GPU if available:

```python
# Check GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

Training on GPU can be 10-50x faster depending on your hardware.

## Performance

Typical training times on CPU:
- K-M Single Layer: < 1 second
- Neural Network: 10-30 seconds (2000 epochs, ~100 samples)

With GPU:
- Neural Network: 2-5 seconds

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the correct directory:
```bash
cd python_version
python main.py --use-initial
```

### Singular Matrix Errors

This occurs when:
- Too few samples
- Samples are too similar (linearly dependent)
- Missing reagent data

**Solution**: Add more diverse training samples.

### Neural Network Overfitting

If predictions are poor on new formulations:
- Increase L2 regularization (`l2_lambda`)
- Reduce hidden layer size
- Add more training samples

## Contributing

This is a port of the original TypeScript application. The original codebase structure and algorithms have been preserved for consistency.

## References

- Kubelka, P., & Munk, F. (1931). "An article on optics of paint layers"
- PyTorch Documentation: https://pytorch.org/docs/
- NumPy Documentation: https://numpy.org/doc/

## License

Same as the original OptiMix project.
