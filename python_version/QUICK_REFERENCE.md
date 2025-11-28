# OptiMix Python - Quick Reference

## Installation

```bash
# Quick setup
./setup.sh

# Manual setup
pip install -r requirements.txt
python test_installation.py
```

## Running the Application

### Basic Usage
```bash
# Use built-in sample data
python main.py --use-initial

# Load from CSV files
python main.py --data-dir ../public

# Headless mode (no plots)
python main.py --use-initial --no-plot
```

## Using as a Python Module

### Import and Train
```python
from types_constants import SampleData, WAVELENGTHS
from services.km_service import train_model, predict_reflectance
import numpy as np

# Create sample data
samples = [
    SampleData(
        id='sample_1',
        name='Sample 1',
        substrate='Paper',
        thickness=4.0,
        spectrum=np.array([...]),  # 31 values
        concentrations={'BiVaO4': 10.0, 'PG': 5.0}
    )
]

# Train models
km_model = train_model(samples, 'single')
nn_model = train_model(samples, 'neural-net')
```

### Make Predictions
```python
# Define formulation
formulation = {
    'BiVaO4': 8.0,  # percent
    'PG': 3.0,
    'PB': 0.5
}
thickness = 4.0  # micrometers

# Predict
km_spectrum = predict_reflectance(formulation, km_model, thickness)
nn_spectrum = predict_reflectance(formulation, nn_model, thickness)

# Results are numpy arrays of 31 wavelengths (400-700nm)
print(f"Predicted reflectance at 550nm: {nn_spectrum[15]:.3f}")
```

### Visualize Results
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(WAVELENGTHS, km_spectrum, label='K-M Model')
plt.plot(WAVELENGTHS, nn_spectrum, label='Neural Network')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend()
plt.show()
```

## Model Types

| Model | String ID | Best For |
|-------|-----------|----------|
| K-M Single Layer | `'single'` | Non-fluorescent, opaque samples |
| K-M Two Layer | `'two-layer'` | Translucent films, coatings |
| Neural Network | `'neural-net'` | Fluorescent, complex formulations |

## Data Format

### Sample Data Structure
```python
SampleData(
    id: str              # Unique identifier
    name: str            # Display name
    substrate: str       # Substrate type (e.g., 'Paper')
    thickness: float     # Film thickness in micrometers
    spectrum: np.ndarray # 31 reflectance values (400-700nm)
    concentrations: dict # {'reagent': percent}
)
```

### CSV Format (Concentrations)
```csv
Sample,Substrate,Thickness (um),BiVaO4 (%),PG (%),PB (%)
S001,Paper,4,10.5,5.2,0.3
```

### CSV Format (Spectra, no header)
```csv
S001,,,L*,a*,b*,,,0.45,0.48,0.51,...,0.82,path,
```
Columns 8-38 contain reflectance at 400-700nm (10nm intervals)

## Common Tasks

### Load Data from CSV
```python
from utils.data_loader import parse_samples_from_csv, load_master_data

# Single CSV file
samples = parse_samples_from_csv('data.csv')

# Master files (conc + spec)
samples = load_master_data('Master conc.csv', 'Master spec.csv')
```

### Save Trained Model
```python
import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(km_model, f)

# Load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

### Batch Predictions
```python
formulations = [
    {'BiVaO4': 10, 'PG': 5},
    {'BiVaO4': 8, 'PG': 7},
    {'BiVaO4': 12, 'PG': 3}
]

predictions = [
    predict_reflectance(f, km_model, 4.0)
    for f in formulations
]
```

### Calculate Color Difference
```python
def calculate_delta_e(spectrum1, spectrum2):
    """Simple mean absolute error"""
    return np.mean(np.abs(spectrum1 - spectrum2))

error = calculate_delta_e(reference_spectrum, predicted_spectrum)
```

## Performance Tips

### GPU Acceleration
```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Neural network training automatically uses GPU if available
```

### Faster Training
```python
# Reduce epochs for faster training (less accurate)
nn_model = train_model(samples, 'neural-net')  # Default: 2000 epochs

# Adjust in neural_network.py:
# train_neural_network(..., epochs=500)  # Faster but less accurate
```

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the python_version directory
cd python_version
python main.py --use-initial
```

### Singular Matrix Error
- Add more diverse training samples
- Remove duplicate samples
- Check for missing concentration data

### Poor Predictions
- Increase training data quantity
- Ensure formulation is within training range
- Try different model types
- Increase neural network size or epochs

### Memory Issues
- Reduce batch size in neural network training
- Use fewer training samples
- Close other applications

## API Reference

### train_model(samples, model_type)
- **samples**: List[SampleData]
- **model_type**: 'single' | 'two-layer' | 'neural-net'
- **returns**: ModelCoefficients

### predict_reflectance(concentrations, model, thickness)
- **concentrations**: Dict[str, float]
- **model**: ModelCoefficients
- **thickness**: float
- **returns**: np.ndarray (31 wavelengths)

## Constants

```python
from types_constants import WAVELENGTHS, REAGENTS_LIST, Rg_DEFAULT

WAVELENGTHS    # [400, 410, 420, ..., 700] nm
REAGENTS_LIST  # ['BiVaO4', 'PG', 'PB', ...]
Rg_DEFAULT     # 0.95 (white paper background)
```

## Getting Help

1. Check [README.md](README.md) for detailed documentation
2. Review [PORTING_NOTES.md](PORTING_NOTES.md) for technical details
3. Run `python test_installation.py` to verify setup
4. Check error messages for specific issues

## Examples Directory Structure

```
python_version/
├── main.py              # Main CLI application
├── test_installation.py # Verification script
└── examples/            # (create your own)
    ├── custom_training.py
    ├── batch_predict.py
    └── model_comparison.py
```
