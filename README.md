# OptiMix - Spectral Formulation Engine

A predictive modeling application for spectral reflectance in pigment formulations, comparing physics-based Kubelka-Munk theory with machine learning approaches.

## Overview

OptiMix is a web-based tool that predicts the spectral reflectance curves of pigment mixtures using two complementary modeling approaches:

1. **Kubelka-Munk (K-M) Single Layer Model** - A physics-based approach grounded in light scattering theory
2. **Neural Network Model** - A data-driven approach that learns non-linear relationships from training data

The application enables users to:
- Simulate spectral reflectance for custom pigment formulations
- Compare predictions from both modeling approaches in real-time
- Visualize spectral curves across the 400-700nm visible range
- Load and analyze sample libraries from CSV data

## How It Works

### Data Flow

```
Master CSV Data → Training → Model Coefficients → Real-time Prediction → Visualization
```

### 1. Data Loading

On startup, the application loads spectral data from two master CSV files located in `public/`:

- **Master conc.csv** - Contains sample metadata and pigment concentrations
- **Master spec - master_sample_library.csv** - Contains measured spectral reflectance curves (400-700nm, 10nm intervals)

The data loader ([utils/loadMasterData.ts](utils/loadMasterData.ts)) merges these datasets by sample ID.

### 2. Model Training

#### Kubelka-Munk Model

The K-M model ([services/kmService.ts](services/kmService.ts)) converts measured reflectance values to K/S ratios using:

```
K/S = (1 - R)² / (2R)
```

Where:
- `K` = absorption coefficient
- `S` = scattering coefficient
- `R` = reflectance

For each wavelength, the model solves a linear system using matrix pseudoinverse to find K/S coefficients for each pigment. The model assumes:
- K/S values are additive (Beer-Lambert-like behavior)
- Linear concentration-to-K/S relationship

**Strengths:**
- Physically interpretable
- Works well for opaque, non-fluorescent samples
- Requires minimal training data

**Limitations:**
- Assumes linearity (fails for fluorescent pigments)
- Cannot capture pigment interactions
- Poor extrapolation outside training range

#### Neural Network Model

The neural network ([utils/neuralNet.ts](utils/neuralNet.ts)) uses a feedforward architecture:

```
Input Layer (N pigments + thickness) → Hidden Layer (128 neurons, ReLU) → Output Layer (31 wavelengths)
```

Training process:
- **Input:** Pigment concentrations + substrate thickness
- **Output:** 31 reflectance values (400-700nm)
- **Architecture:** 1 hidden layer with 128 neurons
- **Activation:** ReLU (hidden), Linear (output)
- **Optimization:** Stochastic Gradient Descent with mini-batches
- **Regularization:** L2 weight decay (λ=0.005)
- **Training:** 2000 epochs, learning rate 0.005

The network learns to map formulation parameters directly to spectral curves, capturing:
- Non-linear concentration effects
- Fluorescence peaks (R > 1.0)
- Pigment interaction effects
- Substrate-specific behavior

**Strengths:**
- Captures fluorescence and non-linear effects
- Learns from data patterns
- Handles complex mixtures

**Limitations:**
- Requires substantial training data
- Black-box (not physically interpretable)
- May overfit with insufficient data

### 3. Prediction

Both models predict reflectance for user-specified formulations:

**K-M Approach:**
```typescript
KS_mix(λ) = Σ [concentration_i × alpha_i(λ)]
R(λ) = 1 + KS - √(KS² + 2KS)
```

**Neural Network Approach:**
```typescript
R = NN([c1, c2, ..., cn, thickness])
```

Predictions update in real-time as users adjust concentration sliders.

### 4. Visualization

The spectral chart ([components/SpectralChart.tsx](components/SpectralChart.tsx)) displays three overlaid curves:

- **Blue line** - Reference spectrum (selected sample)
- **Cyan line** - K-M model prediction
- **Purple line** - Neural network prediction

## Project Structure

```
PredictiveApp/
├── App.tsx                    # Main application component, state management
├── index.tsx                  # React app entry point
├── types.ts                   # TypeScript type definitions
├── constants.ts               # Wavelengths, reagent lists, initial data
│
├── components/
│   └── SpectralChart.tsx      # Recharts visualization component
│
├── services/
│   └── kmService.ts           # K-M model training & prediction
│
├── utils/
│   ├── csvParser.ts           # CSV file parsing utilities
│   ├── loadMasterData.ts      # Master data loading logic
│   ├── matrix.ts              # Linear algebra (pseudoinverse)
│   └── neuralNet.ts           # Neural network implementation
│
├── public/
│   ├── Master conc.csv        # Sample concentrations
│   └── Master spec - ...csv   # Spectral measurements
│
├── DATA_COLLECTION_GUIDE.md   # Guidance for improving model accuracy
└── package.json               # Dependencies & scripts
```

## Getting Started

### Prerequisites

- Node.js (v18 or later)
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PredictiveApp
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser to `http://localhost:5173`

### Building for Production

```bash
npm run build
```

The optimized build will be created in the `dist/` folder.

## Using the Application

### Loading a Reference Sample

1. Use the "Load Reference" dropdown in the sidebar to select a sample
2. The spectral chart updates to show the measured curve
3. Concentration sliders automatically populate with that sample's formulation

### Creating Custom Formulations

1. Adjust the concentration sliders for each pigment (0-30%)
2. Monitor the "Total" percentage (should not exceed 100%)
3. Watch the predictions update in real-time
4. Compare K-M (cyan) vs Neural Network (purple) predictions

### Comparing Models

- **Smooth curves** - Both models agree (high confidence region)
- **Diverging predictions** - Models disagree (check training data coverage)
- **Purple peaks > 1.0** - Neural network detected fluorescence (K-M cannot model this)

## Data Format

### Concentration CSV Format

```csv
Sample,BiVaO4,PG,PB,LY,GXT,SY43,TiO2,SFXC G,...
Sample_001,10.5,5.2,0,0,0,0,0,15.3,...
```

### Spectral CSV Format

```csv
Sample,400,410,420,...,700
Sample_001,0.452,0.478,0.501,...,0.823
```

See [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md) for recommendations on collecting training data.

## Improving Model Accuracy

Model accuracy depends heavily on training data quality and coverage:

### Data Collection Strategies

1. **Dilution Series** - Measure each pigment in intervals or sensible range for formulations used in lab
2. **Binary Mixtures** - Test interactions between pigment pairs
3. **High-Concentration Fluorophores** - Capture extreme or self-quenching behavior

## Technology Stack

- **React** - UI framework
- **TypeScript** - Type safety and developer experience
- **Vite** - Build tool and dev server
- **Recharts** - Data visualization library
- **Tailwind CSS** - Styling framework

## Key Algorithms

### Matrix Pseudoinverse

Used in K-M model to solve overdetermined linear systems:

```typescript
// Solve: Y = X × α
// α = (X^T × X)^-1 × X^T × Y
alpha = pseudoInverse(X) × Y
```

Implementation: [utils/matrix.ts](utils/matrix.ts)

### Backpropagation

Used to train neural network:

```typescript
1. Forward pass: compute predictions
2. Calculate loss: MSE(predicted, actual)
3. Backward pass: compute gradients
4. Update weights: w -= learningRate × gradient
```

Implementation: [utils/neuralNet.ts](utils/neuralNet.ts)



## References

- Kubelka, P., & Munk, F. (1931). "An article on optics of paint layers"
- Goodfellow, I., et al. (2016). "Deep Learning" - Neural network theory
- Sharma, G., et al. (2017). "Digital Color Imaging Handbook" - Color science fundamentals
