"""
Enhanced FastAPI server for PP substrate neural network predictions (V2)
- Spectral reflectance prediction via NN
- Deterministic CIELAB calculation from reflectance
- Deterministic fluorescence calculation
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import torch
import os
import pickle

from optimize_hyperparameters import FlexibleNN
from utils.new_data_loader import load_new_dataset
from fluorescence_nn_predictor import get_fluorescence_predictor
from cielab_calculator import reflectance_to_cielab

app = FastAPI(title="OptiMix PyTorch API V2", version="2.0.0")

# Initialize predictors (lazy-loaded)
FLUORESCENCE_PREDICTOR = None

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
MODEL_CACHE = {
    'model': None,
    'input_mean': None,
    'input_std': None,
    'spectral_mean': None,
    'spectral_std': None,
    'input_size': 5,
    'hidden_size': 64,
    'reagents': ['GXT', 'BiVaO4', 'PG', 'PearlB']
}

MODEL_PATH = 'trained_models/optimized_best_model.pkl'


class PredictionRequest(BaseModel):
    """Request format for predictions with 4 reagents"""
    concentrations: Dict[str, float]  # GXT, BiVaO4, PG, PearlB (as percentages)
    thickness: float = 8.0  # 8μm or 12μm


class CIELABValues(BaseModel):
    """CIELAB color coordinates"""
    L: float  # Lightness (0-100)
    a: float  # Green (-) to Red (+)
    b: float  # Blue (-) to Yellow (+)
    c: float  # Chroma
    h: float  # Hue angle (0-360)


class FluorescenceValues(BaseModel):
    """Fluorescence prediction values"""
    fluorescence_cts: float  # Predicted fluorescence intensity (ct/s)
    fluorescence_area: float  # Background-subtracted area
    gxt_multiplier: float  # Smooth constraint multiplier (0 at 0% GXT, ~1 at ≥5% GXT)
    model_r2: float  # R² of fluorescence model


class PredictionResponse(BaseModel):
    """Response format with spectral + CIELAB + fluorescence predictions"""
    wavelengths: List[int]
    reflectance: List[float]
    cielab: CIELABValues  # Calculated deterministically from reflectance
    fluorescence: FluorescenceValues  # Calculated from concentrations + reflectance
    thickness: float
    model_version: str


class ModelStatus(BaseModel):
    """Model status response"""
    status: str
    model_type: str
    reagents: List[str]
    total_samples: int
    architecture: str
    parameters: int


def load_spectral_model():
    """Load the spectral reflectance neural network model"""
    if MODEL_CACHE['model'] is not None:
        return MODEL_CACHE

    if os.path.exists(MODEL_PATH):
        print(f"Loading spectral NN from {MODEL_PATH}...")
        try:
            with open(MODEL_PATH, 'rb') as f:
                saved_data = pickle.load(f)

            # Handle old model format (model already initialized)
            if 'model' in saved_data:
                MODEL_CACHE['model'] = saved_data['model']
                MODEL_CACHE['input_mean'] = saved_data['input_mean']
                MODEL_CACHE['input_std'] = saved_data['input_std']

                output_mean = saved_data.get('output_mean')
                output_std = saved_data.get('output_std')
                MODEL_CACHE['spectral_mean'] = output_mean[:31] if output_mean is not None else None
                MODEL_CACHE['spectral_std'] = output_std[:31] if output_std is not None else None
                MODEL_CACHE['input_size'] = 5

                config = saved_data.get('config', {})
                hidden_layers = config.get('hidden_layers', [64, 128, 64])
                MODEL_CACHE['hidden_size'] = hidden_layers[0] if hidden_layers else 64

                print(f"✓ Spectral NN loaded")
                print(f"  Architecture: 5 → {hidden_layers} → 31 Spectral")

            else:
                # New model format (need to reconstruct)
                hidden_layers = saved_data['hidden_layers']
                input_size = saved_data.get('input_size', 5)

                model = FlexibleNN(
                    input_size=input_size,
                    hidden_layers=hidden_layers,
                    activation='relu',
                    dropout_rate=0.1,
                    use_batchnorm=True
                )

                model.load_state_dict(saved_data['model_state'])
                model.eval()

                MODEL_CACHE['model'] = model
                MODEL_CACHE['input_mean'] = saved_data['input_mean']
                MODEL_CACHE['input_std'] = saved_data['input_std']
                MODEL_CACHE['spectral_mean'] = saved_data['spectral_mean']
                MODEL_CACHE['spectral_std'] = saved_data['spectral_std']
                MODEL_CACHE['input_size'] = input_size
                MODEL_CACHE['hidden_size'] = hidden_layers[0] if hidden_layers else 64

                print(f"✓ Spectral NN loaded")
                print(f"  Architecture: 5 → {hidden_layers} → 31 Spectral")

            return MODEL_CACHE

        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            raise

    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")


@app.on_event("startup")
async def startup_event():
    """Load models on server startup"""
    global FLUORESCENCE_PREDICTOR
    try:
        load_spectral_model()
        # Load fluorescence NN predictor
        FLUORESCENCE_PREDICTOR = get_fluorescence_predictor()
        print("✓ API server ready (spectral + deterministic CIELAB + fluorescence)")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Server will attempt to load models on first request")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "OptiMix PyTorch API V2 is running",
        "version": "2.0.0",
        "model": "PP Substrate 4-Reagent with Deterministic CIELAB"
    }


@app.get("/api/status", response_model=ModelStatus)
async def get_status():
    """Get model status and capabilities"""
    try:
        model_cache = load_spectral_model()

        # Count samples from dataset
        concentrations_path = '../public/Concentrations.csv'
        spectra_path = '../public/Spectra.csv'

        samples, _, _ = load_new_dataset(concentrations_path, spectra_path)

        # Get architecture based on model type
        hidden_size = model_cache.get('hidden_size', 64)
        input_size = model_cache.get('input_size', 5)
        params = model_cache['model'].count_parameters()

        model_type = "Spectral NN + Deterministic CIELAB"
        architecture = f"5 inputs → 4×{hidden_size} → 31 Spectral (CIELAB calculated deterministically)"

        return ModelStatus(
            status="ready",
            model_type=model_type,
            reagents=model_cache['reagents'],
            total_samples=len(samples),
            architecture=architecture,
            parameters=params
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict spectral reflectance, calculate CIELAB deterministically, and predict fluorescence

    Args:
        request: PredictionRequest with concentrations and thickness

    Returns:
        PredictionResponse with wavelengths, reflectance, CIELAB, and fluorescence
    """
    try:
        model_cache = load_spectral_model()

        # Validate reagents
        expected_reagents = set(model_cache['reagents'])
        provided_reagents = set(request.concentrations.keys())

        if not provided_reagents.issubset(expected_reagents):
            extra_reagents = provided_reagents - expected_reagents
            raise HTTPException(
                status_code=400,
                detail=f"Invalid reagents: {extra_reagents}. Expected: {expected_reagents}"
            )

        # Validate thickness
        if request.thickness not in [8.0, 12.0]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid thickness: {request.thickness}. Must be 8.0 or 12.0 μm"
            )

        # Prepare input features (normalized to 0-1 range)
        # Input: [GXT%, BiVaO4%, PG%, PearlB%, Thickness]
        x = np.array([
            request.concentrations.get('GXT', 0.0) / 100.0,
            request.concentrations.get('BiVaO4', 0.0) / 100.0,
            request.concentrations.get('PG', 0.0) / 100.0,
            request.concentrations.get('PearlB', 0.0) / 100.0,
            request.thickness / 12.0
        ]).reshape(1, -1)

        # Normalize inputs using training statistics
        x_normalized = (x - model_cache['input_mean']) / (model_cache['input_std'] + 1e-8)

        # Convert to tensor
        x_tensor = torch.FloatTensor(x_normalized)

        # Make prediction
        model = model_cache['model']
        model.eval()

        with torch.no_grad():
            output = model(x_tensor)

            # Handle tuple output (model may return spectral and CIELAB)
            if isinstance(output, tuple):
                pred_spectral = output[0].numpy().flatten()
            else:
                pred_spectral = output.numpy().flatten()

            # Take only first 31 values (spectral)
            pred_spectral = pred_spectral[:31]

        # Denormalize spectral predictions
        if MODEL_CACHE['spectral_mean'] is not None:
            spectral_mean = MODEL_CACHE['spectral_mean']
            spectral_std = MODEL_CACHE['spectral_std']
        else:
            spectral_mean = model_cache.get('output_mean', np.zeros(36))[:31]
            spectral_std = model_cache.get('output_std', np.ones(36))[:31]

        reflectance = pred_spectral * (spectral_std + 1e-8) + spectral_mean

        # Calculate CIELAB deterministically from reflectance
        # Note: Reflectance >1.0 can occur due to fluorescence/scattering/measurement artifacts
        # Analysis shows using raw (unclipped) reflectance is 16.3% more accurate (ΔE*ab = 5.57 vs 6.65)
        cielab_values = reflectance_to_cielab(reflectance)

        # Wavelengths (400-700nm, 10nm intervals)
        wavelengths = list(range(400, 710, 10))

        # Calculate fluorescence using separate NN predictor
        fluorescence_result = FLUORESCENCE_PREDICTOR.predict(
            request.concentrations,
            reflectance,
            request.thickness
        )

        return PredictionResponse(
            wavelengths=wavelengths,
            reflectance=reflectance.tolist(),
            cielab=CIELABValues(**cielab_values),
            fluorescence=FluorescenceValues(**fluorescence_result),
            thickness=request.thickness,
            model_version="2.0.0-spectral+deterministic-cielab+fluorescence"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/reagents")
async def get_reagents():
    """Get list of available reagents for PP substrate"""
    try:
        model_cache = load_spectral_model()
        return {
            "reagents": model_cache['reagents'],
            "substrate": "PP (Polypropylene)",
            "thicknesses": [8.0, 12.0],
            "outputs": ["spectral_reflectance", "cielab_L", "cielab_a", "cielab_b", "cielab_c", "cielab_h", "fluorescence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """
    Batch prediction for multiple formulations

    Args:
        requests: List of prediction requests

    Returns:
        List of prediction responses
    """
    try:
        results = []
        for req in requests:
            response = await predict(req)
            results.append(response)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting OptiMix PyTorch API V2...")
    print("Model: Spectral NN + Deterministic CIELAB + Fluorescence")

    # Pre-load model before starting server
    try:
        load_spectral_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
