"""
Enhanced FastAPI server for PP substrate neural network predictions
Serves predictions for 4 reagents (GXT, BiVaO4, PG, PearlB) with CIELAB outputs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
import torch
import os
import pickle

from models.enhanced_neural_network import EnhancedSpectralNN
from optimize_hyperparameters import FlexibleNN
from utils.new_data_loader import load_new_dataset
from fluorescence_predictor import FluorescencePredictor

app = FastAPI(title="OptiMix Enhanced PyTorch API", version="2.0.0")

# Initialize fluorescence predictor
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
    'cielab_mean': None,
    'cielab_std': None,
    'fluorescence_mean': None,
    'fluorescence_std': None,
    'has_fluorescence': False,  # Flag to indicate if model predicts fluorescence
    'use_physics_features': False,
    'input_size': 5,
    'feature_names': None,
    'hidden_size': 64,
    'reagents': ['GXT', 'BiVaO4', 'PG', 'PearlB']
}

MODEL_PATH = 'trained_models/optimized_best_model.pkl'


class EnhancedPredictionRequest(BaseModel):
    """Request format for enhanced predictions with 4 reagents"""
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
    model_r2: float  # R² of fluorescence model


class EnhancedPredictionResponse(BaseModel):
    """Response format with spectral + CIELAB + fluorescence predictions"""
    wavelengths: List[int]
    reflectance: List[float]
    cielab: CIELABValues
    fluorescence: FluorescenceValues  # NEW: Fluorescence predictions
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


def load_enhanced_model():
    """Load the enhanced neural network model"""
    if MODEL_CACHE['model'] is not None:
        return MODEL_CACHE

    if os.path.exists(MODEL_PATH):
        print(f"Loading enhanced NN from {MODEL_PATH}...")
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
                MODEL_CACHE['cielab_mean'] = output_mean[31:] if output_mean is not None else None
                MODEL_CACHE['cielab_std'] = output_std[31:] if output_std is not None else None
                MODEL_CACHE['input_size'] = 5

                config = saved_data.get('config', {})
                hidden_layers = config.get('hidden_layers', [64, 128, 64])
                MODEL_CACHE['hidden_size'] = hidden_layers[0] if hidden_layers else 64

                print(f"✓ Multi-output NN loaded")
                print(f"  Architecture: 5 → {hidden_layers} → [31 Spectral + 5 CIELAB]")

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
                MODEL_CACHE['cielab_mean'] = saved_data['cielab_mean']
                MODEL_CACHE['cielab_std'] = saved_data['cielab_std']
                MODEL_CACHE['input_size'] = input_size
                MODEL_CACHE['hidden_size'] = hidden_layers[0] if hidden_layers else 64

                print(f"✓ Multi-output NN loaded")
                print(f"  Architecture: 5 → {hidden_layers} → [31 Spectral + 5 CIELAB]")

            return MODEL_CACHE

        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            raise

    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")


def denormalize_cielab(cielab_normalized: np.ndarray) -> Dict[str, float]:
    """
    Denormalize CIELAB values from 0-1 range back to original ranges

    Args:
        cielab_normalized: Normalized CIELAB values [L, a, b, c, h]

    Returns:
        Dictionary with denormalized CIELAB values
    """
    L = cielab_normalized[0] * 100.0  # L: 0-100
    a = cielab_normalized[1] * 256.0 - 128.0  # a: -128 to 128
    b = cielab_normalized[2] * 256.0 - 128.0  # b: -128 to 128
    c = cielab_normalized[3] * 150.0  # c: 0-150
    h = cielab_normalized[4] * 360.0  # h: 0-360

    return {
        'L': float(L),
        'a': float(a),
        'b': float(b),
        'c': float(c),
        'h': float(h)
    }


@app.on_event("startup")
async def startup_event():
    """Load model on server startup"""
    global FLUORESCENCE_PREDICTOR
    try:
        load_enhanced_model()
        # Load fluorescence predictor
        FLUORESCENCE_PREDICTOR = FluorescencePredictor()
        print("✓ Enhanced API server ready")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Server will attempt to load model on first request")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "OptiMix Physics-Informed PyTorch API is running",
        "version": "2.1.0",
        "model": "PP Substrate 4-Reagent with Physics-Informed NN + CIELAB"
    }


@app.get("/api/status", response_model=ModelStatus)
async def get_status():
    """Get model status and capabilities"""
    try:
        model_cache = load_enhanced_model()

        # Count samples from dataset
        concentrations_path = '../public/Concentrations.csv'
        spectra_path = '../public/Spectra.csv'

        samples, _, _ = load_new_dataset(concentrations_path, spectra_path)

        # Get architecture based on model type
        hidden_size = model_cache.get('hidden_size', 64)
        input_size = model_cache.get('input_size', 5)
        params = model_cache['model'].count_parameters()

        if model_cache.get('use_physics_features', False):
            model_type = "Physics-Informed Neural Network (Kubelka-Munk)"
            architecture = f"{input_size} features → 4×{hidden_size} → [31 Spectral + 5 CIELAB]"
        else:
            model_type = "Enhanced Multi-Output Neural Network"
            architecture = f"4×{hidden_size} (Shared) → [31 Spectral + 5 CIELAB]"

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


@app.post("/api/predict", response_model=EnhancedPredictionResponse)
async def predict(request: EnhancedPredictionRequest):
    """
    Predict spectral reflectance AND CIELAB values for given formulation

    Args:
        request: EnhancedPredictionRequest with concentrations and thickness

    Returns:
        EnhancedPredictionResponse with wavelengths, reflectance, and CIELAB values
    """
    try:
        model_cache = load_enhanced_model()

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

        # Wavelengths (400-700nm, 10nm intervals)
        wavelengths = list(range(400, 710, 10))

        # Use linear fluorescence predictor
        fluorescence_result = FLUORESCENCE_PREDICTOR.predict(request.concentrations, reflectance, request.thickness)

        return EnhancedPredictionResponse(
            wavelengths=wavelengths,
            reflectance=reflectance.tolist(),
            cielab=CIELABValues(**cielab_values),
            fluorescence=FluorescenceValues(**fluorescence_result),
            thickness=request.thickness,
            model_version="3.0.0-optimized+fluorescence"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/reagents")
async def get_reagents():
    """Get list of available reagents for PP substrate"""
    try:
        model_cache = load_enhanced_model()
        return {
            "reagents": model_cache['reagents'],
            "substrate": "PP (Polypropylene)",
            "thicknesses": [8.0, 12.0],
            "outputs": ["spectral_reflectance", "cielab_L", "cielab_a", "cielab_b", "cielab_c", "cielab_h"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch_predict")
async def batch_predict(requests: List[EnhancedPredictionRequest]):
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
    print("Starting OptiMix Enhanced PyTorch API server...")
    print("Model: PP Substrate (4 reagents) with CIELAB predictions")

    # Pre-load model before starting server
    try:
        load_enhanced_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Different port from old API
        log_level="info"
    )
