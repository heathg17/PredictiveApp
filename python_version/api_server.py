"""
FastAPI server to serve PyTorch neural network predictions
Replaces TypeScript NN in the React frontend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import torch
import os
import pickle

from services.km_service import train_model, predict_reflectance
from utils.data_loader import load_master_data
from types_constants import WAVELENGTHS

app = FastAPI(title="OptiMix PyTorch API", version="1.0.0")

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
    'km_model': None,
    'nn_model': None,
    'reagents': None
}

MODEL_PATH = 'trained_models/pytorch_nn_model.pkl'


class PredictionRequest(BaseModel):
    """Request format for predictions"""
    concentrations: Dict[str, float]  # Reagent name -> percentage
    thickness: float = 4.0
    model_type: str = 'neural-net'  # 'single', 'two-layer', 'neural-net'


class PredictionResponse(BaseModel):
    """Response format for predictions"""
    wavelengths: List[int]
    reflectance: List[float]
    model_type: str


class TrainingStatus(BaseModel):
    """Training status response"""
    status: str
    samples_loaded: int
    samples_with_conc: int
    reagents: List[str]
    model_types: List[str]


def load_or_train_models():
    """Load pre-trained models or train new ones"""
    if MODEL_CACHE['nn_model'] is not None:
        return MODEL_CACHE

    print("Loading models...")

    # Load data
    master_conc_path = '../public/Master conc.csv'
    master_spec_path = '../public/Master spec - master_sample_library.csv'

    if not os.path.exists(master_conc_path):
        raise FileNotFoundError(f"Master concentration file not found: {master_conc_path}")

    samples = load_master_data(master_conc_path, master_spec_path)
    samples_with_conc = [s for s in samples if len(s.concentrations) > 0
                         and any(c > 0 for c in s.concentrations.values())]

    if len(samples_with_conc) == 0:
        raise ValueError("No samples with concentration data found")

    print(f"Loaded {len(samples_with_conc)} samples with concentration data")

    # Check if pre-trained model exists
    if os.path.exists(MODEL_PATH):
        print(f"Loading pre-trained model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            saved_data = pickle.load(f)
            MODEL_CACHE['nn_model'] = saved_data['nn_model']
            MODEL_CACHE['km_model'] = saved_data['km_model']
            MODEL_CACHE['reagents'] = saved_data['reagents']
            print("✓ Pre-trained models loaded successfully")
            return MODEL_CACHE

    # Train new models
    print("Training new models...")
    MODEL_CACHE['km_model'] = train_model(samples_with_conc, 'single')
    MODEL_CACHE['nn_model'] = train_model(samples_with_conc, 'neural-net')

    # Get reagents list
    all_reagents = set()
    for s in samples_with_conc:
        all_reagents.update(s.concentrations.keys())
    MODEL_CACHE['reagents'] = sorted(list(all_reagents))

    # Save trained models
    os.makedirs('trained_models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'nn_model': MODEL_CACHE['nn_model'],
            'km_model': MODEL_CACHE['km_model'],
            'reagents': MODEL_CACHE['reagents']
        }, f)
    print(f"✓ Models trained and saved to {MODEL_PATH}")

    return MODEL_CACHE


@app.on_event("startup")
async def startup_event():
    """Load models on server startup"""
    try:
        load_or_train_models()
        print("✓ API server ready")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Server will train models on first request")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "OptiMix PyTorch API is running",
        "version": "1.0.0"
    }


@app.get("/api/status", response_model=TrainingStatus)
async def get_status():
    """Get training status and available models"""
    try:
        models = load_or_train_models()

        # Count samples
        master_conc_path = '../public/Master conc.csv'
        master_spec_path = '../public/Master spec - master_sample_library.csv'
        samples = load_master_data(master_conc_path, master_spec_path)
        samples_with_conc = [s for s in samples if len(s.concentrations) > 0]

        return TrainingStatus(
            status="ready",
            samples_loaded=len(samples),
            samples_with_conc=len(samples_with_conc),
            reagents=models['reagents'] or [],
            model_types=['single', 'neural-net']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict spectral reflectance for given formulation

    Args:
        request: PredictionRequest containing concentrations, thickness, and model type

    Returns:
        PredictionResponse with wavelengths and predicted reflectance
    """
    try:
        models = load_or_train_models()

        # Select model
        if request.model_type == 'neural-net':
            model = models['nn_model']
        elif request.model_type == 'single':
            model = models['km_model']
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {request.model_type}. Use 'single' or 'neural-net'"
            )

        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Make prediction
        reflectance = predict_reflectance(
            request.concentrations,
            model,
            request.thickness
        )

        # Convert to list for JSON serialization
        reflectance_list = reflectance.tolist() if isinstance(reflectance, np.ndarray) else list(reflectance)

        return PredictionResponse(
            wavelengths=WAVELENGTHS.tolist(),
            reflectance=reflectance_list,
            model_type=request.model_type
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/retrain")
async def retrain_models():
    """Force retrain all models with latest data"""
    try:
        # Clear cache
        MODEL_CACHE['nn_model'] = None
        MODEL_CACHE['km_model'] = None
        MODEL_CACHE['reagents'] = None

        # Delete saved model
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        # Retrain
        models = load_or_train_models()

        return {
            "status": "success",
            "message": "Models retrained successfully",
            "reagents": models['reagents']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")


@app.get("/api/reagents")
async def get_reagents():
    """Get list of available reagents"""
    try:
        models = load_or_train_models()
        return {
            "reagents": models['reagents']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting OptiMix PyTorch API server...")
    print("Training models (this may take a moment)...")

    # Pre-load models before starting server
    load_or_train_models()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
