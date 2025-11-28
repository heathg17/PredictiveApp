"""
Combined Best Model: Baseline NN (Spectral) + Hybrid K-M (CIELAB)

Uses the strengths of each approach:
- Baseline NN for spectral reflectance (better accuracy)
- Hybrid K-M for CIELAB predictions (94% better accuracy)
"""
import numpy as np
import torch
import torch.nn as nn
import pickle


def predict_combined(GXT, BiVaO4, PG, PearlB, thickness,
                     baseline_model, hybrid_model):
    """
    Combined prediction using both models

    Args:
        GXT, BiVaO4, PG, PearlB: Concentrations (%)
        thickness: Coating thickness (μm)
        baseline_model: Trained baseline model dict
        hybrid_model: Trained hybrid K-M model dict

    Returns:
        reflectance: From baseline NN (31 wavelengths)
        cielab: From hybrid K-M model
    """
    # Prepare input for baseline NN
    x_baseline = np.array([
        GXT / 100.0,
        BiVaO4 / 100.0,
        PG / 100.0,
        PearlB / 100.0,
        thickness / 12.0
    ])

    # Get spectral prediction from baseline NN
    x_norm = (x_baseline - baseline_model['input_mean']) / (baseline_model['input_std'] + 1e-8)
    x_tensor = torch.FloatTensor(x_norm).unsqueeze(0)

    baseline_model['model'].eval()
    with torch.no_grad():
        pred_spectral, _ = baseline_model['model'](x_tensor)
        pred_spectral = pred_spectral.numpy().flatten()

    # Denormalize spectral
    spectral_mean = baseline_model.get('output_mean', np.zeros(36))[:31]
    spectral_std = baseline_model.get('output_std', np.ones(36))[:31]
    reflectance = pred_spectral * (spectral_std + 1e-8) + spectral_mean

    # Get CIELAB prediction from hybrid K-M model
    from models.km_hybrid_model import predict_hybrid
    _, cielab = predict_hybrid(
        GXT, BiVaO4, PG, PearlB, thickness,
        hybrid_model['model'],
        hybrid_model['input_mean'],
        hybrid_model['input_std'],
        hybrid_model['output_mean'],
        hybrid_model['output_std']
    )

    return reflectance, cielab


def train_combined_models(X_train, Y_train, X_val, Y_val,
                          baseline_epochs=2000,
                          hybrid_epochs=2000,
                          verbose=True):
    """
    Train both baseline and hybrid models

    Returns:
        baseline_model: Trained baseline NN
        hybrid_model: Trained hybrid K-M model
    """
    from models.enhanced_neural_network import train_enhanced_neural_network
    from models.km_hybrid_model import train_hybrid_model

    print("=" * 100)
    print("TRAINING COMBINED BEST MODEL")
    print("=" * 100)
    print()

    # Train baseline for spectral
    print("Training baseline NN for spectral predictions...")
    baseline_model = train_enhanced_neural_network(
        X_train, Y_train,
        X_val, Y_val,
        hidden_size=64,
        learning_rate=0.001,
        epochs=baseline_epochs,
        batch_size=16,
        l2_lambda=0.001,
        dropout_rate=0.2,
        verbose=verbose
    )

    # Train hybrid for CIELAB
    print("\nTraining hybrid K-M for CIELAB predictions...")
    hybrid_model = train_hybrid_model(
        X_train, Y_train,
        X_val, Y_val,
        hidden_size=32,
        learning_rate=0.001,
        epochs=hybrid_epochs,
        batch_size=16,
        l2_lambda=0.001,
        dropout_rate=0.2,
        verbose=verbose
    )

    return baseline_model, hybrid_model


def save_combined_models(baseline_model, hybrid_model, filepath='trained_models/combined_best_model.pkl'):
    """Save both models to a single file"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    combined = {
        'baseline': baseline_model,
        'hybrid': hybrid_model,
        'model_type': 'combined_spectral_cielab',
        'description': 'Baseline NN for spectral + Hybrid K-M for CIELAB'
    }

    with open(filepath, 'wb') as f:
        pickle.dump(combined, f)

    print(f"✓ Saved combined model to: {filepath}")


def load_combined_models(filepath='trained_models/combined_best_model.pkl'):
    """Load both models from file"""
    with open(filepath, 'rb') as f:
        combined = pickle.load(f)

    return combined['baseline'], combined['hybrid']
