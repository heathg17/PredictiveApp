"""
Comprehensive Hyperparameter Optimization for Neural Network

Tests combinations of:
- Network depth (number of hidden layers)
- Layer widths (neurons per layer)
- Learning rate
- Batch size
- Regularization (L2, dropout)
- Activation functions
- Normalization strategies

Finds the optimal configuration for spectral + CIELAB prediction.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools
import time
import json
import os
from typing import Dict, List, Tuple

from utils.new_data_loader import load_new_dataset, prepare_training_data

# ============================================================================
# Flexible Neural Network Architecture
# ============================================================================

class FlexibleNN(nn.Module):
    """Flexible multi-layer neural network with configurable architecture"""

    def __init__(self, input_size, hidden_layers, activation='relu',
                 dropout_rate=0.2, use_batchnorm=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm

        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            self.activation = nn.ReLU()

        # Build shared layers
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(self.activation)

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        self.shared = nn.Sequential(*layers)

        # Output heads
        final_hidden = hidden_layers[-1] if hidden_layers else input_size

        # Spectral head (31 wavelengths)
        self.spectral_head = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            self.activation,
            nn.Linear(final_hidden // 2, 31)
        )

        # CIELAB head (5 values: L, a, b, c, h)
        self.cielab_head = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            self.activation,
            nn.Linear(final_hidden // 2, 5)
        )

    def forward(self, x):
        shared = self.shared(x)
        spectral = self.spectral_head(shared)
        cielab = self.cielab_head(shared)
        return spectral, cielab

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training Function
# ============================================================================

def train_model(X_train, Y_train, X_val, Y_val, config, verbose=False):
    """
    Train model with given hyperparameter configuration

    Args:
        config: Dict with hyperparameters

    Returns:
        Dict with trained model and metrics
    """
    # Normalize data
    input_mean = np.mean(X_train, axis=0)
    input_std = np.std(X_train, axis=0)
    output_mean = np.mean(Y_train, axis=0)
    output_std = np.std(Y_train, axis=0)

    X_train_norm = (X_train - input_mean) / (input_std + 1e-8)
    X_val_norm = (X_val - input_mean) / (input_std + 1e-8)
    Y_train_norm = (Y_train - output_mean) / (output_std + 1e-8)
    Y_val_norm = (Y_val - output_mean) / (output_std + 1e-8)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_norm)
    Y_train_tensor = torch.FloatTensor(Y_train_norm)
    X_val_tensor = torch.FloatTensor(X_val_norm)
    Y_val_tensor = torch.FloatTensor(Y_val_norm)

    # Create model
    model = FlexibleNN(
        input_size=X_train.shape[1],
        hidden_layers=config['hidden_layers'],
        activation=config['activation'],
        dropout_rate=config['dropout'],
        use_batchnorm=config['use_batchnorm']
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['l2_lambda']
    )

    # Learning rate scheduler
    if config.get('use_lr_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50
        )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 100)

    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        model.train()

        # Mini-batch training
        indices = torch.randperm(len(X_train_tensor))
        for i in range(0, len(indices), config['batch_size']):
            batch_idx = indices[i:i+config['batch_size']]
            X_batch = X_train_tensor[batch_idx]
            Y_batch = Y_train_tensor[batch_idx]

            optimizer.zero_grad()

            pred_spectral, pred_cielab = model(X_batch)
            pred = torch.cat([pred_spectral, pred_cielab], dim=1)

            # Weighted loss (spectral vs CIELAB)
            loss_spectral = nn.MSELoss()(pred_spectral, Y_batch[:, :31])
            loss_cielab = nn.MSELoss()(pred_cielab, Y_batch[:, 31:])

            loss = config['spectral_weight'] * loss_spectral + config['cielab_weight'] * loss_cielab

            loss.backward()

            # Gradient clipping
            if config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])

            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            pred_spectral_val, pred_cielab_val = model(X_val_tensor)

            loss_spectral_val = nn.MSELoss()(pred_spectral_val, Y_val_tensor[:, :31]).item()
            loss_cielab_val = nn.MSELoss()(pred_cielab_val, Y_val_tensor[:, 31:]).item()

            val_loss = config['spectral_weight'] * loss_spectral_val + config['cielab_weight'] * loss_cielab_val

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        # Learning rate scheduling
        if config.get('use_lr_scheduler', False):
            scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    return {
        'model': model,
        'input_mean': input_mean,
        'input_std': input_std,
        'output_mean': output_mean,
        'output_std': output_std,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
        'parameters': model.count_parameters()
    }


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model_data, X_test, Y_test):
    """Evaluate model on test set"""
    model = model_data['model']

    # Normalize
    X_test_norm = (X_test - model_data['input_mean']) / (model_data['input_std'] + 1e-8)
    X_test_tensor = torch.FloatTensor(X_test_norm)

    # Predict
    model.eval()
    with torch.no_grad():
        pred_spectral, pred_cielab = model(X_test_tensor)
        pred_spectral = pred_spectral.numpy()
        pred_cielab = pred_cielab.numpy()

    # Denormalize
    pred_spectral = pred_spectral * (model_data['output_std'][:31] + 1e-8) + model_data['output_mean'][:31]
    pred_cielab = pred_cielab * (model_data['output_std'][31:] + 1e-8) + model_data['output_mean'][31:]

    # Metrics
    Y_test_spectral = Y_test[:, :31]
    Y_test_cielab = Y_test[:, 31:]

    metrics = {
        'spectral_mae': mean_absolute_error(Y_test_spectral.flatten(), pred_spectral.flatten()),
        'spectral_rmse': np.sqrt(mean_squared_error(Y_test_spectral.flatten(), pred_spectral.flatten())),
        'spectral_r2': r2_score(Y_test_spectral.flatten(), pred_spectral.flatten()),
        'cielab_mae': mean_absolute_error(Y_test_cielab.flatten(), pred_cielab.flatten()),
        'cielab_rmse': np.sqrt(mean_squared_error(Y_test_cielab.flatten(), pred_cielab.flatten())),
        'cielab_r2': r2_score(Y_test_cielab.flatten(), pred_cielab.flatten()),
        'combined_score': 0.0  # Will be computed below
    }

    # Combined score (lower is better)
    # Normalize metrics to 0-1 scale and combine
    metrics['combined_score'] = (
        metrics['spectral_mae'] * 10 +  # Scale to ~0.3
        metrics['cielab_mae'] * 1 +      # Scale to ~0.3
        (1 - metrics['spectral_r2']) +   # Convert R² to error (0-1)
        (1 - max(metrics['cielab_r2'], 0))  # Convert R² to error (handle negative)
    )

    return metrics


# ============================================================================
# Hyperparameter Search
# ============================================================================

def hyperparameter_search(X, Y, search_space, n_trials=50, use_kfold=False, random_seed=42):
    """
    Search for optimal hyperparameters

    Args:
        search_space: Dict defining ranges for each hyperparameter
        n_trials: Number of random configurations to try
        use_kfold: Whether to use k-fold cross-validation
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("=" * 100)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 100)
    print(f"\nSearch space:")
    for key, values in search_space.items():
        print(f"  {key}: {values}")
    print(f"\nNumber of trials: {n_trials}")
    print(f"Cross-validation: {'K-Fold (5)' if use_kfold else 'Single Train/Val/Test Split'}")
    print()

    # Split data
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_seed
    )

    results = []

    for trial in range(n_trials):
        print(f"\n[Trial {trial+1}/{n_trials}]")

        # Sample random configuration
        config = {}
        for key, values in search_space.items():
            if key == 'hidden_layers':
                # Sample architecture
                arch_choice = np.random.choice(len(values))
                config[key] = values[arch_choice]
            else:
                config[key] = np.random.choice(values)

        print(f"  Config: {config['hidden_layers']} layers, "
              f"lr={config['learning_rate']}, batch={config['batch_size']}, "
              f"dropout={config['dropout']}, activation={config['activation']}")

        start_time = time.time()

        if use_kfold:
            # K-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            fold_metrics = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
                X_train_fold = X_temp[train_idx]
                Y_train_fold = Y_temp[train_idx]
                X_val_fold = X_temp[val_idx]
                Y_val_fold = Y_temp[val_idx]

                model_data = train_model(X_train_fold, Y_train_fold, X_val_fold, Y_val_fold, config, verbose=False)
                metrics = evaluate_model(model_data, X_val_fold, Y_val_fold)
                fold_metrics.append(metrics)

            # Average metrics across folds
            avg_metrics = {
                key: np.mean([m[key] for m in fold_metrics])
                for key in fold_metrics[0].keys()
            }

            # Train final model on all training data
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_temp, Y_temp, test_size=0.2, random_state=random_seed
            )
            final_model = train_model(X_train, Y_train, X_val, Y_val, config, verbose=False)
            test_metrics = evaluate_model(final_model, X_test, Y_test)

        else:
            # Simple train/val/test split
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_temp, Y_temp, test_size=0.2, random_state=random_seed
            )

            final_model = train_model(X_train, Y_train, X_val, Y_val, config, verbose=False)
            test_metrics = evaluate_model(final_model, X_test, Y_test)
            avg_metrics = test_metrics

        training_time = time.time() - start_time

        print(f"  Results: Spectral MAE={test_metrics['spectral_mae']:.6f}, "
              f"CIELAB MAE={test_metrics['cielab_mae']:.6f}, "
              f"Combined Score={test_metrics['combined_score']:.6f}")
        print(f"  Training time: {training_time:.1f}s, Parameters: {final_model['parameters']}")

        results.append({
            'config': config,
            'test_metrics': test_metrics,
            'avg_metrics': avg_metrics,
            'training_time': training_time,
            'parameters': final_model['parameters'],
            'model_data': final_model if not use_kfold else None  # Save best model
        })

    return results, X_test, Y_test


# ============================================================================
# Main Optimization
# ============================================================================

if __name__ == "__main__":
    RANDOM_SEED = 42

    # Load data
    print("Loading dataset...")
    samples, _, _ = load_new_dataset(
        '../public/Concentrations.csv',
        '../public/Spectra.csv'
    )

    X, Y = prepare_training_data(samples)
    print(f"✓ Dataset loaded: {X.shape[0]} samples\n")

    # Define search space
    search_space = {
        'hidden_layers': [
            [32],           # 1 layer, small
            [64],           # 1 layer, medium
            [128],          # 1 layer, large
            [64, 32],       # 2 layers, decreasing
            [64, 64],       # 2 layers, constant
            [128, 64],      # 2 layers, large to medium
            [64, 64, 32],   # 3 layers, decreasing
            [64, 64, 64],   # 3 layers, constant
            [128, 64, 32],  # 3 layers, large to small
            [64, 128, 64],  # 3 layers, bottleneck
            [128, 128, 64], # 3 layers, wide
            [64, 64, 64, 32],   # 4 layers
            [128, 128, 64, 32], # 4 layers, wide
        ],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.002, 0.005],
        'batch_size': [8, 16, 32],
        'dropout': [0.0, 0.1, 0.2, 0.3],
        'l2_lambda': [0.0, 0.0001, 0.001, 0.01],
        'activation': ['relu', 'leaky_relu', 'elu'],
        'use_batchnorm': [True, False],
        'spectral_weight': [1.0, 1.5, 2.0],
        'cielab_weight': [1.0, 0.5, 0.25],
        'use_lr_scheduler': [True, False],
        'gradient_clip': [0, 1.0, 5.0],
        'epochs': [2000],
        'patience': [100]
    }

    # Run optimization
    results, X_test, Y_test = hyperparameter_search(
        X, Y,
        search_space,
        n_trials=50,  # Test 50 random configurations
        use_kfold=False,
        random_seed=RANDOM_SEED
    )

    # Sort by combined score (lower is better)
    results.sort(key=lambda x: x['test_metrics']['combined_score'])

    # Print top 10 results
    print("\n" + "=" * 100)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 100)

    for i, result in enumerate(results[:10], 1):
        print(f"\n{i}. Combined Score: {result['test_metrics']['combined_score']:.6f}")
        print(f"   Architecture: {result['config']['hidden_layers']}")
        print(f"   Learning Rate: {result['config']['learning_rate']}")
        print(f"   Batch Size: {result['config']['batch_size']}")
        print(f"   Dropout: {result['config']['dropout']}")
        print(f"   L2: {result['config']['l2_lambda']}")
        print(f"   Activation: {result['config']['activation']}")
        print(f"   BatchNorm: {result['config']['use_batchnorm']}")
        print(f"   Spectral MAE: {result['test_metrics']['spectral_mae']:.6f}")
        print(f"   CIELAB MAE: {result['test_metrics']['cielab_mae']:.6f}")
        print(f"   Spectral R²: {result['test_metrics']['spectral_r2']:.6f}")
        print(f"   CIELAB R²: {result['test_metrics']['cielab_r2']:.6f}")
        print(f"   Parameters: {result['parameters']}")
        print(f"   Training Time: {result['training_time']:.1f}s")

    # Save best configuration
    best_result = results[0]

    os.makedirs('results/optimization', exist_ok=True)

    # Save configuration
    with open('results/optimization/best_config.json', 'w') as f:
        json.dump({
            'config': best_result['config'],
            'test_metrics': best_result['test_metrics'],
            'parameters': best_result['parameters'],
            'training_time': best_result['training_time']
        }, f, indent=2)

    print("\n" + "=" * 100)
    print("OPTIMIZATION COMPLETE")
    print("=" * 100)
    print(f"\nBest configuration saved to: results/optimization/best_config.json")
    print(f"\nBest model performance:")
    print(f"  Spectral MAE: {best_result['test_metrics']['spectral_mae']:.6f}")
    print(f"  CIELAB MAE: {best_result['test_metrics']['cielab_mae']:.6f}")
    print(f"  Combined Score: {best_result['test_metrics']['combined_score']:.6f}")

    # Save best model
    if best_result['model_data'] is not None:
        import pickle
        with open('trained_models/optimized_model.pkl', 'wb') as f:
            pickle.dump(best_result['model_data'], f)
        print(f"\nBest model saved to: trained_models/optimized_model.pkl")

    print()
