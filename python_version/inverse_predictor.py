"""
Inverse Prediction: Optimize concentrations to match target reflectance spectrum

Uses gradient-based optimization with the forward neural network model to find
reagent concentrations that produce a target reflectance spectrum.

This is the inverse problem: Spectrum → Concentrations
(vs. the forward problem: Concentrations → Spectrum)
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize, differential_evolution
import pickle
from typing import Dict, List, Tuple, Optional
from optimize_hyperparameters import FlexibleNN


class InversePredictor:
    """
    Optimize reagent concentrations to match a target reflectance spectrum

    Uses the forward neural network model with optimization to solve the
    inverse problem: given a target spectrum, find the concentrations.
    """

    def __init__(self, model_path='trained_models/optimized_best_model.pkl'):
        """Load the forward model for inverse prediction"""

        print(f"Loading forward model from {model_path}...")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Handle old model format (model object stored)
        if 'model' in model_data:
            self.model = model_data['model']
            self.input_mean = model_data['input_mean']
            self.input_std = model_data['input_std']

            output_mean = model_data['output_mean']
            output_std = model_data['output_std']
            self.spectral_mean = output_mean[:31]
            self.spectral_std = output_std[:31]

        else:
            # New model format (state dict stored)
            hidden_layers = model_data['hidden_layers']
            input_size = model_data.get('input_size', 5)

            self.model = FlexibleNN(
                input_size=input_size,
                hidden_layers=hidden_layers,
                activation='relu',
                dropout_rate=0.1,
                use_batchnorm=True
            )
            self.model.load_state_dict(model_data['model_state'])

            self.input_mean = model_data['input_mean']
            self.input_std = model_data['input_std']
            self.spectral_mean = model_data['spectral_mean']
            self.spectral_std = model_data['spectral_std']

        self.model.eval()

        # Convert to torch tensors for faster computation
        self.input_mean_t = torch.FloatTensor(self.input_mean)
        self.input_std_t = torch.FloatTensor(self.input_std)
        self.spectral_mean_t = torch.FloatTensor(self.spectral_mean)
        self.spectral_std_t = torch.FloatTensor(self.spectral_std)

        print(f"✓ Forward model loaded for inverse prediction")

    def forward_predict_spectrum(self, concentrations: np.ndarray) -> np.ndarray:
        """
        Predict spectrum from concentrations using forward model

        Args:
            concentrations: [GXT, BiVaO4, PG, PearlB, thickness]

        Returns:
            Predicted reflectance spectrum (31 values)
        """
        # Normalize input
        x_norm = (concentrations - self.input_mean) / self.input_std

        # Predict
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_norm).unsqueeze(0)
            output = self.model(x_tensor)

            # Handle tuple output (model may return multiple outputs)
            if isinstance(output, tuple):
                output = output[0]

            # Extract spectral output (first 31 values)
            spectral_norm = output[0, :31].numpy()

        # Denormalize
        spectrum = spectral_norm * self.spectral_std + self.spectral_mean

        return spectrum

    def objective_function(
        self,
        concentrations: np.ndarray,
        target_spectrum: np.ndarray,
        regularization: float = 0.0
    ) -> float:
        """
        Objective function to minimize: error between predicted and target spectrum

        Args:
            concentrations: [GXT, BiVaO4, PG, PearlB, thickness]
            target_spectrum: Target reflectance (31 values)
            regularization: L2 regularization weight (encourages lower concentrations)

        Returns:
            Mean squared error + regularization penalty
        """
        # Predict spectrum
        predicted = self.forward_predict_spectrum(concentrations)

        # Calculate MSE
        mse = np.mean((predicted - target_spectrum) ** 2)

        # Add L2 regularization on concentrations (encourages simpler formulations)
        if regularization > 0:
            reg_penalty = regularization * np.sum(concentrations[:4] ** 2)  # Don't regularize thickness
            return mse + reg_penalty

        return mse

    def optimize_concentrations(
        self,
        target_spectrum: np.ndarray,
        thickness: float = 8.0,
        initial_guess: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = 'L-BFGS-B',
        regularization: float = 0.0,
        max_iterations: int = 1000
    ) -> Dict:
        """
        Optimize concentrations to match target spectrum

        Args:
            target_spectrum: Target reflectance (31 values, 400-700nm)
            thickness: Film thickness (8 or 12 μm) - can be fixed or optimized
            initial_guess: Starting concentrations (dict with GXT, BiVaO4, PG, PearlB)
            bounds: Concentration bounds (dict with reagent names)
            method: Optimization method ('L-BFGS-B', 'TNC', 'SLSQP')
            regularization: L2 regularization weight
            max_iterations: Maximum optimization iterations

        Returns:
            Dictionary with optimized concentrations, predicted spectrum, and metrics
        """

        # Default bounds (based on training data ranges)
        if bounds is None:
            bounds = {
                'GXT': (0.0, 25.0),
                'BiVaO4': (0.0, 20.0),
                'PG': (0.0, 2.5),
                'PearlB': (0.0, 15.0),
                'thickness': (thickness, thickness)  # Fix thickness by default
            }
        else:
            # Ensure thickness is in bounds
            if 'thickness' not in bounds:
                bounds['thickness'] = (thickness, thickness)

        # Default initial guess (middle of ranges)
        if initial_guess is None:
            initial_guess = {
                'GXT': 10.0,
                'BiVaO4': 5.0,
                'PG': 1.0,
                'PearlB': 2.0,
                'thickness': thickness
            }
        else:
            initial_guess['thickness'] = thickness

        # Convert to array format
        x0 = np.array([
            initial_guess['GXT'],
            initial_guess['BiVaO4'],
            initial_guess['PG'],
            initial_guess['PearlB'],
            initial_guess['thickness']
        ])

        # Bounds as list of tuples
        bounds_list = [
            bounds['GXT'],
            bounds['BiVaO4'],
            bounds['PG'],
            bounds['PearlB'],
            bounds['thickness']
        ]

        print(f"\nOptimizing concentrations to match target spectrum...")
        print(f"Method: {method}, Regularization: {regularization}")
        print(f"Initial guess: GXT={x0[0]:.1f}%, BiVaO4={x0[1]:.1f}%, PG={x0[2]:.2f}%, PearlB={x0[3]:.2f}%")

        # Optimize
        result = minimize(
            fun=self.objective_function,
            x0=x0,
            args=(target_spectrum, regularization),
            method=method,
            bounds=bounds_list,
            options={'maxiter': max_iterations}
        )

        # Extract optimized concentrations
        optimized = result.x

        # Predict spectrum with optimized concentrations
        predicted_spectrum = self.forward_predict_spectrum(optimized)

        # Calculate metrics
        mse = np.mean((predicted_spectrum - target_spectrum) ** 2)
        mae = np.mean(np.abs(predicted_spectrum - target_spectrum))
        max_error = np.max(np.abs(predicted_spectrum - target_spectrum))
        r2 = 1 - (np.sum((target_spectrum - predicted_spectrum) ** 2) /
                  np.sum((target_spectrum - target_spectrum.mean()) ** 2))

        print(f"\n{'='*70}")
        print(f"OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        print(f"Converged: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Final error (MSE): {mse:.6f}")
        print()
        print(f"Optimized concentrations:")
        print(f"  GXT:    {optimized[0]:6.2f}%")
        print(f"  BiVaO4: {optimized[1]:6.2f}%")
        print(f"  PG:     {optimized[2]:6.2f}%")
        print(f"  PearlB: {optimized[3]:6.2f}%")
        print(f"  Thickness: {optimized[4]:.1f} μm")
        print()
        print(f"Match quality:")
        print(f"  MSE:       {mse:.6f}")
        print(f"  MAE:       {mae:.6f}")
        print(f"  Max error: {max_error:.6f}")
        print(f"  R²:        {r2:.6f}")
        print(f"{'='*70}")

        return {
            'concentrations': {
                'GXT': float(optimized[0]),
                'BiVaO4': float(optimized[1]),
                'PG': float(optimized[2]),
                'PearlB': float(optimized[3]),
                'thickness': float(optimized[4])
            },
            'predicted_spectrum': predicted_spectrum.tolist(),
            'target_spectrum': target_spectrum.tolist(),
            'metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'max_error': float(max_error),
                'r2': float(r2)
            },
            'optimization': {
                'success': result.success,
                'iterations': result.nit,
                'final_objective': float(result.fun),
                'message': result.message
            }
        }

    def global_optimize(
        self,
        target_spectrum: np.ndarray,
        thickness: float = 8.0,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        max_iterations: int = 100
    ) -> Dict:
        """
        Global optimization using differential evolution

        More thorough but slower than local optimization.
        Good for finding global minimum when local methods get stuck.

        Args:
            target_spectrum: Target reflectance (31 values)
            thickness: Film thickness
            bounds: Concentration bounds
            max_iterations: Maximum iterations (population generations)

        Returns:
            Dictionary with optimized concentrations and metrics
        """

        # Default bounds
        if bounds is None:
            bounds = {
                'GXT': (0.0, 25.0),
                'BiVaO4': (0.0, 20.0),
                'PG': (0.0, 2.5),
                'PearlB': (0.0, 15.0),
                'thickness': (thickness, thickness)
            }
        else:
            if 'thickness' not in bounds:
                bounds['thickness'] = (thickness, thickness)

        bounds_list = [
            bounds['GXT'],
            bounds['BiVaO4'],
            bounds['PG'],
            bounds['PearlB'],
            bounds['thickness']
        ]

        print(f"\nGlobal optimization using differential evolution...")
        print(f"Max iterations: {max_iterations}")

        # Global optimization
        result = differential_evolution(
            func=self.objective_function,
            bounds=bounds_list,
            args=(target_spectrum, 0.0),
            maxiter=max_iterations,
            seed=42,
            workers=1
        )

        # Extract and evaluate result
        optimized = result.x
        predicted_spectrum = self.forward_predict_spectrum(optimized)

        # Calculate metrics
        mse = np.mean((predicted_spectrum - target_spectrum) ** 2)
        mae = np.mean(np.abs(predicted_spectrum - target_spectrum))
        max_error = np.max(np.abs(predicted_spectrum - target_spectrum))
        r2 = 1 - (np.sum((target_spectrum - predicted_spectrum) ** 2) /
                  np.sum((target_spectrum - target_spectrum.mean()) ** 2))

        print(f"\n{'='*70}")
        print(f"GLOBAL OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        print(f"Converged: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Final error (MSE): {mse:.6f}")
        print()
        print(f"Optimized concentrations:")
        print(f"  GXT:    {optimized[0]:6.2f}%")
        print(f"  BiVaO4: {optimized[1]:6.2f}%")
        print(f"  PG:     {optimized[2]:6.2f}%")
        print(f"  PearlB: {optimized[3]:6.2f}%")
        print(f"  Thickness: {optimized[4]:.1f} μm")
        print()
        print(f"Match quality:")
        print(f"  MSE:       {mse:.6f}")
        print(f"  MAE:       {mae:.6f}")
        print(f"  Max error: {max_error:.6f}")
        print(f"  R²:        {r2:.6f}")
        print(f"{'='*70}")

        return {
            'concentrations': {
                'GXT': float(optimized[0]),
                'BiVaO4': float(optimized[1]),
                'PG': float(optimized[2]),
                'PearlB': float(optimized[3]),
                'thickness': float(optimized[4])
            },
            'predicted_spectrum': predicted_spectrum.tolist(),
            'target_spectrum': target_spectrum.tolist(),
            'metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'max_error': float(max_error),
                'r2': float(r2)
            },
            'optimization': {
                'success': result.success,
                'iterations': result.nit,
                'final_objective': float(result.fun),
                'message': result.message
            }
        }


if __name__ == "__main__":
    """Test inverse prediction with known samples"""

    print("="*70)
    print("INVERSE PREDICTION TEST")
    print("="*70)

    # Load predictor
    predictor = InversePredictor()

    # Test 1: Use a known training sample as target
    print("\n" + "="*70)
    print("TEST 1: Known Training Sample (T22: 20% GXT, 5% BiVaO4, 1.5% PG)")
    print("="*70)

    # Create target spectrum from known formulation
    known_concentrations = np.array([20.0, 5.0, 1.5, 0.0, 8.0])
    target_spectrum = predictor.forward_predict_spectrum(known_concentrations)

    print(f"\nTarget formulation:")
    print(f"  GXT:    20.0%")
    print(f"  BiVaO4:  5.0%")
    print(f"  PG:      1.5%")
    print(f"  PearlB:  0.0%")
    print(f"  Thickness: 8.0 μm")

    # Optimize to recover concentrations
    result = predictor.optimize_concentrations(
        target_spectrum=target_spectrum,
        thickness=8.0,
        initial_guess={'GXT': 10.0, 'BiVaO4': 2.0, 'PG': 0.5, 'PearlB': 0.0}
    )

    # Test 2: Synthetic target spectrum
    print("\n" + "="*70)
    print("TEST 2: Synthetic Target Spectrum")
    print("="*70)

    # Create a synthetic smooth spectrum
    wavelengths = np.arange(400, 710, 10)
    synthetic_spectrum = 0.3 + 0.5 * np.exp(-((wavelengths - 550) ** 2) / (2 * 50 ** 2))

    print(f"\nSynthetic target: Gaussian peak at 550nm")
    print(f"Reflectance range: {synthetic_spectrum.min():.3f} - {synthetic_spectrum.max():.3f}")

    result2 = predictor.optimize_concentrations(
        target_spectrum=synthetic_spectrum,
        thickness=8.0
    )

    # Test 3: Global optimization
    print("\n" + "="*70)
    print("TEST 3: Global Optimization (Differential Evolution)")
    print("="*70)

    result3 = predictor.global_optimize(
        target_spectrum=target_spectrum,
        thickness=8.0,
        max_iterations=50
    )

    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
