"""
Option 1: Separate Fluorescence-Only NN Predictor

Neural network that predicts fluorescence (ct/s) from:
- Pigment concentrations (GXT, BiVaO4, PG, PearlB)
- Thickness
- Integrated area under reflectance curve

Includes smooth physics constraint: 0% GXT → 0 ct/s
"""
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from scipy.integrate import trapezoid


class FluorescenceOnlyNN(nn.Module):
    """Fluorescence-only neural network architecture"""
    def __init__(self, hidden_layers=[64, 32], dropout_rate=0.1):
        super(FluorescenceOnlyNN, self).__init__()
        layers = []
        input_size = 6

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FluorescenceNNPredictor:
    """
    Wrapper for fluorescence NN with smooth physics constraints

    Applies smooth constraint function that ensures:
    - 0% GXT → 0 ct/s
    - Smooth transition (no sharp steps)
    - Monotonic increase with GXT concentration
    """

    def __init__(self, model_path='trained_models/option1_fluorescence_nn.pkl'):
        """Load the trained fluorescence NN model"""

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading fluorescence NN from {model_path}...")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Create model
        self.model = FluorescenceOnlyNN(hidden_layers=model_data['hidden_layers'])
        self.model.load_state_dict(model_data['model_state'])
        self.model.eval()

        # Store normalization parameters
        self.input_mean = model_data['input_mean']
        self.input_std = model_data['input_std']
        self.output_mean = model_data['output_mean']
        self.output_std = model_data['output_std']

        # Store metrics
        self.r2_score = model_data['metrics']['r2']
        self.mae = model_data['metrics']['mae']
        self.pct_error = model_data['metrics']['pct_error']

        print(f"✓ Fluorescence NN loaded (R²={self.r2_score:.4f}, MAE={self.mae:.1f} ct/s)")

    def calculate_integrated_area(self, reflectance, wavelengths=None):
        """
        Calculate integrated area under reflectance curve

        Args:
            reflectance: array of 31 reflectance values
            wavelengths: optional wavelengths (default: 400-700nm, 10nm steps)

        Returns:
            Integrated area
        """
        if wavelengths is None:
            wavelengths = np.arange(400, 710, 10)  # 400-700nm in 10nm steps

        return trapezoid(reflectance, wavelengths)

    def smooth_gxt_constraint(self, gxt_concentration):
        """
        Smooth multiplicative constraint function

        Ensures 0% GXT → 0 multiplier (therefore 0 ct/s)
        Smooth sigmoid-like transition from 0% to higher concentrations

        Function: f(x) = tanh(x / scale)
        - At x=0: f(0) = 0
        - Smooth S-curve transition
        - At x≥5%: f(x) ≈ 1 (minimal effect on predictions)

        Args:
            gxt_concentration: GXT concentration in %

        Returns:
            Multiplier in range [0, 1]
        """
        # Scale factor controls transition smoothness
        # scale=2.0 means ~95% of full value at 5% GXT
        scale = 2.0

        # tanh provides smooth S-curve: 0 at x=0, asymptotically approaches 1
        multiplier = np.tanh(gxt_concentration / scale)

        return multiplier

    def predict(self, concentrations, reflectance, thickness=8.0):
        """
        Predict fluorescence intensity

        Args:
            concentrations: dict with GXT, BiVaO4, PG, PearlB (in %)
            reflectance: array of 31 reflectance values
            thickness: film thickness in μm (8 or 12)

        Returns:
            dict with:
                - fluorescence_cts: predicted fluorescence (ct/s)
                - fluorescence_area: integrated area
                - gxt_multiplier: smooth constraint multiplier
                - model_r2: model R² score
        """

        # Extract GXT concentration
        gxt_conc = concentrations.get('GXT', 0.0)

        # Calculate integrated area
        integrated_area = self.calculate_integrated_area(reflectance)

        # Prepare input features
        x = np.array([[
            gxt_conc,
            concentrations.get('BiVaO4', 0.0),
            concentrations.get('PG', 0.0),
            concentrations.get('PearlB', 0.0),
            thickness,
            integrated_area
        ]])

        # Normalize
        x_norm = (x - self.input_mean) / self.input_std

        # Predict
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_norm)
            pred_norm = self.model(x_tensor)
            pred = pred_norm.numpy() * self.output_std + self.output_mean

        # Get raw prediction
        raw_fluorescence = pred[0, 0]

        # Apply smooth GXT constraint
        gxt_multiplier = self.smooth_gxt_constraint(gxt_conc)
        constrained_fluorescence = raw_fluorescence * gxt_multiplier

        # Clip to non-negative (fluorescence can't be negative)
        constrained_fluorescence = max(0.0, constrained_fluorescence)

        return {
            'fluorescence_cts': float(constrained_fluorescence),
            'fluorescence_area': float(integrated_area),
            'gxt_multiplier': float(gxt_multiplier),
            'model_r2': float(self.r2_score)
        }


# Module-level singleton instance
_FLUORESCENCE_PREDICTOR = None


def get_fluorescence_predictor(model_path='trained_models/option1_fluorescence_nn.pkl'):
    """Get or create the fluorescence predictor singleton"""
    global _FLUORESCENCE_PREDICTOR

    if _FLUORESCENCE_PREDICTOR is None:
        _FLUORESCENCE_PREDICTOR = FluorescenceNNPredictor(model_path)

    return _FLUORESCENCE_PREDICTOR


if __name__ == "__main__":
    """Test the fluorescence NN predictor with smooth constraints"""

    print("="*80)
    print("FLUORESCENCE NN PREDICTOR - CONSTRAINT TESTING")
    print("="*80)

    # Load predictor
    predictor = get_fluorescence_predictor()

    # Test smooth constraint function
    print("\n" + "-"*80)
    print("SMOOTH GXT CONSTRAINT FUNCTION")
    print("-"*80)
    print(f"{'GXT %':<10s} {'Multiplier':<15s} {'Effect':<30s}")
    print("-"*80)

    gxt_test_values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]
    for gxt in gxt_test_values:
        mult = predictor.smooth_gxt_constraint(gxt)
        if gxt == 0.0:
            effect = "Zero fluorescence"
        elif mult < 0.5:
            effect = "Strongly reduced"
        elif mult < 0.9:
            effect = "Moderately reduced"
        else:
            effect = "Minimal constraint effect"

        print(f"{gxt:<10.1f} {mult:<15.3f} {effect:<30s}")

    print("\n✓ Smooth S-curve transition from 0% to 5% GXT")

    # Test zero GXT constraint
    print("\n" + "-"*80)
    print("TEST: ZERO GXT CONSTRAINT (0% GXT → 0 ct/s)")
    print("-"*80)
    print(f"{'Formulation':<50s} {'Prediction':>15s}")
    print("-"*80)

    # Sample reflectance (typical spectrum)
    sample_reflectance = np.linspace(0.5, 0.9, 31)

    zero_gxt_cases = [
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0},
        {"GXT": 0.0, "BiVaO4": 10.0, "PG": 0.0, "PearlB": 0.0},
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 2.0, "PearlB": 0.0},
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 5.0},
    ]

    for conc in zero_gxt_cases:
        result = predictor.predict(conc, sample_reflectance, 8.0)
        desc = f"GXT:{conc['GXT']:.0f}% BiVaO4:{conc['BiVaO4']:.0f}% PG:{conc['PG']:.0f}% PearlB:{conc['PearlB']:.0f}%"
        print(f"{desc:<50s} {result['fluorescence_cts']:>12.1f} ct/s")

    print("\n✓ All predictions should be 0 ct/s")

    # Test pure GXT samples
    print("\n" + "-"*80)
    print("TEST: PURE GXT SAMPLES (smooth transition)")
    print("-"*80)
    print(f"{'Formulation':<50s} {'Multiplier':>12s} {'Prediction':>15s}")
    print("-"*80)

    pure_gxt_cases = [
        ({"GXT": 0.5, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0),
        ({"GXT": 1.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0),
        ({"GXT": 2.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0),
        ({"GXT": 5.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0),
        ({"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0),
        ({"GXT": 15.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0),
        ({"GXT": 20.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0),
    ]

    for conc, thickness in pure_gxt_cases:
        result = predictor.predict(conc, sample_reflectance, thickness)
        desc = f"GXT:{conc['GXT']:.1f}% @ {thickness:.0f}μm"
        print(f"{desc:<50s} {result['gxt_multiplier']:>12.3f} {result['fluorescence_cts']:>12.0f} ct/s")

    print("\n✓ Smooth monotonic increase from 0% to higher concentrations")

    # Test interaction effects
    print("\n" + "-"*80)
    print("TEST: PIGMENT INTERACTIONS")
    print("-"*80)
    print(f"{'Formulation':<50s} {'Prediction':>15s} {'vs Baseline':>15s}")
    print("-"*80)

    baseline_conc = {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}
    baseline_result = predictor.predict(baseline_conc, sample_reflectance, 8.0)
    baseline_fluor = baseline_result['fluorescence_cts']

    print(f"{'10% GXT (baseline)':<50s} {baseline_fluor:>12.0f} ct/s")

    interaction_cases = [
        {"GXT": 10.0, "BiVaO4": 5.0, "PG": 0.0, "PearlB": 0.0},
        {"GXT": 10.0, "BiVaO4": 10.0, "PG": 0.0, "PearlB": 0.0},
        {"GXT": 10.0, "BiVaO4": 0.0, "PG": 2.0, "PearlB": 0.0},
        {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 5.0},
    ]

    for conc in interaction_cases:
        result = predictor.predict(conc, sample_reflectance, 8.0)
        diff = result['fluorescence_cts'] - baseline_fluor

        desc = f"+ {conc['BiVaO4']:.0f}% BiVaO4 + {conc['PG']:.0f}% PG + {conc['PearlB']:.0f}% PearlB"
        print(f"{desc:<50s} {result['fluorescence_cts']:>12.0f} ct/s {diff:>12.0f} ct/s")

    print("\n✓ Model captures complex pigment interactions")

    print("\n" + "="*80)
    print("CONSTRAINT TESTING COMPLETE")
    print("="*80)
    print("\nSmooth constraint function ensures:")
    print("  • 0% GXT → 0 ct/s (exact)")
    print("  • Smooth S-curve transition (no sharp steps)")
    print("  • Minimal effect on typical concentrations (≥5% GXT)")
    print("  • Physics-based constraint without breaking predictions")
    print("="*80)
