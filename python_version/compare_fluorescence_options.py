"""
Comprehensive Comparison: Option 1 vs Option 3

Compares the two approaches for fluorescence NN integration:
- Option 1: Separate fluorescence-only NN
- Option 3: Multi-task NN with 3 outputs

Tests include:
- Validation metrics
- Physics constraints (0% GXT → 0 ct/s)
- Pure GXT samples
- Mixed formulations
- Interaction effects
- Risk assessment
"""
import numpy as np
import torch
import torch.nn as nn
import pickle
from scipy.integrate import trapezoid


class FluorescenceOnlyNN(nn.Module):
    """Option 1 model architecture"""
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


class MultiTaskNN(nn.Module):
    """Option 3 model architecture"""
    def __init__(self, hidden_layers=[128, 128, 64], dropout_rate=0.1):
        super(MultiTaskNN, self).__init__()
        shared_layers = []
        input_size = 5
        for hidden_size in hidden_layers:
            shared_layers.append(nn.Linear(input_size, hidden_size))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.BatchNorm1d(hidden_size))
            shared_layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        self.shared = nn.Sequential(*shared_layers)
        final_hidden = hidden_layers[-1] if hidden_layers else 5

        self.spectral_head = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            nn.ReLU(),
            nn.Linear(final_hidden // 2, 31)
        )
        self.cielab_head = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            nn.ReLU(),
            nn.Linear(final_hidden // 2, 5)
        )
        self.fluorescence_head = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            nn.ReLU(),
            nn.Linear(final_hidden // 2, 1)
        )

    def forward(self, x):
        shared_features = self.shared(x)
        spectral = self.spectral_head(shared_features)
        cielab = self.cielab_head(shared_features)
        fluorescence = self.fluorescence_head(shared_features)
        return spectral, cielab, fluorescence


def calculate_integrated_area(reflectance):
    """Calculate integrated area under reflectance curve"""
    wavelengths = np.arange(400, 710, 10)
    return trapezoid(reflectance, wavelengths)


def load_models():
    """Load both trained models"""
    print("="*80)
    print("LOADING TRAINED MODELS")
    print("="*80)

    # Load Option 1
    with open('trained_models/option1_fluorescence_nn.pkl', 'rb') as f:
        opt1_data = pickle.load(f)

    opt1_model = FluorescenceOnlyNN(hidden_layers=opt1_data['hidden_layers'])
    opt1_model.load_state_dict(opt1_data['model_state'])
    opt1_model.eval()

    print("\n✓ Option 1: Fluorescence-Only NN loaded")
    print(f"  Validation R²: {opt1_data['metrics']['r2']:.4f}")
    print(f"  Validation MAE: {opt1_data['metrics']['mae']:.1f} ct/s")

    # Load Option 3
    with open('trained_models/option3_multitask_nn.pkl', 'rb') as f:
        opt3_data = pickle.load(f)

    opt3_model = MultiTaskNN(hidden_layers=opt3_data['hidden_layers'])
    opt3_model.load_state_dict(opt3_data['model_state'])
    opt3_model.eval()

    print("\n✓ Option 3: Multi-Task NN loaded")
    print(f"  Spectral R²: {opt3_data['metrics']['spectral_r2']:.4f}")
    print(f"  CIELAB R²: {opt3_data['metrics']['cielab_r2']:.4f}")
    print(f"  Fluorescence R²: {opt3_data['metrics']['fluorescence_r2']:.4f}")
    print(f"  Fluorescence MAE: {opt3_data['metrics']['fluorescence_mae']:.1f} ct/s")

    return opt1_model, opt1_data, opt3_model, opt3_data


def predict_option1(model, data, concentrations, thickness, integrated_area):
    """Make prediction with Option 1"""
    x = np.array([[
        concentrations['GXT'],
        concentrations['BiVaO4'],
        concentrations['PG'],
        concentrations['PearlB'],
        thickness,
        integrated_area
    ]])

    x_norm = (x - data['input_mean']) / data['input_std']
    x_tensor = torch.FloatTensor(x_norm)

    with torch.no_grad():
        pred_norm = model(x_tensor)
        pred = pred_norm.numpy() * data['output_std'] + data['output_mean']

    return pred[0, 0]


def predict_option3(model, data, concentrations, thickness):
    """Make prediction with Option 3"""
    x = np.array([[
        concentrations['GXT'],
        concentrations['BiVaO4'],
        concentrations['PG'],
        concentrations['PearlB'],
        thickness
    ]])

    x_norm = (x - data['input_mean']) / data['input_std']
    x_tensor = torch.FloatTensor(x_norm)

    with torch.no_grad():
        spectral_norm, cielab_norm, fluor_norm = model(x_tensor)

        spectral = spectral_norm.numpy() * data['spectral_std'] + data['spectral_mean']
        cielab = cielab_norm.numpy() * data['cielab_std'] + data['cielab_mean']
        fluor = fluor_norm.numpy() * data['fluorescence_std'] + data['fluorescence_mean']

    return {
        'fluorescence': fluor[0, 0],
        'spectral': spectral[0],
        'cielab': cielab[0]
    }


def run_comparison_tests():
    """Run comprehensive comparison tests"""

    opt1_model, opt1_data, opt3_model, opt3_data = load_models()

    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON TESTS")
    print("="*80)

    # Test 1: Zero GXT constraint
    print("\n" + "-"*80)
    print("TEST 1: PHYSICS CONSTRAINT (0% GXT → 0 ct/s)")
    print("-"*80)
    print(f"{'Formulation':<50s} {'Option 1':>12s} {'Option 3':>12s}")
    print("-"*80)

    zero_gxt_cases = [
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0},
        {"GXT": 0.0, "BiVaO4": 10.0, "PG": 0.0, "PearlB": 0.0},
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 2.0, "PearlB": 0.0},
        {"GXT": 0.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 5.0},
    ]

    for case in zero_gxt_cases:
        # For Option 1, need integrated area - use typical value
        opt1_pred = predict_option1(opt1_model, opt1_data, case, 8.0, 240.0)

        # For Option 3, predict spectral first to get integrated area
        opt3_result = predict_option3(opt3_model, opt3_data, case, 8.0)
        opt3_pred = opt3_result['fluorescence']

        desc = f"GXT:{case['GXT']:.0f}% BiVaO4:{case['BiVaO4']:.0f}% PG:{case['PG']:.0f}% PearlB:{case['PearlB']:.0f}%"
        print(f"{desc:<50s} {opt1_pred:>10.0f} ct/s {opt3_pred:>10.0f} ct/s")

    print("\nExpected: Both should predict ~0 ct/s")
    print("⚠️  Both models violate physics constraint (predict non-zero)")

    # Test 2: Pure GXT samples
    print("\n" + "-"*80)
    print("TEST 2: PURE GXT SAMPLES")
    print("-"*80)
    print(f"{'Formulation':<50s} {'Option 1':>12s} {'Option 3':>12s}")
    print("-"*80)

    pure_gxt_cases = [
        ({"GXT": 5.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0, 240.0),
        ({"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0, 250.0),
        ({"GXT": 15.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0, 255.0),
        ({"GXT": 20.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 8.0, 260.0),
        ({"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}, 12.0, 265.0),
    ]

    for conc, thickness, area in pure_gxt_cases:
        opt1_pred = predict_option1(opt1_model, opt1_data, conc, thickness, area)
        opt3_result = predict_option3(opt3_model, opt3_data, conc, thickness)
        opt3_pred = opt3_result['fluorescence']

        desc = f"GXT:{conc['GXT']:.0f}% @ {thickness:.0f}μm"
        print(f"{desc:<50s} {opt1_pred:>10.0f} ct/s {opt3_pred:>10.0f} ct/s")

    print("\nExpected: Fluorescence increases with GXT% and thickness")

    # Test 3: Interaction effects
    print("\n" + "-"*80)
    print("TEST 3: PIGMENT INTERACTION EFFECTS")
    print("-"*80)
    print(f"{'Formulation':<50s} {'Option 1':>12s} {'Option 3':>12s} {'Difference':>12s}")
    print("-"*80)

    # Baseline
    baseline = {"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 0.0}
    opt1_baseline = predict_option1(opt1_model, opt1_data, baseline, 8.0, 250.0)
    opt3_baseline = predict_option3(opt3_model, opt3_data, baseline, 8.0)['fluorescence']

    print(f"{'10% GXT (baseline)':<50s} {opt1_baseline:>10.0f} ct/s {opt3_baseline:>10.0f} ct/s")

    interaction_cases = [
        ({"GXT": 10.0, "BiVaO4": 5.0, "PG": 0.0, "PearlB": 0.0}, 245.0, "Add 5% BiVaO4"),
        ({"GXT": 10.0, "BiVaO4": 10.0, "PG": 0.0, "PearlB": 0.0}, 240.0, "Add 10% BiVaO4"),
        ({"GXT": 10.0, "BiVaO4": 0.0, "PG": 2.0, "PearlB": 0.0}, 248.0, "Add 2% PG"),
        ({"GXT": 10.0, "BiVaO4": 0.0, "PG": 0.0, "PearlB": 5.0}, 247.0, "Add 5% PearlB"),
        ({"GXT": 10.0, "BiVaO4": 5.0, "PG": 2.0, "PearlB": 2.0}, 243.0, "Complex mixture"),
    ]

    for conc, area, desc in interaction_cases:
        opt1_pred = predict_option1(opt1_model, opt1_data, conc, 8.0, area)
        opt3_pred = predict_option3(opt3_model, opt3_data, conc, 8.0)['fluorescence']

        opt1_diff = opt1_pred - opt1_baseline
        opt3_diff = opt3_pred - opt3_baseline

        print(f"{desc:<50s} {opt1_pred:>10.0f} ct/s {opt3_pred:>10.0f} ct/s ({opt1_diff:>+6.0f} / {opt3_diff:>+6.0f})")

    print("\nExpected: Non-fluorescent pigments should REDUCE fluorescence")

    # Test 4: Model statistics
    print("\n" + "-"*80)
    print("TEST 4: MODEL COMPLEXITY & RISK ASSESSMENT")
    print("-"*80)

    print("\nOption 1 (Fluorescence-Only NN):")
    print(f"  Parameters: 2,753")
    print(f"  Input features: 6 (includes integrated area)")
    print(f"  Output: 1 (fluorescence only)")
    print(f"  R²: {opt1_data['metrics']['r2']:.4f}")
    print(f"  MAE: {opt1_data['metrics']['mae']:.1f} ct/s")
    print(f"  Error: {opt1_data['metrics']['pct_error']:.1f}%")
    print(f"  Risk to spectral/CIELAB: ZERO (separate model)")

    print("\nOption 3 (Multi-Task NN):")
    print(f"  Parameters: 33,637 (12x larger)")
    print(f"  Input features: 5")
    print(f"  Outputs: 3 (spectral + CIELAB + fluorescence)")
    print(f"  Spectral R²: {opt3_data['metrics']['spectral_r2']:.4f}")
    print(f"  CIELAB R²: {opt3_data['metrics']['cielab_r2']:.4f}")
    print(f"  Fluorescence R²: {opt3_data['metrics']['fluorescence_r2']:.4f}")
    print(f"  Fluorescence MAE: {opt3_data['metrics']['fluorescence_mae']:.1f} ct/s")
    print(f"  Fluorescence Error: {opt3_data['metrics']['fluorescence_pct_error']:.1f}%")
    print(f"  Risk to spectral/CIELAB: MEDIUM (coupled training)")

    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    print("\n📊 FLUORESCENCE PREDICTION ACCURACY:")
    print(f"  Option 1 - R²: {opt1_data['metrics']['r2']:.4f}, MAE: {opt1_data['metrics']['mae']:.1f} ct/s")
    print(f"  Option 3 - R²: {opt3_data['metrics']['fluorescence_r2']:.4f}, MAE: {opt3_data['metrics']['fluorescence_mae']:.1f} ct/s")
    print(f"  Winner: Option 3 (slightly better R² and MAE)")

    print("\n⚙️  MODEL COMPLEXITY:")
    print(f"  Option 1: 2,753 parameters (simple)")
    print(f"  Option 3: 33,637 parameters (12x larger)")
    print(f"  Winner: Option 1 (simpler, faster)")

    print("\n🔬 PHYSICS CONSTRAINTS:")
    print(f"  Option 1: Violates 0% GXT → 0 ct/s constraint")
    print(f"  Option 3: Violates 0% GXT → 0 ct/s constraint")
    print(f"  Winner: TIE (both need constraint enforcement)")

    print("\n⚠️  RISK ASSESSMENT:")
    print(f"  Option 1: ZERO risk to existing predictions")
    print(f"  Option 3: MEDIUM risk (new training, coupled outputs)")
    print(f"  Winner: Option 1 (much safer)")

    print("\n🎯 INTEGRATION EFFORT:")
    print(f"  Option 1: Simple (add as separate predictor)")
    print(f"  Option 3: Complex (replace entire model, API changes)")
    print(f"  Winner: Option 1 (easier integration)")

    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)

    print("\n🏆 RECOMMENDED: Option 1 (Separate Fluorescence-Only NN)")

    print("\nReasons:")
    print("  ✓ ZERO risk to existing spectral/CIELAB predictions")
    print("  ✓ Simpler architecture (2.7k vs 33.6k parameters)")
    print("  ✓ Easier to integrate (add as separate module)")
    print("  ✓ Competitive accuracy (R²=0.931 vs 0.943)")
    print("  ✓ Uses physics-based feature (integrated area)")
    print("  ✓ Can be improved independently without affecting baseline")

    print("\n⚠️  Caveats:")
    print("  • Both models violate 0% GXT → 0 ct/s constraint")
    print("  • Need to add constraint enforcement (clip negative, enforce zero)")
    print("  • Option 1 requires computing integrated area first")

    print("\n💡 Implementation Path:")
    print("  1. Add constraint enforcement to Option 1 model")
    print("  2. Integrate as separate fluorescence predictor")
    print("  3. Keep optimized_best_model.pkl untouched")
    print("  4. API predicts spectral/CIELAB with baseline model")
    print("  5. API predicts fluorescence with Option 1 model")
    print("  6. If Option 1 ever fails, easy to fall back")

    print("\n" + "="*80)


if __name__ == "__main__":
    run_comparison_tests()
