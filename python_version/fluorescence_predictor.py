"""
Fluorescence Predictor Module

Predicts fluorescence (ct/s) from:
1. Pigment concentrations (GXT, BiVaO4, PG, PearlB)
2. Background-subtracted reflectance area (calculated from NN predictions)

This is a POST-PROCESSING step that doesn't modify the core NN.
"""

import numpy as np
import pickle
from scipy.integrate import trapezoid

# PP Background reflectance
PP_BACKGROUND = np.array([
    6.32230798e-1, 8.24865798e-1, 9.64723766e-1, 9.74486450e-1, 9.75640754e-1,
    9.78552401e-1, 9.75615660e-1, 9.77584740e-1, 9.89334762e-1, 9.98248855e-1,
    1.00193385e+0, 1.00288971e+0, 1.00431999e+0, 1.00225039e+0, 1.00409655e+0,
    1.00195293e+0, 1.00040476e+0, 1.00163217e+0, 1.00373669e+0, 1.00462874e+0,
    1.00526512e+0, 1.00307862e+0, 9.99236067e-1, 9.90266760e-1, 9.87492502e-1,
    9.90339776e-1, 9.88663038e-1, 9.86457149e-1, 9.83821054e-1, 9.81393794e-1,
    9.78041152e-1
])

WAVELENGTHS = np.arange(400, 710, 10)  # 400-700nm in 10nm intervals


class FluorescencePredictor:
    """
    Predicts fluorescence from concentrations and reflectance spectra

    This uses a simple linear regression trained on the OPTI dataset
    """

    def __init__(self, model_path='trained_models/fluorescence_predictor_cts.pkl'):
        """Load the fluorescence prediction model for ct/s values"""
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.coefficients = data['coefficients']
                self.intercept = data['intercept']
                self.feature_names = data['feature_names']
                self.r2_score = data.get('r2_score', None)
                self.mae = data.get('mae', None)
            print(f"✓ Fluorescence predictor loaded (R² = {self.r2_score:.3f}, MAE = {self.mae:.1f} ct/s)")
        except FileNotFoundError:
            print(f"⚠ Fluorescence predictor model not found at {model_path}")
            print(f"  Using default coefficients (will be less accurate)")
            # Default coefficients (based on simple GXT + thickness model)
            self.coefficients = np.array([202.57, 0.0, 0.0, 0.0, 69.07])  # GXT, BiVaO4, PG, PearlB, thickness
            self.intercept = 2152.6
            self.feature_names = ['GXT', 'BiVaO4', 'PG', 'PearlB', 'thickness']
            self.r2_score = 0.964
            self.mae = 300.9

    def calculate_background_subtracted_area(self, reflectance):
        """
        Calculate fluorescence area from reflectance spectrum

        Args:
            reflectance: Array of 31 reflectance values (400-700nm, 10nm intervals)

        Returns:
            Background-subtracted fluorescence area
        """
        if len(reflectance) != 31:
            raise ValueError(f"Expected 31 reflectance values, got {len(reflectance)}")

        # Subtract PP background
        reflectance_sub = np.array(reflectance) - PP_BACKGROUND

        # Only consider positive values (area above normalized baseline)
        reflectance_sub[reflectance_sub < 0] = 0

        # Calculate area using trapezoidal integration
        area = trapezoid(reflectance_sub, WAVELENGTHS)

        return area

    def predict(self, concentrations, reflectance, thickness=8.0):
        """
        Predict fluorescence (ct/s)

        Args:
            concentrations: Dict with keys 'GXT', 'BiVaO4', 'PG', 'PearlB' (in %)
            reflectance: List/array of 31 reflectance values
            thickness: Film thickness in μm (8.0 or 12.0)

        Returns:
            Predicted fluorescence in ct/s
        """
        # Calculate fluorescence area from reflectance (for reference)
        fluor_area = self.calculate_background_subtracted_area(reflectance)

        # GXT is the fluorescent pigment - no GXT means no fluorescence
        gxt_conc = concentrations.get('GXT', 0.0)
        if gxt_conc == 0.0:
            return {
                'fluorescence_cts': 0.0,
                'fluorescence_area': float(fluor_area),
                'model_r2': self.r2_score
            }

        # Build feature vector: [GXT%, BiVaO4%, PG%, PearlB%, thickness]
        # Current model primarily uses GXT and thickness
        features = np.array([
            gxt_conc,
            concentrations.get('BiVaO4', 0.0),
            concentrations.get('PG', 0.0),
            concentrations.get('PearlB', 0.0),
            thickness
        ])

        # Linear prediction
        fluorescence = np.dot(self.coefficients, features) + self.intercept

        # Ensure non-negative
        fluorescence = max(0.0, fluorescence)

        return {
            'fluorescence_cts': float(fluorescence),
            'fluorescence_area': float(fluor_area),
            'model_r2': self.r2_score
        }


def train_fluorescence_predictor(save_path='trained_models/fluorescence_predictor.pkl'):
    """
    Train a simple linear model to predict fluorescence from OPTI dataset

    Features: GXT%, BiVaO4%, PG%, PearlB%, fluorescence_area
    Target: Mean fluorescence intensity (ct/s proxy)
    """
    print("Training fluorescence predictor...")

    # Load the fluorescence analysis results
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split

    results_path = 'results/fluorescence/fluorescence_analysis.csv'
    df = pd.read_csv(results_path)

    # Prepare features: concentrations + background-subtracted area
    X = df[['GXT', 'BiVaO4', 'PG', 'PearlB', 'area_above_0_sub']].values

    # Target: mean fluorescence (as proxy for ct/s)
    y = df['mean_fluor_sub'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"✓ Model trained:")
    print(f"  R² = {r2:.3f}")
    print(f"  MAE = {mae:.4f}")
    print(f"  Coefficients: {model.coef_}")
    print(f"  Intercept: {model.intercept_:.4f}")

    # Save model
    model_data = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'feature_names': ['GXT', 'BiVaO4', 'PG', 'PearlB', 'fluor_area'],
        'r2_score': r2,
        'mae': mae
    }

    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Saved to: {save_path}")

    return model_data


if __name__ == "__main__":
    # Train the fluorescence predictor
    train_fluorescence_predictor()
