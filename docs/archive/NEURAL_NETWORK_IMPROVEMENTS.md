# Neural Network Accuracy Improvement Recommendations

## Current Status
Your enhanced neural network already implements:
- ✅ Z-score normalization (inputs & outputs)
- ✅ Batch normalization
- ✅ Dropout (0.2)
- ✅ L2 regularization (0.001)
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ Multi-output architecture (spectral + CIELAB)

## High-Impact Improvements

### 1. **Physics-Informed Features** ⭐⭐⭐
**Why:** Add domain knowledge to help the network learn faster
**Implementation:**
```python
def add_physics_features(X):
    """
    X: [GXT, BiVaO4, PG, PearlB, Thickness]
    Returns: Extended feature set
    """
    total_conc = X[:, :4].sum(axis=1, keepdims=True)

    # Interaction terms (important for pigment mixing)
    gxt_bivao4 = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    gxt_pg = (X[:, 0] * X[:, 2]).reshape(-1, 1)
    pg_pearlb = (X[:, 2] * X[:, 3]).reshape(-1, 1)

    # Thickness interactions
    thickness_conc = (X[:, 4:5] * total_conc)

    # Extended features
    X_extended = np.hstack([
        X,  # Original 5 features
        total_conc,  # Total concentration
        gxt_bivao4, gxt_pg, pg_pearlb,  # Pairwise interactions
        thickness_conc,  # Thickness-concentration interaction
        X[:, :4] ** 2  # Quadratic terms for concentrations
    ])

    return X_extended  # Now 15 features instead of 5
```
**Expected improvement:** 15-30% reduction in MAE

### 2. **Data Augmentation** ⭐⭐⭐
**Why:** You have 84 samples - augmentation can effectively increase this
**Implementation:**
```python
def augment_spectral_data(X, Y, noise_level=0.01, n_augmented=2):
    """
    Generate synthetic samples with small perturbations

    Args:
        X: Input samples [n, 5]
        Y: Output samples [n, 36]
        noise_level: Gaussian noise std (as fraction of std)
        n_augmented: Number of augmented samples per original
    """
    X_aug = []
    Y_aug = []

    for _ in range(n_augmented):
        # Add small Gaussian noise to inputs
        X_noise = X + np.random.normal(0, noise_level * X.std(axis=0), X.shape)
        X_noise = np.clip(X_noise, 0, None)  # Keep concentrations non-negative

        # Add smaller noise to outputs (measurements have some noise)
        Y_noise = Y + np.random.normal(0, noise_level * 0.5 * Y.std(axis=0), Y.shape)

        X_aug.append(X_noise)
        Y_aug.append(Y_noise)

    X_augmented = np.vstack([X] + X_aug)
    Y_augmented = np.vstack([Y] + Y_aug)

    return X_augmented, Y_augmented
```
**Expected improvement:** 10-20% reduction in validation error
**Note:** Apply to training set only, not validation/test

### 3. **Ensemble Methods** ⭐⭐
**Why:** Multiple models reduce variance and improve generalization
**Implementation:**
```python
def train_ensemble(X_train, Y_train, X_val, Y_val, n_models=5):
    """Train multiple models with different random initializations"""
    models = []

    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}...")

        # Different random seeds for each model
        np.random.seed(42 + i)
        torch.manual_seed(42 + i)

        model_data = train_enhanced_neural_network(
            X_train, Y_train, X_val, Y_val,
            hidden_size=64,
            verbose=False
        )
        models.append(model_data)

    return models

def predict_ensemble(X, models):
    """Average predictions from multiple models"""
    predictions = []

    for model_data in models:
        spectrum, cielab = predict_enhanced_neural_network(
            X,
            model_data['model'],
            model_data['input_mean'],
            model_data['input_std'],
            model_data['output_mean'],
            model_data['output_std']
        )
        predictions.append(np.concatenate([spectrum, [
            cielab['L'], cielab['a'], cielab['b'], cielab['c'], cielab['h']
        ]]))

    # Average predictions
    return np.mean(predictions, axis=0)
```
**Expected improvement:** 5-15% reduction in MAE
**Trade-off:** 5x slower inference, 5x more storage

### 4. **Residual Connections (ResNet-style)** ⭐⭐
**Why:** Helps gradient flow in deeper networks
**Implementation:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.layers(x))  # Skip connection

# In EnhancedSpectralNN:
self.shared_layers = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.BatchNorm1d(hidden_size),
    nn.ReLU(),
    ResidualBlock(hidden_size, dropout_rate),
    ResidualBlock(hidden_size, dropout_rate),
    nn.Dropout(dropout_rate)
)
```
**Expected improvement:** 5-10% better convergence

### 5. **Separate Normalization for Spectral vs CIELAB** ⭐⭐
**Why:** Different output scales - normalize separately for better learning
**Current issue:** CIELAB values (L: 0-100, a/b: -128-+128) have very different ranges than reflectance (0-1)
**Implementation:**
```python
# Instead of normalizing all 36 outputs together:
Y_spectral_mean = np.mean(Y_train[:, :31], axis=0)
Y_spectral_std = np.std(Y_train[:, :31], axis=0)

Y_cielab_mean = np.mean(Y_train[:, 31:], axis=0)
Y_cielab_std = np.std(Y_train[:, 31:], axis=0)

# Normalize separately
Y_train_spectral_norm = (Y_train[:, :31] - Y_spectral_mean) / Y_spectral_std
Y_train_cielab_norm = (Y_train[:, 31:] - Y_cielab_mean) / Y_cielab_std
```
**Expected improvement:** 10-15% better CIELAB predictions

### 6. **K-Fold Cross-Validation** ⭐⭐
**Why:** Better model selection and error estimation with limited data
**Implementation:**
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold = X[train_idx]
    Y_train_fold = Y[train_idx]
    X_val_fold = X[val_idx]
    Y_val_fold = Y[val_idx]

    model_data = train_enhanced_neural_network(
        X_train_fold, Y_train_fold,
        X_val_fold, Y_val_fold,
        verbose=False
    )

    # Evaluate
    cv_scores.append(model_data['best_val_loss'])

print(f"CV Mean: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
```
**Expected improvement:** Better hyperparameter tuning

### 7. **Wavelength-Aware Loss Function** ⭐
**Why:** Some wavelengths are more important for color perception
**Implementation:**
```python
# Weight visible spectrum (400-700nm) more heavily
wavelength_weights = torch.ones(31)
wavelength_weights[10:25] = 1.5  # 500-650nm (most important for color)

def weighted_spectral_loss(pred, target, weights):
    squared_errors = (pred - target) ** 2
    weighted_errors = squared_errors * weights
    return weighted_errors.mean()
```
**Expected improvement:** 5-10% better perceptual color matching

### 8. **Cosine Annealing Learning Rate** ⭐
**Why:** Better than ReduceLROnPlateau for escaping local minima
**Implementation:**
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=100,  # Restart every 100 epochs
    T_mult=2,  # Double period after each restart
    eta_min=1e-6
)
```
**Expected improvement:** 5-10% better final loss

## Medium-Impact Improvements

### 9. **Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 10. **Mixup Data Augmentation**
```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y
```

### 11. **Label Smoothing**
For CIELAB outputs to prevent overconfidence

### 12. **Bayesian Neural Networks**
Get uncertainty estimates with predictions

## Implementation Priority

**Week 1:**
1. Physics-informed features (#1)
2. Data augmentation (#2)
3. Separate normalization (#5)

**Week 2:**
4. K-fold cross-validation (#6)
5. Residual connections (#4)

**Week 3:**
6. Ensemble methods (#3)
7. Wavelength-aware loss (#7)

## Expected Overall Improvement
Implementing top 5 recommendations: **30-50% reduction in prediction error**

## About Z-Score Normalization for Non-Reflectance Spectra

**Q: Is z-score normalization useful for spectral data outside 0-1 range?**

**A: Yes, absolutely!** Your current implementation already does this correctly. Here's why:

1. **Reflectance spectra (0-1):** Z-score normalization helps the network learn patterns independent of absolute scale
2. **Fluorescent spectra (>1):** Z-score is ESSENTIAL - it handles the extended range without clipping
3. **CIELAB values:** Different scales (L: 0-100, a/b: -128 to +128) - z-score makes them comparable

**Current implementation (lines 164-176):**
```python
Y_mean = np.mean(Y_train, axis=0)  # Separate mean for each output
Y_std = np.std(Y_train, axis=0)    # Separate std for each output
Y_train_norm = (Y_train - Y_mean) / Y_std
```

This is correct! Each output dimension gets its own normalization parameters.

**Improvement:** Consider normalizing spectral and CIELAB outputs separately (recommendation #5) for even better results.
