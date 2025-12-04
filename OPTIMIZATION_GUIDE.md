# Neural Network Hyperparameter Optimization

## What's Being Optimized

The comprehensive hyperparameter optimization is currently testing **50 random configurations** across the following parameters:

### Architecture Parameters
- **Hidden Layers**: 13 different architectures
  - Single layer: [32], [64], [128]
  - Two layers: [64,32], [64,64], [128,64]
  - Three layers: [64,64,32], [64,64,64], [128,64,32], [64,128,64], [128,128,64]
  - Four layers: [64,64,64,32], [128,128,64,32]

### Training Parameters
- **Learning Rate**: 0.0001, 0.0005, 0.001, 0.002, 0.005
- **Batch Size**: 8, 16, 32
- **Dropout**: 0.0, 0.1, 0.2, 0.3
- **L2 Regularization**: 0.0, 0.0001, 0.001, 0.01

### Model Features
- **Activation Function**: ReLU, Leaky ReLU, ELU
- **Batch Normalization**: True, False
- **Learning Rate Scheduler**: True, False
- **Gradient Clipping**: 0, 1.0, 5.0

### Loss Weighting
- **Spectral Weight**: 1.0, 1.5, 2.0
- **CIELAB Weight**: 1.0, 0.5, 0.25

## Evaluation Metrics

Each configuration is evaluated on multiple metrics:

1. **Spectral Reflectance**
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R² (Coefficient of Determination)

2. **CIELAB Color**
   - MAE
   - RMSE
   - R²

3. **Combined Score** (lower is better)
   - Weighted combination of all metrics
   - Used to rank configurations

## Search Space Size

Total possible combinations: 13 × 5 × 3 × 4 × 4 × 3 × 2 × 3 × 3 × 2 × 3 = **~1.5 million combinations**

Testing 50 random samples (0.003% of space) using **random search** which is proven to be more efficient than grid search for high-dimensional spaces.

## Expected Duration

- **Per trial**: ~3-8 seconds (depending on architecture size and early stopping)
- **50 trials**: ~3-7 minutes total
- **With early stopping**: Most trials will stop before 2000 epochs

## Output Files

When complete, the optimization will save:

1. **`results/optimization/best_config.json`**
   - Best hyperparameter configuration
   - Test set performance metrics
   - Model parameters count
   - Training time

2. **`trained_models/optimized_model.pkl`**
   - Trained model with best configuration
   - Ready to deploy to production

## What Happens Next

Once optimization completes:

1. **Review top 10 configurations** - See which architectures performed best
2. **Analyze patterns** - Identify what features matter most
3. **Deploy best model** - Update API server to use optimized model
4. **Compare to baseline** - Verify improvement over current model

## Current Baseline Performance

For comparison, the current baseline model:
- Architecture: [64, 64, 64, 64] (4 layers)
- Spectral MAE: ~0.033
- CIELAB MAE: ~0.366
- Parameters: 18,724
- Training time: ~4.5s

**Goal**: Beat this performance with optimal hyperparameters

## Monitoring Progress

You can check optimization progress with:
```bash
cd python_version
tail -f optimization.log  # If logging was enabled
```

Or check the background process output periodically to see which trial is running.

## Key Insights from Search Space

### Why These Parameters?

1. **Architecture variety**: Tests shallow vs deep, narrow vs wide networks
2. **Learning rate range**: From very conservative (0.0001) to aggressive (0.005)
3. **Regularization**: Balances overfitting prevention vs model capacity
4. **Activation functions**: ReLU (standard), Leaky ReLU (avoids dead neurons), ELU (smooth gradient)
5. **Loss weighting**: Allows prioritizing spectral vs CIELAB accuracy

### Expected Findings

Based on neural network theory and the dataset size (84 samples):

- **Smaller networks likely to perform better** (less overfitting)
- **Moderate dropout (0.1-0.2)** probably optimal
- **Learning rate ~0.001** typically best starting point
- **Batch normalization** should help with training stability
- **L2 regularization** important with small dataset

## After Optimization

### If Optimization Improves Performance

1. Update API server to use optimized model
2. Re-run with more trials (100+) for fine-tuning
3. Consider ensemble methods (train 5 models with best config)

### If Baseline Still Best

1. Current architecture already near-optimal
2. Focus on:
   - Collecting more training data
   - Data augmentation strategies
   - Physics-informed features
   - Hybrid approaches (K-M + NN)

## Hybrid K-M Approach

As discovered earlier, the hybrid Kubelka-Munk approach shows:
- **94% better CIELAB predictions** (MAE 0.021 vs 0.366)
- **69% worse spectral predictions** (MAE 0.056 vs 0.033)

**Recommendation**: Use optimized NN for spectral + hybrid K-M for CIELAB
- Gets best of both worlds
- More physically interpretable
- Better color accuracy

## Next Steps After Results

1. ✅ Review top 10 configurations
2. ✅ Train final model with best config
3. ✅ Integrate with API server
4. ✅ Test predictions in UI
5. ✅ Compare against baseline
6. ✅ Document improvements
7. ⚠️ Consider hybrid approach for CIELAB

---

**Status**: Optimization running... (~3-7 minutes remaining)
