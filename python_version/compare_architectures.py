"""
Compare shallow vs deep neural network architectures
Shallow: 1 layer of 128 neurons
Deep: 4 layers of 32 neurons
"""
import numpy as np
import time
from utils.data_loader import load_master_data
from models.neural_network import train_neural_network, predict_neural_network, SpectralNN
from models.deep_neural_network import train_deep_neural_network, predict_deep_neural_network
from utils.data_augmentation import create_mixup_augmented_dataset
from types_constants import WAVELENGTHS

print("=" * 80)
print("ARCHITECTURE COMPARISON: SHALLOW vs DEEP")
print("=" * 80)
print()

# Load data
print("Loading data...")
samples = load_master_data('../public/Master conc.csv', '../public/Master spec - master_sample_library.csv')
samples_with_conc = [s for s in samples if len(s.concentrations) > 0 and any(c > 0 for c in s.concentrations.values())]
print(f"Loaded {len(samples_with_conc)} samples with concentration data")
print()

# Apply mix-up augmentation
augmented = create_mixup_augmented_dataset(samples_with_conc, n_synthetic=20)
print()

# Get reagents
reagents = sorted(list(set(r for s in augmented for r in s.concentrations.keys())))

# Prepare training data
C = np.array([[s.concentrations.get(r, 0) / 100.0 for r in reagents] for s in augmented])
X = np.column_stack([C, np.array([s.thickness for s in augmented])])
Y = np.array([s.spectrum for s in augmented])

print(f"Training data shape: X={X.shape}, Y={Y.shape}")
print()

# Architecture 1: Shallow (1 layer of 128 neurons)
print("=" * 80)
print("ARCHITECTURE 1: SHALLOW (1×128)")
print("=" * 80)
print()

start_time = time.time()
shallow_model_data = train_neural_network(
    X, Y,
    hidden_size=128,
    learning_rate=0.005,
    epochs=2000,
    batch_size=8,
    l2_lambda=0.005,
    dropout_rate=0.2
)
shallow_train_time = time.time() - start_time

shallow_params = shallow_model_data['model'].fc1.weight.numel() + \
                 shallow_model_data['model'].fc1.bias.numel() + \
                 shallow_model_data['model'].fc2.weight.numel() + \
                 shallow_model_data['model'].fc2.bias.numel()

print(f"Training time: {shallow_train_time:.2f} seconds")
print(f"Total parameters: {shallow_params:,}")
print()

# Architecture 2: Deep (4 layers of 32 neurons)
print("=" * 80)
print("ARCHITECTURE 2: DEEP (4×32)")
print("=" * 80)
print()

start_time = time.time()
deep_model_data = train_deep_neural_network(
    X, Y,
    hidden_sizes=[32, 32, 32, 32],
    learning_rate=0.005,
    epochs=2000,
    batch_size=8,
    l2_lambda=0.005,
    dropout_rate=0.2
)
deep_train_time = time.time() - start_time

print(f"Training time: {deep_train_time:.2f} seconds")
print(f"Total parameters: {deep_model_data['param_count']:,}")
print()

# Architecture 3: Very Deep (6 layers of 32 neurons)
print("=" * 80)
print("ARCHITECTURE 3: VERY DEEP (6×32)")
print("=" * 80)
print()

start_time = time.time()
very_deep_model_data = train_deep_neural_network(
    X, Y,
    hidden_sizes=[32, 32, 32, 32, 32, 32],
    learning_rate=0.005,
    epochs=2000,
    batch_size=8,
    l2_lambda=0.005,
    dropout_rate=0.2
)
very_deep_train_time = time.time() - start_time

print(f"Training time: {very_deep_train_time:.2f} seconds")
print(f"Total parameters: {very_deep_model_data['param_count']:,}")
print()

# Test all models
print("=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)
print()

# Test on original samples (not augmented)
test_samples = samples_with_conc[:10]  # First 10 samples

shallow_errors = []
deep_errors = []
very_deep_errors = []

for sample in test_samples:
    input_vec = np.array([sample.concentrations.get(r, 0) / 100.0 for r in reagents] + [sample.thickness])

    # Shallow prediction
    shallow_pred = predict_neural_network(
        input_vec,
        shallow_model_data['model'],
        shallow_model_data['input_mean'],
        shallow_model_data['input_std'],
        shallow_model_data['output_mean'],
        shallow_model_data['output_std']
    )
    shallow_error = np.mean(np.abs(shallow_pred - sample.spectrum))
    shallow_errors.append(shallow_error)

    # Deep prediction
    deep_pred = predict_deep_neural_network(
        input_vec,
        deep_model_data['model'],
        deep_model_data['input_mean'],
        deep_model_data['input_std'],
        deep_model_data['output_mean'],
        deep_model_data['output_std']
    )
    deep_error = np.mean(np.abs(deep_pred - sample.spectrum))
    deep_errors.append(deep_error)

    # Very deep prediction
    very_deep_pred = predict_deep_neural_network(
        input_vec,
        very_deep_model_data['model'],
        very_deep_model_data['input_mean'],
        very_deep_model_data['input_std'],
        very_deep_model_data['output_mean'],
        very_deep_model_data['output_std']
    )
    very_deep_error = np.mean(np.abs(very_deep_pred - sample.spectrum))
    very_deep_errors.append(very_deep_error)

    print(f"{sample.name:15} | Shallow: {shallow_error:.6f} | Deep: {deep_error:.6f} | Very Deep: {very_deep_error:.6f}")

print()
print("Summary Statistics:")
print("-" * 80)
print(f"{'Architecture':<20} | {'Avg MAE':<12} | {'Std Dev':<12} | {'Parameters':<12} | {'Train Time':<12}")
print("-" * 80)
print(f"{'Shallow (1×128)':<20} | {np.mean(shallow_errors):<12.6f} | {np.std(shallow_errors):<12.6f} | {shallow_params:<12,} | {shallow_train_time:<12.2f}s")
print(f"{'Deep (4×32)':<20} | {np.mean(deep_errors):<12.6f} | {np.std(deep_errors):<12.6f} | {deep_model_data['param_count']:<12,} | {deep_train_time:<12.2f}s")
print(f"{'Very Deep (6×32)':<20} | {np.mean(very_deep_errors):<12.6f} | {np.std(very_deep_errors):<12.6f} | {very_deep_model_data['param_count']:<12,} | {very_deep_train_time:<12.2f}s")
print()

# Determine winner
architectures = [
    ('Shallow (1×128)', np.mean(shallow_errors)),
    ('Deep (4×32)', np.mean(deep_errors)),
    ('Very Deep (6×32)', np.mean(very_deep_errors))
]
winner = min(architectures, key=lambda x: x[1])

print("=" * 80)
print("WINNER")
print("=" * 80)
print(f"Best architecture: {winner[0]}")
print(f"Average MAE: {winner[1]:.6f}")
print()

# Analysis
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()
print("Pros and Cons:")
print()
print("SHALLOW (1×128):")
print("  + Simple architecture, easy to train")
print("  + Fewer layers = less risk of vanishing gradients")
print("  + More parameters = higher capacity")
print("  - May overfit with small datasets")
print()
print("DEEP (4×32):")
print("  + Hierarchical feature learning")
print("  + Fewer parameters = better generalization")
print("  + More regularization from depth")
print("  - Harder to train (vanishing gradients)")
print("  - May underfit complex patterns")
print()
print("VERY DEEP (6×32):")
print("  + Maximum hierarchical learning")
print("  + Best for very complex patterns")
print("  - Risk of vanishing gradients")
print("  - May need more training data")
print()

print("Recommendation:")
if winner[0] == 'Shallow (1×128)':
    print("  ✓ Use SHALLOW architecture for this dataset")
    print("  Reason: Better performance with current data size")
elif winner[0] == 'Deep (4×32)':
    print("  ✓ Use DEEP (4×32) architecture")
    print("  Reason: Best balance of performance and generalization")
else:
    print("  ✓ Use VERY DEEP (6×32) architecture")
    print("  Reason: Best performance for complex patterns")

print()
print("Note: Results may vary with different random initializations")
print("Consider running multiple trials and averaging results")
