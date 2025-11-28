import pickle

with open('trained_models/fluorescence_model.pkl', 'rb') as f:
    data = pickle.load(f)

print("Model Data Keys:")
for key in data.keys():
    print(f"  - {key}")

print("\nChecking normalization stats:")
print(f"  input_mean: {data.get('input_mean', 'MISSING')}")
print(f"  input_std: {data.get('input_std', 'MISSING')}")
print(f"  output_mean: {data.get('output_mean', 'MISSING')}")
print(f"  output_std: {data.get('output_std', 'MISSING')}")
