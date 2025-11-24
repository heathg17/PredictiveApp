/**
 * Simple feedforward neural network for spectral prediction
 * Architecture: Input -> Hidden (ReLU) -> Output (Linear)
 */

// Activation functions
export const relu = (x: number): number => Math.max(0, x);
export const reluDerivative = (x: number): number => x > 0 ? 1 : 0;

// Initialize weights with Xavier initialization
export const initializeWeights = (inputSize: number, outputSize: number): number[][] => {
  const limit = Math.sqrt(6 / (inputSize + outputSize));
  const weights: number[][] = [];
  for (let i = 0; i < outputSize; i++) {
    weights[i] = [];
    for (let j = 0; j < inputSize; j++) {
      weights[i][j] = (Math.random() * 2 - 1) * limit;
    }
  }
  return weights;
};

// Forward pass through a layer
export const forwardLayer = (
  input: number[],
  weights: number[][],
  bias: number[],
  activation?: (x: number) => number
): number[] => {
  const output: number[] = [];
  for (let i = 0; i < weights.length; i++) {
    let sum = bias[i];
    for (let j = 0; j < input.length; j++) {
      sum += weights[i][j] * input[j];
    }
    output[i] = activation ? activation(sum) : sum;
  }
  return output;
};

// Normalize data
export const normalize = (
  data: number[][],
  mean?: number[],
  std?: number[]
): { normalized: number[][], mean: number[], std: number[] } => {
  const numFeatures = data[0].length;

  if (!mean || !std) {
    // Compute mean and std
    mean = new Array(numFeatures).fill(0);
    std = new Array(numFeatures).fill(0);

    // Mean
    for (const row of data) {
      for (let j = 0; j < numFeatures; j++) {
        mean[j] += row[j];
      }
    }
    for (let j = 0; j < numFeatures; j++) {
      mean[j] /= data.length;
    }

    // Std
    for (const row of data) {
      for (let j = 0; j < numFeatures; j++) {
        std[j] += Math.pow(row[j] - mean[j], 2);
      }
    }
    for (let j = 0; j < numFeatures; j++) {
      std[j] = Math.sqrt(std[j] / data.length);
      if (std[j] < 1e-8) std[j] = 1.0; // Prevent division by zero
    }
  }

  // Normalize
  const normalized = data.map(row =>
    row.map((val, j) => (val - mean![j]) / std![j])
  );

  return { normalized, mean, std };
};

// Denormalize data
export const denormalize = (
  normalized: number[],
  mean: number[],
  std: number[]
): number[] => {
  return normalized.map((val, i) => val * std[i] + mean[i]);
};

// Train neural network with backpropagation
export const trainNeuralNetwork = (
  X: number[][], // [samples x features]
  Y: number[][], // [samples x outputs]
  hiddenSize: number = 64,
  learningRate: number = 0.001,
  epochs: number = 500,
  batchSize: number = 8,
  l2Lambda: number = 0.001 // L2 regularization
): {
  weights1: number[][],
  bias1: number[],
  weights2: number[][],
  bias2: number[],
  inputMean: number[],
  inputStd: number[],
  outputMean: number[],
  outputStd: number[]
} => {
  console.log(`Training neural network: ${X[0].length} inputs -> ${hiddenSize} hidden -> ${Y[0].length} outputs`);

  // Normalize inputs and outputs
  const { normalized: X_norm, mean: inputMean, std: inputStd } = normalize(X);
  const { normalized: Y_norm, mean: outputMean, std: outputStd } = normalize(Y);

  const inputSize = X[0].length;
  const outputSize = Y[0].length;

  // Initialize network
  let weights1 = initializeWeights(inputSize, hiddenSize);
  let bias1 = new Array(hiddenSize).fill(0);
  let weights2 = initializeWeights(hiddenSize, outputSize);
  let bias2 = new Array(outputSize).fill(0);

  // Training loop
  const numSamples = X_norm.length;

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;

    // Shuffle data
    const indices = Array.from({ length: numSamples }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    // Mini-batch training
    for (let batchStart = 0; batchStart < numSamples; batchStart += batchSize) {
      const batchEnd = Math.min(batchStart + batchSize, numSamples);
      const currentBatchSize = batchEnd - batchStart;

      // Accumulate gradients
      const gradW1 = weights1.map(row => new Array(inputSize).fill(0));
      const gradB1 = new Array(hiddenSize).fill(0);
      const gradW2 = weights2.map(row => new Array(hiddenSize).fill(0));
      const gradB2 = new Array(outputSize).fill(0);

      for (let b = batchStart; b < batchEnd; b++) {
        const idx = indices[b];
        const x = X_norm[idx];
        const y = Y_norm[idx];

        // Forward pass
        const hidden = forwardLayer(x, weights1, bias1, relu);
        const output = forwardLayer(hidden, weights2, bias2);

        // Compute loss
        const loss = output.reduce((sum, pred, i) => sum + Math.pow(pred - y[i], 2), 0) / output.length;
        totalLoss += loss;

        // Backward pass
        // Output layer gradients
        const outputError = output.map((pred, i) => pred - y[i]);

        for (let i = 0; i < outputSize; i++) {
          gradB2[i] += outputError[i];
          for (let j = 0; j < hiddenSize; j++) {
            gradW2[i][j] += outputError[i] * hidden[j];
          }
        }

        // Hidden layer gradients
        const hiddenError = new Array(hiddenSize).fill(0);
        for (let j = 0; j < hiddenSize; j++) {
          for (let i = 0; i < outputSize; i++) {
            hiddenError[j] += outputError[i] * weights2[i][j];
          }
          hiddenError[j] *= reluDerivative(hidden[j]);
        }

        for (let j = 0; j < hiddenSize; j++) {
          gradB1[j] += hiddenError[j];
          for (let k = 0; k < inputSize; k++) {
            gradW1[j][k] += hiddenError[j] * x[k];
          }
        }
      }

      // Update weights with averaged gradients + L2 regularization
      const lr = learningRate / currentBatchSize;

      for (let i = 0; i < hiddenSize; i++) {
        bias1[i] -= lr * gradB1[i];
        for (let j = 0; j < inputSize; j++) {
          // L2 regularization: weight decay
          weights1[i][j] -= lr * (gradW1[i][j] + l2Lambda * weights1[i][j]);
        }
      }

      for (let i = 0; i < outputSize; i++) {
        bias2[i] -= lr * gradB2[i];
        for (let j = 0; j < hiddenSize; j++) {
          // L2 regularization: weight decay
          weights2[i][j] -= lr * (gradW2[i][j] + l2Lambda * weights2[i][j]);
        }
      }
    }

    // Log progress
    if (epoch % 100 === 0 || epoch === epochs - 1) {
      const avgLoss = totalLoss / numSamples;
      console.log(`Epoch ${epoch}/${epochs}, Loss: ${avgLoss.toFixed(6)}`);
    }
  }

  console.log("Neural network training complete");

  return {
    weights1,
    bias1,
    weights2,
    bias2,
    inputMean,
    inputStd,
    outputMean,
    outputStd
  };
};

// Predict using trained network
export const predictNeuralNetwork = (
  input: number[],
  weights1: number[][],
  bias1: number[],
  weights2: number[][],
  bias2: number[],
  inputMean: number[],
  inputStd: number[],
  outputMean: number[],
  outputStd: number[],
  clipMin: number = -0.1,
  clipMax: number = 1.5
): number[] => {
  // Normalize input
  const x_norm = input.map((val, i) => (val - inputMean[i]) / inputStd[i]);

  // Forward pass
  const hidden = forwardLayer(x_norm, weights1, bias1, relu);
  const output_norm = forwardLayer(hidden, weights2, bias2);

  // Denormalize output
  const output = denormalize(output_norm, outputMean, outputStd);

  // Clip to reasonable range (prevent extreme fluorescence peaks)
  return output.map(val => Math.max(clipMin, Math.min(clipMax, val)));
};
