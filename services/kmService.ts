import { SampleData, ModelCoefficients, ModelType } from '../types';
import { pseudoInverse, multiply } from '../utils/matrix';
import { WAVELENGTHS, Rg_DEFAULT } from '../constants';
import { trainNeuralNetwork, predictNeuralNetwork } from '../utils/neuralNet';

// --- Fluorescence Detection ---

/**
 * Detect if a sample contains fluorescent pigments
 * Fluorescent samples have R > 1 at some wavelengths
 */
export const hasFluorescence = (spectrum: number[]): boolean => {
  return spectrum.some(r => r > 1.0);
};

/**
 * Extract fluorescence component from spectrum
 * For fluorescent samples: F(λ) = max(0, R(λ) - 1)
 * This is the excess reflectance due to fluorescence emission
 */
export const extractFluorescence = (spectrum: number[]): {
  reflectance: number[],
  fluorescence: number[]
} => {
  const reflectance = spectrum.map(r => Math.min(r, 1.0));
  const fluorescence = spectrum.map(r => Math.max(0, r - 1.0));
  return { reflectance, fluorescence };
};

// --- Single Layer Logic ---

export const reflectanceToKS = (R: number): number => {
  // For fluorescent materials, clamp to valid range
  const rSafe = Math.max(0.001, Math.min(0.999, R));
  return Math.pow(1 - rSafe, 2) / (2 * rSafe);
};

export const ksToReflectance = (KS: number): number => {
  const term = Math.sqrt(Math.pow(KS, 2) + 2 * KS);
  return 1 + KS - term;
};

// --- Two Layer Logic (From Python Script) ---

const twoLayerForward = (K_mix: number, S_mix: number, Rg: number, X: number): number => {
  const S = Math.max(1e-6, S_mix);
  const K = Math.max(0, K_mix);

  const a = 1 + K / S;
  const b = Math.sqrt(Math.pow(a, 2) - 1);
  
  // Limit exp power to avoid overflow/underflow
  const bSX = Math.max(-50, Math.min(50, b * S * X));
  
  const exp_bSX = Math.exp(bSX);
  const exp_neg = Math.exp(-bSX);

  const num = (1 - Rg * (a - b)) * exp_bSX * (a + b) - (1 - Rg * (a + b)) * exp_neg * (a - b);
  const den = (a + b - Rg * (a - b)) * exp_bSX - (a - b - Rg * (a + b)) * exp_neg;

  if (den === 0) return 0;
  const R = num / den;
  return Math.max(0, Math.min(1, R));
};

// --- Training ---

export const trainModel = (samples: SampleData[], type: ModelType): ModelCoefficients => {
  if (samples.length === 0) throw new Error("No samples to train on.");

  // Filter samples to only those with concentration data
  const samplesWithConc = samples.filter(s => {
    const concs = Object.values(s.concentrations);
    return concs.length > 0 && concs.some(c => c > 0);
  });

  if (samplesWithConc.length === 0) {
    throw new Error("No samples with concentration data found. Cannot train model.");
  }

  console.log(`Filtered to ${samplesWithConc.length} samples with concentration data (from ${samples.length} total)`);

  // Detect fluorescent samples
  const fluorescentSamples = samplesWithConc.filter(s => hasFluorescence(s.spectrum));
  console.log(`Found ${fluorescentSamples.length} fluorescent samples (R > 1.0)`);

  // Separate spectra into reflectance and fluorescence components
  const processedSamples = samplesWithConc.map(s => {
    const { reflectance, fluorescence } = extractFluorescence(s.spectrum);
    return {
      ...s,
      reflectance,
      fluorescence,
      isFluorescent: hasFluorescence(s.spectrum)
    };
  });

  const reagentSet = new Set<string>();
  samplesWithConc.forEach(s => {
    Object.keys(s.concentrations).forEach(k => {
      // Only add reagents with non-zero concentration
      if ((s.concentrations[k] || 0) > 0) {
        reagentSet.add(k);
      }
    });
  });
  const reagents = Array.from(reagentSet).sort();

  console.log(`Training with ${samplesWithConc.length} samples and ${reagents.length} reagents:`, reagents.join(', '));

  // Concentration Matrix [Samples x Reagents]
  const C = samplesWithConc.map(s => reagents.map(r => (s.concentrations[r] || 0) / 100.0));

  if (type === 'single') {
    // Analytical Solution for K/S using reflectance component only
    const KS_measured = processedSamples.map(s => s.reflectance.map(r => reflectanceToKS(r)));

    try {
      const C_pinv = pseudoInverse(C);
      const Alpha = multiply(C_pinv, KS_measured);

      const alphaMap: Record<string, number[]> = {};
      reagents.forEach((r, idx) => {
        alphaMap[r] = Alpha[idx];
      });

      // Train fluorescence model if fluorescent samples exist
      let fluorMap: Record<string, number[]> | undefined = undefined;
      if (fluorescentSamples.length > 0) {
        console.log("Training fluorescence emission model...");
        const F_measured = processedSamples.map(s => s.fluorescence);
        const Fluor = multiply(C_pinv, F_measured);

        fluorMap = {};
        reagents.forEach((r, idx) => {
          fluorMap![r] = Fluor[idx];
        });
        console.log("Fluorescence model trained successfully");
      }

      console.log("Single-layer model trained successfully");
      return {
        type: 'single',
        alpha: alphaMap,
        fluorescence: fluorMap,
        wavelengths: WAVELENGTHS
      };
    } catch (err) {
      console.error("Failed to compute pseudo-inverse:", err);
      throw new Error("Failed to train single-layer model. Try using fewer samples or check data quality.");
    }
  } else if (type === 'two-layer') {
    // Two-Layer Optimization (Numerical Gradient Descent)
    return trainTwoLayerModel(processedSamples, reagents, C, fluorescentSamples.length > 0);
  } else {
    // Neural Network (Black-box model)
    return trainNeuralNetModel(samplesWithConc, reagents, C);
  }
};

const trainTwoLayerModel = (
  samples: any[], // processed samples with reflectance/fluorescence
  reagents: string[],
  C: number[][],
  hasFluorescent: boolean
): ModelCoefficients => {
  const numWavelengths = WAVELENGTHS.length;
  const numReagents = reagents.length;
  
  const K_map: Record<string, number[]> = {};
  const S_map: Record<string, number[]> = {};
  const F_map: Record<string, number[]> = {};

  reagents.forEach(r => {
    K_map[r] = new Array(numWavelengths).fill(0);
    S_map[r] = new Array(numWavelengths).fill(0);
    if (hasFluorescent) {
      F_map[r] = new Array(numWavelengths).fill(0);
    }
  });

  // Perform optimization for each wavelength independently
  for (let w = 0; w < numWavelengths; w++) {
    // Target Reflectances for this wavelength (use reflectance component only)
    const R_targets = samples.map(s => s.reflectance[w]);
    
    // Thicknesses
    const X_vals = samples.map(s => s.thickness);

    // Initial Guess: Better starting values based on typical K-M values
    // For most pigments: K ~ 0.5-5.0, S ~ 5.0-50.0
    // Params vector: [K_1...K_n, S_1...S_n]
    let params = [...new Array(numReagents).fill(1.0), ...new Array(numReagents).fill(10.0)];

    // Adaptive Gradient Descent with momentum
    const initialLR = 0.001;
    const iterations = 200;
    const momentum = 0.9;
    let velocity = new Array(2 * numReagents).fill(0); 

    for (let iter = 0; iter < iterations; iter++) {
      const grads = new Array(2 * numReagents).fill(0);
      
      // Compute Gradients
      for (let sIdx = 0; sIdx < samples.length; sIdx++) {
        const concs = C[sIdx]; // Vector of concentrations
        const X = X_vals[sIdx];
        const R_tgt = R_targets[sIdx];
        
        // Current K_mix, S_mix
        let K_mix = 0, S_mix = 0;
        for (let r = 0; r < numReagents; r++) {
          K_mix += concs[r] * params[r];
          S_mix += concs[r] * params[r + numReagents];
        }

        const R_pred = twoLayerForward(K_mix, S_mix, Rg_DEFAULT, X);
        const error = R_pred - R_tgt;

        // Finite difference approximation for gradient 
        // (Simplifies implementation vs analytical derivation)
        const epsilon = 1e-4;
        
        for (let r = 0; r < numReagents; r++) {
           // dL/dK_r
           // Approximation: dR/dK * conc
           const R_k_plus = twoLayerForward(K_mix + epsilon * concs[r], S_mix, Rg_DEFAULT, X);
           const dR_dK = (R_k_plus - R_pred) / epsilon;
           grads[r] += 2 * error * dR_dK;

           // dL/dS_r
           const R_s_plus = twoLayerForward(K_mix, S_mix + epsilon * concs[r], Rg_DEFAULT, X);
           const dR_dS = (R_s_plus - R_pred) / epsilon;
           grads[r + numReagents] += 2 * error * dR_dS;
        }
      }

      // Normalize gradients to prevent explosion
      const gradNorm = Math.sqrt(grads.reduce((sum, g) => sum + g * g, 0));
      const normalizedGrads = gradNorm > 1 ? grads.map(g => g / gradNorm) : grads;

      // Update Params with momentum
      const learningRate = initialLR / (1 + iter / 100); // Decay learning rate
      for (let i = 0; i < params.length; i++) {
        velocity[i] = momentum * velocity[i] - learningRate * normalizedGrads[i];
        params[i] += velocity[i];

        // Enforce bounds (positive coefficients with minimum values)
        // K should be >= 0, S should be > 0
        if (i < numReagents) {
          params[i] = Math.max(0.01, params[i]); // K values
        } else {
          params[i] = Math.max(0.1, params[i]);  // S values
        }
      }

      // Early stopping if gradients are very small
      if (gradNorm < 1e-6 && iter > 50) {
        break;
      }
    }

    // Log optimization progress for this wavelength
    if (w % 10 === 0) {
      const avgK = params.slice(0, numReagents).reduce((a, b) => a + b, 0) / numReagents;
      const avgS = params.slice(numReagents).reduce((a, b) => a + b, 0) / numReagents;
      console.log(`λ=${WAVELENGTHS[w]}nm: avg K=${avgK.toFixed(3)}, avg S=${avgS.toFixed(3)}`);
    }

    // Store optimized values
    reagents.forEach((r, rIdx) => {
      K_map[r][w] = params[rIdx];
      S_map[r][w] = params[rIdx + numReagents];
    });

    // Train fluorescence separately if needed
    if (hasFluorescent) {
      const F_targets = samples.map(s => s.fluorescence[w]);
      // Simple linear regression for fluorescence emission
      // F = sum(c_i * F_i) where F_i is fluorescence coefficient per reagent
      const C_pinv = pseudoInverse(C);
      const F_coeffs = multiply(C_pinv, F_targets.map(f => [f])).map(row => row[0]);
      reagents.forEach((r, rIdx) => {
        F_map[r][w] = Math.max(0, F_coeffs[rIdx]); // Fluorescence must be non-negative
      });
    }
  }

  console.log("Two-layer model trained successfully");

  return {
    type: 'two-layer',
    K: K_map,
    S: S_map,
    fluorescence: hasFluorescent ? F_map : undefined,
    wavelengths: WAVELENGTHS
  };
};

const trainNeuralNetModel = (
  samples: SampleData[],
  reagents: string[],
  C: number[][]
): ModelCoefficients => {
  console.log("Training neural network (black-box model)...");

  // Prepare training data
  // Input: [concentration1, concentration2, ..., thickness]
  // Output: [R400, R410, ..., R700]

  const X: number[][] = samples.map((s, idx) => {
    return [...C[idx], s.thickness];
  });

  const Y: number[][] = samples.map(s => s.spectrum);

  // Train neural network with improved hyperparameters
  const {
    weights1,
    bias1,
    weights2,
    bias2,
    inputMean,
    inputStd,
    outputMean,
    outputStd
  } = trainNeuralNetwork(
    X,
    Y,
    128,  // Hidden layer size (increased for better capacity)
    0.005, // Learning rate (reduced for stability)
    2000, // Epochs (increased for convergence)
    Math.min(8, Math.max(4, Math.floor(samples.length / 5))), // Batch size
    0.005 // L2 regularization (prevents overfitting)
  );

  return {
    type: 'neural-net',
    neuralNet: {
      reagents,
      weights1,
      bias1,
      weights2,
      bias2,
      inputMean,
      inputStd,
      outputMean,
      outputStd
    },
    wavelengths: WAVELENGTHS
  };
};

export const predictReflectance = (
  concentrations: Record<string, number>,
  model: ModelCoefficients,
  thickness: number = 4.0
): number[] => {
  const { type, wavelengths } = model;

  if (type === 'neural-net' && model.neuralNet) {
    // Neural network prediction
    const { reagents, weights1, bias1, weights2, bias2, inputMean, inputStd, outputMean, outputStd } = model.neuralNet;

    // Prepare input: [conc1, conc2, ..., thickness]
    const input = reagents.map(r => (concentrations[r] || 0) / 100.0);
    input.push(thickness);

    return predictNeuralNetwork(
      input,
      weights1,
      bias1,
      weights2,
      bias2,
      inputMean,
      inputStd,
      outputMean,
      outputStd
    );
  }

  // K-M based predictions
  const predictedR = new Array(wavelengths.length).fill(0);

  for (let i = 0; i < wavelengths.length; i++) {
    let baseReflectance = 0;

    if (type === 'single') {
      // Single Layer K-M
      let ksSum = 0;
      Object.entries(concentrations).forEach(([reagent, percent]) => {
        const unitKS = model.alpha?.[reagent]?.[i] || 0;
        ksSum += (percent / 100.0) * unitKS;
      });
      ksSum = Math.max(0.0001, ksSum);
      baseReflectance = ksToReflectance(ksSum);
    } else {
      // Two-Layer K-M
      let kSum = 0;
      let sSum = 0;
      Object.entries(concentrations).forEach(([reagent, percent]) => {
        const unitK = model.K?.[reagent]?.[i] || 0;
        const unitS = model.S?.[reagent]?.[i] || 0;
        kSum += (percent / 100.0) * unitK;
        sSum += (percent / 100.0) * unitS;
      });
      baseReflectance = twoLayerForward(kSum, sSum, Rg_DEFAULT, thickness);
    }

    // Add fluorescence contribution if model has fluorescence coefficients
    let fluorescence = 0;
    if (model.fluorescence) {
      Object.entries(concentrations).forEach(([reagent, percent]) => {
        const unitF = model.fluorescence?.[reagent]?.[i] || 0;
        fluorescence += (percent / 100.0) * unitF;
      });
    }

    // Total reflectance = base reflectance + fluorescence emission
    predictedR[i] = baseReflectance + fluorescence;
  }

  return predictedR;
};