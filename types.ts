export interface SpectralPoint {
  wavelength: number;
  reflectance: number;
}

export interface SampleData {
  id: string;
  name: string;
  spectrum: number[]; // Array of 31 values (400-700nm)
  concentrations: Record<string, number>; // Reagent name -> %
  thickness: number;
  substrate: string;
}

export interface Reagent {
  name: string;
  color: string;
}

export type ModelType = 'single' | 'two-layer' | 'neural-net';

export interface SimulationResult {
  wavelengths: number[];
  predictedSpectrum: number[];
  targetSpectrum?: number[];
}

export interface ModelCoefficients {
  type: ModelType;
  // Single Layer: Alpha = K/S
  alpha?: Record<string, number[]>;
  // Two Layer: Separate K and S
  K?: Record<string, number[]>;
  S?: Record<string, number[]>;
  // Fluorescence emission coefficients (for fluorescent pigments)
  fluorescence?: Record<string, number[]>;
  // Neural Network: weights and biases
  neuralNet?: {
    reagents: string[];
    weights1: number[][]; // Input to hidden
    bias1: number[];
    weights2: number[][]; // Hidden to output
    bias2: number[];
    inputMean: number[];
    inputStd: number[];
    outputMean: number[];
    outputStd: number[];
  };
  wavelengths: number[];
}