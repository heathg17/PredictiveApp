/**
 * Enhanced PyTorch API Client
 * Connects React frontend to Enhanced Python FastAPI backend
 * PP Substrate with 4 reagents, CIELAB color predictions, and Fluorescence
 */

const API_BASE_URL = 'http://localhost:8001';

export interface CIELABValues {
  L: number;  // Lightness (0-100)
  a: number;  // Green (-) to Red (+)
  b: number;  // Blue (-) to Yellow (+)
  c: number;  // Chroma (color intensity)
  h: number;  // Hue angle (0-360 degrees)
}

export interface FluorescenceValues {
  fluorescence_cts: number;    // Predicted fluorescence intensity (ct/s)
  fluorescence_area: number;    // Background-subtracted area
  gxt_multiplier: number;       // Smooth constraint multiplier (0 at 0% GXT, ~1 at ≥5% GXT)
  model_r2: number;             // R² of fluorescence model
}

export interface PredictionRequest {
  concentrations: { [reagent: string]: number };
  thickness: number;
  model_type?: 'single' | 'neural-net';
}

export interface PredictionResponse {
  wavelengths: number[];
  reflectance: number[];
  cielab: CIELABValues;
  fluorescence: FluorescenceValues;
  thickness: number;
  model_version: string;
}

export interface APIStatus {
  status: string;
  samples_loaded: number;
  samples_with_conc: number;
  reagents: string[];
  model_types: string[];
}

/**
 * Get API status and available models
 */
export async function getAPIStatus(): Promise<APIStatus> {
  const response = await fetch(`${API_BASE_URL}/api/status`);
  if (!response.ok) {
    throw new Error(`API status check failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Predict spectral reflectance, CIELAB color coordinates, and fluorescence using enhanced PyTorch neural network
 */
export async function predictSpectrum(
  concentrations: { [reagent: string]: number },
  thickness: number = 8.0,
  modelType: 'single' | 'neural-net' = 'neural-net'
): Promise<PredictionResponse> {
  const request: PredictionRequest = {
    concentrations,
    thickness,
  };

  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(`Prediction failed: ${error.detail || response.statusText}`);
  }

  return response.json();
}

/**
 * Get list of available reagents
 */
export async function getReagents(): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/api/reagents`);
  if (!response.ok) {
    throw new Error(`Failed to fetch reagents: ${response.statusText}`);
  }
  const data = await response.json();
  return data.reagents;
}

/**
 * Force retrain models with latest data
 */
export async function retrainModels(): Promise<{ status: string; message: string; reagents: string[] }> {
  const response = await fetch(`${API_BASE_URL}/api/retrain`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(`Retrain failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Check if API is available
 */
export async function isAPIAvailable(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/`, {
      method: 'GET',
      signal: AbortSignal.timeout(2000), // 2 second timeout
    });
    return response.ok;
  } catch {
    return false;
  }
}

export interface InversePredictionRequest {
  target_spectrum: number[];  // 31 reflectance values (400-700nm, 10nm steps)
  thickness?: number;          // Film thickness (8 or 12 μm)
  method?: 'global' | 'local'; // Optimization method
  initial_guess?: { [reagent: string]: number };
  bounds?: { [reagent: string]: [number, number] };
}

export interface InversePredictionResponse {
  concentrations: {
    GXT: number;
    BiVaO4: number;
    PG: number;
    PearlB: number;
    thickness: number;
  };
  predicted_spectrum: number[];
  target_spectrum: number[];
  metrics: {
    mse: number;
    mae: number;
    max_error: number;
    r2: number;
  };
  optimization: {
    success: boolean;
    iterations: number;
    final_objective: number;
    message: string;
  };
  method: string;
}

/**
 * Inverse prediction: Find reagent concentrations that produce a target spectrum
 * Uses gradient-based optimization with the forward neural network model
 */
export async function inversePrediction(
  targetSpectrum: number[],
  thickness: number = 8.0,
  method: 'global' | 'local' = 'global'
): Promise<InversePredictionResponse> {
  const request: InversePredictionRequest = {
    target_spectrum: targetSpectrum,
    thickness,
    method,
  };

  const response = await fetch(`${API_BASE_URL}/api/inverse_predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(`Inverse prediction failed: ${error.detail || response.statusText}`);
  }

  return response.json();
}
