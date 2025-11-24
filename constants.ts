import { SampleData } from './types';

export const WAVELENGTHS = Array.from({ length: 31 }, (_, i) => 400 + i * 10);

// Background Reflectance (Rg) - approximated as White Paper (0.95 across spectrum)
// In a full app, this would be dynamic per wavelength.
export const Rg_DEFAULT = 0.95;

// A subset of the data provided in the prompt to initialize the app state
export const INITIAL_SAMPLES: SampleData[] = [
  {
    id: 'Eren 1',
    name: 'Eren 1',
    substrate: 'Paper',
    thickness: 4,
    spectrum: [0.408,0.425,0.439,0.445,0.448,0.461,0.5,0.574,0.695,0.822,0.908,0.94,0.945,0.939,0.929,0.912,0.895,0.883,0.876,0.873,0.872,0.871,0.874,0.872,0.876,0.885,0.887,0.886,0.884,0.883,0.883],
    concentrations: { 'BiVaO4': 10.24, 'PG': 0, 'PB': 0.49, 'LY': 0, 'GXT': 0, 'SY43': 0, 'TiO2': 0 }
  },
  {
    id: 'Eren 2',
    name: 'Eren 2',
    substrate: 'Paper',
    thickness: 4,
    spectrum: [0.535,0.562,0.584,0.591,0.596,0.614,0.651,0.715,0.808,0.894,0.946,0.966,0.973,0.974,0.975,0.97,0.964,0.958,0.95,0.937,0.92,0.904,0.896,0.886,0.883,0.886,0.893,0.903,0.91,0.914,0.918],
    concentrations: { 'BiVaO4': 5.01, 'PG': 2.00, 'PB': 0, 'LY': 0, 'GXT': 0, 'SY43': 0, 'TiO2': 0 }
  },
  {
    id: 'Eren 3',
    name: 'Eren 3',
    substrate: 'Paper',
    thickness: 4,
    spectrum: [0.572,0.619,0.674,0.708,0.747,0.793,0.83,0.862,0.886,0.896,0.888,0.865,0.829,0.793,0.75,0.7,0.653,0.619,0.591,0.57,0.548,0.532,0.529,0.528,0.531,0.541,0.55,0.56,0.566,0.571,0.578],
    concentrations: { 'BiVaO4': 0, 'PG': 5.34, 'PB': 2.10, 'LY': 0, 'GXT': 0, 'SY43': 0, 'TiO2': 0 }
  },
  {
    id: 'F023A',
    name: 'F023A',
    substrate: 'Paper',
    thickness: 4,
    spectrum: [0.31366,0.31989,0.31112,0.30121,0.29100,0.29264,0.32367,0.40172,0.55635,0.74409,0.88964,0.94396,0.93101,0.89507,0.84239,0.78506,0.73032,0.69126,0.66603,0.65195,0.64272,0.63998,0.64883,0.65401,0.66363,0.67912,0.68286,0.68110,0.67746,0.67652,0.68005],
    concentrations: { 'LY': 20, 'GXT': 0, 'SY43': 0, 'BiVaO4': 9, 'PG': 0, 'PB': 1, 'SY160': 2 }
  },
  {
    id: 'F046C',
    name: 'F046C',
    substrate: 'Paper',
    thickness: 4,
    spectrum: [0.44454,0.45870,0.45979,0.45515,0.45015,0.45659,0.48355,0.54286,0.66701,0.84231,0.99576,1.06293,1.07183,1.05909,1.04518,1.03086,1.02060,1.01552,1.01339,1.00997,1.00658,1.00167,0.99724,0.98800,0.98548,0.98944,0.99051,0.99096,0.98995,0.98867,0.98734],
    concentrations: { 'LY': 0, 'GXT': 30, 'BiVaO4': 9, 'TiO2': 1 }
  }
];

// Unique reagents found in initial set
export const REAGENTS_LIST = [
  'BiVaO4', 'PG', 'PB', 'LY', 'GXT', 'SY43', 'TiO2', 'SY160', 'SFXC G', 'SFXC P', 'P Gold'
];