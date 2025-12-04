export const WAVELENGTHS = Array.from({ length: 31 }, (_, i) => 400 + i * 10);

// Background Reflectance (Rg) - approximated as White Paper (0.95 across spectrum)
// In a full app, this would be dynamic per wavelength.
export const Rg_DEFAULT = 0.95;

// PP Substrate reagents (4 reagents for new model)
export const REAGENTS_LIST = [
  'GXT', 'BiVaO4', 'PG', 'PearlB'
];

// Available thicknesses for PP substrate (μm)
export const THICKNESS_OPTIONS = [8.0, 12.0];