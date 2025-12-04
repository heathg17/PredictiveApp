import React from 'react';
import { FluorescenceValues } from '../services/pytorchApi';

interface FluorescenceDisplayProps {
  fluorescence: FluorescenceValues | null;
  title?: string;
}

export default function FluorescenceDisplay({ fluorescence, title = 'Fluorescence Prediction' }: FluorescenceDisplayProps) {
  if (!fluorescence) {
    return (
      <div className="bg-slate-850 p-6 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-300 mb-4 border-b border-slate-700 pb-2">
          {title}
        </h3>
        <p className="text-xs text-slate-500">No prediction available</p>
      </div>
    );
  }

  // Get intensity description based on fluorescence value (ct/s)
  const getIntensityDescription = (cts: number): { label: string; color: string } => {
    if (cts === 0) return { label: 'None', color: 'text-slate-500' };
    if (cts < 2000) return { label: 'Very Low', color: 'text-blue-400' };
    if (cts < 4000) return { label: 'Low', color: 'text-cyan-450' };
    if (cts < 6000) return { label: 'Moderate', color: 'text-green-400' };
    if (cts < 8000) return { label: 'High', color: 'text-yellow-400' };
    return { label: 'Very High', color: 'text-orange-400' };
  };

  const intensity = getIntensityDescription(fluorescence.fluorescence_cts);

  // Format numbers for display
  const formatCts = (val: number) => val.toFixed(0); // Show whole ct/s values
  const formatArea = (val: number) => val.toFixed(3);
  const formatR2 = (val: number) => (val * 100).toFixed(1);

  return (
    <div className="bg-slate-850 p-6 rounded-lg border border-slate-700">
      <div className="flex justify-between items-center mb-4 border-b border-slate-700 pb-2">
        <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" title="Fluorescence active" />
          <span className={`text-xs font-medium ${intensity.color}`}>{intensity.label}</span>
        </div>
      </div>

      <div className="space-y-4">
        {/* Fluorescence Intensity */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-medium text-slate-400">Fluorescence Intensity</span>
            <span className="text-lg font-mono font-bold text-yellow-400">
              {formatCts(fluorescence.fluorescence_cts)} ct/s
            </span>
          </div>
          <div className="w-full h-3 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-slate-600 via-cyan-500 to-yellow-400 transition-all duration-500"
              style={{ width: `${Math.min(100, (fluorescence.fluorescence_cts / 10000) * 100)}%` }}
            />
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Predicted counts per second (ct/s)
          </p>
        </div>

        {/* Background-Subtracted Area */}
        <div className="pt-3 border-t border-slate-700">
          <div className="flex justify-between items-center">
            <span className="text-xs font-medium text-slate-400">Fluorescence Area</span>
            <span className="text-sm font-mono text-cyan-450">
              {formatArea(fluorescence.fluorescence_area)}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Background-subtracted reflectance area (400-700nm)
          </p>
        </div>

        {/* Model Performance */}
        <div className="pt-3 border-t border-slate-700">
          <div className="flex justify-between items-center">
            <span className="text-xs font-medium text-slate-400">Model Accuracy (R²)</span>
            <span className="text-sm font-mono text-green-400">
              {formatR2(fluorescence.model_r2)}%
            </span>
          </div>
          <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden mt-2">
            <div
              className="h-full bg-green-400"
              style={{ width: `${fluorescence.model_r2 * 100}%` }}
            />
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Variance explained by fluorescence predictor
          </p>
        </div>
      </div>

      {/* Summary */}
      <div className="mt-4 pt-4 border-t border-slate-700">
        <p className="text-xs text-slate-400">
          <span className="font-medium">Summary:</span>{' '}
          {fluorescence.fluorescence_cts === 0
            ? 'No fluorescence (GXT concentration is 0%)'
            : `${intensity.label} fluorescence intensity predicted from pigment concentrations and spectral characteristics.`}
          {fluorescence.fluorescence_cts > 5000 && ' Higher concentrations of fluorescent pigments (e.g., GXT) contribute to increased fluorescence.'}
        </p>
      </div>
    </div>
  );
}
