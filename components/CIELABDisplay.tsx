import React from 'react';
import { CIELABValues } from '../services/pytorchApi';

interface CIELABDisplayProps {
  cielab: CIELABValues | null;
  title?: string;
}

export default function CIELABDisplay({ cielab, title = 'CIELAB Color' }: CIELABDisplayProps) {
  if (!cielab) {
    return (
      <div className="bg-slate-850 p-6 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-300 mb-4 border-b border-slate-700 pb-2">
          {title}
        </h3>
        <p className="text-xs text-slate-500">No prediction available</p>
      </div>
    );
  }

  // Convert CIELAB to approximate RGB for color preview (rough approximation)
  const getColorPreview = (L: number, a: number, b: number): string => {
    // Very rough CIELAB to RGB conversion for visualization purposes
    // In production, use a proper color space conversion library
    const lightness = L / 100;
    const redComponent = Math.max(0, Math.min(255, 128 + a * 2));
    const greenComponent = Math.max(0, Math.min(255, 128 - a * 2));
    const blueComponent = Math.max(0, Math.min(255, 128 - b * 2));

    return `rgb(${Math.round(redComponent * lightness)}, ${Math.round(greenComponent * lightness)}, ${Math.round(blueComponent * lightness)})`;
  };

  const colorPreview = getColorPreview(cielab.L, cielab.a, cielab.b);

  // Interpret the hue angle
  const getHueDescription = (h: number): string => {
    if (h >= 0 && h < 45) return 'Red-Yellow';
    if (h >= 45 && h < 90) return 'Yellow';
    if (h >= 90 && h < 135) return 'Yellow-Green';
    if (h >= 135 && h < 180) return 'Green';
    if (h >= 180 && h < 225) return 'Green-Blue';
    if (h >= 225 && h < 270) return 'Blue';
    if (h >= 270 && h < 315) return 'Blue-Purple';
    return 'Red';
  };

  return (
    <div className="bg-slate-850 p-6 rounded-lg border border-slate-700">
      <div className="flex justify-between items-center mb-4 border-b border-slate-700 pb-2">
        <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
        <div
          className="w-8 h-8 rounded-full border-2 border-slate-600 shadow-lg"
          style={{ backgroundColor: colorPreview }}
          title="Approximate color preview"
        />
      </div>

      <div className="space-y-3">
        {/* L - Lightness */}
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-medium text-slate-400">L (Lightness)</span>
            <span className="text-sm font-mono text-white">{cielab.L.toFixed(2)}</span>
          </div>
          <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-black to-white"
              style={{ width: `${cielab.L}%` }}
            />
          </div>
          <p className="text-xs text-slate-500 mt-1">
            {cielab.L > 90 ? 'Very Light' : cielab.L > 70 ? 'Light' : cielab.L > 50 ? 'Medium' : cielab.L > 30 ? 'Dark' : 'Very Dark'}
          </p>
        </div>

        {/* a - Green to Red */}
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-medium text-slate-400">a (Green ↔ Red)</span>
            <span className="text-sm font-mono text-white">{cielab.a.toFixed(2)}</span>
          </div>
          <div className="w-full h-2 bg-gradient-to-r from-green-500 via-gray-400 to-red-500 rounded-full relative">
            <div
              className="absolute top-0 w-1 h-full bg-white border border-slate-900"
              style={{ left: `${((cielab.a + 128) / 256) * 100}%` }}
            />
          </div>
          <p className="text-xs text-slate-500 mt-1">
            {cielab.a < -10 ? 'Strong Green' : cielab.a < -2 ? 'Green' : cielab.a < 2 ? 'Neutral' : cielab.a < 10 ? 'Red' : 'Strong Red'}
          </p>
        </div>

        {/* b - Blue to Yellow */}
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-medium text-slate-400">b (Blue ↔ Yellow)</span>
            <span className="text-sm font-mono text-white">{cielab.b.toFixed(2)}</span>
          </div>
          <div className="w-full h-2 bg-gradient-to-r from-blue-500 via-gray-400 to-yellow-400 rounded-full relative">
            <div
              className="absolute top-0 w-1 h-full bg-white border border-slate-900"
              style={{ left: `${((cielab.b + 128) / 256) * 100}%` }}
            />
          </div>
          <p className="text-xs text-slate-500 mt-1">
            {cielab.b < -10 ? 'Strong Blue' : cielab.b < -2 ? 'Blue' : cielab.b < 2 ? 'Neutral' : cielab.b < 10 ? 'Yellow' : 'Strong Yellow'}
          </p>
        </div>

        {/* c - Chroma (Color Intensity) */}
        <div className="pt-2 border-t border-slate-700">
          <div className="flex justify-between items-center">
            <span className="text-xs font-medium text-slate-400">c (Chroma)</span>
            <span className="text-sm font-mono text-cyan-450">{cielab.c.toFixed(2)}</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            {cielab.c < 10 ? 'Low intensity (grayish)' : cielab.c < 40 ? 'Moderate intensity' : cielab.c < 70 ? 'High intensity' : 'Very intense (vibrant)'}
          </p>
        </div>

        {/* h - Hue Angle */}
        <div>
          <div className="flex justify-between items-center">
            <span className="text-xs font-medium text-slate-400">h (Hue Angle)</span>
            <span className="text-sm font-mono text-purple-400">{cielab.h.toFixed(1)}°</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            {getHueDescription(cielab.h)}
          </p>
        </div>
      </div>

      {/* Color Summary */}
      <div className="mt-4 pt-4 border-t border-slate-700">
        <p className="text-xs text-slate-400">
          <span className="font-medium">Summary:</span>{' '}
          {cielab.L > 90 ? 'Very light' : cielab.L > 70 ? 'Light' : 'Dark'}{' '}
          {getHueDescription(cielab.h).toLowerCase()}{' '}
          {cielab.c > 40 ? 'with high intensity' : 'with moderate intensity'}
        </p>
      </div>
    </div>
  );
}
