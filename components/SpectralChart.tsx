import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { SpectralPoint } from '../types';

interface SpectralChartProps {
  data: SpectralPoint[];
  singleSimData?: SpectralPoint[];
  neuralSimData?: SpectralPoint[];
  showTarget?: boolean;
}

const SpectralChart: React.FC<SpectralChartProps> = ({ data, singleSimData, neuralSimData }) => {
  // Merge data for display
  const chartData = data.map((point, index) => ({
    wavelength: point.wavelength,
    measured: point.reflectance,
    kmSingle: singleSimData ? (singleSimData[index]?.reflectance ?? undefined) : undefined,
    neural: neuralSimData ? (neuralSimData[index]?.reflectance ?? undefined) : undefined
  }));

  // Debug: log data so developer can inspect in browser console
  // (Remove or comment out after debugging.)
  // eslint-disable-next-line no-console
  console.log('SpectralChart - chartData length', chartData.length, { chartData, singleSimData, neuralSimData });

  if (data.length === 0) {
    return (
      <div className="w-full h-96 bg-slate-850 rounded-lg p-4 shadow-lg border border-slate-700 flex items-center justify-center">
        <span className="text-sm text-slate-400">No reference spectral data available</span>
      </div>
    );
  }

  return (
    <div className="w-full h-96 bg-slate-850 rounded-lg p-4 shadow-lg border border-slate-700">
      <h3 className="text-cyan-450 font-semibold mb-4">Spectral Curve (400nm - 700nm)</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis 
            dataKey="wavelength" 
            type="number" 
            domain={[400, 700]} 
            stroke="#94a3b8"
            label={{ value: 'Wavelength (nm)', position: 'insideBottomRight', offset: -5, fill: '#94a3b8' }}
          />
          <YAxis 
            domain={[0, 1.1]} 
            stroke="#94a3b8"
            label={{ value: 'Reflectance (0-1)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
            itemStyle={{ color: '#e2e8f0' }}
            formatter={(value: any) => (typeof value === 'number' ? value.toFixed(3) : value)}
          />
          <Legend verticalAlign="top" height={36}/>
          
          {/* User's Selection / Target */}
          <Line 
            type="monotone" 
            dataKey="measured" 
            stroke="#94a3b8" 
            strokeWidth={2} 
            dot={false} 
            name="Reference / Measured"
            strokeDasharray="5 5"
          />

          {/* K-M Single Simulation */}
          {singleSimData && (
            <Line
              type="monotone"
              dataKey="kmSingle"
              stroke="#22d3ee"
              strokeWidth={2.5}
              dot={false}
              name="K-M Single"
            />
          )}

          {/* Neural Network Simulation */}
          {neuralSimData && (
            <Line
              type="monotone"
              dataKey="neural"
              stroke="#a78bfa"
              strokeWidth={2.5}
              dot={false}
              name="Neural Network"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SpectralChart;
