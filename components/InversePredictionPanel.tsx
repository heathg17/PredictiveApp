import React, { useState } from 'react';
import { inversePrediction } from '../services/pytorchApi';

interface InversePredictionResult {
  concentrations: {
    GXT: number;
    BiVaO4: number;
    PG: number;
    PearlB: number;
    thickness: number;
  };
  predicted_spectrum: number[];
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

interface InversePredictionPanelProps {
  targetSpectrum?: number[];  // Can pass from reference sample
}

export default function InversePredictionPanel({ targetSpectrum: initialSpectrum }: InversePredictionPanelProps) {
  const [targetSpectrum, setTargetSpectrum] = useState<string>(
    initialSpectrum ? initialSpectrum.join(',') : ''
  );
  const [thickness, setThickness] = useState<number>(8.0);
  const [method, setMethod] = useState<'global' | 'local'>('global');
  const [result, setResult] = useState<InversePredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const parseSpectrum = (input: string): number[] => {
    // Try to detect NIX CSV format
    if (input.includes('R400 nm') || input.includes('sep=;')) {
      return parseNixCSV(input);
    }

    // Otherwise, parse as simple numeric values
    // Remove any non-numeric characters except commas, periods, minus signs, and scientific notation
    const cleaned = input
      .replace(/[^\d,.\-eE+\s]/g, '')
      .trim();

    // Split by commas or whitespace
    const values = cleaned
      .split(/[,\s]+/)
      .map(s => s.trim())
      .filter(s => s.length > 0)
      .map(s => parseFloat(s))
      .filter(n => !isNaN(n));

    return values;
  };

  const parseNixCSV = (csv: string): number[] => {
    // Split into lines
    const lines = csv.split('\n').map(l => l.trim()).filter(l => l.length > 0);

    // Find the header line (contains column names)
    const headerLine = lines.find(line => line.includes('R400 nm') || line.includes('R410 nm'));
    if (!headerLine) {
      throw new Error('Could not find reflectance columns (R400 nm, R410 nm, etc.) in CSV header');
    }

    // Parse header to find reflectance column indices
    const headers = headerLine.split(';').map(h => h.trim());
    const reflectanceColumns: number[] = [];

    for (let i = 0; i < headers.length; i++) {
      const header = headers[i];
      // Match R400 nm through R700 nm (31 wavelengths at 10nm intervals)
      if (header.match(/^R\d{3}\s*nm$/i)) {
        reflectanceColumns.push(i);
      }
    }

    if (reflectanceColumns.length === 0) {
      throw new Error('Could not find any reflectance columns (R###nm) in CSV');
    }

    // Find the data line (first line after header that doesn't start with "Index" or "sep=")
    const dataLine = lines.find(line =>
      line.includes(';') &&
      !line.startsWith('sep=') &&
      !line.startsWith('Index') &&
      !line.startsWith('Export') &&
      !line.startsWith('Exported')
    );

    if (!dataLine) {
      throw new Error('Could not find data row in CSV');
    }

    // Parse data line
    const values = dataLine.split(';').map(v => v.trim());

    // Extract reflectance values
    const spectrum: number[] = [];
    for (const colIndex of reflectanceColumns) {
      if (colIndex < values.length) {
        const val = parseFloat(values[colIndex]);
        if (!isNaN(val)) {
          spectrum.push(val);
        }
      }
    }

    return spectrum;
  };

  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // Parse target spectrum (handles both comma-separated and space-separated)
      const spectrum = parseSpectrum(targetSpectrum);

      if (spectrum.length !== 31) {
        throw new Error(`Expected 31 wavelength values (400-700nm in 10nm steps), got ${spectrum.length}. Please check your input.`);
      }

      // Validate range (reflectance should be 0-1.5 typically)
      const min = Math.min(...spectrum);
      const max = Math.max(...spectrum);
      if (min < -0.5 || max > 2.0) {
        console.warn(`Warning: Reflectance values outside typical range [0, 1.5]: min=${min.toFixed(3)}, max=${max.toFixed(3)}`);
      }

      // Call API
      const response = await inversePrediction(spectrum, thickness, method);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inverse prediction failed');
      console.error('Inverse prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadSampleSpectrum = (sampleName: string) => {
    // Example spectra for quick testing (from actual data)
    const exampleSpectra: Record<string, string> = {
      'GXT10': '0.74,0.82,0.87,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.10,1.11,1.12,1.13,1.14,1.15,1.16',
      'BiVaO10': '0.35,0.52,0.68,0.79,0.85,0.89,0.92,0.94,0.95,0.96,0.97,0.98,0.98,0.99,0.99,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.99,0.99,0.98,0.98,0.97,0.96,0.95,0.94,0.93',
      'BiVa14.5': `sep=;
Export Date/Time;Mon Dec 01 2025 16:26:13 GMT+0000 (Coordinated Universal Time),User Email;heath@quantumbase.com,Saved Library Name;52B
Exported From; Saved Library
Index;Original Library Name;Saved Collection Name;Original Color Name;User Color Name;Scan Type;DE00 to Scan;Match Degree;Scan Reference Index;Nix Device;Note;Date Saved;Illuminant;Observer;Measurement Mode;L;a;b;L;c;h;X;Y;Z;sRGB R;sRGB G;sRGB B;HEX;Density Status;Density C;Density M;Density Y;Density K;R400 nm;R410 nm;R420 nm;R430 nm;R440 nm;R450 nm;R460 nm;R470 nm;R480 nm;R490 nm;R500 nm;R510 nm;R520 nm;R530 nm;R540 nm;R550 nm;R560 nm;R570 nm;R580 nm;R590 nm;R600 nm;R610 nm;R620 nm;R630 nm;R640 nm;R650 nm;R660 nm;R670 nm;R680 nm;R690 nm;R700 nm
1;"-";"52B";"-";"BiVa14.5";Scan;"";"-";"-";Nix Spectro 2;"-";Mon Dec 01 2025 16:25:50 GMT+0000 (Coordinated Universal Time);D65;10;M2;9.56613345e+1;-1.47243388e+1;7.57529824e+1;9.56613345e+1;7.71707230e+1;1.00999600e+2;7.70391992e-1;8.91937605e-1;2.13541054e-1;255;247;84;#fff754;T;0.01;0.02;0.55;0.02;7.51436949e-2;1.01863652e-1;1.09080561e-1;1.11522414e-1;1.10768981e-1;1.10703900e-1;1.43559173e-1;2.27857828e-1;4.01280135e-1;6.30051911e-1;8.24566782e-1;9.18725610e-1;9.53758538e-1;9.64454114e-1;9.69377875e-1;9.68674183e-1;9.67285216e-1;9.68302429e-1;9.70476985e-1;9.72729623e-1;9.73887563e-1;9.72111642e-1;9.69249487e-1;9.61184561e-1;9.58986819e-1;9.61581767e-1;9.60153878e-1;9.58117008e-1;9.55364108e-1;9.52922165e-1;9.49153721e-1`
    };

    if (exampleSpectra[sampleName]) {
      setTargetSpectrum(exampleSpectra[sampleName]);
    }
  };

  return (
    <div className="inverse-prediction-panel" style={{ maxWidth: '1200px' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <h3 style={{ margin: '0 0 0.5rem 0', color: '#e2e8f0', fontSize: '1.1rem', fontWeight: 600 }}>
          🔬 Inverse Prediction
        </h3>
        <p style={{ margin: 0, color: '#94a3b8', fontSize: '0.875rem' }}>
          Find reagent concentrations that produce a target reflectance spectrum
        </p>
      </div>

      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{
          display: 'block',
          marginBottom: '0.5rem',
          fontWeight: 500,
          color: '#cbd5e1',
          fontSize: '0.9rem'
        }}>
          Target Spectrum (31 values: 400-700nm @ 10nm intervals)
        </label>
        <textarea
          value={targetSpectrum}
          onChange={(e) => setTargetSpectrum(e.target.value)}
          placeholder="Paste spectrum data here - supports multiple formats:&#10;&#10;• NIX CSV (copy/paste entire export from BiVa14.5.csv)&#10;• Raw values: 0.075,0.102,0.109,0.112,...&#10;• Space-separated: 0.075 0.102 0.109 0.112...&#10;• Scientific notation: 7.51e-2,1.02e-1,...&#10;&#10;Just paste and click 'Find Concentrations'!"
          rows={6}
          style={{
            width: '100%',
            padding: '0.75rem',
            background: '#1e293b',
            border: '1px solid #475569',
            borderRadius: '6px',
            fontFamily: 'ui-monospace, monospace',
            fontSize: '0.85rem',
            color: '#e2e8f0',
            resize: 'vertical',
            lineHeight: '1.5'
          }}
        />
        <div style={{
          marginTop: '0.75rem',
          display: 'flex',
          gap: '0.5rem',
          flexWrap: 'wrap',
          alignItems: 'center'
        }}>
          <span style={{ fontSize: '0.8rem', color: '#64748b', marginRight: '0.5rem' }}>
            Quick load:
          </span>
          <button
            onClick={() => loadSampleSpectrum('GXT10')}
            style={{
              padding: '0.4rem 0.75rem',
              fontSize: '0.8rem',
              background: '#334155',
              color: '#94a3b8',
              border: '1px solid #475569',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#475569';
              e.currentTarget.style.color = '#e2e8f0';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = '#334155';
              e.currentTarget.style.color = '#94a3b8';
            }}
          >
            GXT10 (10% Yellow)
          </button>
          <button
            onClick={() => loadSampleSpectrum('BiVaO10')}
            style={{
              padding: '0.4rem 0.75rem',
              fontSize: '0.8rem',
              background: '#334155',
              color: '#94a3b8',
              border: '1px solid #475569',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#475569';
              e.currentTarget.style.color = '#e2e8f0';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = '#334155';
              e.currentTarget.style.color = '#94a3b8';
            }}
          >
            BiVaO10 (10% Vanadate)
          </button>
          <button
            onClick={() => loadSampleSpectrum('BiVa14.5')}
            style={{
              padding: '0.4rem 0.75rem',
              fontSize: '0.8rem',
              background: '#334155',
              color: '#94a3b8',
              border: '1px solid #475569',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#475569';
              e.currentTarget.style.color = '#e2e8f0';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = '#334155';
              e.currentTarget.style.color = '#94a3b8';
            }}
          >
            BiVa14.5 (NIX CSV)
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr auto', gap: '1rem', marginBottom: '1.5rem', alignItems: 'end' }}>
        <div>
          <label style={{
            display: 'block',
            marginBottom: '0.5rem',
            fontWeight: 500,
            color: '#cbd5e1',
            fontSize: '0.9rem'
          }}>
            Thickness
          </label>
          <select
            value={thickness}
            onChange={(e) => setThickness(parseFloat(e.target.value))}
            style={{
              width: '100%',
              padding: '0.6rem',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: '6px',
              color: '#e2e8f0',
              fontSize: '0.9rem',
              cursor: 'pointer'
            }}
          >
            <option value={8.0}>8 μm</option>
            <option value={12.0}>12 μm</option>
          </select>
        </div>

        <div>
          <label style={{
            display: 'block',
            marginBottom: '0.5rem',
            fontWeight: 500,
            color: '#cbd5e1',
            fontSize: '0.9rem'
          }}>
            Optimization Method
          </label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as 'global' | 'local')}
            style={{
              width: '100%',
              padding: '0.6rem',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: '6px',
              color: '#e2e8f0',
              fontSize: '0.9rem',
              cursor: 'pointer'
            }}
          >
            <option value="global">Global (Differential Evolution) - Thorough</option>
            <option value="local">Local (L-BFGS-B) - Faster</option>
          </select>
        </div>

        <button
          onClick={handlePredict}
          disabled={isLoading || !targetSpectrum.trim()}
          style={{
            padding: '0.6rem 1.5rem',
            background: isLoading ? '#475569' : 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            fontSize: '0.95rem',
            fontWeight: 600,
            cursor: isLoading || !targetSpectrum.trim() ? 'not-allowed' : 'pointer',
            whiteSpace: 'nowrap',
            boxShadow: isLoading ? 'none' : '0 2px 8px rgba(59, 130, 246, 0.3)',
            opacity: isLoading || !targetSpectrum.trim() ? 0.6 : 1,
            transition: 'all 0.2s'
          }}
        >
          {isLoading ? '🔄 Optimizing...' : '🎯 Find Concentrations'}
        </button>
      </div>

      {error && (
        <div style={{
          marginTop: '1rem',
          padding: '1rem',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '6px',
          color: '#fca5a5',
          fontSize: '0.9rem'
        }}>
          <strong style={{ color: '#ef4444' }}>⚠️ Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: '2rem' }}>
          <h4 style={{
            margin: '0 0 1.25rem 0',
            color: '#e2e8f0',
            fontSize: '1.05rem',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <span style={{ color: '#10b981' }}>✅</span> Optimization Complete
          </h4>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            <div style={{
              padding: '1.25rem',
              background: 'rgba(34, 197, 94, 0.1)',
              border: '1px solid rgba(34, 197, 94, 0.3)',
              borderRadius: '8px'
            }}>
              <h5 style={{ margin: '0 0 1rem 0', color: '#86efac', fontSize: '0.9rem', fontWeight: 600 }}>
                💧 Reagent Concentrations
              </h5>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontFamily: 'ui-monospace, monospace', fontSize: '0.9rem' }}>
                <div style={{ color: '#cbd5e1' }}>
                  <span style={{ color: '#94a3b8' }}>GXT:</span> <strong style={{ color: '#fbbf24' }}>{result.concentrations.GXT.toFixed(2)}%</strong>
                </div>
                <div style={{ color: '#cbd5e1' }}>
                  <span style={{ color: '#94a3b8' }}>BiVaO4:</span> <strong style={{ color: '#fbbf24' }}>{result.concentrations.BiVaO4.toFixed(2)}%</strong>
                </div>
                <div style={{ color: '#cbd5e1' }}>
                  <span style={{ color: '#94a3b8' }}>PG:</span> <strong style={{ color: '#22c55e' }}>{result.concentrations.PG.toFixed(2)}%</strong>
                </div>
                <div style={{ color: '#cbd5e1' }}>
                  <span style={{ color: '#94a3b8' }}>PearlB:</span> <strong style={{ color: '#60a5fa' }}>{result.concentrations.PearlB.toFixed(2)}%</strong>
                </div>
                <div style={{ gridColumn: '1 / -1', color: '#cbd5e1', marginTop: '0.25rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(34, 197, 94, 0.2)' }}>
                  <span style={{ color: '#94a3b8' }}>Thickness:</span> <strong>{result.concentrations.thickness.toFixed(1)} μm</strong>
                </div>
              </div>
            </div>

            <div style={{
              padding: '1.25rem',
              background: 'rgba(59, 130, 246, 0.1)',
              border: '1px solid rgba(59, 130, 246, 0.3)',
              borderRadius: '8px'
            }}>
              <h5 style={{ margin: '0 0 1rem 0', color: '#93c5fd', fontSize: '0.9rem', fontWeight: 600 }}>
                📊 Match Quality
              </h5>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontFamily: 'ui-monospace, monospace', fontSize: '0.9rem' }}>
                <div style={{ color: '#cbd5e1' }}>
                  <span style={{ color: '#94a3b8' }}>R²:</span> <strong style={{ color: result.metrics.r2 >= 0.95 ? '#22c55e' : '#f59e0b' }}>{result.metrics.r2.toFixed(4)}</strong>
                </div>
                <div style={{ color: '#cbd5e1' }}>
                  <span style={{ color: '#94a3b8' }}>MSE:</span> <strong>{result.metrics.mse.toFixed(6)}</strong>
                </div>
                <div style={{ color: '#cbd5e1' }}>
                  <span style={{ color: '#94a3b8' }}>MAE:</span> <strong>{result.metrics.mae.toFixed(6)}</strong>
                </div>
                <div style={{ color: '#cbd5e1' }}>
                  <span style={{ color: '#94a3b8' }}>Max Δ:</span> <strong>{result.metrics.max_error.toFixed(6)}</strong>
                </div>
              </div>
            </div>
          </div>

          <div style={{
            padding: '1rem',
            background: 'rgba(148, 163, 184, 0.1)',
            border: '1px solid rgba(148, 163, 184, 0.2)',
            borderRadius: '8px',
            fontSize: '0.85rem'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1', lineHeight: '1.8' }}>
              <div>
                <span style={{ color: '#94a3b8' }}>Method:</span> <strong>{result.method === 'global' ? 'Differential Evolution' : 'L-BFGS-B'}</strong>
              </div>
              <div>
                <span style={{ color: '#94a3b8' }}>Status:</span> <strong style={{ color: result.optimization.success ? '#22c55e' : '#ef4444' }}>
                  {result.optimization.success ? '✓ Converged' : '✗ Failed'}
                </strong>
              </div>
              <div>
                <span style={{ color: '#94a3b8' }}>Iterations:</span> <strong>{result.optimization.iterations}</strong>
              </div>
              <div>
                <span style={{ color: '#94a3b8' }}>Objective:</span> <strong>{result.optimization.final_objective.toFixed(6)}</strong>
              </div>
            </div>
          </div>

          {result.metrics.r2 < 0.95 && (
            <div style={{
              marginTop: '1rem',
              padding: '0.875rem',
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '6px',
              fontSize: '0.85rem',
              color: '#fcd34d',
              lineHeight: '1.6'
            }}>
              <strong style={{ color: '#f59e0b' }}>⚠️ Low R² Warning:</strong> The match quality (R² = {result.metrics.r2.toFixed(3)}) suggests the target spectrum may not be perfectly achievable with these reagents. This could indicate: (1) non-unique solutions exist, (2) target is outside the formulation space, or (3) more optimization iterations needed.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
