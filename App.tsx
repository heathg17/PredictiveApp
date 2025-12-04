import React, { useState, useEffect, useMemo } from 'react';
import { SampleData, ModelCoefficients, SpectralPoint, ModelType } from './types';
import { REAGENTS_LIST, WAVELENGTHS, THICKNESS_OPTIONS } from './constants';
import { trainModel, predictReflectance } from './services/kmService';
import { predictSpectrum, isAPIAvailable, CIELABValues, FluorescenceValues } from './services/pytorchApi';
import SpectralChart from './components/SpectralChart';
import CIELABDisplay from './components/CIELABDisplay';
import FluorescenceDisplay from './components/FluorescenceDisplay';
import { parseSamplesFromCSV } from './utils/csvParser';
import { loadPPData } from './utils/loadPPData';

// Reagent input component with buffered typing
interface ReagentInputProps {
  reagent: string;
  value: number;
  onChange: (value: number) => void;
}

function ReagentInput({ reagent, value, onChange }: ReagentInputProps) {
  const [inputValue, setInputValue] = useState<string>(value.toFixed(2));

  // Update local state when prop value changes (from slider)
  useEffect(() => {
    setInputValue(value.toFixed(2));
  }, [value]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Allow free typing without validation
    setInputValue(e.target.value);
  };

  const commitValue = () => {
    // Validate and commit on blur or Enter
    const val = parseFloat(inputValue);
    if (!isNaN(val) && val >= 0 && val <= 30) {
      onChange(val);
    } else {
      // Reset to current valid value if invalid
      setInputValue(value.toFixed(2));
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      commitValue();
      e.currentTarget.blur();
    }
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <label className="text-xs font-medium text-slate-300">{reagent}</label>
        <div className="flex items-center space-x-2">
          <input
            type="number"
            min="0"
            max="30"
            step="0.1"
            value={inputValue}
            onChange={handleInputChange}
            onBlur={commitValue}
            onKeyDown={handleKeyDown}
            className="w-20 bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-sm text-white text-right focus:outline-none focus:border-cyan-450 focus:ring-1 focus:ring-cyan-450"
          />
          <span className="text-sm font-mono text-slate-400">%</span>
        </div>
      </div>
      <input
        type="range"
        min="0"
        max="30"
        step="0.1"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-450"
      />
    </div>
  );
}

export default function App() {
  // --- State ---
  const [samples, setSamples] = useState<SampleData[]>([]);
  const [singleModel, setSingleModel] = useState<ModelCoefficients | null>(null);
  const [neuralModel, setNeuralModel] = useState<ModelCoefficients | null>(null);
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [isLoadingData, setIsLoadingData] = useState<boolean>(true);
  const [apiAvailable, setApiAvailable] = useState<boolean>(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // Simulation State
  const [mixConcentrations, setMixConcentrations] = useState<Record<string, number>>({});
  const [thickness, setThickness] = useState<number>(8.0);
  const [singleSimResult, setSingleSimResult] = useState<number[]>([]);
  const [neuralSimResult, setNeuralSimResult] = useState<number[]>([]);
  const [cielabResult, setCielabResult] = useState<CIELABValues | null>(null);
  const [fluorescenceResult, setFluorescenceResult] = useState<FluorescenceValues | null>(null);

  // --- Effects ---

  // 0. Check if PyTorch API is available
  useEffect(() => {
    const checkAPI = async () => {
      const available = await isAPIAvailable();
      setApiAvailable(available);
      if (!available) {
        setApiError('PyTorch API not available. Start API server: cd python_version && python3 api_server.py');
        console.warn('PyTorch API not available');
      } else {
        console.log('✓ PyTorch API is available');
      }
    };
    checkAPI();
  }, []);

  // 1. Load PP Substrate Data on Mount
  useEffect(() => {
    const loadData = async () => {
      setIsLoadingData(true);
      const ppSamples = await loadPPData();
      if (ppSamples.length > 0) {
        setSamples(ppSamples);
        setSelectedSampleId(ppSamples[0].id);
        console.log(`Loaded ${ppSamples.length} PP substrate samples from Concentrations.csv and Spectra.csv`);
      } else {
        console.warn('Failed to load PP substrate data, using initial samples');
      }
      setIsLoadingData(false);
    };
    loadData();
  }, []);

  // 2. Train K-M Model Only (Neural network is now server-side)
  useEffect(() => {
    try {
      console.log(`Training K-M model on ${samples.length} samples...`);
      const singleTrained = trainModel(samples, 'single');
      setSingleModel(singleTrained);

      // Mark neural model as "available via API"
      setNeuralModel({ type: 'neural-net' } as ModelCoefficients);

      // Initialize sliders if first run
      if (Object.keys(mixConcentrations).length === 0 && samples.length > 0) {
        setMixConcentrations(samples[0].concentrations);
      }
    } catch (err) {
      console.error("Training failed:", err);
    }
  }, [samples]);

  // 3. Run Simulations (K-M locally, Neural via API)
  useEffect(() => {
    // K-M prediction (local)
    if (singleModel) {
      const pred = predictReflectance(mixConcentrations, singleModel, thickness);
      setSingleSimResult(pred);
    }

    // PyTorch neural network prediction (API) - now includes CIELAB and Fluorescence
    if (apiAvailable && Object.keys(mixConcentrations).length > 0) {
      predictSpectrum(mixConcentrations, thickness, 'neural-net')
        .then(response => {
          setNeuralSimResult(response.reflectance);
          setCielabResult(response.cielab);
          setFluorescenceResult(response.fluorescence);
          setApiError(null);
        })
        .catch(err => {
          console.error('PyTorch API prediction failed:', err);
          setApiError(err.message);
          // Fallback: clear results
          setNeuralSimResult([]);
          setCielabResult(null);
          setFluorescenceResult(null);
        });
    } else if (!apiAvailable) {
      // Clear results if API not available
      setNeuralSimResult([]);
      setCielabResult(null);
      setFluorescenceResult(null);
    }
  }, [mixConcentrations, singleModel, thickness, apiAvailable]);

  // --- Handlers ---
  
  const handleSliderChange = (reagent: string, value: number) => {
    setMixConcentrations(prev => ({
      ...prev,
      [reagent]: value
    }));
  };

  const handleSampleSelect = (id: string) => {
    setSelectedSampleId(id);
    const s = samples.find(sample => sample.id === id);
    if (s) {
      setMixConcentrations(s.concentrations);
      setThickness(s.thickness);
    }
  };

  // --- File Upload Handler ---
  const handleCSVUpload = async (file: File | null) => {
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = parseSamplesFromCSV(text);
      if (parsed.length === 0) {
        alert('No samples parsed from CSV. Check file format.');
        return;
      }
      setSamples(parsed);
      setSelectedSampleId(parsed[0].id);
    } catch (err) {
      console.error('Failed to parse CSV', err);
      alert('Failed to parse CSV. See console for details.');
    }
  };

  // --- Data Prep ---
  const selectedSample = samples.find(s => s.id === selectedSampleId);
  
  const chartRefData: SpectralPoint[] = useMemo(() => {
    if (!selectedSample) return [];
    return selectedSample.spectrum.map((r, i) => ({
      wavelength: WAVELENGTHS[i],
      reflectance: r
    }));
  }, [selectedSample]);

  const chartSingleSimData: SpectralPoint[] = useMemo(() => {
    if (singleSimResult.length === 0) return [];
    return singleSimResult.map((r, i) => ({
      wavelength: WAVELENGTHS[i],
      reflectance: r
    }));
  }, [singleSimResult]);

  const chartNeuralSimData: SpectralPoint[] = useMemo(() => {
    if (neuralSimResult.length === 0) return [];
    return neuralSimResult.map((r, i) => ({
      wavelength: WAVELENGTHS[i],
      reflectance: r
    }));
  }, [neuralSimResult]);

  const totalConc = ((Object.values(mixConcentrations) as number[]) || []).reduce((a, b) => a + b, 0);

  // Show loading state
  if (isLoadingData) {
    return (
      <div className="min-h-screen bg-slate-900 text-slate-200 font-sans flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-450 mb-4"></div>
          <p className="text-lg text-slate-400">Loading PP substrate dataset...</p>
          <p className="text-sm text-slate-500 mt-2">Reading Concentrations.csv and Spectra.csv</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 font-sans flex flex-col md:flex-row">
      
      {/* Sidebar */}
      <aside className="w-full md:w-80 bg-slate-850 border-r border-slate-700 flex flex-col h-screen sticky top-0 overflow-hidden">
        <div className="p-6 border-b border-slate-700">
          <h1 className="text-2xl font-bold text-cyan-450 tracking-tight">OptiMix</h1>
          <p className="text-xs text-slate-400 mt-1">Spectral Formulation Engine</p>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          
          {/* Controls */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Load Reference</label>
              <select 
                value={selectedSampleId} 
                onChange={(e) => handleSampleSelect(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-450"
              >
                {samples.map(s => (
                  <option key={s.id} value={s.id}>{s.name}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Thickness Selector */}
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Coating Thickness</label>
            <select
              value={thickness}
              onChange={(e) => setThickness(parseFloat(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-450"
            >
              {THICKNESS_OPTIONS.map(t => (
                <option key={t} value={t}>{t}μm {t === 12.0 ? '(Heavy)' : '(Standard)'}</option>
              ))}
            </select>
          </div>

          {/* Formulation Mixers */}
          <div>
            <div className="flex justify-between items-end mb-4">
              <label className="block text-sm font-medium text-cyan-450">Formulation</label>
              <span className={`text-xs font-mono ${totalConc > 100 ? 'text-red-400' : 'text-slate-400'}`}>
                Total: {totalConc.toFixed(2)}%
              </span>
            </div>

            <div className="space-y-5">
              {REAGENTS_LIST.map(reagent => (
                <ReagentInput
                  key={reagent}
                  reagent={reagent}
                  value={mixConcentrations[reagent] || 0}
                  onChange={(val) => handleSliderChange(reagent, val)}
                />
              ))}
            </div>
          </div>
        </div>
        
        <div className="p-4 border-t border-slate-700 bg-slate-900">
          <button 
            className="w-full py-2 px-4 bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold rounded transition-colors text-sm border border-slate-600"
            onClick={() => setMixConcentrations({})}
          >
            Clear Mixture
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-6 md:p-12 overflow-y-auto">
        
        <header className="mb-8">
          <div className="flex justify-between items-start">
            <div>
              <h2 className="text-3xl font-light text-white">Simulation Dashboard</h2>
              <p className="text-slate-400 mt-2 max-w-2xl">
                Comparing <span className="text-cyan-450 font-medium">Single-Layer K-M</span> (physics-based) vs <span className="text-purple-400 font-medium">Neural Network</span> (data-driven) models in real-time.
              </p>
            </div>
            <div className="flex items-center space-x-2 bg-slate-800 px-4 py-2 rounded-full border border-slate-700">
              <div className={`w-2 h-2 rounded-full ${singleModel && apiAvailable ? 'bg-green-400 animate-pulse' : 'bg-red-500'}`}></div>
              <span className="text-xs font-medium text-slate-300">
                {apiAvailable ? 'PyTorch API Connected' : 'PyTorch API Offline'}
              </span>
            </div>
          </div>
        </header>

        {/* API Status Warning */}
        {!apiAvailable && (
          <div className="mb-6 bg-red-900/20 border border-red-700 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <div className="text-red-400 text-xl">⚠️</div>
              <div>
                <h3 className="text-red-400 font-semibold mb-1">PyTorch API Not Available</h3>
                <p className="text-sm text-slate-300 mb-2">
                  The PyTorch neural network predictions require the Python API server.
                </p>
                <code className="text-xs bg-slate-800 px-2 py-1 rounded text-cyan-400">
                  cd python_version && python3 enhanced_api_server.py
                </code>
              </div>
            </div>
          </div>
        )}

        {/* Visualization */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          <div className="lg:col-span-2">
            <SpectralChart
              data={chartRefData}
              singleSimData={chartSingleSimData}
              neuralSimData={chartNeuralSimData}
            />
          </div>

          <div className="space-y-6">
             {/* CIELAB Color Display */}
             <CIELABDisplay cielab={cielabResult} title="Predicted Color (CIELAB)" />

             {/* Fluorescence Display */}
             <FluorescenceDisplay fluorescence={fluorescenceResult} title="Fluorescence Prediction" />

             <div className="bg-slate-850 p-6 rounded-lg border border-slate-700">
               <h3 className="text-sm font-semibold text-slate-300 mb-4 border-b border-slate-700 pb-2">Model Comparison</h3>
               <div className="space-y-4">
                 <div>
                   <h4 className="text-xs font-bold text-cyan-450 uppercase tracking-wider mb-2">K-M Single Layer</h4>
                   <div className="space-y-2 text-xs text-slate-400">
                     <div className="flex justify-between">
                       <span>Method:</span>
                       <span className="text-slate-200">Matrix P-Inv</span>
                     </div>
                     <div className="flex justify-between">
                       <span>Variables:</span>
                       <span className="text-slate-200">K/S (Alpha)</span>
                     </div>
                     <p className="text-xs text-slate-500 pt-2 leading-relaxed">
                       Physics-based linear model. Best for opaque samples.
                     </p>
                   </div>
                 </div>

                 <div className="border-t border-slate-700 pt-4">
                   <h4 className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-2">Enhanced PyTorch NN</h4>
                   <div className="space-y-2 text-xs text-slate-400">
                     <div className="flex justify-between">
                       <span>Engine:</span>
                       <span className="text-slate-200">PyTorch 4×64</span>
                     </div>
                     <div className="flex justify-between">
                       <span>Architecture:</span>
                       <span className="text-slate-200">Dual-Output</span>
                     </div>
                     <div className="flex justify-between">
                       <span>Substrate:</span>
                       <span className="text-slate-200">PP</span>
                     </div>
                     <div className="flex justify-between">
                       <span>Outputs:</span>
                       <span className="text-slate-200">Spectral + CIELAB</span>
                     </div>
                     <div className="flex justify-between">
                       <span>Status:</span>
                       <span className={apiAvailable ? "text-green-400" : "text-red-400"}>
                         {apiAvailable ? 'API Connected' : 'Offline'}
                       </span>
                     </div>
                     <p className="text-xs text-slate-500 pt-2 leading-relaxed">
                       Multi-output neural network (18,724 params) with dual heads for spectral and CIELAB predictions. Trained on 84 PP substrate samples (8μm & 12μm).
                     </p>
                   </div>
                 </div>
               </div>
             </div>
          </div>
        </div>


        {/* Data Table */}
        <div className="bg-slate-850 rounded-lg border border-slate-700 overflow-hidden">
           <div className="px-6 py-4 border-b border-slate-700 flex justify-between items-center">
             <h3 className="font-semibold text-slate-200">Training Dataset</h3>
             <span className="text-xs text-slate-500">{samples.length} Samples Loaded</span>
           </div>
           <div className="overflow-x-auto">
             <table className="w-full text-sm text-left text-slate-400">
               <thead className="text-xs text-slate-300 uppercase bg-slate-800">
                 <tr>
                   <th className="px-6 py-3">Sample</th>
                   <th className="px-6 py-3">GXT (%)</th>
                  <th className="px-6 py-3">BiVaO4 (%)</th>
                   <th className="px-6 py-3">PG (%)</th>
                   <th className="px-6 py-3">PearlB (%)</th>
                   <th className="px-6 py-3">R @ 520nm</th>
                 </tr>
               </thead>
               <tbody>
                 {samples.map((s, idx) => (
                   <tr key={idx} className="border-b border-slate-700 hover:bg-slate-800/50">
                     <td className="px-6 py-4 font-medium text-slate-200">{s.name}</td>
                     <td className="px-6 py-4 font-mono">{s.concentrations['GXT']?.toFixed(2) || '0.00'}</td>
                    <td className="px-6 py-4 font-mono">{s.concentrations['BiVaO4']?.toFixed(2) || '0.00'}</td>
                     <td className="px-6 py-4 font-mono">{s.concentrations['PG']?.toFixed(2) || '0.00'}</td>
                     <td className="px-6 py-4 font-mono">{s.concentrations['PearlB']?.toFixed(2) || '0.00'}</td>
                     <td className="px-6 py-4 text-cyan-450 font-mono">{s.spectrum[12]?.toFixed(4)}</td>
                   </tr>
                 ))}
               </tbody>
             </table>
           </div>
        </div>
      </main>
    </div>
  );
}