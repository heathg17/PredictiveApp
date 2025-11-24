import { SampleData } from '../types';
import { REAGENTS_LIST, WAVELENGTHS } from '../constants';

// Split CSV line by commas not inside quotes
const splitCSVLine = (line: string) => line.split(/,(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)/).map(s => s.replace(/^\"|\"$/g, '').trim());

const isNumeric = (v: string) => v !== '' && !Number.isNaN(Number(v));

export function parseSamplesFromCSV(content: string): SampleData[] {
  const lines = content.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length < 2) return [];

  const header = splitCSVLine(lines[0]).map(h => h.trim());
  const rows = lines.slice(1).map(r => splitCSVLine(r));

  // Map header -> index
  const idx: Record<string, number> = {};
  header.forEach((h, i) => idx[h] = i);

  // Heuristics to find spectrum columns
  const spectrumIndices: number[] = [];
  header.forEach((h, i) => {
    const hLow = h.toLowerCase();
    if (/^\d{3}(nm)?$/.test(hLow) || /^r[_-]?\d{3}$/.test(hLow) || hLow.startsWith('r_') || hLow.startsWith('reflect') || hLow.includes('spec')) {
      spectrumIndices.push(i);
    }
  });

  // If not detected, try to find a contiguous block of numeric columns of length close to WAVELENGTHS.length
  if (spectrumIndices.length < Math.min(5, WAVELENGTHS.length)) {
    const numericMask = header.map((_, col) => rows.filter(r => r[col] !== undefined).length > 0 && rows.reduce((c, r) => c + (isNumeric(r[col]) ? 1 : 0), 0));
    // find longest contiguous block where most values numeric and within 0..1
    let bestStart = -1;
    let bestLen = 0;
    for (let start = 0; start < header.length; start++) {
      for (let len = 1; len <= header.length - start; len++) {
        const cols = Array.from({ length: len }, (_, k) => start + k);
        const numericCount = cols.reduce((s, c) => s + rows.reduce((a, r) => a + (isNumeric(r[c]) ? 1 : 0), 0), 0);
        const total = cols.length * rows.length;
        if (rows.length > 0 && numericCount / total > 0.6 && len > bestLen) {
          bestStart = start;
          bestLen = len;
        }
      }
    }
    if (bestStart >= 0 && bestLen > 0) {
      for (let i = bestStart; i < bestStart + bestLen; i++) spectrumIndices.push(i);
    }
  }

  // Build samples
  const samples: SampleData[] = rows.map((r, rowIdx) => {
    const get = (name: string) => {
      if (idx[name] !== undefined) return r[idx[name]];
      return undefined;
    };

    const id = get('id') || get('ID') || get('name') || `row-${rowIdx + 1}`;
    const name = get('name') || id;
    const substrate = get('substrate') || 'Unknown';
    const thicknessStr = get('thickness') || get('Thickness') || get('X') || get('x') || '';
    const thickness = thicknessStr ? Number(thicknessStr) : 4;

    // Spectrum
    let spectrum: number[] = [];
    if (spectrumIndices.length > 0) {
      spectrum = spectrumIndices.map(i => {
        const v = r[i];
        const num = Number(v);
        return Number.isFinite(num) ? num : 0;
      });
      // If spectrum length differs from expected, pad or slice to match WAVELENGTHS
      if (spectrum.length !== WAVELENGTHS.length) {
        if (spectrum.length > WAVELENGTHS.length) spectrum = spectrum.slice(0, WAVELENGTHS.length);
        else {
          while (spectrum.length < WAVELENGTHS.length) spectrum.push(0);
        }
      }
    } else {
      spectrum = new Array(WAVELENGTHS.length).fill(0);
    }

    // Concentrations: find columns matching reagent names or numeric columns not used for spectrum/thickness
    const concentrations: Record<string, number> = {};
    const usedCols = new Set<number>(spectrumIndices);
    if (idx['thickness'] !== undefined) usedCols.add(idx['thickness']);

    header.forEach((h, col) => {
      const hTrim = h.trim();
      // Exact reagent name
      if (REAGENTS_LIST.includes(hTrim)) {
        const val = Number(r[col]);
        concentrations[hTrim] = Number.isFinite(val) ? val : 0;
        usedCols.add(col);
        return;
      }
      // If numeric column and not in usedCols, treat as concentration candidate
      if (!usedCols.has(col) && rows.some(row => isNumeric(row[col]))) {
        // use header as reagent name if short
        if (hTrim.length > 0 && hTrim.length <= 20 && !/^\d+$/.test(hTrim)) {
          const val = Number(r[col]);
          concentrations[hTrim] = Number.isFinite(val) ? val : 0;
          usedCols.add(col);
        }
      }
    });

    return {
      id: String(id),
      name: String(name),
      substrate: String(substrate),
      thickness,
      spectrum,
      concentrations
    } as SampleData;
  });

  return samples;
}
