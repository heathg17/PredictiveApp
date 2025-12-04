/**
 * Load PP substrate data from Concentrations.csv and Spectra.csv
 *
 * Concentrations.csv format:
 * Sample ID,GXT-10 (%),BiVaO4 (%),PG (%),PearlB (%)
 *
 * Spectra.csv format (semicolon-separated):
 * Line 1: sep=;
 * Line 2-3: metadata
 * Line 4: Header with column names
 * Line 5+: Data rows
 */
import { SampleData } from '../types';

export async function loadPPData(): Promise<SampleData[]> {
  try {
    // Load both CSV files
    const concResponse = await fetch('/Concentrations.csv');
    const spectraResponse = await fetch('/Spectra.csv');

    if (!concResponse.ok || !spectraResponse.ok) {
      console.error('Failed to load PP substrate data files');
      console.error(`  Concentrations.csv: ${concResponse.status}`);
      console.error(`  Spectra.csv: ${spectraResponse.status}`);
      return [];
    }

    const concText = await concResponse.text();
    const spectraText = await spectraResponse.text();

    // Parse Concentrations.csv
    const concLines = concText.split(/\r?\n/).filter(line => line.trim().length > 0);
    if (concLines.length < 2) {
      console.error('Concentrations.csv is empty or has no data rows');
      return [];
    }

    const concHeader = concLines[0].split(',').map(h => h.trim());
    console.log('Concentration header:', concHeader);

    // Build map of sample -> concentrations
    const concentrations = new Map<string, Record<string, number>>();

    for (let i = 1; i < concLines.length; i++) {
      const values = concLines[i].split(',').map(v => v.trim());
      if (values.length < 5) {
        console.warn(`Skipping concentration row ${i}: insufficient columns`);
        continue;
      }

      const sampleId = values[0];

      // Parse concentrations by column (handle GXT-10 or GXT)
      const concData: Record<string, number> = {
        'GXT': parseFloat(values[1].replace('%', '')) || 0,
        'BiVaO4': parseFloat(values[2].replace('%', '')) || 0,
        'PG': parseFloat(values[3].replace('%', '')) || 0,
        'PearlB': parseFloat(values[4].replace('%', '')) || 0,
      };

      concentrations.set(sampleId, concData);

      // Also store normalized versions for matching
      const normalized = sampleId.replace(/[_\s]/g, '');
      if (normalized !== sampleId) {
        concentrations.set(normalized, concData);
      }

      // Handle OPTI samples with space variation (OPTI6 and OPTI 6)
      if (sampleId.startsWith('OPTI ')) {
        const withoutSpace = sampleId.replace(/\s+/g, '');
        concentrations.set(withoutSpace, concData);
      }
    }

    console.log(`✓ Loaded ${concentrations.size} concentration entries`);

    // Parse Spectra.csv (semicolon-separated, skip first 3 rows)
    const spectraLines = spectraText.split(/\r?\n/).filter(line => line.trim().length > 0);
    if (spectraLines.length < 5) {
      console.error('Spectra.csv is empty or has insufficient rows');
      return [];
    }

    // Line 0: sep=;
    // Line 1-2: metadata
    // Line 3: Header
    // Line 4+: Data
    const spectraHeader = spectraLines[3].split(';').map(h => h.trim().replace(/^"|"$/g, ''));
    console.log(`Spectra header has ${spectraHeader.length} columns`);

    // Find wavelength column indices (R400 nm through R700 nm)
    const wavelengthIndices: number[] = [];
    for (let i = 0; i < spectraHeader.length; i++) {
      const header = spectraHeader[i];
      if (header.startsWith('R') && header.includes('nm')) {
        wavelengthIndices.push(i);
      }
    }

    console.log(`Found ${wavelengthIndices.length} wavelength columns`);

    if (wavelengthIndices.length !== 31) {
      console.error(`Expected 31 wavelength columns, found ${wavelengthIndices.length}`);
    }

    // Find other column indices
    const nameIndex = spectraHeader.findIndex(h => h === 'User Color Name');

    if (nameIndex === -1) {
      console.error('Could not find "User Color Name" column in Spectra.csv');
      console.error('Available columns:', spectraHeader.slice(0, 10));
      return [];
    }

    console.log(`User Color Name is at column ${nameIndex}`);

    const samples: SampleData[] = [];
    const seenSamples = new Map<string, number>(); // Track duplicates

    // Parse spectral data (starting from line 4, index 4)
    for (let i = 4; i < spectraLines.length; i++) {
      const values = spectraLines[i].split(';').map(v => v.trim().replace(/^"|"$/g, ''));
      if (values.length < 10) {
        console.warn(`Skipping spectral row ${i + 1}: insufficient columns (${values.length})`);
        continue;
      }

      const sampleName = values[nameIndex] || `Sample_${i}`;
      if (!sampleName || sampleName === '-') continue;

      // Determine thickness from name (H suffix = 12μm, otherwise 8μm)
      const thickness = sampleName.endsWith('H') ? 12.0 : 8.0;

      // Get base name for concentration lookup (remove H suffix if present)
      let baseName = sampleName.endsWith('H') ? sampleName.slice(0, -1) : sampleName;

      // Try multiple matching strategies
      let sampleConc = concentrations.get(baseName)
                    || concentrations.get(sampleName)
                    || concentrations.get(baseName.replace(/[_\s]/g, ''))
                    || concentrations.get(sampleName.replace(/[_\s]/g, ''));

      // Special handling for OPTI samples
      if (!sampleConc && baseName.startsWith('OPTI')) {
        const optiNum = baseName.replace('OPTI', '').replace(/[_\s]/g, '');
        sampleConc = concentrations.get(`OPTI ${optiNum}`)
                  || concentrations.get(`OPTI${optiNum}`);
      }

      if (!sampleConc) {
        console.warn(`No concentration data for ${sampleName} (base: ${baseName})`);
        continue;
      }

      // Extract spectrum
      const spectrum: number[] = [];
      for (const idx of wavelengthIndices) {
        const value = parseFloat(values[idx]);
        spectrum.push(isNaN(value) ? 0 : value);
      }

      // Validate spectrum length
      if (spectrum.length !== 31) {
        console.warn(`Invalid spectrum length for ${sampleName}: ${spectrum.length} (expected 31)`);
        continue;
      }

      // Check if spectrum is valid (not all zeros)
      const validValues = spectrum.filter(v => v > 0).length;
      if (validValues < 15) {
        console.warn(`Spectrum for ${sampleName} has too few valid values: ${validValues}`);
        continue;
      }

      const sampleId = `${baseName}_${thickness}um`;

      const newSample: SampleData = {
        id: sampleId,
        name: sampleName,
        substrate: 'PP',
        thickness,
        spectrum,
        concentrations: sampleConc
      };

      // Handle duplicates (keep the one with more valid spectral values)
      if (seenSamples.has(sampleId)) {
        const existingIdx = seenSamples.get(sampleId)!;
        const existingSample = samples[existingIdx];
        const existingValidValues = existingSample.spectrum.filter(v => v > 0).length;

        if (validValues > existingValidValues) {
          console.log(`Replacing duplicate ${sampleId}: ${existingValidValues} → ${validValues} valid values`);
          samples[existingIdx] = newSample;
        } else {
          console.log(`Skipping duplicate ${sampleId}: keeping existing with ${existingValidValues} valid values`);
        }
      } else {
        seenSamples.set(sampleId, samples.length);
        samples.push(newSample);
      }
    }

    console.log(`✓ Loaded ${samples.length} PP substrate samples from CSV files`);
    console.log(`  8μm samples: ${samples.filter(s => s.thickness === 8).length}`);
    console.log(`  12μm samples: ${samples.filter(s => s.thickness === 12).length}`);

    return samples;

  } catch (error) {
    console.error('Error loading PP substrate data:', error);
    return [];
  }
}
