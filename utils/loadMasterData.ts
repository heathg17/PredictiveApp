import { SampleData } from '../types';
import { WAVELENGTHS } from '../constants';

/**
 * Merges concentration data and spectral data from the master CSVs
 *
 * Master conc.csv format:
 * Sample,Substrate,Thickness (um),LY (%),GXT (%),...
 *
 * Master spec CSV format (no header):
 * SampleName,empty,empty,L*,a*,b*,empty,empty,R400,R410,...,R700,FilePath,empty
 */
export async function loadMasterData(): Promise<SampleData[]> {
  try {
    // Load both CSV files
    const concResponse = await fetch('/Master conc.csv');
    const specResponse = await fetch('/Master spec - master_sample_library.csv');

    if (!concResponse.ok || !specResponse.ok) {
      console.error('Failed to load master CSV files');
      return [];
    }

    const concText = await concResponse.text();
    const specText = await specResponse.text();

    // Parse concentration data
    const concLines = concText.split(/\r?\n/).filter(line => line.trim().length > 0);
    const concHeader = concLines[0].split(',').map(h => h.trim());
    const concRows = concLines.slice(1);

    // Build map of sample -> concentrations
    const concMap = new Map<string, { substrate: string; thickness: number; concentrations: Record<string, number> }>();

    for (const row of concRows) {
      const cells = row.split(',').map(c => c.trim());
      const sampleName = cells[0];
      const substrate = cells[1] || 'Unknown';
      const thickness = parseFloat(cells[2]) || 4;

      const concentrations: Record<string, number> = {};
      // Start from index 3 (after Sample, Substrate, Thickness)
      for (let i = 3; i < concHeader.length; i++) {
        const reagentName = concHeader[i].replace(/\s*\(%\)\s*$/, '').trim(); // Remove " (%)" suffix
        const value = parseFloat(cells[i]);
        if (!isNaN(value)) {
          concentrations[reagentName] = value;
        }
      }

      concMap.set(sampleName, { substrate, thickness, concentrations });
    }

    // Parse spectral data (no header row)
    const specLines = specText.split(/\r?\n/).filter(line => line.trim().length > 0);
    const samples: SampleData[] = [];
    const seenSamples = new Map<string, number>(); // Track sample names and their index

    for (const line of specLines) {
      const cells = line.split(',').map(c => c.trim());
      const sampleName = cells[0];

      // Skip if no sample name
      if (!sampleName) continue;

      // Extract spectrum (columns 8-38 should be the 31 reflectance values for 400-700nm)
      const spectrum: number[] = [];
      for (let i = 8; i < 39 && i < cells.length; i++) {
        const value = parseFloat(cells[i]);
        spectrum.push(isNaN(value) ? 0 : value);
      }

      // Ensure we have exactly 31 wavelength values
      while (spectrum.length < 31) {
        spectrum.push(0);
      }
      if (spectrum.length > 31) {
        spectrum.length = 31;
      }

      // Check if spectrum is valid (not mostly negative or zero)
      const validValues = spectrum.filter(v => v > 0).length;
      const isValidSpectrum = validValues >= 15; // At least half the values should be positive

      // Get concentration data from map, or use empty if not found
      const concData = concMap.get(sampleName) || {
        substrate: 'Unknown',
        thickness: 4,
        concentrations: {}
      };

      const newSample: SampleData = {
        id: sampleName,
        name: sampleName,
        substrate: concData.substrate,
        thickness: concData.thickness,
        spectrum: spectrum,
        concentrations: concData.concentrations
      };

      // Handle duplicates: if we've seen this sample before, only keep the one with valid spectrum
      if (seenSamples.has(sampleName)) {
        const existingIdx = seenSamples.get(sampleName)!;
        const existingSample = samples[existingIdx];

        // Check if existing sample has valid spectrum
        const existingValidValues = existingSample.spectrum.filter(v => v > 0).length;
        const existingIsValid = existingValidValues >= 15;

        // Replace existing if current is valid and existing is not, or if both are valid but current has more positive values
        if (isValidSpectrum && (!existingIsValid || validValues > existingValidValues)) {
          console.log(`Replacing duplicate sample ${sampleName}: existing had ${existingValidValues} valid values, new has ${validValues}`);
          samples[existingIdx] = newSample;
        } else {
          console.log(`Skipping duplicate sample ${sampleName}: keeping existing with ${existingValidValues} valid values`);
        }
      } else {
        // New sample, add it
        seenSamples.set(sampleName, samples.length);
        samples.push(newSample);
      }
    }

    console.log(`Loaded ${samples.length} samples from master CSVs`);
    return samples;

  } catch (error) {
    console.error('Error loading master data:', error);
    return [];
  }
}
