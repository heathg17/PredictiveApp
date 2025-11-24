/**
 * Simple Matrix Utilities for Client-Side K-M Calculations
 */

// Create a zero-filled matrix
export const zeros = (rows: number, cols: number): number[][] => {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
};

// Transpose a matrix
export const transpose = (A: number[][]): number[][] => {
  const rows = A.length;
  const cols = A[0].length;
  const AT = zeros(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      AT[j][i] = A[i][j];
    }
  }
  return AT;
};

// Multiply two matrices
export const multiply = (A: number[][], B: number[][]): number[][] => {
  const m = A.length;
  const n = A[0].length;
  const p = B[0].length;
  if (n !== B.length) {
    throw new Error(`Matrix dimension mismatch: ${m}x${n} vs ${B.length}x${p}`);
  }
  const C = zeros(m, p);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
  return C;
};

// Multiply Matrix by Vector
export const multiplyVector = (A: number[][], v: number[]): number[] => {
  const res = multiply(A, v.map(x => [x]));
  return res.map(row => row[0]);
};

// Pseudo-inverse using (A^T * A)^-1 * A^T
// Note: This involves inverting a square matrix.
// Since we are in client-side JS, we will use a basic Gaussian elimination for inversion.
// This is sufficient for training on ~20 reagents.
export const pseudoInverse = (A: number[][]): number[][] => {
  const AT = transpose(A);
  const ATA = multiply(AT, A);
  const ATA_inv = invertMatrix(ATA);
  return multiply(ATA_inv, AT);
};

// Basic Gaussian Elimination for Matrix Inversion with pivoting
export const invertMatrix = (M: number[][]): number[][] => {
  const n = M.length;
  // Create augmented matrix [M | I]
  const A = M.map(row => [...row]);
  const I = zeros(n, n);
  for(let i=0; i<n; i++) I[i][i] = 1;

  // Combine
  const aug = A.map((row, i) => [...row, ...I[i]]);

  for (let i = 0; i < n; i++) {
    // Partial pivoting: find row with largest absolute value in column i
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(aug[k][i]) > Math.abs(aug[maxRow][i])) {
        maxRow = k;
      }
    }

    // Swap rows if needed
    if (maxRow !== i) {
      [aug[i], aug[maxRow]] = [aug[maxRow], aug[i]];
    }

    // Check for singularity
    let pivot = aug[i][i];
    if (Math.abs(pivot) < 1e-10) {
      throw new Error(`Matrix is singular or nearly singular at row ${i}. Cannot invert.`);
    }

    // Scale pivot row
    for (let j = 0; j < 2 * n; j++) {
      aug[i][j] /= pivot;
    }

    // Eliminate column
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = aug[k][i];
        for (let j = 0; j < 2 * n; j++) {
          aug[k][j] -= factor * aug[i][j];
        }
      }
    }
  }

  // Extract inverse
  const inv = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      inv[i][j] = aug[i][j + n];
    }
  }
  return inv;
};
