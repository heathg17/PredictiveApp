
// TypeScript Neural Network Predictions
// Copy this into the browser console while running npm run dev

const testFormulations = [
  {
    "name": "Test 1: Yellow Dominant",
    "concentrations": {
      "BiVaO4": 10.0,
      "LY": 5.0
    },
    "thickness": 4.0
  },
  {
    "name": "Test 2: Green Mix",
    "concentrations": {
      "PG": 8.0,
      "LY": 3.0
    },
    "thickness": 4.0
  },
  {
    "name": "Test 3: Blue Tint",
    "concentrations": {
      "PB": 2.0,
      "TiO2": 1.0
    },
    "thickness": 4.0
  },
  {
    "name": "Test 4: Fluorescent High",
    "concentrations": {
      "GXT": 25.0,
      "BiVaO4": 5.0
    },
    "thickness": 4.0
  }
];

console.log("TypeScript Neural Network Predictions:");
console.log("======================================\n");

// You'll need to manually trigger predictions in the UI
// or extract the prediction logic to run here

testFormulations.forEach(test => {
  console.log(`${test.name}:`);
  console.log(`  Formulation: ${JSON.stringify(test.concentrations)}`);
  console.log(`  Set these values in the UI sliders and observe the purple curve`);
  console.log("");
});

console.log("Compare the purple line (Neural Network) in the UI with PyTorch predictions");
