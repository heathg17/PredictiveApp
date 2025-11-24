module.exports = {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
    './App.tsx',
    './components/**/*.{ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        slate: {
          850: '#1e293b',
          900: '#0f172a'
        },
        cyan: {
          450: '#22d3ee'
        }
      }
    }
  },
  plugins: []
}
