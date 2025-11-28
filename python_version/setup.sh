#!/bin/bash
# Setup script for OptiMix Python/PyTorch version

echo "============================================================"
echo "OptiMix - Spectral Formulation Engine Setup"
echo "Python/PyTorch Version"
echo "============================================================"
echo ""

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"
echo ""

# Check pip
echo "Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi
echo "✓ pip3 is available"
echo ""

# Create virtual environment (optional but recommended)
read -p "Create a virtual environment? (recommended) [Y/n]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv

    echo "Activating virtual environment..."
    source venv/bin/activate

    echo "✓ Virtual environment created and activated"
    echo ""
    echo "To activate it later, run: source venv/bin/activate"
    echo "To deactivate, run: deactivate"
    echo ""
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Check for GPU support
echo "Checking for GPU support..."
python3 -c "import torch; print('✓ GPU available' if torch.cuda.is_available() else '  CPU only (no GPU detected)')"
echo ""

# Run installation test
echo "Running installation tests..."
python3 test_installation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Setup complete! ✓"
    echo "============================================================"
    echo ""
    echo "Quick start:"
    echo "  1. Run with initial data:"
    echo "     python3 main.py --use-initial"
    echo ""
    echo "  2. Run without plots (headless):"
    echo "     python3 main.py --use-initial --no-plot"
    echo ""
    echo "  3. Load master CSV files:"
    echo "     python3 main.py --data-dir ../public"
    echo ""
    echo "See README.md for more information."
    echo "============================================================"
else
    echo ""
    echo "❌ Installation tests failed. Please check the error messages above."
    exit 1
fi
