#!/bin/bash

# BYOVLA-Kinova Installation Script
# This script sets up the complete environment for BYOVLA-Kinova integration

set -e  # Exit on any error

echo "🚀 BYOVLA-Kinova Installation Script"
echo "=================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "byovla_kinova.py" ]; then
    echo "❌ Please run this script from the directory containing byovla_kinova.py"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Create conda environment
echo ""
echo "📦 Creating conda environment 'byovla_kinova'..."
if conda env list | grep -q "byovla_kinova"; then
    echo "⚠️  Environment 'byovla_kinova' already exists"
    read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n byovla_kinova
    else
        echo "🔄 Using existing environment"
    fi
fi

if ! conda env list | grep -q "byovla_kinova"; then
    echo "🔨 Creating new environment..."
    conda env create -f environment_byovla_kinova.yml
fi

# Activate environment
echo ""
echo "🔌 Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate byovla_kinova

echo "✅ Environment activated: $(conda info --envs | grep '*')"

# Install Kinova API
echo ""
echo "🤖 Installing Kinova API..."
if [ -f "kortex_api-2.6.0.post3-py3-none-any.whl" ]; then
    python3 setup_kinova.py
else
    echo "⚠️  Kinova API wheel file not found"
    echo "Please ensure kortex_api-2.6.0.post3-py3-none-any.whl is in the current directory"
fi

# Install additional dependencies
echo ""
echo "📚 Installing additional dependencies..."
pip install -r requirements_byovla_kinova.txt

# Test installation
echo ""
echo "🧪 Testing installation..."
python3 test_byovla_kinova.py

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Connect your Kinova robot"
echo "2. Connect your RealSense camera"
echo "3. Update robot IP address in byovla_kinova.py if needed"
echo "4. Run: python3 byovla_kinova.py --instruction 'your task'"
echo ""
echo "📖 For more information, see README_byovla_kinova.md"
echo ""
echo "🔧 To activate the environment in the future:"
echo "   conda activate byovla_kinova" 