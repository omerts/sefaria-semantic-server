#!/bin/bash
# Quick setup script

set -e

echo "=========================================="
echo "Torah Source Finder - Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "⚠ Warning: Python 3.8+ required. Current version: $PYTHON_VERSION"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ./.venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ./.venv/bin/activate

# Check and install uv
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    # Update pip first (required for uv installation)
    pip install --upgrade pip --quiet || {
        echo "✗ Error updating pip"
        exit 1
    }
    
    # Install uv
    pip install uv --quiet || {
        echo ""
        echo "✗ Error installing uv"
        echo ""
        echo "Try installing manually:"
        echo "  pip install uv"
        echo "or:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    }
    echo "✓ uv installed"
else
    echo "✓ uv found"
fi

# Install dependencies with uv
echo "Installing dependencies with uv..."
# Use uv pip install to avoid building the project as a package
uv pip install -r requirements.txt --native-tls || {
    echo ""
    echo "✗ Error installing dependencies with uv"
    echo ""
    echo "Try one of the following:"
    echo "1. Update uv: pip install --upgrade uv"
    echo "2. Check internet connection"
    echo "3. Try again: uv pip install -r requirements.txt --native-tls"
    exit 1
}

echo "✓ Installation completed with uv"

# Create directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/normalized
mkdir -p data/chunks
mkdir -p models
mkdir -p training

echo ""
echo "=========================================="
echo "✓ Setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant"
echo "2. Activate venv and run the pipeline:"
echo "   source ./.venv/bin/activate"
echo "   export DISABLE_SSL_VERIFY=1  # If you have SSL certificate issues"
echo "   python run_pipeline.py"
echo "3. Or run steps manually according to README.md"
echo ""
echo "Note: If you have SSL issues with uv, use ./.venv instead of 'uv run'"
echo ""
