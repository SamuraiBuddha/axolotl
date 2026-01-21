#!/bin/bash

# Axolotl Setup Script
# This script sets up Axolotl with proper Python version and dependencies

set -e  # Exit on error

echo "======================================"
echo "       Axolotl Setup Script"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3.11 is available
check_python() {
    echo "Checking for Python 3.11..."
    
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        echo -e "${GREEN}✓ Python 3.11 found${NC}"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [ "$PYTHON_VERSION" = "3.11" ]; then
            PYTHON_CMD="python3"
            echo -e "${GREEN}✓ Python 3.11 found${NC}"
        else
            echo -e "${YELLOW}⚠ Python 3.11 not found. Current version: $(python3 --version)${NC}"
            echo "Please install Python 3.11:"
            echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev"
            echo "  Or use pyenv: pyenv install 3.11.9"
            exit 1
        fi
    else
        echo -e "${RED}✗ Python not found${NC}"
        exit 1
    fi
}

# Check CUDA availability
check_cuda() {
    echo "Checking CUDA installation..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
            echo -e "${GREEN}✓ CUDA $CUDA_VERSION found${NC}"
        else
            echo -e "${YELLOW}⚠ CUDA toolkit not found. Installing PyTorch with CUDA support anyway...${NC}"
        fi
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    else
        echo -e "${YELLOW}⚠ No NVIDIA GPU detected. Installing CPU version...${NC}"
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    fi
}

# Create virtual environment
setup_venv() {
    VENV_DIR="venv_axolotl"
    
    if [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Virtual environment already exists. Do you want to recreate it? (y/n)${NC}"
        read -r response
        if [[ "$response" = "y" ]]; then
            rm -rf "$VENV_DIR"
        else
            echo "Using existing virtual environment..."
            return
        fi
    fi
    
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv $VENV_DIR
    echo -e "${GREEN}✓ Virtual environment created${NC}"
}

# Install dependencies
install_deps() {
    echo "Activating virtual environment..."
    source venv_axolotl/bin/activate
    
    echo "Upgrading pip and essential tools..."
    pip install --upgrade pip
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja
    
    # Install PyTorch first (required for some dependencies)
    echo "Installing PyTorch..."
    if [ "$TORCH_INDEX" = "https://download.pytorch.org/whl/cpu" ]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    else
        pip install torch torchvision torchaudio
    fi
    
    echo "Installing Axolotl..."
    
    # Check if we're in development mode
    echo -e "${YELLOW}Install in development mode? (recommended for making changes) (y/n)${NC}"
    read -r dev_mode
    
    if [[ "$dev_mode" = "y" ]]; then
        echo "Installing in development mode..."
        pip install -e ".[flash-attn,deepspeed]" --no-build-isolation
        echo "Installing development dependencies..."
        if [ -f requirements-dev.txt ]; then
            pip install -r requirements-dev.txt
        fi
        if [ -f requirements-tests.txt ]; then
            pip install -r requirements-tests.txt
        fi
    else
        echo "Installing in production mode..."
        pip install ".[flash-attn,deepspeed]" --no-build-isolation
    fi
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Download example configs
download_examples() {
    echo "Downloading example configurations..."
    source venv_axolotl/bin/activate
    axolotl fetch examples
    axolotl fetch deepspeed_configs
    echo -e "${GREEN}✓ Examples downloaded${NC}"
}

# Main execution
main() {
    check_python
    check_cuda
    setup_venv
    install_deps
    download_examples
    
    echo ""
    echo "======================================"
    echo -e "${GREEN}     Setup Complete!${NC}"
    echo "======================================"
    echo ""
    echo "To activate the environment, run:"
    echo -e "${YELLOW}  source venv_axolotl/bin/activate${NC}"
    echo ""
    echo "Quick start commands:"
    echo "  axolotl train examples/llama-3/lora-1b.yml    # Train a small model"
    echo "  axolotl fetch examples                        # Get more examples"
    echo "  ./axolotl_helper.sh                           # Use helper script"
    echo ""
}

# Run main function
main