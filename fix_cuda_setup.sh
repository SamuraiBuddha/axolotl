#!/bin/bash

# Fix CUDA Setup for Flash Attention

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================"
echo "   Fixing CUDA Setup for Axolotl"
echo "======================================${NC}"
echo ""

# Check current CUDA status
echo "Checking CUDA installation..."

if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✓ nvcc found: $(nvcc --version | head -1)${NC}"
    CUDA_PATH=$(dirname $(dirname $(which nvcc)))
else
    echo -e "${YELLOW}⚠ nvcc not found${NC}"
    CUDA_PATH=""
fi

# Check for CUDA in common locations
for path in /opt/cuda /usr/local/cuda /usr/local/cuda-* /usr; do
    if [ -f "$path/bin/nvcc" ]; then
        echo -e "${GREEN}Found CUDA at: $path${NC}"
        CUDA_PATH="$path"
        break
    fi
done

echo ""
echo "Choose an option:"
echo "1) Install CUDA toolkit (recommended for full functionality)"
echo "2) Skip Flash Attention (use xformers instead - slightly slower)"
echo "3) Use pre-built wheels (if available)"
echo ""
echo -n "Enter choice (1-3): "
read choice

case $choice in
    1)
        echo -e "${GREEN}Installing CUDA toolkit...${NC}"
        
        # For Arch/EndeavourOS
        echo "Installing CUDA packages..."
        sudo pacman -S --needed cuda cuda-tools
        
        # Set environment variables
        echo -e "${YELLOW}Setting up environment variables...${NC}"
        
        # Add to current session
        export CUDA_HOME=/opt/cuda
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        
        # Add to bashrc for persistence
        echo "" >> ~/.bashrc
        echo "# CUDA configuration" >> ~/.bashrc
        echo "export CUDA_HOME=/opt/cuda" >> ~/.bashrc
        echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        
        echo -e "${GREEN}✓ CUDA toolkit installed!${NC}"
        echo ""
        echo "Now run:"
        echo "  source ~/.bashrc"
        echo "  ./setup_axolotl.sh"
        ;;
        
    2)
        echo -e "${YELLOW}Installing Axolotl without Flash Attention...${NC}"
        
        # Activate virtual environment
        source venv_axolotl/bin/activate
        
        # Install without flash-attn
        echo "Installing Axolotl with xformers only..."
        pip install -e ".[deepspeed]" --no-build-isolation
        
        # Install xformers separately if needed
        pip install xformers
        
        echo -e "${GREEN}✓ Installed without Flash Attention${NC}"
        echo ""
        echo -e "${YELLOW}Note: Training will use xformers instead of Flash Attention.${NC}"
        echo "This is slightly slower but doesn't require CUDA compilation."
        ;;
        
    3)
        echo -e "${BLUE}Attempting to use pre-built wheels...${NC}"
        
        # Activate virtual environment
        source venv_axolotl/bin/activate
        
        # Try to install pre-built flash-attn wheel
        echo "Checking for pre-built Flash Attention wheels..."
        
        # Get CUDA version from PyTorch
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', ''))")
        
        if [ -z "$CUDA_VERSION" ]; then
            echo -e "${RED}PyTorch is CPU-only. Installing GPU version first...${NC}"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            CUDA_VERSION="121"
        fi
        
        echo "Detected CUDA version: $CUDA_VERSION"
        
        # Try to install pre-built wheel
        pip install flash-attn --no-build-isolation
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Pre-built wheel installed successfully!${NC}"
            
            # Continue with axolotl installation
            pip install -e ".[deepspeed]" --no-build-isolation
        else
            echo -e "${YELLOW}Pre-built wheel not available. Falling back to option 2...${NC}"
            pip install -e ".[deepspeed]" --no-build-isolation
            pip install xformers
        fi
        ;;
        
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""

# Test the installation
echo "Testing Axolotl installation..."
source venv_axolotl/bin/activate

if python -c "import axolotl" 2>/dev/null; then
    echo -e "${GREEN}✓ Axolotl imported successfully!${NC}"
    
    # Check available optimizations
    echo ""
    echo "Available optimizations:"
    python -c "
try:
    import flash_attn
    print('  ✓ Flash Attention')
except:
    print('  ✗ Flash Attention (not available)')
    
try:
    import xformers
    print('  ✓ xformers')
except:
    print('  ✗ xformers (not available)')
    
try:
    import deepspeed
    print('  ✓ DeepSpeed')
except:
    print('  ✗ DeepSpeed (not available)')
    
import torch
if torch.cuda.is_available():
    print(f'  ✓ CUDA: {torch.cuda.get_device_name(0)}')
else:
    print('  ✗ CUDA (not available)')
"
else
    echo -e "${RED}✗ Failed to import Axolotl${NC}"
fi

echo ""
echo "Next steps:"
echo "1. Download example configs: axolotl fetch examples"
echo "2. Try a quick training: axolotl train examples/llama-3/lora-1b.yml"
echo "3. Or launch the GUI: ./launch_gui.sh"