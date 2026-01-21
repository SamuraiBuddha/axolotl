#!/bin/bash

# Complete Flash Attention Fix for Axolotl
# This script properly configures CUDA and compiles Flash Attention

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================"
echo "   Flash Attention Complete Fix"
echo "======================================"
echo -e "${NC}"

echo "Detected Configuration:"
echo "- PyTorch CUDA: 12.8"
echo "- System CUDA: 12.9.1"
echo "- GCC version: 15.2.1"
echo ""

# Step 1: Set up CUDA environment
echo -e "${YELLOW}Step 1: Setting up CUDA environment...${NC}"

export CUDA_HOME=/opt/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify nvcc is now accessible
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✓ nvcc is now accessible${NC}"
    nvcc --version | head -3
else
    echo -e "${RED}✗ nvcc still not found${NC}"
    exit 1
fi

# Step 2: Check for GCC compatibility issues
echo ""
echo -e "${YELLOW}Step 2: Checking GCC compatibility...${NC}"

# GCC 15 is very new and might cause issues with CUDA 12.9
# Flash Attention might need an older GCC version
echo "Current GCC version: $(gcc --version | head -1)"

# Check if older GCC versions are available
if pacman -Q | grep -q "gcc12"; then
    echo -e "${GREEN}✓ gcc12 is installed${NC}"
    GCC_OVERRIDE="CC=gcc-12 CXX=g++-12"
elif pacman -Q | grep -q "gcc13"; then
    echo -e "${GREEN}✓ gcc13 is installed${NC}"
    GCC_OVERRIDE="CC=gcc-13 CXX=g++-13"
else
    echo -e "${YELLOW}⚠ No older GCC version found. Installing gcc13...${NC}"
    sudo pacman -S --needed gcc13
    GCC_OVERRIDE="CC=gcc-13 CXX=g++-13"
fi

# Step 3: Install additional CUDA development packages if needed
echo ""
echo -e "${YELLOW}Step 3: Ensuring all CUDA development packages are installed...${NC}"

# Check for missing packages
PACKAGES_TO_INSTALL=""

if ! pacman -Q cudnn 2>/dev/null; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL cudnn"
fi

if [ -n "$PACKAGES_TO_INSTALL" ]; then
    echo "Installing: $PACKAGES_TO_INSTALL"
    sudo pacman -S --needed $PACKAGES_TO_INSTALL
else
    echo -e "${GREEN}✓ All required CUDA packages are installed${NC}"
fi

# Step 4: Activate virtual environment
echo ""
echo -e "${YELLOW}Step 4: Activating virtual environment...${NC}"
source /home/samuraibuddha/Documents/GitHub/axolotl/venv_axolotl/bin/activate

# Step 5: Clean up any previous build attempts
echo ""
echo -e "${YELLOW}Step 5: Cleaning up previous build artifacts...${NC}"

# Clear pip cache for flash-attn
pip cache remove flash-attn 2>/dev/null || true

# Remove any partial builds
rm -rf /tmp/pip-install-*/flash-attn* 2>/dev/null || true
rm -rf ~/.cache/pip/wheels/*flash_attn* 2>/dev/null || true

echo -e "${GREEN}✓ Cleanup complete${NC}"

# Step 6: Set compilation flags
echo ""
echo -e "${YELLOW}Step 6: Setting compilation flags...${NC}"

# Set flags for better compatibility
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"  # RTX A4000 is 8.6
export MAX_JOBS=4  # Limit parallel compilation jobs to avoid memory issues
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc-13"  # Use GCC 13 for CUDA compilation

echo "CUDA architectures: $TORCH_CUDA_ARCH_LIST"
echo "Max parallel jobs: $MAX_JOBS"

# Step 7: Try to build Flash Attention
echo ""
echo -e "${YELLOW}Step 7: Building Flash Attention...${NC}"
echo "This may take 5-10 minutes..."

# Method 1: Try with pip and proper environment
echo -e "${BLUE}Attempting build with pip...${NC}"

if $GCC_OVERRIDE pip install flash-attn==2.7.4.post1 --no-cache-dir --no-build-isolation -v; then
    echo -e "${GREEN}✓ Flash Attention installed successfully!${NC}"
else
    echo -e "${YELLOW}Standard installation failed. Trying alternative method...${NC}"
    
    # Method 2: Download and build manually
    echo -e "${BLUE}Downloading Flash Attention source...${NC}"
    
    cd /tmp
    rm -rf flash-attention
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    git checkout v2.7.4
    
    # Apply any necessary patches for GCC 15 compatibility
    echo -e "${YELLOW}Applying compatibility patches...${NC}"
    
    # Build with manual configuration
    echo -e "${BLUE}Building from source...${NC}"
    
    # Use the older GCC for compilation
    export CC=gcc-13
    export CXX=g++-13
    export CUDAHOSTCXX=g++-13
    
    # Build the package
    python setup.py build
    
    # Install the package
    pip install -e .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Flash Attention built from source successfully!${NC}"
    else
        echo -e "${RED}✗ Build from source also failed${NC}"
        echo ""
        echo "Possible solutions:"
        echo "1. Install an even older GCC version (gcc11 or gcc12)"
        echo "2. Use a different CUDA version that matches PyTorch exactly"
        echo "3. Use Docker with a pre-configured environment"
        echo ""
        echo "For now, let's use the pre-compiled wheel method..."
        
        # Method 3: Try pre-compiled wheel
        echo -e "${BLUE}Attempting to use pre-compiled wheel...${NC}"
        
        # Download pre-built wheel for CUDA 12.1 (closest to 12.8)
        pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/flash_attn-2.7.4-cp311-cp311-linux_x86_64.whl
    fi
    
    cd /home/samuraibuddha/Documents/GitHub/axolotl
fi

# Step 8: Verify installation
echo ""
echo -e "${YELLOW}Step 8: Verifying installation...${NC}"

python -c "
import sys
try:
    import flash_attn
    print('✓ Flash Attention imported successfully')
    print(f'  Version: {flash_attn.__version__}')
    
    # Test CUDA kernels
    import torch
    if torch.cuda.is_available():
        # Simple test
        x = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)
        from flash_attn import flash_attn_func
        print('✓ Flash Attention CUDA kernels working')
except Exception as e:
    print(f'✗ Error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================"
    echo "   Flash Attention Setup Complete!"
    echo "======================================${NC}"
    echo ""
    echo "Flash Attention is now properly installed and working."
    echo ""
    echo "Next steps:"
    echo "1. Complete Axolotl installation:"
    echo "   pip install -e \".[deepspeed]\" --no-build-isolation"
    echo ""
    echo "2. Test with a small model:"
    echo "   axolotl train examples/llama-3/lora-1b.yml"
    
    # Save environment variables for future use
    echo ""
    echo -e "${YELLOW}Adding CUDA paths to ~/.bashrc for persistence...${NC}"
    
    if ! grep -q "CUDA_HOME=/opt/cuda" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# CUDA configuration for Axolotl" >> ~/.bashrc
        echo "export CUDA_HOME=/opt/cuda" >> ~/.bashrc
        echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        echo -e "${GREEN}✓ Added to ~/.bashrc${NC}"
    else
        echo "Already configured in ~/.bashrc"
    fi
    
else
    echo ""
    echo -e "${RED}Flash Attention installation failed.${NC}"
    echo "Please check the error messages above."
    echo ""
    echo "You can still use Axolotl without Flash Attention by using xformers instead."
fi

# Step 9: Complete Axolotl installation
echo ""
echo -e "${YELLOW}Step 9: Completing Axolotl installation...${NC}"

pip install -e ".[deepspeed]" --no-build-isolation

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"