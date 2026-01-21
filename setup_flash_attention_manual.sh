#!/bin/bash

# Manual Flash Attention Setup for Axolotl
# This script sets up and builds Flash Attention properly

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================"
echo "   Flash Attention Setup"
echo "======================================"
echo -e "${NC}"

# First, check if gcc14 is available (it's already installed)
echo -e "${YELLOW}Checking compiler versions...${NC}"

if [ -f /usr/bin/gcc-14 ]; then
    echo -e "${GREEN}✓ gcc-14 is available${NC}"
    USE_GCC="/usr/bin/gcc-14"
    USE_GXX="/usr/bin/g++-14"
else
    echo -e "${YELLOW}Using default GCC (version 15)${NC}"
    echo "Note: This might cause compilation issues"
    USE_GCC="/usr/bin/gcc"
    USE_GXX="/usr/bin/g++"
fi

# Set up CUDA environment
echo ""
echo -e "${YELLOW}Setting up CUDA environment...${NC}"

export CUDA_HOME=/opt/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify nvcc
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✓ nvcc found: $(nvcc --version | grep release | cut -d' ' -f5,6)${NC}"
else
    echo -e "${RED}✗ nvcc not found. Please ensure CUDA is installed.${NC}"
    exit 1
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"
source /home/samuraibuddha/Documents/GitHub/axolotl/venv_axolotl/bin/activate

# Clean previous attempts
echo ""
echo -e "${YELLOW}Cleaning previous build attempts...${NC}"
pip uninstall flash-attn -y 2>/dev/null || true
rm -rf ~/.cache/pip/wheels/*flash_attn* 2>/dev/null || true

# Set compilation environment
echo ""
echo -e "${YELLOW}Setting compilation environment...${NC}"

# Use gcc-14 if available
export CC=$USE_GCC
export CXX=$USE_GXX
export CUDAHOSTCXX=$USE_GXX

# Set CUDA architectures for RTX A4000 (compute capability 8.6)
export TORCH_CUDA_ARCH_LIST="8.6"

# Limit compilation threads to avoid memory issues
export MAX_JOBS=4

# Show configuration
echo ""
echo "Configuration:"
echo "  CC: $CC ($(${CC} --version | head -1))"
echo "  CXX: $CXX"
echo "  CUDA: $(nvcc --version | grep release | cut -d' ' -f5,6)"
echo "  CUDA architectures: $TORCH_CUDA_ARCH_LIST"
echo "  Max parallel jobs: $MAX_JOBS"

# Method 1: Try installing with pip
echo ""
echo -e "${BLUE}Method 1: Installing Flash Attention with pip...${NC}"
echo "This will take 5-15 minutes to compile..."

if pip install flash-attn==2.7.4.post1 --no-cache-dir --no-build-isolation; then
    echo -e "${GREEN}✓ Flash Attention installed successfully!${NC}"
    SUCCESS=true
else
    echo -e "${YELLOW}Pip installation failed. Trying alternative method...${NC}"
    SUCCESS=false
    
    # Method 2: Try with pre-built wheel
    echo ""
    echo -e "${BLUE}Method 2: Trying pre-built wheel...${NC}"
    
    # Get Python version
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
    
    # Try to download a compatible wheel
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/flash_attn-2.7.4-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"
    
    echo "Attempting to download wheel from: $WHEEL_URL"
    
    if wget -q --spider $WHEEL_URL 2>/dev/null; then
        pip install $WHEEL_URL
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Pre-built wheel installed successfully!${NC}"
            SUCCESS=true
        fi
    else
        echo -e "${YELLOW}No compatible pre-built wheel found${NC}"
    fi
fi

if [ "$SUCCESS" = false ]; then
    # Method 3: Build from source with patches
    echo ""
    echo -e "${BLUE}Method 3: Building from source with compatibility patches...${NC}"
    
    cd /tmp
    rm -rf flash-attention
    
    echo "Cloning Flash Attention repository..."
    git clone https://github.com/Dao-AILab/flash-attention.git --quiet
    cd flash-attention
    git checkout v2.7.4 --quiet
    
    # Apply compatibility fixes
    echo -e "${YELLOW}Applying compatibility patches...${NC}"
    
    # Create a setup patch for better compatibility
    cat > setup_patch.py << 'EOF'
import os
import sys

# Force specific compiler flags for compatibility
os.environ['NVCC_APPEND_FLAGS'] = '-allow-unsupported-compiler'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
os.environ['MAX_JOBS'] = '4'

# Import and run original setup
sys.path.insert(0, '.')
import setup
EOF
    
    echo "Building Flash Attention from source..."
    python setup_patch.py build
    
    if [ $? -eq 0 ]; then
        echo "Installing..."
        pip install -e .
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Built and installed from source successfully!${NC}"
            SUCCESS=true
        fi
    fi
    
    cd /home/samuraibuddha/Documents/GitHub/axolotl
fi

# Verify installation
echo ""
echo -e "${YELLOW}Verifying Flash Attention installation...${NC}"

python -c "
import sys
try:
    import flash_attn
    print('✓ Flash Attention module imported')
    print(f'  Version: {flash_attn.__version__}')
    
    import torch
    if torch.cuda.is_available():
        # Create test tensors
        batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device='cuda', dtype=torch.float16)
        
        from flash_attn import flash_attn_func
        output = flash_attn_func(q, k, v, causal=True)
        print('✓ Flash Attention CUDA kernels working')
        print(f'  Output shape: {output.shape}')
    sys.exit(0)
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'✗ Runtime error: {e}')
    sys.exit(1)
"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================"
    echo "   Success! Flash Attention is working!"
    echo "======================================${NC}"
    
    # Save environment settings
    ENV_FILE="/home/samuraibuddha/Documents/GitHub/axolotl/.env"
    echo "# CUDA Environment for Flash Attention" > $ENV_FILE
    echo "export CUDA_HOME=/opt/cuda" >> $ENV_FILE
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> $ENV_FILE
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> $ENV_FILE
    
    echo ""
    echo "Environment saved to .env file"
    echo ""
    echo "Next steps:"
    echo "1. Source the environment: source .env"
    echo "2. Complete Axolotl installation:"
    echo "   pip install -e . --no-build-isolation"
    echo "3. Download examples:"
    echo "   axolotl fetch examples"
    echo "4. Test with a small model:"
    echo "   axolotl train examples/llama-3/lora-1b.yml"
else
    echo ""
    echo -e "${RED}Flash Attention installation failed${NC}"
    echo ""
    echo "Troubleshooting options:"
    echo ""
    echo "1. Install with xformers instead (recommended):"
    echo "   pip install xformers"
    echo "   pip install -e . --no-build-isolation"
    echo ""
    echo "2. Try with Docker (most reliable):"
    echo "   docker pull winglian/axolotl:main-latest"
    echo ""
    echo "3. Report the issue:"
    echo "   https://github.com/axolotl-ai-cloud/axolotl/issues"
fi