#!/bin/bash
# Quick test to verify WSL2 Ubuntu environment

echo "=== WSL2 Environment Check ==="
echo

# Check if in WSL
if grep -q microsoft /proc/version; then
    echo "✓ Running in WSL2"
else
    echo "✗ Not running in WSL2"
fi

# Check distro name
echo "Distro: $WSL_DISTRO_NAME"

# Check Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "OS: $NAME $VERSION"
fi

# Check Python 3.11
if command -v python3.11 &> /dev/null; then
    echo "✓ Python 3.11 found: $(python3.11 --version)"
else
    echo "✗ Python 3.11 not found - install with: sudo apt install python3.11"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU available"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "! No NVIDIA GPU detected (CPU training will be slower)"
fi

# Check disk space
echo
echo "Disk space on /mnt/c:"
df -h /mnt/c | grep -E "Filesystem|/mnt/c"

echo
echo "Ready to proceed? Run: ./wsl_setup.sh"
