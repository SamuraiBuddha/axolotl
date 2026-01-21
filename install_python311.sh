#!/bin/bash

# Python 3.11 Installation Helper for Axolotl

set -e

echo "======================================"
echo "   Python 3.11 Installation Helper"
echo "======================================"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
else
    echo "Cannot detect OS. Please install Python 3.11 manually."
    exit 1
fi

install_ubuntu() {
    echo "Installing Python 3.11 on Ubuntu/Debian..."
    
    # Check if we need sudo
    if [ "$EUID" -ne 0 ]; then 
        SUDO="sudo"
    else
        SUDO=""
    fi
    
    # Add deadsnakes PPA for older Ubuntu versions
    if [ "$VERSION_ID" \< "23.04" ]; then
        echo "Adding deadsnakes PPA..."
        $SUDO apt update
        $SUDO apt install -y software-properties-common
        $SUDO add-apt-repository -y ppa:deadsnakes/ppa
    fi
    
    # Install Python 3.11
    echo "Installing Python 3.11 packages..."
    $SUDO apt update
    $SUDO apt install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils
    
    # Install pip for Python 3.11
    echo "Installing pip for Python 3.11..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
    
    echo "✓ Python 3.11 installed successfully!"
}

install_fedora() {
    echo "Installing Python 3.11 on Fedora/RHEL..."
    
    if [ "$EUID" -ne 0 ]; then 
        SUDO="sudo"
    else
        SUDO=""
    fi
    
    $SUDO dnf install -y python3.11 python3.11-devel
    echo "✓ Python 3.11 installed successfully!"
}

install_arch() {
    echo "Installing Python 3.11 on Arch Linux..."
    
    if [ "$EUID" -ne 0 ]; then 
        SUDO="sudo"
    else
        SUDO=""
    fi
    
    # Python 3.11 should be in the main repos
    $SUDO pacman -S --noconfirm python311
    echo "✓ Python 3.11 installed successfully!"
}

install_pyenv() {
    echo "Installing Python 3.11 using pyenv (universal method)..."
    echo "This method works on most Unix-like systems."
    echo ""
    
    # Check if pyenv is installed
    if ! command -v pyenv &> /dev/null; then
        echo "Installing pyenv..."
        curl https://pyenv.run | bash
        
        # Add to shell config
        echo "" >> ~/.bashrc
        echo '# Pyenv configuration' >> ~/.bashrc
        echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        
        # Load pyenv for this session
        export PATH="$HOME/.pyenv/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
    fi
    
    # Install build dependencies
    echo "Installing build dependencies..."
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        sudo apt update
        sudo apt install -y make build-essential libssl-dev zlib1g-dev \
            libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
            libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
            libffi-dev liblzma-dev
    elif [ "$OS" = "fedora" ] || [ "$OS" = "rhel" ]; then
        sudo dnf groupinstall -y "Development Tools"
        sudo dnf install -y gcc zlib-devel bzip2 bzip2-devel readline-devel \
            sqlite sqlite-devel openssl-devel tk-devel libffi-devel xz-devel
    fi
    
    # Install Python 3.11.9
    echo "Installing Python 3.11.9..."
    pyenv install 3.11.9
    
    # Set as local version for this directory
    pyenv local 3.11.9
    
    echo "✓ Python 3.11.9 installed successfully via pyenv!"
    echo ""
    echo "NOTE: You may need to restart your shell or run:"
    echo "  source ~/.bashrc"
}

# Main installation logic
case "$OS" in
    ubuntu|debian|pop|mint|elementary)
        install_ubuntu
        ;;
    fedora|rhel|centos|rocky|almalinux)
        install_fedora
        ;;
    arch|manjaro|endeavouros)
        install_arch
        ;;
    *)
        echo "Your OS ($OS) is not directly supported."
        echo "Would you like to try installing via pyenv? (recommended) (y/n)"
        read -r response
        if [[ "$response" = "y" ]]; then
            install_pyenv
        else
            echo "Please install Python 3.11 manually for your system."
            echo "Visit: https://www.python.org/downloads/"
            exit 1
        fi
        ;;
esac

# Verify installation
echo ""
echo "Verifying installation..."
if python3.11 --version &> /dev/null; then
    echo "✓ Success! Python $(python3.11 --version) is installed."
    echo ""
    echo "You can now run: ./setup_axolotl.sh"
else
    echo "⚠ Python 3.11 not found in PATH."
    echo "You may need to restart your shell or add it to PATH manually."
fi