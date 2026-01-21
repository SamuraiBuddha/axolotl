#!/bin/bash

# Python 3.11 Installation for Arch/EndeavourOS

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}======================================"
echo "   Python 3.11 Installation"
echo "   for EndeavourOS/Arch Linux"
echo "======================================${NC}"
echo ""

echo "Choose installation method:"
echo "1) pyenv (Recommended - manages multiple Python versions)"
echo "2) AUR with yay (System-wide installation)"
echo "3) Build from source"
echo ""
echo -n "Enter your choice (1-3): "
read choice

case $choice in
    1)
        echo -e "${GREEN}Installing Python 3.11 via pyenv...${NC}"
        echo ""
        
        # Install pyenv dependencies
        echo "Installing build dependencies..."
        sudo pacman -S --needed --noconfirm base-devel openssl zlib xz tk

        # Install pyenv
        if [ ! -d "$HOME/.pyenv" ]; then
            echo "Installing pyenv..."
            curl https://pyenv.run | bash
            
            # Add to bashrc
            echo "" >> ~/.bashrc
            echo '# Pyenv configuration' >> ~/.bashrc
            echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
            echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
            echo 'eval "$(pyenv init -)"' >> ~/.bashrc
            
            # Also add to current session
            export PYENV_ROOT="$HOME/.pyenv"
            export PATH="$PYENV_ROOT/bin:$PATH"
            eval "$(pyenv init -)"
        else
            echo "pyenv is already installed"
            export PYENV_ROOT="$HOME/.pyenv"
            export PATH="$PYENV_ROOT/bin:$PATH"
            eval "$(pyenv init -)"
        fi
        
        # Install Python 3.11.9
        echo -e "${YELLOW}Installing Python 3.11.9 (this may take a few minutes)...${NC}"
        pyenv install -s 3.11.9
        
        # Set as local version for this directory
        cd /home/samuraibuddha/Documents/GitHub/axolotl
        pyenv local 3.11.9
        
        echo -e "${GREEN}✓ Python 3.11.9 installed successfully!${NC}"
        echo ""
        echo "Python 3.11 is now available in this directory."
        echo "Run: python --version"
        echo ""
        echo -e "${YELLOW}IMPORTANT: Restart your terminal or run:${NC}"
        echo "  source ~/.bashrc"
        echo ""
        echo "Then run ./setup_axolotl.sh again"
        ;;
        
    2)
        echo -e "${GREEN}Installing Python 3.11 from AUR...${NC}"
        
        # Check if yay is installed
        if ! command -v yay &> /dev/null; then
            echo "Installing yay first..."
            sudo pacman -S --needed git base-devel
            git clone https://aur.archlinux.org/yay.git /tmp/yay
            cd /tmp/yay
            makepkg -si --noconfirm
            cd -
        fi
        
        # Try to install python311 from AUR
        echo "Searching for Python 3.11 in AUR..."
        yay -S python311 --noconfirm
        
        if command -v python3.11 &> /dev/null; then
            echo -e "${GREEN}✓ Python 3.11 installed successfully!${NC}"
        else
            echo -e "${RED}Failed to install from AUR. Try method 1 (pyenv) instead.${NC}"
            exit 1
        fi
        ;;
        
    3)
        echo -e "${GREEN}Building Python 3.11 from source...${NC}"
        
        # Install build dependencies
        sudo pacman -S --needed base-devel zlib readline sqlite openssl libffi bzip2 xz tk

        # Download and build Python
        cd /tmp
        wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
        tar xzf Python-3.11.9.tgz
        cd Python-3.11.9
        
        ./configure --enable-optimizations --with-ensurepip=install
        make -j $(nproc)
        sudo make altinstall
        
        echo -e "${GREEN}✓ Python 3.11 installed successfully!${NC}"
        echo "Python 3.11 is available as: python3.11"
        ;;
        
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "1. If you used pyenv, restart your terminal or run: source ~/.bashrc"
echo "2. Run: ./setup_axolotl.sh"