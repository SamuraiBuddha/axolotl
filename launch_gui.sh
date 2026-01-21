#!/bin/bash

# Axolotl GUI Launcher

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ASCII art
echo -e "${GREEN}"
echo "    ___             __      __  __"
echo "   / _ | __ __ ___  / /__   / /_/ /"
echo "  / __ |\ \ // _ \/ / _ \ / __/ / "
echo " /_/ |_//_\_\\\\___/_/\___/ \__/_/  "
echo "         Training GUI v1.0"
echo -e "${NC}"
echo ""

# Check for virtual environment
if [ -d "venv_axolotl" ]; then
    echo -e "${GREEN}✓ Found Axolotl virtual environment${NC}"
    source venv_axolotl/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Using current virtual environment: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}⚠ No virtual environment detected${NC}"
    echo "It's recommended to use a virtual environment."
    echo "Run ./setup_axolotl.sh first to set everything up."
    echo ""
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        exit 0
    fi
fi

# Install GUI dependencies if needed
echo "Checking GUI dependencies..."

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Flask and required packages..."
    pip install flask flask-cors psutil pyyaml
fi

# Check if psutil is installed
if ! python -c "import psutil" 2>/dev/null; then
    echo "Installing psutil..."
    pip install psutil
fi

# Kill any existing GUI process on port 5000
echo "Checking for existing GUI processes..."
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}Found existing process on port 5000${NC}"
    echo "Stop it and start new GUI? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        kill $(lsof -Pi :5000 -sTCP:LISTEN -t) 2>/dev/null
        sleep 2
    else
        echo "Please stop the existing process first"
        exit 1
    fi
fi

# Copy the Axolotl icons if they don't exist
if [ ! -f "gui_static/img/axolotl_icon.svg" ]; then
    echo "Copying Axolotl icons..."
    mkdir -p gui_static/img
    
    # Copy SVG symbol
    if [ -f "image/axolotl_symbol_digital_black.svg" ]; then
        cp image/axolotl_symbol_digital_black.svg gui_static/img/axolotl_icon.svg
    fi
    
    # Also copy PNG if it exists
    if [ -f "image/axolotl.png" ]; then
        cp image/axolotl.png gui_static/img/axolotl_logo.png
    fi
fi

# Start the GUI
echo ""
echo -e "${GREEN}Starting Axolotl GUI...${NC}"
echo "======================================"
echo "  Web Interface: http://localhost:5000"
echo "  API Endpoint:  http://localhost:5000/api"
echo "======================================"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch the Flask app
python axolotl_gui.py