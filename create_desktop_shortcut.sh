#!/bin/bash

# Create Desktop Shortcut for Axolotl GUI

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Creating Axolotl GUI Desktop Shortcut...${NC}"

# Get the current directory
AXOLOTL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Detect desktop directory
if [ -d "$HOME/Desktop" ]; then
    DESKTOP_DIR="$HOME/Desktop"
elif [ -d "$HOME/desktop" ]; then
    DESKTOP_DIR="$HOME/desktop"
elif [ -n "$XDG_DESKTOP_DIR" ]; then
    DESKTOP_DIR="$XDG_DESKTOP_DIR"
else
    echo -e "${RED}Could not find desktop directory${NC}"
    echo "Please specify your desktop directory:"
    read -r DESKTOP_DIR
    if [ ! -d "$DESKTOP_DIR" ]; then
        echo -e "${RED}Directory does not exist: $DESKTOP_DIR${NC}"
        exit 1
    fi
fi

echo "Desktop directory: $DESKTOP_DIR"

# Convert SVG to PNG for the icon (if needed)
ICON_PATH="$AXOLOTL_DIR/gui_static/img/axolotl_icon.svg"
PNG_ICON_PATH="$AXOLOTL_DIR/gui_static/img/axolotl_icon.png"

# Try to convert SVG to PNG if tools are available
if command -v convert &> /dev/null; then
    echo "Converting SVG to PNG icon..."
    convert -background none -resize 256x256 "$ICON_PATH" "$PNG_ICON_PATH" 2>/dev/null || true
elif command -v rsvg-convert &> /dev/null; then
    echo "Converting SVG to PNG icon using rsvg-convert..."
    rsvg-convert -w 256 -h 256 "$ICON_PATH" -o "$PNG_ICON_PATH" 2>/dev/null || true
elif command -v inkscape &> /dev/null; then
    echo "Converting SVG to PNG icon using inkscape..."
    inkscape "$ICON_PATH" --export-png="$PNG_ICON_PATH" --export-width=256 --export-height=256 2>/dev/null || true
else
    echo -e "${YELLOW}No SVG converter found. Using SVG directly.${NC}"
    echo "To get a PNG icon, install one of: imagemagick, librsvg2-bin, or inkscape"
fi

# Use PNG if it exists, otherwise use SVG
if [ -f "$PNG_ICON_PATH" ]; then
    FINAL_ICON_PATH="$PNG_ICON_PATH"
else
    FINAL_ICON_PATH="$ICON_PATH"
fi

# Create the desktop entry file
DESKTOP_FILE="$DESKTOP_DIR/axolotl-gui.desktop"

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Axolotl Training GUI
Comment=Web-based interface for Axolotl LLM training
Icon=$FINAL_ICON_PATH
Exec=bash -c "cd '$AXOLOTL_DIR' && ./launch_gui.sh"
Terminal=true
Categories=Development;Education;Science;
Keywords=AI;ML;LLM;Training;Machine Learning;
StartupNotify=true
EOF

# Make the desktop file executable
chmod +x "$DESKTOP_FILE"

# For GNOME/KDE, also install to applications menu
APPLICATIONS_DIR="$HOME/.local/share/applications"
if [ -d "$APPLICATIONS_DIR" ]; then
    cp "$DESKTOP_FILE" "$APPLICATIONS_DIR/"
    echo -e "${GREEN}✓ Added to applications menu${NC}"
fi

# Trust the desktop file on Ubuntu/GNOME (if gio is available)
if command -v gio &> /dev/null; then
    gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
fi

# For KDE, mark as trusted
if [ -n "$KDE_FULL_SESSION" ]; then
    kwriteconfig5 --file "$DESKTOP_FILE" --group "Desktop Entry" --key "Trusted" true 2>/dev/null || true
fi

echo -e "${GREEN}✓ Desktop shortcut created successfully!${NC}"
echo ""
echo "Location: $DESKTOP_FILE"
echo ""
echo "You should now see 'Axolotl Training GUI' on your desktop."
echo "Double-click it to launch the GUI."
echo ""

# Create a start menu entry for Windows WSL users
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo -e "${YELLOW}Detected WSL environment${NC}"
    
    # Get Windows user profile path
    WIN_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')
    WIN_DESKTOP="/mnt/c/Users/$WIN_USER/Desktop"
    
    if [ -d "$WIN_DESKTOP" ]; then
        # Create a Windows batch file to launch the GUI
        WIN_LAUNCHER="$WIN_DESKTOP/Axolotl GUI.bat"
        cat > "$WIN_LAUNCHER" << 'WINEOF'
@echo off
title Axolotl Training GUI
echo Starting Axolotl Training GUI...
wsl.exe bash -c "cd ~/Documents/GitHub/axolotl && ./launch_gui.sh"
pause
WINEOF
        
        # Convert line endings for Windows
        unix2dos "$WIN_LAUNCHER" 2>/dev/null || sed -i 's/$/\r/' "$WIN_LAUNCHER"
        
        echo -e "${GREEN}✓ Windows desktop shortcut created!${NC}"
        echo "Location: $WIN_LAUNCHER"
    fi
fi

# Create a launcher script for easy terminal access
LAUNCHER_SCRIPT="$HOME/.local/bin/axolotl-gui"
mkdir -p "$HOME/.local/bin"

cat > "$LAUNCHER_SCRIPT" << EOF
#!/bin/bash
cd "$AXOLOTL_DIR"
./launch_gui.sh
EOF

chmod +x "$LAUNCHER_SCRIPT"

echo -e "${GREEN}✓ Command-line launcher created!${NC}"
echo "You can now run 'axolotl-gui' from any terminal."
echo ""

# Final message
echo "======================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "======================================"
echo ""
echo "Launch options:"
echo "1. Double-click the desktop icon"
echo "2. Run 'axolotl-gui' in terminal"
echo "3. Find in your applications menu"
echo ""