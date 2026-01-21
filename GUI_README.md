# Axolotl Training GUI

A modern web-based graphical interface for Axolotl LLM training.

![Axolotl Symbol](image/axolotl_symbol_digital_black.svg)

## Features

- 🎯 **Visual Training Management** - Start, monitor, and manage training jobs
- 📝 **Configuration Editor** - Create and edit YAML configs with syntax highlighting
- 📊 **Real-time Monitoring** - Track GPU, CPU, and memory usage
- 📦 **Model Browser** - View and manage trained models
- 🗂️ **Dataset Manager** - Browse local and HuggingFace datasets
- 🚀 **Quick Train** - One-click training with preset configurations
- 📜 **Live Log Streaming** - Watch training progress in real-time

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-gui.txt
```

### 2. Launch the GUI

**Option A: Use the desktop shortcut**
- Double-click the "Axolotl Training GUI" icon on your desktop

**Option B: Use the launcher script**
```bash
./launch_gui.sh
```

**Option C: Use the command-line launcher**
```bash
axolotl-gui  # Available after running create_desktop_shortcut.sh
```

**Option D: Run directly**
```bash
python axolotl_gui.py
```

### 3. Access the Interface

Open your web browser and navigate to:
```
http://localhost:5000
```

## Interface Overview

### Dashboard
- System resource monitoring
- Active training jobs
- Recent models
- Quick actions

### Training Manager
- Select configuration files
- Start/stop training jobs
- Monitor training logs in real-time
- Configure multi-GPU and DeepSpeed options

### Configuration Editor
- Create new configs from templates
- Edit existing configurations
- Validate YAML syntax
- Auto-save functionality

### Model Browser
- View all trained models
- Check model sizes and creation dates
- Export models
- Use models for inference

### Dataset Manager
- Browse local dataset files
- Search HuggingFace datasets
- Add custom datasets

## Keyboard Shortcuts

- `Ctrl+S` - Save current configuration (in editor)
- `Ctrl+N` - Create new configuration
- `Ctrl+Enter` - Start training
- `Esc` - Close modal dialogs

## Configuration Templates

The GUI includes pre-configured templates for:

- **LoRA Training** - Low memory, fast training
- **QLoRA Training** - Ultra-low memory with 4-bit quantization
- **Full Fine-tuning** - Best quality, requires more resources

## System Requirements

- Python 3.11+
- Modern web browser (Chrome, Firefox, Safari, Edge)
- For training: NVIDIA GPU with CUDA support

## Troubleshooting

### GUI Won't Start
```bash
# Check if port 5000 is already in use
lsof -i :5000

# Kill existing process if needed
kill $(lsof -Pi :5000 -sTCP:LISTEN -t)
```

### Can't See Desktop Icon
```bash
# Recreate desktop shortcut
./create_desktop_shortcut.sh
```

### Missing Dependencies
```bash
# Install all GUI requirements
pip install flask flask-cors psutil pyyaml
```

## Development

### Project Structure
```
axolotl/
├── axolotl_gui.py          # Flask backend
├── gui_templates/          # HTML templates
│   └── index.html         # Main interface
├── gui_static/            # Static assets
│   ├── css/
│   │   └── style.css     # Styling
│   ├── js/
│   │   └── app.js        # Frontend logic
│   └── img/
│       └── axolotl_icon.svg  # Logo
└── launch_gui.sh          # Launch script
```

### API Endpoints

- `GET /` - Main interface
- `GET /api/system/info` - System information
- `GET /api/configs/list` - List configurations
- `POST /api/configs/save` - Save configuration
- `POST /api/training/start` - Start training
- `GET /api/training/{id}/logs` - Get training logs
- `GET /api/models/list` - List models
- `GET /api/datasets/list` - List datasets

## Support

For issues or questions:
- Check the [Axolotl Documentation](https://docs.axolotl.ai/)
- Join the [Discord Community](https://discord.gg/HhrNrHJPRb)
- Open an issue on [GitHub](https://github.com/axolotl-ai-cloud/axolotl)

## License

Apache 2.0 - Same as Axolotl