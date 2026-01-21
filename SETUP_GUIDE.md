# Axolotl Quick Setup Guide

## Prerequisites

1. **Python 3.11** (Required)
   ```bash
   # Check if you have Python 3.11
   python3.11 --version
   
   # If not installed, install it:
   # Ubuntu/Debian:
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3.11-dev
   
   # Or using pyenv (recommended):
   curl https://pyenv.run | bash
   pyenv install 3.11.9
   pyenv local 3.11.9
   ```

2. **NVIDIA GPU** (Recommended for training)
   - CUDA 11.8+ or 12.1 recommended
   - At least 8GB VRAM for small models, 24GB+ for larger ones

## Quick Installation

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
./setup_axolotl.sh
```

This will:
- Check Python 3.11 availability
- Create a virtual environment
- Install all dependencies
- Download example configurations

### Option 2: Manual Installation

```bash
# Create virtual environment
python3.11 -m venv venv_axolotl
source venv_axolotl/bin/activate

# Install dependencies
pip install -U pip
pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja

# Install Axolotl with extras
pip install -e ".[flash-attn,deepspeed]" --no-build-isolation

# Download examples
axolotl fetch examples
```

## Using the Helper Script

After setup, use the interactive helper for easy access to all features:

```bash
./axolotl_helper.sh
```

Features:
- 🚀 Quick training with pre-configured models
- 📝 Create custom configurations
- 🧪 Run tests
- 📊 Launch TensorBoard monitoring
- 🔧 Validate configurations
- And more!

## Quick Start Examples

### 1. Train Your First Model (Small, Fast)
```bash
# Activate environment
source venv_axolotl/bin/activate

# Train a 1B parameter model with LoRA
axolotl train examples/llama-3/lora-1b.yml
```

### 2. Use the Helper Menu
```bash
./axolotl_helper.sh
# Select option 1 for quick training
```

### 3. Train with Custom Data
```bash
# Create a config file (use helper option 10)
./axolotl_helper.sh

# Or create manually
cat > my_config.yml << 'EOF'
base_model: NousResearch/Llama-3.2-1B
load_in_8bit: true
adapter: lora

datasets:
  - path: your_data.jsonl
    type: alpaca

output_dir: ./outputs/my_model
num_epochs: 3
micro_batch_size: 2
learning_rate: 0.0003
EOF

# Train
axolotl train my_config.yml
```

## Common Training Configurations

### LoRA (Low memory, fast)
```yaml
load_in_8bit: true
adapter: lora
```

### QLoRA (Very low memory)
```yaml
load_in_4bit: true
adapter: qlora
```

### Full Fine-tuning (Best quality, high memory)
```yaml
# Remove adapter and load_in_Xbit lines
```

## Memory Requirements

| Model Size | LoRA  | QLoRA | Full Fine-tune |
|------------|-------|-------|----------------|
| 1B params  | ~6GB  | ~4GB  | ~10GB          |
| 7B params  | ~14GB | ~8GB  | ~28GB          |
| 13B params | ~24GB | ~12GB | ~52GB          |

## Troubleshooting

### Python 3.11 Not Found
```bash
# Install using deadsnakes PPA (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### CUDA/GPU Issues
```bash
# Check GPU status
nvidia-smi

# Install with CPU support only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Out of Memory Errors
- Reduce `micro_batch_size` in config (try 1)
- Increase `gradient_accumulation_steps` 
- Use QLoRA instead of LoRA
- Enable `gradient_checkpointing: true`

## Next Steps

1. Read the [documentation](https://docs.axolotl.ai/)
2. Explore example configs in `examples/` directory
3. Join the [Discord community](https://discord.gg/HhrNrHJPRb)
4. Check out advanced features like multi-GPU training

## Helper Script Commands

You can also use the helper script directly:

```bash
# Direct training
./axolotl_helper.sh train my_config.yml

# Preprocess only
./axolotl_helper.sh preprocess my_config.yml

# Run inference
./axolotl_helper.sh inference my_config.yml
```

Happy training! 🦎