# Swarm Architecture Training in WSL2

## Quick Start

1. **Open WSL2 Ubuntu**:
   ```bash
   # From Windows Terminal or PowerShell:
   wsl
   ```

2. **Navigate to the swarm directory**:
   ```bash
   cd /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl/training_approaches/swarm_architecture
   ```

3. **Run the setup script**:
   ```bash
   bash wsl_setup.sh
   ```

This will:
- Install Python 3.11 and dependencies
- Create a virtual environment
- Install PyTorch, Transformers, Axolotl
- Download all base models (SmolLM-135M, SmolLM-360M, Qwen2.5-0.5B)
- Configure everything for training

4. **Train the swarm** (after setup completes):
   ```bash
   python train_swarm_wsl.py
   ```

## Detailed Steps

### Step 1: Enter WSL2 Environment

```bash
# Check WSL2 is running
wsl --list --running

# Enter Ubuntu
wsl -d Ubuntu

# Verify you're in Linux
uname -a
# Should show: Linux ... Microsoft ... WSL2
```

### Step 2: Install Dependencies (if setup script fails)

```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install build tools
sudo apt install build-essential git curl
```

### Step 3: Manual Setup (if needed)

```bash
# Navigate to project
cd /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl/training_approaches/swarm_architecture

# Create virtual environment
python3.11 -m venv swarm_env
source swarm_env/bin/activate

# Install packages
pip install torch transformers accelerate peft bitsandbytes datasets

# Install axolotl
git clone https://github.com/OpenAccess-AI-Collective/axolotl /tmp/axolotl
cd /tmp/axolotl
pip install -e .
cd -
```

### Step 4: Training Individual Models

```bash
# Activate environment
source swarm_env/bin/activate

# Configure accelerate (first time only)
accelerate config
# Choose: No distributed training, No DeepSpeed, CPU or GPU, FP16

# Train intent parser
accelerate launch -m axolotl.cli.train intent_parser/config.yaml

# Train error recognizer  
accelerate launch -m axolotl.cli.train error_recognizer/config.yaml

# Train context manager
accelerate launch -m axolotl.cli.train context_manager/config.yaml
```

### Step 5: Monitor Training

From another terminal:
```bash
# Watch training progress
wsl -d Ubuntu
cd /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl/training_approaches/swarm_architecture
tail -f intent_parser/output/trainer_state.json
```

## GPU Support (if available)

### Check NVIDIA GPU in WSL2:
```bash
nvidia-smi
```

If GPU is available:
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Expected Training Times

### On CPU:
- Intent Parser (135M): ~2-3 hours
- Error Recognizer (135M): ~2-3 hours  
- Context Manager (135M): ~1-2 hours
- API Mappers (500M): ~3-4 hours
- Orchestrator (360M): ~4-5 hours

### On GPU (RTX 3060+):
- All models: 30-60 minutes each

## Troubleshooting

### Issue: "Command not found"
```bash
# Ensure virtual environment is activated
source swarm_env/bin/activate
which python  # Should show swarm_env path
```

### Issue: "Out of memory"
Edit config files to reduce batch size:
```yaml
micro_batch_size: 1  # Reduce from 4
gradient_accumulation_steps: 8  # Increase to compensate
```

### Issue: "Permission denied"
```bash
# Fix permissions
chmod -R 755 .
chmod +x *.py *.sh
```

### Issue: "Module not found"
```bash
# Reinstall in virtual environment
source swarm_env/bin/activate
pip install --upgrade --force-reinstall transformers accelerate
```

## Using Trained Models

After training completes:

```bash
# Test the models
python test_swarm_wsl.py

# Copy models back to Windows (optional)
cp -r intent_parser/output /mnt/c/Users/JordanEhrig/Desktop/intent_parser_model
```

## Integration with Docker Desktop

Since Docker Desktop uses WSL2:

```bash
# Build swarm container
docker build -t swarm-ai .

# Run with GPU support
docker run --gpus all -v $(pwd):/workspace swarm-ai

# Or use docker-compose
docker-compose up
```

## Next Steps

1. **After training**, the models will be in:
   - `intent_parser/output/`
   - `error_recognizer/output/`
   - `context_manager/output/`
   - etc.

2. **Deploy the swarm**:
   ```bash
   python scripts/swarm_coordinator.py
   ```

3. **Run production swarm**:
   ```bash
   # In WSL2 with models loaded
   python -m uvicorn swarm_api:app --host 0.0.0.0 --port 8000
   ```

   Access from Windows: `http://localhost:8000`

## Tips

- WSL2 file access is faster within `/home/username/` than `/mnt/c/`
- Consider copying project to Linux filesystem for faster training
- Use `tmux` or `screen` for long training sessions
- WSL2 can access Windows GPU with proper drivers installed

The swarm will train successfully in WSL2 without any Python 3.13 compatibility issues!
