# Quick Start: Swarm Training in WSL2 Ubuntu

## Step 1: Open Your Ubuntu Instance (NOT docker-desktop)

```bash
# From Windows Terminal, PowerShell, or CMD:
wsl -d Ubuntu

# Or if Ubuntu is your default:
wsl
```

**Important**: Make sure you're in Ubuntu, not docker-desktop:
```bash
# Check which instance you're in:
cat /etc/os-release
# Should show Ubuntu, not Docker Desktop Linux
```

## Step 2: Quick Setup Commands

```bash
# Navigate to your project
cd /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl/training_approaches/swarm_architecture

# Make setup script executable
chmod +x wsl_setup.sh

# Run the automated setup
./wsl_setup.sh
```

## Step 3: Train the Swarm

After setup completes:
```bash
# Make sure environment is activated
source swarm_env/bin/activate

# Start training all models
python train_swarm_wsl.py
```

## Manual Setup (if automated fails)

```bash
# 1. Enter Ubuntu (NOT docker-desktop)
wsl -d Ubuntu

# 2. Navigate to project
cd /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl/training_approaches/swarm_architecture

# 3. Install Python 3.11
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# 4. Create virtual environment
python3.11 -m venv swarm_env
source swarm_env/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install torch transformers accelerate peft bitsandbytes datasets

# 6. Install axolotl
cd /tmp
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .
cd -

# 7. Download models
python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM-135M', local_dir='models/smollm-135m')"

# 8. Train
accelerate launch -m axolotl.cli.train intent_parser/config.yaml
```

## Verify You're in the Right Place

```bash
# Check WSL instance name
echo $WSL_DISTRO_NAME
# Should output: Ubuntu

# Check Python version
python3.11 --version
# Should show: Python 3.11.x

# Check current directory
pwd
# Should show: /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl/training_approaches/swarm_architecture
```

## Tips for WSL2 Ubuntu Instance

1. **Keep Docker and Ubuntu separate**: Docker Desktop manages its own instance, don't install training tools there

2. **Ubuntu instance is for development**: This is where you install Python, ML libraries, etc.

3. **You can still use Docker from Ubuntu**:
   ```bash
   # In Ubuntu instance, Docker commands still work:
   docker ps
   docker run hello-world
   ```

4. **Better performance**: Copy files to Linux filesystem:
   ```bash
   # Optional: Copy to Linux home for faster I/O
   cp -r /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl ~/axolotl
   cd ~/axolotl/training_approaches/swarm_architecture
   ```

## Check GPU Access (if you have NVIDIA GPU)

```bash
# In Ubuntu instance:
nvidia-smi

# If it works, you'll see your GPU
# If not, install NVIDIA drivers in Windows and WSL2 GPU support
```

Ready to train! Just run:
```bash
wsl -d Ubuntu
cd /mnt/c/Users/JordanEhrig/Documents/GitHub/axolotl/training_approaches/swarm_architecture
./wsl_setup.sh
```
