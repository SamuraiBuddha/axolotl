#!/usr/bin/env python3
"""
Quick WSL check and setup helper
Runs commands in WSL from Windows
"""

import subprocess
import os
import sys

def run_wsl_command(command):
    """Run a command in WSL and return output"""
    try:
        result = subprocess.run(
            ["wsl", "-e", "bash", "-c", command],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_wsl():
    """Check if WSL is available and working"""
    print("Checking WSL availability...")
    
    # Test basic WSL
    code, stdout, stderr = run_wsl_command("echo 'WSL is working!'")
    if code != 0:
        print(f"❌ WSL not available: {stderr}")
        return False
        
    print(f"✅ {stdout.strip()}")
    
    # Check Ubuntu version
    code, stdout, stderr = run_wsl_command("lsb_release -d")
    if code == 0:
        print(f"✅ {stdout.strip()}")
    
    # Check Python
    code, stdout, stderr = run_wsl_command("python3 --version")
    if code == 0:
        print(f"✅ Python: {stdout.strip()}")
        
    # Check conda
    code, stdout, stderr = run_wsl_command("conda --version")
    if code == 0:
        print(f"✅ Conda: {stdout.strip()}")
    else:
        print("⚠️  Conda not found (will need to install)")
        
    return True

def setup_swarm_in_wsl():
    """Set up swarm training in WSL"""
    print("\n=== Setting up Swarm Training in WSL ===")
    
    # Create directory
    print("\n1. Creating swarm directory in WSL...")
    run_wsl_command("mkdir -p ~/swarm_training")
    
    # Copy files
    print("\n2. Copying swarm architecture files...")
    windows_path = os.path.abspath(".")
    wsl_path = windows_path.replace("C:", "/mnt/c").replace("\\", "/")
    
    cmd = f"cp -r {wsl_path} ~/swarm_training/"
    code, stdout, stderr = run_wsl_command(cmd)
    if code == 0:
        print("✅ Files copied successfully")
    else:
        print(f"❌ Copy failed: {stderr}")
        
    # Create setup script
    print("\n3. Creating setup script...")
    setup_script = """
cd ~/swarm_training/swarm_architecture

# Check if conda environment exists
if conda env list | grep -q "swarm"; then
    echo "Swarm environment already exists"
else
    echo "Creating swarm conda environment..."
    conda create -n swarm python=3.11 -y
fi

echo "Setup complete! Next steps:"
echo "1. Open WSL terminal"
echo "2. cd ~/swarm_training/swarm_architecture"
echo "3. conda activate swarm"
echo "4. Follow the training guide"
"""
    
    code, stdout, stderr = run_wsl_command(f"echo '{setup_script}' > ~/swarm_setup.sh")
    run_wsl_command("chmod +x ~/swarm_setup.sh")
    
    print("\n✅ Setup complete!")
    print("\nTo start training:")
    print("1. Open WSL terminal (or type 'wsl' in this terminal)")
    print("2. Run: bash ~/swarm_setup.sh")
    print("3. Follow the WSL_TRAINING_GUIDE.md")

def quick_train_test():
    """Quick test to see if we can train in WSL"""
    print("\n=== Quick WSL Training Test ===")
    
    test_commands = [
        ("Check GPU", "python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"),
        ("Check memory", "free -h | grep Mem"),
        ("Check disk space", "df -h | grep -E '/$|/home'"),
    ]
    
    for name, cmd in test_commands:
        print(f"\n{name}:")
        code, stdout, stderr = run_wsl_command(cmd)
        if code == 0:
            print(stdout.strip())
        else:
            print(f"Not available: {stderr.strip()}")

if __name__ == "__main__":
    if not check_wsl():
        print("\nWSL is required for training. Please install WSL2 with Ubuntu.")
        sys.exit(1)
        
    setup_swarm_in_wsl()
    quick_train_test()
    
    print("\n" + "="*50)
    print("WSL is ready for swarm training!")
    print("Open a WSL terminal and follow the guide.")
    print("="*50)
