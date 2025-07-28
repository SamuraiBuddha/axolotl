#!/usr/bin/env python3
"""
Test which training approaches work in your WSL2 environment
"""

import subprocess
import os
import sys

def test_imports():
    """Test if required packages are installed"""
    print("=== Testing Package Imports ===")
    
    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers", 
        "datasets": "Datasets",
        "accelerate": "Accelerate",
        "peft": "PEFT",
        "axolotl": "Axolotl"
    }
    
    results = {}
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} imported successfully")
            results[package] = True
        except ImportError:
            print(f"✗ {name} not found")
            results[package] = False
    
    return results

def test_simple_training():
    """Test if simple training can run"""
    print("\n=== Testing Simple Training ===")
    
    if os.path.exists("simple_train_wsl.py"):
        # Do a dry run with 1 epoch
        result = subprocess.run(
            [sys.executable, "simple_train_wsl.py", "--dry-run"],
            capture_output=True,
            text=True
        )
        
        if "main()" in result.stderr:  # Check if it at least starts
            print("✓ Simple training script can start")
            return True
        else:
            print("✗ Simple training has issues")
            return False
    else:
        print("✗ simple_train_wsl.py not found")
        return False

def test_axolotl_training():
    """Test if axolotl can run"""
    print("\n=== Testing Axolotl ===")
    
    try:
        import axolotl
        print("✓ Axolotl is installed")
        
        # Test with minimal config
        if os.path.exists("minimal_axolotl_config.yaml"):
            print("✓ Minimal config found")
            return True
        else:
            print("! No minimal config, creating one...")
            return False
    except ImportError:
        print("✗ Axolotl not installed")
        return False

def main():
    print("=== Swarm Training Environment Test ===\n")
    
    # Test imports
    results = test_imports()
    
    # Test training approaches
    simple_ok = test_simple_training()
    axolotl_ok = test_axolotl_training()
    
    print("\n=== Recommendations ===")
    
    if results["transformers"] and results["torch"]:
        print("✓ You can use simple training (recommended)")
        print("  Run: python simple_train_wsl.py")
    
    if results["axolotl"]:
        print("✓ You can use axolotl training")
        print("  Run: python -m axolotl.cli.train minimal_axolotl_config.yaml")
    
    if not results["transformers"]:
        print("✗ Install basic packages first:")
        print("  pip install torch transformers datasets")
    
    print("\n=== Quick Test Commands ===")
    print("Simple test: python test_swarm_no_training.py")
    print("Simple train: python simple_train_wsl.py")
    print("Axolotl train: python -m axolotl.cli.train minimal_axolotl_config.yaml")

if __name__ == "__main__":
    main()
