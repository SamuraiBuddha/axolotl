#!/usr/bin/env python3
"""
Quick test to see if we can run axolotl training directly
"""

import subprocess
import sys
from pathlib import Path

print("Testing axolotl direct execution...")
print(f"Python version: {sys.version}")

# Try to import axolotl
try:
    import axolotl
    print(f"✅ Axolotl imported successfully from: {axolotl.__file__}")
except ImportError as e:
    print(f"❌ Failed to import axolotl: {e}")
    sys.exit(1)

# Try to run a simple training command
config_path = Path(__file__).parent.parent / "intent_parser" / "config.yaml"

if config_path.exists():
    print(f"\nTrying to train with config: {config_path}")
    
    # Direct Python execution
    cmd = [sys.executable, "-m", "axolotl.cli.train", str(config_path)]
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Training command executed successfully!")
        else:
            print(f"❌ Training failed with code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
    except Exception as e:
        print(f"❌ Exception running training: {e}")
else:
    print(f"❌ Config file not found at {config_path}")

# Also test if we can at least load a SmolLM model
print("\n\nTesting model loading capabilities...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Try to load tokenizer (lightweight test)
    print("Attempting to load a tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Small test model
    print("✅ Tokenizer loaded successfully")
    
except Exception as e:
    print(f"❌ Failed to load test model: {e}")
