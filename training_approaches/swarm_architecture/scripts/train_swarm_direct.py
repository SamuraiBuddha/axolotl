#!/usr/bin/env python3
"""
Direct training script for swarm models
Bypasses accelerate issues on Windows/Python 3.13
"""

import os
import sys
import json
from pathlib import Path

# Add axolotl to Python path
axolotl_path = Path("C:/Users/JordanEhrig/Documents/GitHub/axolotl")
sys.path.insert(0, str(axolotl_path / "src"))

# Now we can import axolotl
try:
    from axolotl.cli.train import do_cli
    from axolotl.common.cli import TrainerCliArgs
    print("Successfully imported axolotl")
except ImportError as e:
    print(f"Failed to import axolotl: {e}")
    print("Trying alternative approach...")
    
    # Alternative: Run as subprocess with proper paths
    os.chdir(axolotl_path)
    import subprocess
    
    def train_with_subprocess(config_path):
        cmd = [sys.executable, "-m", "axolotl.cli.train", config_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0

def train_model(model_name, config_path):
    """Train a single swarm model"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"Config: {config_path}")
    print(f"{'='*50}")
    
    # Change to axolotl directory for proper paths
    original_dir = os.getcwd()
    os.chdir(axolotl_path)
    
    try:
        # Use subprocess approach for compatibility
        result = train_with_subprocess(config_path)
        if result:
            print(f"SUCCESS: {model_name} trained successfully")
        else:
            print(f"FAILED: {model_name} training failed")
        return result
    finally:
        os.chdir(original_dir)

def main():
    """Train all swarm models"""
    swarm_base = Path(__file__).parent.parent
    
    # Models to train in order
    models = [
        ("intent_parser", swarm_base / "intent_parser" / "config.yaml"),
        ("error_recognizer", swarm_base / "error_recognizer" / "config.yaml"),
        ("context_manager", swarm_base / "context_manager" / "config.yaml"),
    ]
    
    print("SWARM TRAINING PIPELINE")
    print(f"Base directory: {swarm_base}")
    print(f"Axolotl directory: {axolotl_path}")
    
    results = {}
    for model_name, config_path in models:
        if config_path.exists():
            # Check for training data
            data_files = list(config_path.parent.glob("*.jsonl"))
            if data_files:
                print(f"\nFound training data: {data_files[0].name}")
                results[model_name] = train_model(model_name, str(config_path))
            else:
                print(f"\nERROR: No training data found for {model_name}")
                results[model_name] = False
        else:
            print(f"\nERROR: Config not found: {config_path}")
            results[model_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    successful = sum(results.values())
    total = len(results)
    print(f"Models trained successfully: {successful}/{total}")
    
    for model, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {model}: {status}")
    
    # Save results
    report_path = swarm_base / "training_results.json"
    with open(report_path, "w") as f:
        json.dump({
            "results": results,
            "successful": successful,
            "total": total
        }, f, indent=2)
    
    print(f"\nResults saved to: {report_path}")

if __name__ == "__main__":
    # Set environment variables for better compatibility
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages
    
    main()
