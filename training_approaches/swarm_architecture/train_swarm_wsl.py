#!/usr/bin/env python3
"""
Swarm Training Script for WSL2
Trains all micro-models in the swarm architecture
"""

import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

def train_model(name, config_path):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"Config: {config_path}")
    print(f"Started: {datetime.now()}")
    print(f"{'='*60}")
    
    cmd = ["accelerate", "launch", "-m", "axolotl.cli.train", config_path]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {name} trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} training failed: {e}")
        return False

def main():
    # Configure accelerate
    print("Configuring accelerate...")
    subprocess.run(["accelerate", "config", "default"], check=True)
    
    # Models to train
    models = [
        ("Intent Parser", "intent_parser/config.yaml"),
        ("Error Recognizer", "error_recognizer/config.yaml"),
        ("Context Manager", "context_manager/config.yaml"),
        ("Docker API Mapper", "api_mappers/docker_mapper_config.yaml"),
        ("Orchestrator", "orchestrator/config.yaml"),
    ]
    
    results = {}
    for name, config in models:
        if Path(config).exists():
            results[name] = train_model(name, config)
        else:
            print(f"Config not found: {config}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    successful = sum(results.values())
    total = len(results)
    print(f"Models trained: {successful}/{total}")
    
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} - {model}")

if __name__ == "__main__":
    main()
