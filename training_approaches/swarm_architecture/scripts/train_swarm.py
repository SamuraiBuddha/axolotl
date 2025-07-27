#!/usr/bin/env python3
"""
Swarm Architecture Training Pipeline
Trains all micro-models in sequence or parallel
"""

import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

BASE_PATH = Path(__file__).parent.parent

MODELS = {
    "intent_parser": {
        "config": "intent_parser/config.yaml",
        "priority": 1,
        "estimated_time": "30 minutes"
    },
    "context_manager": {
        "config": "context_manager/config.yaml",
        "priority": 2,
        "estimated_time": "20 minutes"
    },
    "error_recognizer": {
        "config": "error_recognizer/config.yaml", 
        "priority": 1,
        "estimated_time": "25 minutes"
    },
    "api_mapper_docker": {
        "config": "api_mappers/docker_mapper_config.yaml",
        "priority": 3,
        "estimated_time": "15 minutes"
    },
    "orchestrator": {
        "config": "orchestrator/config.yaml",
        "priority": 4,
        "estimated_time": "45 minutes"
    }
}

def download_base_models():
    """Download all required base models"""
    print("=== Downloading Base Models ===")
    
    base_models = [
        ("HuggingFaceTB/SmolLM-135M", "models/smollm-135m"),
        ("HuggingFaceTB/SmolLM-360M", "models/smollm-360m"),
        ("Qwen/Qwen2.5-0.5B-Instruct", "models/qwen2.5-0.5b")
    ]
    
    for model_id, local_dir in base_models:
        local_path = BASE_PATH / local_dir
        if not local_path.exists():
            print(f"Downloading {model_id}...")
            cmd = f"huggingface-cli download {model_id} --local-dir {local_path}"
            subprocess.run(cmd, shell=True)
        else:
            print(f"{model_id} already downloaded")

def train_model(model_name: str, config_path: str):
    """Train a single model"""
    print(f"\n=== Training {model_name} ===")
    print(f"Config: {config_path}")
    print(f"Started: {datetime.now()}")
    
    full_config_path = BASE_PATH / config_path
    
    # Check if training data exists
    training_dir = full_config_path.parent
    training_data = training_dir / "training_data.jsonl"
    
    if not training_data.exists():
        print(f"ERROR: No training data found at {training_data}")
        return False
        
    # Run training
    cmd = [
        "python", "-m", "accelerate", "launch",
        "-m", "axolotl.cli.train",
        str(full_config_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {model_name} training completed successfully")
            return True
        else:
            print(f"❌ {model_name} training failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")
        return False

def train_sequential():
    """Train models one by one in priority order"""
    print("Training models sequentially...")
    
    # Sort by priority
    sorted_models = sorted(MODELS.items(), key=lambda x: x[1]["priority"])
    
    results = {}
    for model_name, model_info in sorted_models:
        success = train_model(model_name, model_info["config"])
        results[model_name] = success
        
    return results

def train_parallel(max_workers: int = 2):
    """Train models in parallel (requires multiple GPUs)"""
    print(f"Training models in parallel with {max_workers} workers...")
    
    with mp.Pool(max_workers) as pool:
        tasks = [(name, info["config"]) for name, info in MODELS.items()]
        results = pool.starmap(train_model, tasks)
        
    return dict(zip(MODELS.keys(), results))

def generate_training_report(results: dict):
    """Generate a training report"""
    report_path = BASE_PATH / "training_report.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "summary": {
            "total": len(results),
            "successful": sum(results.values()),
            "failed": len(results) - sum(results.values())
        }
    }
    
    for model_name, success in results.items():
        output_dir = BASE_PATH / MODELS[model_name]["config"].replace("config.yaml", "output")
        
        report["models"][model_name] = {
            "trained": success,
            "config": MODELS[model_name]["config"],
            "output_exists": output_dir.exists(),
            "estimated_time": MODELS[model_name]["estimated_time"]
        }
        
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nTraining report saved to: {report_path}")
    print(f"Summary: {report['summary']['successful']}/{report['summary']['total']} models trained successfully")

def main():
    """Main training pipeline"""
    print("=== Swarm Architecture Training Pipeline ===")
    print(f"Base path: {BASE_PATH}")
    print(f"Models to train: {len(MODELS)}")
    
    # Step 1: Download base models
    download_base_models()
    
    # Step 2: Generate training data if needed
    training_data_script = BASE_PATH / "scripts" / "generate_swarm_training_data.py"
    if training_data_script.exists():
        print("\n=== Generating Training Data ===")
        subprocess.run(["python", str(training_data_script)])
    
    # Step 3: Train models
    print("\n=== Starting Training ===")
    
    # Check if we have accelerate configured
    try:
        subprocess.run(["accelerate", "config", "--help"], capture_output=True)
    except:
        print("Accelerate not configured. Running: accelerate config")
        subprocess.run(["python", "-m", "accelerate", "config"])
    
    # Choose training mode
    if mp.cpu_count() >= 4 and torch.cuda.device_count() > 1:
        print("Multiple GPUs detected. Using parallel training.")
        results = train_parallel()
    else:
        print("Using sequential training.")
        results = train_sequential()
    
    # Step 4: Generate report
    generate_training_report(results)
    
    print("\n=== Training Pipeline Complete ===")

if __name__ == "__main__":
    import torch  # Import here to check GPU count
    main()
