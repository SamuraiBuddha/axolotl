#!/usr/bin/env python3
"""
Quick check of swarm training status
"""

import os
from datetime import datetime

def check_swarm_status():
    """Check status of all swarm components"""
    print("=== Swarm Architecture Status ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    components = [
        {
            "name": "Intent Parser",
            "data_path": "intent_parser/training_data.jsonl",
            "model_path": "intent_parser/simple_output",
            "description": "Extracts user intent from natural language"
        },
        {
            "name": "Error Recognizer",
            "data_path": "error_recognizer/training_data.jsonl", 
            "model_path": "error_recognizer/simple_output",
            "description": "Identifies and categorizes errors"
        },
        {
            "name": "Context Manager",
            "data_path": "context_manager/training_data.jsonl",
            "model_path": "context_manager/simple_output",
            "description": "Tracks conversation state"
        },
        {
            "name": "Docker API Mapper",
            "data_path": "api_mappers/docker_mapper_data.jsonl",
            "model_path": "api_mappers/docker_output",
            "description": "Converts intents to Docker commands"
        },
        {
            "name": "Orchestrator",
            "data_path": "orchestrator/training_data.jsonl",
            "model_path": "orchestrator/simple_output",
            "description": "Coordinates all models"
        }
    ]
    
    trained_count = 0
    data_count = 0
    
    for comp in components:
        print(f"\n{comp['name']}:")
        print(f"  Purpose: {comp['description']}")
        
        # Check training data
        if os.path.exists(comp['data_path']):
            with open(comp['data_path'], 'r') as f:
                lines = sum(1 for _ in f)
            print(f"  ✓ Training data: {lines} examples")
            data_count += 1
        else:
            print(f"  ✗ Training data: Not found")
            
        # Check trained model
        if os.path.exists(comp['model_path']):
            print(f"  ✓ Model: Trained and ready")
            trained_count += 1
        else:
            print(f"  ✗ Model: Not trained yet")
    
    print(f"\n{'='*40}")
    print(f"Summary: {trained_count}/{len(components)} models trained")
    print(f"Training data: {data_count}/{len(components)} datasets available")
    
    if trained_count < len(components):
        print("\nTo train missing models:")
        print("  python train_all_simple.py")
    else:
        print("\nAll models trained! Test with:")
        print("  python test_trained_swarm.py")
        print("  python full_swarm_demo.py")

if __name__ == "__main__":
    check_swarm_status()
