#!/usr/bin/env python3
"""
Train all swarm components using simple approach
"""

from simple_train_wsl import train_simple_model
import os

def main():
    """Train all swarm components"""
    print("=== Training All Swarm Components ===\n")
    
    components = [
        {
            "name": "Intent Parser",
            "data": "intent_parser/training_data.jsonl",
            "output": "intent_parser/simple_output",
            "status": "✓ Already trained"
        },
        {
            "name": "Error Recognizer", 
            "data": "error_recognizer/training_data.jsonl",
            "output": "error_recognizer/simple_output",
            "status": "Ready to train"
        },
        {
            "name": "Context Manager",
            "data": "context_manager/training_data.jsonl", 
            "output": "context_manager/simple_output",
            "status": "Ready to train"
        }
    ]
    
    for component in components:
        print(f"\n{'='*60}")
        print(f"{component['name']}: {component['status']}")
        print(f"{'='*60}")
        
        if os.path.exists(component['output']):
            print(f"✓ Already exists at {component['output']}")
            continue
            
        if os.path.exists(component['data']):
            print(f"Training on {component['data']}...")
            train_simple_model(
                model_name="gpt2",
                data_path=component['data'],
                output_dir=component['output']
            )
        else:
            print(f"✗ No training data found at {component['data']}")
    
    print("\n=== Training Complete ===")
    print("\nTo test all models, run: python test_trained_swarm.py")

if __name__ == "__main__":
    main()
