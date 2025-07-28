#!/usr/bin/env python3
"""
Test trained swarm models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def test_model(model_path, test_input):
    """Test a trained model"""
    print(f"\nTesting model at: {model_path}")
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return
        
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Test generation
    inputs = tokenizer(test_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_input}")
    print(f"Output: {response}")

def main():
    print("=== Testing Swarm Models ===")
    
    # Test intent parser
    test_model(
        "intent_parser/output",
        "User: create a new python file\nAssistant:"
    )
    
    # Test error recognizer
    test_model(
        "error_recognizer/output",
        "Error: FileNotFoundError: No such file or directory\nAnalysis:"
    )

if __name__ == "__main__":
    main()
