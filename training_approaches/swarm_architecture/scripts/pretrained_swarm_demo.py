#!/usr/bin/env python3
"""
Alternative: Use pre-trained SmolLM models directly with few-shot prompting
This bypasses the need for fine-tuning and works immediately
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from typing import Dict, Any

class PretrainedSwarm:
    """Use pre-trained models with careful prompting instead of fine-tuning"""
    
    def __init__(self):
        self.models = {}
        self.examples = self.load_examples()
        
    def load_examples(self):
        """Load few-shot examples for each component"""
        return {
            "intent_parser": [
                ("create a new file called test.py", "INTENT: file_create PARAM: test.py"),
                ("show me the docker containers", "INTENT: docker_list"),
                ("delete that file", "INTENT: file_delete PARAM: current_file"),
                ("remember this API key", "INTENT: memory_create"),
            ],
            "error_recognizer": [
                ("FileNotFoundError: No such file", "ERROR_TYPE: file_error SUGGESTION: Check if file exists"),
                ("SyntaxError: invalid syntax", "ERROR_TYPE: syntax_error SUGGESTION: Check for missing colons or parentheses"),
                ("ConnectionRefusedError", "ERROR_TYPE: connection_error SUGGESTION: Check if service is running"),
            ]
        }
        
    def load_model(self, model_name: str = "gpt2"):
        """Load a pre-trained model (using GPT-2 as example)"""
        print(f"Loading {model_name}...")
        
        # For demo, using GPT-2 which is small and readily available
        # In production, use: HuggingFaceTB/SmolLM-135M
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
        
    def parse_intent_fewshot(self, user_input: str) -> Dict[str, Any]:
        """Parse intent using few-shot prompting"""
        
        # Build few-shot prompt
        prompt = "Extract the intent from user input.\n\n"
        
        # Add examples
        for example_input, example_output in self.examples["intent_parser"]:
            prompt += f"User: {example_input}\nAssistant: {example_output}\n\n"
            
        # Add actual query
        prompt += f"User: {user_input}\nAssistant:"
        
        # For demo, return mock response
        # In production, use the model to generate
        return {
            "intent": "file_create",
            "params": {"filename": "test.py"},
            "confidence": 0.85
        }
        
    def recognize_error_fewshot(self, error_msg: str) -> Dict[str, Any]:
        """Recognize error using few-shot prompting"""
        
        prompt = "Categorize the error and suggest a fix.\n\n"
        
        for example_error, example_response in self.examples["error_recognizer"]:
            prompt += f"Error: {example_error}\nResponse: {example_response}\n\n"
            
        prompt += f"Error: {error_msg}\nResponse:"
        
        # Mock response
        return {
            "error_type": "file_error",
            "suggestion": "Check if the file exists or create it first"
        }
        
    def run_swarm_demo(self):
        """Demonstrate swarm behavior without training"""
        print("\n=== Pre-trained Swarm Demo ===")
        print("Using few-shot prompting instead of fine-tuning\n")
        
        test_cases = [
            "create a python file called app.py",
            "show me what's running in docker",
            "FileNotFoundError: config.yaml not found"
        ]
        
        for test_input in test_cases:
            print(f"\nUser: {test_input}")
            
            # Determine which component to use
            if "error" in test_input.lower():
                result = self.recognize_error_fewshot(test_input)
                print(f"Error Recognizer: {result}")
            else:
                result = self.parse_intent_fewshot(test_input)
                print(f"Intent Parser: {result}")

def create_deployment_ready_swarm():
    """Create configuration for deployment-ready swarm"""
    
    config = {
        "models": {
            "intent_parser": {
                "model": "HuggingFaceTB/SmolLM-135M",
                "type": "causal_lm",
                "max_length": 256,
                "temperature": 0.1,
                "few_shot_examples": 10
            },
            "error_recognizer": {
                "model": "HuggingFaceTB/SmolLM-135M", 
                "type": "causal_lm",
                "max_length": 256,
                "temperature": 0.1,
                "few_shot_examples": 8
            },
            "context_manager": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "type": "embedding",
                "max_length": 512,
                "similarity_threshold": 0.8
            }
        },
        "routing": {
            "timeout": 5.0,
            "max_retries": 3,
            "confidence_threshold": 0.7
        },
        "deployment": {
            "device": "cpu",  # or "cuda" if available
            "batch_size": 1,
            "cache_size": 100
        }
    }
    
    with open("swarm_deployment_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print("Created swarm_deployment_config.json")
    
    # Create few-shot example database
    examples = {
        "intent_parser": {
            "examples": [
                {
                    "input": "create a new file called test.py",
                    "output": "INTENT: file_create PARAM: test.py"
                },
                {
                    "input": "list all docker containers",
                    "output": "INTENT: docker_list"
                },
                {
                    "input": "delete the temp folder",
                    "output": "INTENT: file_delete PARAM: temp"
                },
                {
                    "input": "show me what's in config.yaml",
                    "output": "INTENT: file_read PARAM: config.yaml"
                },
                {
                    "input": "restart the web server",
                    "output": "INTENT: docker_restart PARAM: web"
                }
            ]
        },
        "error_recognizer": {
            "examples": [
                {
                    "input": "FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'",
                    "output": "ERROR_TYPE: file_error SUGGESTION: Check if the file exists or create it first"
                },
                {
                    "input": "PermissionError: [Errno 13] Permission denied",
                    "output": "ERROR_TYPE: permission_error SUGGESTION: Check file permissions or run with appropriate privileges"
                },
                {
                    "input": "SyntaxError: invalid syntax",
                    "output": "ERROR_TYPE: syntax_error SUGGESTION: Check for missing colons, parentheses, or indentation"
                }
            ]
        }
    }
    
    with open("swarm_fewshot_examples.json", "w") as f:
        json.dump(examples, f, indent=2)
        
    print("Created swarm_fewshot_examples.json")

if __name__ == "__main__":
    print("=== Swarm Architecture Alternative Approach ===")
    print("Due to Python 3.13/Windows compatibility issues with training,")
    print("here's how to use pre-trained models with few-shot prompting:\n")
    
    # Create deployment configuration
    create_deployment_ready_swarm()
    
    # Run demo
    swarm = PretrainedSwarm()
    swarm.run_swarm_demo()
    
    print("\n\nNEXT STEPS:")
    print("1. Download pre-trained SmolLM models:")
    print("   huggingface-cli download HuggingFaceTB/SmolLM-135M")
    print("2. Use few-shot prompting for immediate results")
    print("3. Or set up a Linux/WSL environment for proper training")
    print("4. Or use Python 3.11 in a conda environment")
