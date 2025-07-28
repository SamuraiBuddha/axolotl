#!/usr/bin/env python3
"""
Full swarm demonstration with all trained models
"""

from transformers import pipeline
import os
import json

class SwarmDemo:
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load all trained models"""
        print("=== Loading Swarm Models ===")
        
        models_to_load = [
            ("intent_parser", "intent_parser/simple_output"),
            ("error_recognizer", "error_recognizer/simple_output"),
            ("context_manager", "context_manager/simple_output")
        ]
        
        for name, path in models_to_load:
            if os.path.exists(path):
                print(f"✓ Loading {name}...")
                self.models[name] = pipeline('text-generation', model=path, device=-1)
            else:
                print(f"✗ {name} not found at {path}")
                
    def process_user_input(self, user_input):
        """Process input through the swarm"""
        print(f"\n{'='*60}")
        print(f"User: {user_input}")
        print(f"{'='*60}")
        
        # Step 1: Intent Parser
        if "intent_parser" in self.models:
            prompt = f"User: {user_input}\nAssistant:"
            result = self.models["intent_parser"](prompt, max_length=50, temperature=0.1)
            intent = result[0]['generated_text'].split("Assistant:")[-1].strip().split("\n")[0]
            print(f"\n[Intent Parser] → {intent}")
            
            # Extract intent type
            if "INTENT:" in intent:
                intent_type = intent.split("INTENT:")[1].split()[0]
                print(f"  Detected: {intent_type}")
        
        # Step 2: Context Manager (if needed)
        if "context_manager" in self.models and "file" in user_input:
            context_prompt = f"Current context: {{}}\nNew action: {intent}\nContext:"
            result = self.models["context_manager"](context_prompt, max_length=50, temperature=0.1)
            context = result[0]['generated_text'].split("Context:")[-1].strip()
            print(f"\n[Context Manager] → Tracking: {context}")
        
        # Step 3: Simulate execution and error
        if "create" in user_input.lower():
            error_msg = "FileNotFoundError: Parent directory does not exist"
            print(f"\n[System] → Error: {error_msg}")
            
            # Step 4: Error Recognizer
            if "error_recognizer" in self.models:
                error_prompt = f"Error: {error_msg}\nAnalysis:"
                result = self.models["error_recognizer"](error_prompt, max_length=80, temperature=0.1)
                analysis = result[0]['generated_text'].split("Analysis:")[-1].strip()
                print(f"\n[Error Recognizer] → {analysis}")
                
        print(f"\n[Swarm Coordinator] → Complete!")
        
    def run_test_suite(self):
        """Run comprehensive tests"""
        test_cases = [
            "create a new python file called app.py",
            "show me all docker containers",
            "delete the temporary files",
            "read the configuration",
            "fix this error I'm getting",
            "list all the files in the current directory"
        ]
        
        print("\n=== Running Swarm Test Suite ===")
        for test in test_cases:
            self.process_user_input(test)
            
    def interactive_mode(self):
        """Interactive swarm demo"""
        print("\n=== Interactive Swarm Mode ===")
        print("Type 'exit' to quit")
        print("Try commands like: 'create a file', 'show docker status', etc.\n")
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'exit':
                break
            self.process_user_input(user_input)

def main():
    print("=== Full Swarm Architecture Demonstration ===")
    print("Multiple specialized AI models working together\n")
    
    # Check which models are available
    models_status = [
        ("Intent Parser", "intent_parser/simple_output"),
        ("Error Recognizer", "error_recognizer/simple_output"),
        ("Context Manager", "context_manager/simple_output")
    ]
    
    print("Model Status:")
    ready_count = 0
    for name, path in models_status:
        if os.path.exists(path):
            print(f"✓ {name}: Ready")
            ready_count += 1
        else:
            print(f"✗ {name}: Not trained yet")
            
    if ready_count == 0:
        print("\nNo models found! Train them first with:")
        print("  python simple_train_wsl.py")
        print("  python train_all_simple.py")
        return
        
    # Create swarm demo
    swarm = SwarmDemo()
    
    # Run tests
    print("\n1. Running automated tests...")
    swarm.run_test_suite()
    
    # Interactive mode
    print("\n2. Starting interactive mode...")
    swarm.interactive_mode()
    
    print("\n=== Swarm Architecture Summary ===")
    print(f"✓ {ready_count} specialized models working together")
    print(f"✓ Total parameters: ~{ready_count * 124}M (vs 7B+ for single model)")
    print(f"✓ Can run on edge devices ($200 hardware)")
    print(f"✓ Modular and updateable")

if __name__ == "__main__":
    main()
