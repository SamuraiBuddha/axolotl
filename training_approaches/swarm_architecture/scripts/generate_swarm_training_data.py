#!/usr/bin/env python3
"""
Swarm Architecture Training Data Generator
Generates comprehensive training data for all micro-models in the swarm
"""

import json
import random
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Base path for swarm architecture
BASE_PATH = Path(__file__).parent.parent

# Intent categories and variations
INTENT_CATEGORIES = {
    "file_operations": {
        "intents": ["file_create", "file_read", "file_write", "file_delete", "file_search", "file_list"],
        "variations": [
            ("create a new file called {}", "file_create"),
            ("make a {} file", "file_create"),
            ("generate a file named {}", "file_create"),
            ("read the contents of {}", "file_read"),
            ("show me what's in {}", "file_read"),
            ("what does {} contain?", "file_read"),
            ("write {} to the file", "file_write"),
            ("add {} to that file", "file_write"),
            ("delete {}", "file_delete"),
            ("remove the {} file", "file_delete"),
            ("search for {} in files", "file_search"),
            ("find {} in the codebase", "file_search"),
            ("list all {} files", "file_list"),
            ("show me the {} in this directory", "file_list")
        ]
    },
    "docker_operations": {
        "intents": ["docker_list", "docker_start", "docker_stop", "docker_logs", "docker_exec"],
        "variations": [
            ("list all containers", "docker_list"),
            ("show docker containers", "docker_list"),
            ("what's running in docker?", "docker_list"),
            ("start the {} container", "docker_start"),
            ("spin up {}", "docker_start"),
            ("stop {}", "docker_stop"),
            ("shut down the {} container", "docker_stop"),
            ("show logs for {}", "docker_logs"),
            ("what's {} saying?", "docker_logs"),
            ("execute {} in the container", "docker_exec"),
            ("run {} inside docker", "docker_exec")
        ]
    },
    "memory_operations": {
        "intents": ["memory_create", "memory_read", "memory_update", "memory_search"],
        "variations": [
            ("remember that {}", "memory_create"),
            ("save this: {}", "memory_create"),
            ("store the fact that {}", "memory_create"),
            ("what did I tell you about {}?", "memory_read"),
            ("recall information on {}", "memory_read"),
            ("update the {} information", "memory_update"),
            ("change what you know about {}", "memory_update"),
            ("search memory for {}", "memory_search"),
            ("find anything about {} in memory", "memory_search")
        ]
    },
    "code_operations": {
        "intents": ["code_explain", "code_refactor", "code_test", "code_debug"],
        "variations": [
            ("explain this code: {}", "code_explain"),
            ("what does {} do?", "code_explain"),
            ("refactor {} for better performance", "code_refactor"),
            ("improve this: {}", "code_refactor"),
            ("write tests for {}", "code_test"),
            ("test the {} function", "code_test"),
            ("debug {}", "code_debug"),
            ("fix the issue in {}", "code_debug")
        ]
    },
    "system_operations": {
        "intents": ["system_status", "system_restart", "system_config", "system_logs"],
        "variations": [
            ("check system status", "system_status"),
            ("how's the system doing?", "system_status"),
            ("restart {}", "system_restart"),
            ("reboot the {} service", "system_restart"),
            ("configure {}", "system_config"),
            ("change {} settings", "system_config"),
            ("show system logs", "system_logs"),
            ("what do the logs say?", "system_logs")
        ]
    }
}

# Common file names and paths for realistic examples
FILE_NAMES = ["config.yaml", "app.py", "main.js", "README.md", "package.json", "Dockerfile", 
              "test.py", "index.html", "style.css", "data.json", ".env", "requirements.txt"]

CONTAINER_NAMES = ["web", "db", "redis", "nginx", "api", "worker", "postgres", "mysql"]

CODE_SNIPPETS = ["this function", "the loop", "that class", "the API endpoint", "this algorithm"]

# Error patterns for error recognizer
ERROR_PATTERNS = [
    {
        "error": "FileNotFoundError: [Errno 2] No such file or directory: '{}'",
        "category": "file_error",
        "suggestion": "Check if the file exists or create it first"
    },
    {
        "error": "PermissionError: [Errno 13] Permission denied: '{}'",
        "category": "permission_error",
        "suggestion": "Check file permissions or run with appropriate privileges"
    },
    {
        "error": "docker: Error response from daemon: No such container: {}",
        "category": "docker_error",
        "suggestion": "Container doesn't exist. List containers to see available ones"
    },
    {
        "error": "SyntaxError: invalid syntax",
        "category": "syntax_error",
        "suggestion": "Check for missing colons, parentheses, or indentation"
    },
    {
        "error": "ImportError: No module named '{}'",
        "category": "import_error",
        "suggestion": "Install the missing module or check the import path"
    },
    {
        "error": "ConnectionRefusedError: [Errno 111] Connection refused",
        "category": "connection_error",
        "suggestion": "Check if the service is running and the port is correct"
    }
]

class SwarmDataGenerator:
    def __init__(self):
        self.intent_data = []
        self.context_data = []
        self.api_mapper_data = {
            "docker": [],
            "file": [],
            "memory": [],
            "system": []
        }
        self.error_data = []
        self.orchestrator_data = []
        
    def generate_intent_data(self, num_examples: int = 500):
        """Generate intent parsing training data"""
        print("Generating intent parser training data...")
        
        for category, data in INTENT_CATEGORIES.items():
            for variation_template, intent in data["variations"]:
                # Generate multiple examples with different parameters
                for _ in range(num_examples // len(data["variations"]) // len(INTENT_CATEGORIES)):
                    if "{}" in variation_template:
                        # Fill in template with appropriate data
                        if "file" in category:
                            param = random.choice(FILE_NAMES)
                        elif "docker" in category:
                            param = random.choice(CONTAINER_NAMES)
                        elif "code" in category:
                            param = random.choice(CODE_SNIPPETS)
                        else:
                            param = f"example_{random.randint(1, 100)}"
                        
                        user_input = variation_template.format(param)
                        assistant_output = f"INTENT: {intent} PARAM: {param}"
                    else:
                        user_input = variation_template
                        assistant_output = f"INTENT: {intent}"
                    
                    self.intent_data.append({
                        "messages": [
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": assistant_output}
                        ]
                    })
        
        # Add ambiguous cases
        ambiguous_cases = [
            ("show me the file", "CLARIFY: Which file would you like to see?"),
            ("stop it", "CLARIFY: What would you like me to stop?"),
            ("run that again", "CLARIFY: What would you like me to run again?"),
            ("fix this", "CLARIFY: What needs to be fixed?")
        ]
        
        for user_input, assistant_output in ambiguous_cases:
            self.intent_data.append({
                "messages": [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant_output}
                ]
            })
            
    def generate_context_manager_data(self, num_examples: int = 300):
        """Generate context management training data"""
        print("Generating context manager training data...")
        
        # Multi-turn conversations with state tracking
        conversations = [
            [
                ("create a file called test.py", {"current_file": "test.py", "action": "created"}),
                ("write hello world to it", {"current_file": "test.py", "action": "write", "referring_to": "current_file"}),
                ("now read it", {"current_file": "test.py", "action": "read", "referring_to": "current_file"})
            ],
            [
                ("list docker containers", {"action": "list", "context": "docker"}),
                ("start the first one", {"action": "start", "referring_to": "first_container", "needs": "previous_list"}),
                ("show its logs", {"action": "logs", "referring_to": "current_container"})
            ],
            [
                ("remember that port 8080 is for testing", {"memory_key": "port_8080", "value": "testing"}),
                ("what port should I use?", {"action": "recall", "referring_to": "port_8080"}),
                ("change it to 8081", {"action": "update", "memory_key": "port_8080", "new_value": "8081"})
            ]
        ]
        
        for conversation in conversations:
            for i, (user_input, context_state) in enumerate(conversation):
                self.context_data.append({
                    "messages": [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": f"CONTEXT: {json.dumps(context_state)} TURN: {i+1}"}
                    ]
                })
                
    def generate_api_mapper_data(self):
        """Generate API mapping training data for each service"""
        print("Generating API mapper training data...")
        
        # Docker API mappings
        docker_mappings = [
            ("docker_list_containers", "docker ps -a --format json"),
            ("docker_start PARAM:web", "docker start web"),
            ("docker_stop PARAM:db", "docker stop db"),
            ("docker_logs PARAM:nginx", "docker logs nginx --tail 50"),
            ("docker_exec PARAM:api CMD:ps aux", "docker exec api ps aux")
        ]
        
        for intent, api_call in docker_mappings:
            self.api_mapper_data["docker"].append({
                "messages": [
                    {"role": "user", "content": intent},
                    {"role": "assistant", "content": f"API_CALL: {api_call}"}
                ]
            })
            
        # File API mappings
        file_mappings = [
            ("file_create PARAM:test.py", "Path('test.py').touch()"),
            ("file_read PARAM:config.yaml", "Path('config.yaml').read_text()"),
            ("file_write PARAM:data.json CONTENT:{'key':'value'}", "Path('data.json').write_text(json.dumps({'key':'value'}))"),
            ("file_delete PARAM:temp.txt", "Path('temp.txt').unlink()"),
            ("file_list PARAM:*.py", "list(Path('.').glob('*.py'))")
        ]
        
        for intent, api_call in file_mappings:
            self.api_mapper_data["file"].append({
                "messages": [
                    {"role": "user", "content": intent},
                    {"role": "assistant", "content": f"API_CALL: {api_call}"}
                ]
            })

    def generate_error_recognizer_data(self):
        """Generate error recognition training data"""
        print("Generating error recognizer training data...")
        
        for pattern in ERROR_PATTERNS:
            # Generate variations with different parameters
            for _ in range(20):
                if "{}" in pattern["error"]:
                    param = random.choice(FILE_NAMES + CONTAINER_NAMES)
                    error_msg = pattern["error"].format(param)
                else:
                    error_msg = pattern["error"]
                    
                self.error_data.append({
                    "messages": [
                        {"role": "user", "content": error_msg},
                        {"role": "assistant", "content": f"ERROR_TYPE: {pattern['category']} SUGGESTION: {pattern['suggestion']}"}
                    ]
                })
        
        # Add stack traces and multi-line errors
        complex_errors = [
            {
                "error": "Traceback (most recent call last):\n  File 'app.py', line 10\n    import missing_module\nImportError: No module named 'missing_module'",
                "response": "ERROR_TYPE: import_error SUGGESTION: Install missing_module with pip or check import name"
            },
            {
                "error": "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
                "response": "ERROR_TYPE: type_error SUGGESTION: Convert types to match - use str() or int()"
            }
        ]
        
        for error_case in complex_errors:
            self.error_data.append({
                "messages": [
                    {"role": "user", "content": error_case["error"]},
                    {"role": "assistant", "content": error_case["response"]}
                ]
            })
            
    def generate_orchestrator_data(self):
        """Generate orchestrator routing and coordination data"""
        print("Generating orchestrator training data...")
        
        # Complex multi-step operations
        complex_operations = [
            {
                "user": "create a python file with a hello world function and test it",
                "routing": [
                    "ROUTE: intent_parser -> file_create",
                    "ROUTE: api_mapper_file -> create test.py",
                    "ROUTE: intent_parser -> file_write", 
                    "ROUTE: api_mapper_file -> write function",
                    "ROUTE: intent_parser -> code_test",
                    "ROUTE: api_mapper_system -> python test.py"
                ]
            },
            {
                "user": "check if the web container is running and restart it if needed",
                "routing": [
                    "ROUTE: intent_parser -> docker_list",
                    "ROUTE: api_mapper_docker -> list containers",
                    "ANALYZE: web container status",
                    "DECISION: container stopped",
                    "ROUTE: intent_parser -> docker_start",
                    "ROUTE: api_mapper_docker -> start web"
                ]
            },
            {
                "user": "find all TODO comments and create a task list",
                "routing": [
                    "ROUTE: intent_parser -> code_search",
                    "ROUTE: api_mapper_file -> search TODO",
                    "AGGREGATE: found 5 TODOs",
                    "ROUTE: intent_parser -> file_create",
                    "ROUTE: api_mapper_file -> create tasks.md",
                    "FORMAT: generate markdown list"
                ]
            }
        ]
        
        for operation in complex_operations:
            self.orchestrator_data.append({
                "messages": [
                    {"role": "user", "content": operation["user"]},
                    {"role": "assistant", "content": " -> ".join(operation["routing"])}
                ]
            })
            
        # Add error handling scenarios
        error_scenarios = [
            {
                "user": "read a file that doesn't exist",
                "routing": [
                    "ROUTE: intent_parser -> file_read",
                    "ROUTE: api_mapper_file -> read nonexistent.txt",
                    "ERROR: FileNotFoundError",
                    "ROUTE: error_recognizer -> analyze error",
                    "DECISION: suggest creating file first",
                    "RESPOND: File doesn't exist. Would you like me to create it?"
                ]
            }
        ]
        
        for scenario in error_scenarios:
            self.orchestrator_data.append({
                "messages": [
                    {"role": "user", "content": scenario["user"]},
                    {"role": "assistant", "content": " -> ".join(scenario["routing"])}
                ]
            })
            
    def save_all_data(self):
        """Save all generated data to appropriate directories"""
        # Intent parser
        intent_path = BASE_PATH / "intent_parser" / "training_data.jsonl"
        with open(intent_path, 'w') as f:
            for item in self.intent_data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(self.intent_data)} intent examples to {intent_path}")
        
        # Context manager
        context_path = BASE_PATH / "context_manager" / "training_data.jsonl"
        os.makedirs(context_path.parent, exist_ok=True)
        with open(context_path, 'w') as f:
            for item in self.context_data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(self.context_data)} context examples to {context_path}")
        
        # API mappers
        for api_type, data in self.api_mapper_data.items():
            if data:  # Only save if we have data
                mapper_path = BASE_PATH / "api_mappers" / f"{api_type}_mapper_data.jsonl"
                os.makedirs(mapper_path.parent, exist_ok=True)
                with open(mapper_path, 'w') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')
                print(f"Saved {len(data)} {api_type} API examples to {mapper_path}")
        
        # Error recognizer
        error_path = BASE_PATH / "error_recognizer" / "training_data.jsonl"
        os.makedirs(error_path.parent, exist_ok=True)
        with open(error_path, 'w') as f:
            for item in self.error_data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(self.error_data)} error examples to {error_path}")
        
        # Orchestrator
        orchestrator_path = BASE_PATH / "orchestrator" / "training_data.jsonl"
        os.makedirs(orchestrator_path.parent, exist_ok=True)
        with open(orchestrator_path, 'w') as f:
            for item in self.orchestrator_data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(self.orchestrator_data)} orchestrator examples to {orchestrator_path}")
        
    def generate_evaluation_data(self):
        """Generate separate evaluation datasets (20% of training size)"""
        print("\nGenerating evaluation datasets...")
        
        # Simple split - in production, ensure no overlap
        for component, data in [
            ("intent_parser", self.intent_data),
            ("context_manager", self.context_data),
            ("error_recognizer", self.error_data),
            ("orchestrator", self.orchestrator_data)
        ]:
            if data:
                eval_size = min(len(data) - 1, max(10, len(data) // 5))  # 20% or minimum 10, but not more than available
                eval_data = random.sample(data, eval_size)
                
                eval_path = BASE_PATH / component / "eval_data.jsonl"
                with open(eval_path, 'w') as f:
                    for item in eval_data:
                        f.write(json.dumps(item) + '\n')
                print(f"Saved {len(eval_data)} eval examples for {component}")
                
    def generate_all(self):
        """Generate all training data"""
        print("=== Swarm Architecture Training Data Generation ===")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
        
        # Generate data for each component
        self.generate_intent_data(num_examples=500)
        self.generate_context_manager_data(num_examples=300)
        self.generate_api_mapper_data()
        self.generate_error_recognizer_data()
        self.generate_orchestrator_data()
        
        # Save all data
        self.save_all_data()
        
        # Generate evaluation sets
        self.generate_evaluation_data()
        
        print("\n=== Generation Complete ===")
        print(f"Total training examples: {len(self.intent_data) + len(self.context_data) + sum(len(d) for d in self.api_mapper_data.values()) + len(self.error_data) + len(self.orchestrator_data)}")
        

if __name__ == "__main__":
    generator = SwarmDataGenerator()
    generator.generate_all()
