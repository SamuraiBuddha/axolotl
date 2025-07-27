#!/usr/bin/env python3
"""
Enhanced Swarm Training Data Generator
Adds edge cases, error handling, and domain-specific examples
"""

import sys
sys.path.append('..')
from generate_swarm_training_data import SwarmDataGenerator, INTENT_CATEGORIES
import json
import random
from pathlib import Path

class EnhancedSwarmDataGenerator(SwarmDataGenerator):
    """Enhanced generator with more comprehensive examples"""
    
    def __init__(self):
        super().__init__()
        self.domain_specific_intents = self.load_domain_intents()
        
    def load_domain_intents(self):
        """Load domain-specific intents"""
        return {
            "development": {
                "git_operations": [
                    ("commit these changes with message {}", "git_commit"),
                    ("push to main branch", "git_push"),
                    ("create a new branch called {}", "git_branch"),
                    ("merge {} into main", "git_merge"),
                    ("show git status", "git_status"),
                    ("stash current changes", "git_stash")
                ],
                "testing": [
                    ("run all tests", "run_tests"),
                    ("test the {} function", "test_specific"),
                    ("generate coverage report", "coverage_report"),
                    ("run integration tests", "test_integration")
                ],
                "debugging": [
                    ("set breakpoint at line {}", "debug_breakpoint"),
                    ("show call stack", "debug_stack"),
                    ("inspect variable {}", "debug_inspect"),
                    ("continue execution", "debug_continue")
                ]
            },
            "devops": {
                "kubernetes": [
                    ("scale {} to 3 replicas", "k8s_scale"),
                    ("show pod status", "k8s_pods"),
                    ("deploy to production", "k8s_deploy"),
                    ("rollback last deployment", "k8s_rollback")
                ],
                "monitoring": [
                    ("show CPU usage", "monitor_cpu"),
                    ("check memory consumption", "monitor_memory"),
                    ("view error logs", "monitor_logs"),
                    ("set up alert for {}", "monitor_alert")
                ]
            },
            "data_science": {
                "analysis": [
                    ("load {} dataset", "data_load"),
                    ("show data summary", "data_summary"),
                    ("plot correlation matrix", "data_correlations"),
                    ("perform PCA analysis", "data_pca")
                ],
                "ml_operations": [
                    ("train model on {}", "ml_train"),
                    ("evaluate model performance", "ml_evaluate"),
                    ("tune hyperparameters", "ml_tune"),
                    ("export model to {}", "ml_export")
                ]
            }
        }
        
    def generate_edge_cases(self):
        """Generate edge cases and difficult examples"""
        print("Generating edge cases...")
        
        edge_cases = [
            # Typos and misspellings
            ("craete a new file", "INTENT: file_create CONFIDENCE: 0.8"),
            ("delte that", "INTENT: file_delete CONFIDENCE: 0.7"),
            ("shwo me the logs", "INTENT: docker_logs CONFIDENCE: 0.8"),
            
            # Ambiguous pronouns
            ("delete it", "CLARIFY: What would you like to delete?"),
            ("run that again", "CLARIFY: What would you like to run?"),
            ("fix this", "CLARIFY: What needs to be fixed?"),
            
            # Multiple possible intents
            ("show me the file", "CLARIFY: Which file? Or did you mean list files?"),
            ("check the status", "CLARIFY: Status of what? System, Docker, or Git?"),
            
            # Natural language variations
            ("can you please create a new file called test.py", "INTENT: file_create PARAM: test.py"),
            ("I need to see what's in the config", "INTENT: file_read PARAM: config"),
            ("let's spin up the database", "INTENT: docker_start PARAM: database"),
            
            # Compound requests (should split)
            ("create a file and then read it", "INTENT: compound_operation ACTIONS: file_create,file_read"),
            ("stop all containers and restart the web server", "INTENT: compound_operation ACTIONS: docker_stop_all,docker_start"),
            
            # Emotional or conversational
            ("ugh, just delete everything", "CLARIFY: This would delete all files. Please confirm specific items to delete."),
            ("thanks, now show me the logs", "INTENT: docker_logs ACKNOWLEDGE: thanks"),
            
            # Technical jargon
            ("kubectl get pods", "INTENT: k8s_pods"),
            ("git push origin main", "INTENT: git_push PARAMS: origin main"),
            ("docker-compose up -d", "INTENT: docker_compose_up PARAMS: detached"),
            
            # Partial commands
            ("cd ..", "INTENT: change_directory PARAM: .."),
            ("ls -la", "INTENT: file_list PARAMS: all,long"),
            ("grep TODO", "INTENT: file_search PARAM: TODO")
        ]
        
        for user_input, expected_output in edge_cases:
            self.intent_data.append({
                "messages": [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": expected_output}
                ]
            })
            
    def generate_context_edge_cases(self):
        """Generate complex context management scenarios"""
        print("Generating context edge cases...")
        
        complex_contexts = [
            # Context switching
            [
                ("show docker containers", {"context": "docker", "action": "list"}),
                ("now show me files", {"context": "files", "previous_context": "docker"}),
                ("go back to docker", {"context": "docker", "restored": True})
            ],
            
            # Nested operations
            [
                ("create a folder called src", {"current_folder": "src", "action": "create"}),
                ("go into it", {"path": "./src", "action": "cd"}),
                ("create main.py here", {"current_folder": "src", "file": "main.py"}),
                ("go back", {"path": "..", "current_folder": "."})
            ],
            
            # Long-term memory
            [
                ("remember that port 8080 is for dev", {"memory": {"port_8080": "dev"}}),
                ("and 8081 is for staging", {"memory": {"port_8080": "dev", "port_8081": "staging"}}),
                ("what ports did I mention?", {"recall": ["port_8080", "port_8081"]}),
                ("forget about staging", {"memory": {"port_8080": "dev"}, "removed": "port_8081"})
            ],
            
            # Error recovery context
            [
                ("read config.yaml", {"action": "file_read", "target": "config.yaml"}),
                ("ERROR: File not found", {"error": True, "last_action_failed": True}),
                ("create it first", {"action": "file_create", "target": "config.yaml", "recovery": True}),
                ("now read it", {"action": "file_read", "target": "config.yaml", "retry": True})
            ]
        ]
        
        for conversation in complex_contexts:
            for i, (user_input, context_update) in enumerate(conversation):
                self.context_data.append({
                    "messages": [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": f"CONTEXT: {json.dumps(context_update)} TURN: {i+1} MEMORY_SIZE: {len(context_update)}"}
                    ]
                })
                
    def generate_api_edge_cases(self):
        """Generate API mapping edge cases"""
        print("Generating API mapping edge cases...")
        
        # Complex docker operations
        docker_complex = [
            ("docker_compose PARAM:up FLAGS:detached,build", "docker-compose up -d --build"),
            ("docker_exec PARAM:web CMD:python manage.py migrate", "docker exec web python manage.py migrate"),
            ("docker_logs PARAM:api FLAGS:follow,tail=100", "docker logs -f --tail=100 api"),
            ("docker_prune TYPE:volumes FLAGS:force", "docker volume prune -f"),
            ("docker_build PATH:. TAG:myapp:latest", "docker build -t myapp:latest .")
        ]
        
        for intent, api_call in docker_complex:
            self.api_mapper_data["docker"].append({
                "messages": [
                    {"role": "user", "content": intent},
                    {"role": "assistant", "content": f"API_CALL: {api_call}"}
                ]
            })
            
        # Complex file operations
        file_complex = [
            ("file_find PATTERN:*.py EXCLUDE:.git", "find . -name '*.py' -not -path './.git/*'"),
            ("file_chmod PARAM:script.sh MODE:755", "chmod 755 script.sh"),
            ("file_copy SOURCE:template.txt DEST:output.txt", "shutil.copy('template.txt', 'output.txt')"),
            ("file_archive PATH:project FORMAT:zip", "shutil.make_archive('project', 'zip', 'project')"),
            ("file_watch PATH:src ACTION:reload", "watchdog.observers.Observer() # monitor src/")
        ]
        
        for intent, api_call in file_complex:
            self.api_mapper_data["file"].append({
                "messages": [
                    {"role": "user", "content": intent},
                    {"role": "assistant", "content": f"API_CALL: {api_call}"}
                ]
            })
            
    def generate_orchestrator_edge_cases(self):
        """Generate complex orchestration scenarios"""
        print("Generating orchestrator edge cases...")
        
        complex_orchestrations = [
            {
                "user": "set up a new python project with git and tests",
                "routing": [
                    "ROUTE: intent_parser -> compound_operation",
                    "PLAN: create_folder -> git_init -> create_files -> create_tests",
                    "ROUTE: api_mapper_file -> mkdir new_project",
                    "ROUTE: api_mapper_git -> git init",
                    "ROUTE: api_mapper_file -> create setup.py",
                    "ROUTE: api_mapper_file -> create requirements.txt",
                    "ROUTE: api_mapper_file -> mkdir tests",
                    "ROUTE: api_mapper_file -> create test_main.py",
                    "AGGREGATE: Project structure created",
                    "RESPOND: Created new Python project with git and test structure"
                ]
            },
            {
                "user": "analyze why the app is slow and fix it",
                "routing": [
                    "ROUTE: intent_parser -> performance_analysis",
                    "ROUTE: api_mapper_docker -> docker stats",
                    "ANALYZE: High CPU usage detected",
                    "ROUTE: api_mapper_docker -> docker logs api",
                    "ROUTE: error_recognizer -> analyze logs",
                    "FINDING: Database queries taking too long",
                    "ROUTE: intent_parser -> suggest_fix",
                    "PLAN: Add database index",
                    "ROUTE: api_mapper_docker -> docker exec db",
                    "EXECUTE: CREATE INDEX on slow_table",
                    "VERIFY: Performance improved",
                    "RESPOND: Added database index to improve query performance"
                ]
            },
            {
                "user": "deploy if all tests pass",
                "routing": [
                    "ROUTE: intent_parser -> conditional_operation",
                    "CONDITION: Check test status",
                    "ROUTE: api_mapper_system -> python -m pytest",
                    "ANALYZE: Exit code 0 (success)",
                    "CONDITION_MET: Tests passed",
                    "ROUTE: intent_parser -> deploy",
                    "ROUTE: api_mapper_docker -> docker build",
                    "ROUTE: api_mapper_docker -> docker tag",
                    "ROUTE: api_mapper_docker -> docker push",
                    "ROUTE: api_mapper_k8s -> kubectl apply",
                    "RESPOND: All tests passed. Deployment completed successfully."
                ]
            }
        ]
        
        for orchestration in complex_orchestrations:
            self.orchestrator_data.append({
                "messages": [
                    {"role": "user", "content": orchestration["user"]},
                    {"role": "assistant", "content": " -> ".join(orchestration["routing"])}
                ]
            })
            
    def generate_failure_scenarios(self):
        """Generate failure and recovery scenarios"""
        print("Generating failure scenarios...")
        
        failures = [
            # Network failures
            {
                "error": "requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection'))",
                "response": "ERROR_TYPE: connection_error SUGGESTION: Check network connectivity and retry. The remote server may be down."
            },
            
            # Permission failures  
            {
                "error": "PermissionError: [Errno 13] Permission denied: '/etc/config'",
                "response": "ERROR_TYPE: permission_error SUGGESTION: Insufficient permissions. Try with sudo or check file ownership."
            },
            
            # Resource exhaustion
            {
                "error": "MemoryError: Unable to allocate 8.00 GiB for an array",
                "response": "ERROR_TYPE: memory_error SUGGESTION: Out of memory. Reduce data size or increase available RAM."
            },
            
            # Timeout errors
            {
                "error": "TimeoutError: Operation timed out after 30 seconds",
                "response": "ERROR_TYPE: timeout_error SUGGESTION: Operation took too long. Check system load or increase timeout limit."
            },
            
            # Version conflicts
            {
                "error": "ImportError: cannot import name 'TypedDict' from 'typing' (Python 3.7)",
                "response": "ERROR_TYPE: version_error SUGGESTION: Feature requires Python 3.8+. Upgrade Python or use backport package."
            }
        ]
        
        for failure in failures:
            self.error_data.append({
                "messages": [
                    {"role": "user", "content": failure["error"]},
                    {"role": "assistant", "content": failure["response"]}
                ]
            })
            
    def generate_all_enhanced(self):
        """Generate all enhanced training data"""
        print("\n=== Enhanced Swarm Training Data Generation ===")
        
        # Generate base data
        self.generate_intent_data(num_examples=800)  # More examples
        self.generate_context_manager_data(num_examples=500)
        self.generate_api_mapper_data()
        self.generate_error_recognizer_data()
        self.generate_orchestrator_data()
        
        # Add enhanced data
        self.generate_edge_cases()
        self.generate_context_edge_cases()
        self.generate_api_edge_cases()
        self.generate_orchestrator_edge_cases()
        self.generate_failure_scenarios()
        
        # Add domain-specific intents
        for domain, categories in self.domain_specific_intents.items():
            for category, intents in categories.items():
                for template, intent_type in intents:
                    if "{}" in template:
                        param = f"{domain}_{category}_example"
                        user_input = template.format(param)
                        output = f"INTENT: {intent_type} PARAM: {param}"
                    else:
                        user_input = template
                        output = f"INTENT: {intent_type}"
                        
                    self.intent_data.append({
                        "messages": [
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": output}
                        ]
                    })
        
        # Save everything
        self.save_all_data()
        self.generate_evaluation_data()
        
        print(f"\nTotal enhanced examples: {len(self.intent_data) + len(self.context_data) + sum(len(d) for d in self.api_mapper_data.values()) + len(self.error_data) + len(self.orchestrator_data)}")
        print("Enhanced training data generation complete!")

if __name__ == "__main__":
    generator = EnhancedSwarmDataGenerator()
    generator.generate_all_enhanced()
