#!/usr/bin/env python3
"""
Swarm Intelligence Coordinator
Loads and orchestrates all micro-models for intelligent task execution
"""

import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from queue import Queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Inter-model communication message"""
    id: str
    source: str
    target: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    trace: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ModelCapability:
    """Describes what a micro-model can do"""
    model_name: str
    intents: List[str]
    max_tokens: int
    avg_response_time: float

class MicroModel:
    """Base class for all micro-models in the swarm"""
    
    def __init__(self, model_path: str, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.capabilities = []
        self.message_queue = Queue()
        self.response_times = []
        
        # Load model with 4-bit quantization
        logger.info(f"Loading {model_name} from {model_path}")
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load the model with optimization settings"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate response from the model"""
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistency
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Track performance
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        return response
        
    def publish_capabilities(self) -> ModelCapability:
        """Publish what this model can do"""
        avg_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0.1
        return ModelCapability(
            model_name=self.model_name,
            intents=self.capabilities,
            max_tokens=50,
            avg_response_time=avg_time
        )

class IntentParser(MicroModel):
    """Extracts intent from user input"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path, "intent_parser")
        self.capabilities = ["file_*", "docker_*", "memory_*", "code_*", "system_*"]
        
    def parse_intent(self, user_input: str) -> Dict[str, Any]:
        """Extract intent and parameters from user input"""
        prompt = f"User: {user_input}\nAssistant:"
        response = self.generate(prompt, max_tokens=30)
        
        # Parse response
        if "INTENT:" in response:
            parts = response.split("INTENT:")[1].strip().split()
            intent = parts[0]
            params = {}
            
            if "PARAM:" in response:
                param_part = response.split("PARAM:")[1].strip()
                params["param"] = param_part.split()[0]
                
            return {"intent": intent, "params": params, "confidence": 0.9}
        elif "CLARIFY:" in response:
            return {"intent": "clarify", "message": response.split("CLARIFY:")[1].strip()}
        else:
            return {"intent": "unknown", "confidence": 0.3}

class ContextManager(MicroModel):
    """Maintains conversation state"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path, "context_manager")
        self.context_state = {}
        self.conversation_history = []
        
    def update_context(self, message: Message) -> Dict[str, Any]:
        """Update context based on message"""
        self.conversation_history.append(message)
        
        # Generate context update
        prompt = f"Current context: {json.dumps(self.context_state)}\nNew action: {message.action} {message.params}\nUpdated context:"
        response = self.generate(prompt, max_tokens=50)
        
        # Parse context update
        try:
            if "CONTEXT:" in response:
                context_json = response.split("CONTEXT:")[1].strip()
                self.context_state.update(json.loads(context_json))
        except:
            pass
            
        return self.context_state

class ErrorRecognizer(MicroModel):
    """Recognizes and categorizes errors"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path, "error_recognizer")
        self.error_patterns = []
        
    def analyze_error(self, error_message: str) -> Dict[str, Any]:
        """Analyze error and suggest fixes"""
        prompt = f"Error: {error_message}\nAnalysis:"
        response = self.generate(prompt, max_tokens=50)
        
        result = {"error_type": "unknown", "suggestion": "Check the error message"}
        
        if "ERROR_TYPE:" in response and "SUGGESTION:" in response:
            error_type = response.split("ERROR_TYPE:")[1].split("SUGGESTION:")[0].strip()
            suggestion = response.split("SUGGESTION:")[1].strip()
            result = {"error_type": error_type, "suggestion": suggestion}
            
        return result

class SwarmOrchestrator:
    """Main orchestrator that coordinates all micro-models"""
    
    def __init__(self, config_path: str = "swarm_config.json"):
        self.models: Dict[str, MicroModel] = {}
        self.message_bus = asyncio.Queue()
        self.routing_table = {}
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path: str) -> Dict:
        """Load swarm configuration"""
        default_config = {
            "models": {
                "intent_parser": {
                    "path": "./training_approaches/swarm_architecture/intent_parser/output",
                    "enabled": True
                },
                "context_manager": {
                    "path": "./training_approaches/swarm_architecture/context_manager/output",
                    "enabled": True
                },
                "error_recognizer": {
                    "path": "./training_approaches/swarm_architecture/error_recognizer/output",
                    "enabled": True
                }
            },
            "communication": {
                "timeout": 5.0,
                "max_retries": 3
            }
        }
        
        if Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)
        return default_config
        
    async def initialize_swarm(self):
        """Initialize all micro-models"""
        logger.info("Initializing swarm architecture...")
        
        for model_name, model_config in self.config["models"].items():
            if model_config["enabled"] and Path(model_config["path"]).exists():
                try:
                    if model_name == "intent_parser":
                        self.models[model_name] = IntentParser(model_config["path"])
                    elif model_name == "context_manager":
                        self.models[model_name] = ContextManager(model_config["path"])
                    elif model_name == "error_recognizer":
                        self.models[model_name] = ErrorRecognizer(model_config["path"])
                    
                    # Publish capabilities
                    capability = self.models[model_name].publish_capabilities()
                    self.routing_table[model_name] = capability
                    logger.info(f"Loaded {model_name} with capabilities: {capability.intents}")
                    
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    
    async def route_message(self, message: Message) -> Optional[Message]:
        """Route message to appropriate model"""
        target_model = self.models.get(message.target)
        
        if not target_model:
            logger.warning(f"Unknown target: {message.target}")
            return None
            
        # Add to trace
        message.trace.append(f"{message.source}->{message.target}")
        
        # Process based on target
        if message.target == "intent_parser" and isinstance(target_model, IntentParser):
            result = target_model.parse_intent(message.params.get("input", ""))
            return Message(
                id=f"{message.id}_response",
                source="intent_parser",
                target=message.source,
                action="intent_parsed",
                params=result,
                confidence=result.get("confidence", 0.5),
                trace=message.trace
            )
            
        elif message.target == "context_manager" and isinstance(target_model, ContextManager):
            context = target_model.update_context(message)
            return Message(
                id=f"{message.id}_response",
                source="context_manager", 
                target=message.source,
                action="context_updated",
                params={"context": context},
                trace=message.trace
            )
            
        elif message.target == "error_recognizer" and isinstance(target_model, ErrorRecognizer):
            analysis = target_model.analyze_error(message.params.get("error", ""))
            return Message(
                id=f"{message.id}_response",
                source="error_recognizer",
                target=message.source,
                action="error_analyzed",
                params=analysis,
                trace=message.trace
            )
            
        return None
        
    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input through the swarm"""
        logger.info(f"Processing: {user_input}")
        
        # Step 1: Parse intent
        intent_msg = Message(
            id="user_1",
            source="user",
            target="intent_parser",
            action="parse",
            params={"input": user_input}
        )
        
        intent_response = await self.route_message(intent_msg)
        
        if not intent_response:
            return {"error": "Intent parser failed"}
            
        intent_data = intent_response.params
        logger.info(f"Intent: {intent_data}")
        
        # Step 2: Update context
        context_msg = Message(
            id="user_2",
            source="orchestrator",
            target="context_manager",
            action=intent_data.get("intent", "unknown"),
            params=intent_data.get("params", {}),
            trace=intent_response.trace
        )
        
        context_response = await self.route_message(context_msg)
        
        # Step 3: Execute action (would route to API mappers in full implementation)
        result = {
            "intent": intent_data,
            "context": context_response.params if context_response else {},
            "trace": context_response.trace if context_response else intent_response.trace,
            "status": "completed"
        }
        
        return result
        
    async def run_interactive(self):
        """Run interactive swarm session"""
        await self.initialize_swarm()
        
        print("\n=== Swarm Intelligence Active ===")
        print(f"Loaded models: {list(self.models.keys())}")
        print("Type 'exit' to quit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                    
                result = await self.process_user_input(user_input)
                
                print(f"\nSwarm Response:")
                print(f"Intent: {result['intent']}")
                print(f"Context: {result['context']}")
                print(f"Trace: {' -> '.join(result['trace'])}")
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                
        print("\nSwarm shutting down...")

# Deployment script
class SwarmDeployment:
    """Deploy swarm to production"""
    
    @staticmethod
    def create_docker_compose():
        """Generate docker-compose.yml for swarm deployment"""
        compose = """version: '3.8'

services:
  intent_parser:
    image: swarm-intent-parser:latest
    volumes:
      - ./models/intent_parser:/model
    environment:
      - MODEL_PATH=/model
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "5001:5000"
      
  context_manager:
    image: swarm-context-manager:latest
    volumes:
      - ./models/context_manager:/model
    environment:
      - MODEL_PATH=/model
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "5002:5000"
      
  error_recognizer:
    image: swarm-error-recognizer:latest
    volumes:
      - ./models/error_recognizer:/model
    environment:
      - MODEL_PATH=/model
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "5003:5000"
      
  orchestrator:
    image: swarm-orchestrator:latest
    depends_on:
      - intent_parser
      - context_manager
      - error_recognizer
    environment:
      - INTENT_PARSER_URL=http://intent_parser:5000
      - CONTEXT_MANAGER_URL=http://context_manager:5000
      - ERROR_RECOGNIZER_URL=http://error_recognizer:5000
    ports:
      - "5000:5000"
      
  message_bus:
    image: redis:alpine
    ports:
      - "6379:6379"
"""
        
        with open("docker-compose.yml", "w") as f:
            f.write(compose)
            
        print("Created docker-compose.yml for swarm deployment")

if __name__ == "__main__":
    # For testing, run interactive mode
    orchestrator = SwarmOrchestrator()
    asyncio.run(orchestrator.run_interactive())
