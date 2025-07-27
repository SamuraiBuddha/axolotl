# Swarm Architecture Setup Guide

## Folder Structure
```
axolotl/
├── training_approaches/
│   ├── swarm_architecture/
│   │   ├── intent_parser/       # 10M params - extracts user intent
│   │   ├── context_manager/     # 5M params - maintains state
│   │   ├── api_mappers/         # 20M params each - API-specific
│   │   ├── error_recognizer/    # 5M params - pattern matching
│   │   └── orchestrator/        # 50M params - coordinates others
│   ├── monolithic_baseline/     # Traditional single model approach
│   └── hybrid_approach/         # Mix of small and large models
```

## Small Models vs Embedding Models

### Small Language Models (What we want):
- **Purpose**: Full text generation, reasoning, and understanding
- **Size**: 10M - 1B parameters (vs 7B+ for "normal" models)
- **Examples**: 
  - TinyLlama (1.1B)
  - Phi-3.5-mini (3.8B - still big for us)
  - SmolLM (135M - 1.7B) 
  - MobileLLM (125M - 350M)
  - OpenELM (270M - 3B)
- **Can**: Generate text, answer questions, reason
- **Download via**: Hugging Face model hub

### Embedding Models (Different purpose):
- **Purpose**: Convert text to vectors for similarity/search
- **Size**: Usually 100M - 500M parameters
- **Examples**: all-MiniLM-L6-v2, e5-small
- **Can**: Only create vector representations
- **Cannot**: Generate text or have conversations

## Recommended Small Models for Swarm Architecture:

1. **SmolLM-135M** (HuggingFaceTB/SmolLM-135M)
   - Excellent for intent parsing
   - Fast inference, low memory
   
2. **TinyLlama-1.1B** (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
   - Good for orchestrator role
   - Still coherent at this size

3. **Qwen2.5-0.5B** (Qwen/Qwen2.5-0.5B-Instruct)
   - Great balance of size/capability
   - Good for API mappers

4. **MobileLLM-125M** (facebook/mobilellm-125m)
   - Ultra-efficient
   - Perfect for error recognition

## Download Commands:

```bash
# Using Hugging Face CLI
pip install huggingface-hub

# Download SmolLM for intent parsing
huggingface-cli download HuggingFaceTB/SmolLM-135M --local-dir ./models/smollm-135m

# Download Qwen2.5 for API mapping  
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/qwen2.5-0.5b

# Using Python
from huggingface_hub import snapshot_download
snapshot_download("HuggingFaceTB/SmolLM-135M", local_dir="./models/smollm-135m")
```

## Axolotl Config Structure:

Each approach folder contains:
- `config.yaml` - Training configuration
- `training_data.jsonl` - Specific training data
- `output/` - Model checkpoints
- `eval_data.jsonl` - Evaluation set

To switch approaches:
1. Edit `master_swarm_config.yaml` and change `current_approach`
2. OR directly use: `accelerate launch -m axolotl.cli.train ./training_approaches/swarm_architecture/intent_parser/config.yaml`
