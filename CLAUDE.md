# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Axolotl is a tool for streamlining post-training for various AI models. It supports full fine-tuning, parameter-efficient tuning (LoRA/QLoRA), supervised fine-tuning (SFT), instruction tuning, and alignment techniques for models like LLaMA, Mistral, Mixtral, and more.

## Common Development Tasks

### Installation for Development

```bash
# Install with development dependencies
pip3 install -U packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation -e ".[flash-attn,deepspeed]"

# Install test dependencies
pip3 install -r requirements-tests.txt
pip3 install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/e2e/test_llama.py

# Run with coverage
pytest --cov=axolotl tests/

# Run E2E tests (requires GPU)
pytest tests/e2e/ -v

# Run single GPU CI tests
python cicd/single_gpu.py

# Run multi-GPU CI tests
./cicd/multigpu.sh
```

### Linting and Code Quality

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Individual linters
black src/ tests/
isort src/ tests/
flake8 src/ tests/
pylint src/
mypy src/
bandit --ini .bandit -r src/

# Auto-fix formatting issues
black src/ tests/
isort src/ tests/
```

### Training a Model

```bash
# Fetch example configs
axolotl fetch examples

# Train with a config
axolotl train examples/llama-3/lora-1b.yml

# Train with accelerate for multi-GPU
accelerate launch -m axolotl.cli.train examples/llama-3/lora-8b.yml

# Preprocess datasets only
axolotl preprocess examples/llama-3/lora-1b.yml

# Run inference
axolotl inference examples/llama-3/lora-1b.yml --lora_model_dir="./outputs/lora-out"

# Merge LoRA weights
axolotl merge-lora examples/llama-3/lora-1b.yml --lora_model_dir="./outputs/lora-out"
```

## Architecture and Key Components

### Core Structure

- **`src/axolotl/`**: Main package directory
  - **`cli/`**: Command-line interface entry points (train, preprocess, inference, etc.)
  - **`core/`**: Core functionality
    - `builders/`: Model and trainer builders (causal, RL)
    - `trainers/`: Training implementations (DPO, GRPO, etc.)
    - `datasets/`: Dataset handling and transformations
  - **`prompt_strategies/`**: Different prompting strategies (Alpaca, ChatML, etc.)
  - **`utils/`**: Utility functions
    - `config/`: Configuration handling and validation
    - `schemas/`: Pydantic schemas for configuration validation
    - `collators/`: Data collation for batching
  - **`integrations/`**: External library integrations (Liger, Cut Cross Entropy, etc.)
  - **`monkeypatch/`**: Runtime patches for optimization

### Configuration System

Axolotl uses YAML configuration files with Pydantic validation. Key configuration areas:
- Model selection and loading (`base_model`, `model_type`)
- Training parameters (`learning_rate`, `num_epochs`, `micro_batch_size`)
- Dataset configuration (`datasets`, `dataset_prepared_path`)
- Optimization techniques (`adapter`, `load_in_4bit/8bit`, `flash_attention`)
- Hardware settings (`gradient_accumulation_steps`, `deepspeed`, `fsdp`)

### Dataset Processing Pipeline

1. **Loading**: Supports various formats (Alpaca, ShareGPT, completion, etc.)
2. **Prompt Strategy**: Applies formatting based on model type
3. **Tokenization**: Converts to model-specific tokens
4. **Packing**: Optionally packs multiple examples for efficiency
5. **Collation**: Batches examples for training

### Training Flow

1. **Setup**: Load config → Initialize model/tokenizer → Prepare datasets
2. **Training**: Use HuggingFace Trainer with custom callbacks and optimizations
3. **Saving**: Store model weights, LoRA adapters, or merged models
4. **Evaluation**: Run validation during/after training

## Key Files and Directories

- `examples/`: Configuration examples for different models and techniques
- `deepspeed_configs/`: DeepSpeed ZeRO optimization configs
- `tests/`: Comprehensive test suite (unit, integration, E2E)
- `cicd/`: CI/CD scripts and Docker configurations
- `docs/`: Quarto-based documentation source

## Development Tips

1. Always validate YAML configs before training - the schema validation will catch most issues
2. Use `dataset_prepared_path` to cache preprocessed datasets between runs
3. Start with small models (1B-3B) for testing configurations
4. Monitor GPU memory usage - adjust `micro_batch_size` and `gradient_accumulation_steps` accordingly
5. For multi-GPU training, use FSDP for full fine-tuning or DeepSpeed for LoRA/QLoRA
6. Enable `flash_attention` for significant memory and speed improvements on compatible GPUs