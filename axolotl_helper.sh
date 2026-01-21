#!/bin/bash

# Axolotl Helper Script - Easy commands for common tasks

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Check if virtual environment exists
check_venv() {
    if [ ! -d "venv_axolotl" ]; then
        echo -e "${RED}Virtual environment not found!${NC}"
        echo "Please run: ./setup_axolotl.sh first"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    check_venv
    source venv_axolotl/bin/activate
}

# Display menu
show_menu() {
    clear
    echo -e "${BLUE}======================================"
    echo "        Axolotl Helper Menu"
    echo "======================================"
    echo -e "${NC}"
    echo "1)  Quick Train (1B LoRA model - good for testing)"
    echo "2)  Train with Custom Config"
    echo "3)  Preprocess Dataset Only"
    echo "4)  Run Inference"
    echo "5)  Merge LoRA Weights"
    echo "6)  Launch Training Monitor (TensorBoard)"
    echo "7)  Run Tests"
    echo "8)  Update Axolotl"
    echo "9)  Show GPU Status"
    echo "10) Create New Config from Template"
    echo "11) Validate Config File"
    echo "12) Clean Output Directories"
    echo "0)  Exit"
    echo ""
}

# Quick train function
quick_train() {
    activate_venv
    echo -e "${GREEN}Starting quick training with 1B LoRA model...${NC}"
    echo "This will train a small model for testing purposes."
    
    if [ ! -f "examples/llama-3/lora-1b.yml" ]; then
        echo "Fetching examples first..."
        axolotl fetch examples
    fi
    
    axolotl train examples/llama-3/lora-1b.yml
}

# Train with custom config
custom_train() {
    activate_venv
    echo -e "${YELLOW}Enter path to your config file:${NC}"
    read -r config_path
    
    if [ ! -f "$config_path" ]; then
        echo -e "${RED}Config file not found: $config_path${NC}"
        return
    fi
    
    echo -e "${YELLOW}Use multi-GPU with accelerate? (y/n):${NC}"
    read -r use_accelerate
    
    if [[ "$use_accelerate" = "y" ]]; then
        accelerate launch -m axolotl.cli.train "$config_path"
    else
        axolotl train "$config_path"
    fi
}

# Preprocess dataset
preprocess_data() {
    activate_venv
    echo -e "${YELLOW}Enter path to your config file:${NC}"
    read -r config_path
    
    if [ ! -f "$config_path" ]; then
        echo -e "${RED}Config file not found: $config_path${NC}"
        return
    fi
    
    axolotl preprocess "$config_path"
    echo -e "${GREEN}Dataset preprocessed successfully!${NC}"
}

# Run inference
run_inference() {
    activate_venv
    echo -e "${YELLOW}Enter path to your config file:${NC}"
    read -r config_path
    
    if [ ! -f "$config_path" ]; then
        echo -e "${RED}Config file not found: $config_path${NC}"
        return
    fi
    
    echo -e "${YELLOW}Enter path to LoRA model directory (or press Enter for base model):${NC}"
    read -r lora_dir
    
    if [ -n "$lora_dir" ]; then
        axolotl inference "$config_path" --lora_model_dir="$lora_dir"
    else
        axolotl inference "$config_path"
    fi
}

# Merge LoRA weights
merge_lora() {
    activate_venv
    echo -e "${YELLOW}Enter path to your config file:${NC}"
    read -r config_path
    
    if [ ! -f "$config_path" ]; then
        echo -e "${RED}Config file not found: $config_path${NC}"
        return
    fi
    
    echo -e "${YELLOW}Enter path to LoRA model directory:${NC}"
    read -r lora_dir
    
    if [ ! -d "$lora_dir" ]; then
        echo -e "${RED}LoRA directory not found: $lora_dir${NC}"
        return
    fi
    
    axolotl merge-lora "$config_path" --lora_model_dir="$lora_dir"
    echo -e "${GREEN}LoRA weights merged successfully!${NC}"
}

# Launch TensorBoard
launch_tensorboard() {
    activate_venv
    echo -e "${YELLOW}Enter path to output directory (default: ./outputs):${NC}"
    read -r output_dir
    output_dir=${output_dir:-./outputs}
    
    if [ ! -d "$output_dir" ]; then
        echo -e "${RED}Output directory not found: $output_dir${NC}"
        return
    fi
    
    echo -e "${GREEN}Launching TensorBoard on http://localhost:6006${NC}"
    tensorboard --logdir="$output_dir"
}

# Run tests
run_tests() {
    activate_venv
    echo -e "${YELLOW}Select test type:${NC}"
    echo "1) All tests"
    echo "2) Quick tests (no GPU required)"
    echo "3) E2E tests (GPU required)"
    echo "4) Specific test file"
    read -r test_choice
    
    case $test_choice in
        1)
            pytest tests/
            ;;
        2)
            pytest tests/ -k "not e2e"
            ;;
        3)
            pytest tests/e2e/
            ;;
        4)
            echo -e "${YELLOW}Enter test file path:${NC}"
            read -r test_file
            pytest "$test_file"
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
}

# Update Axolotl
update_axolotl() {
    activate_venv
    echo -e "${YELLOW}Updating Axolotl...${NC}"
    git pull
    pip install -U -e ".[flash-attn,deepspeed]" --no-build-isolation
    echo -e "${GREEN}Axolotl updated successfully!${NC}"
}

# Show GPU status
show_gpu_status() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
    else
        echo -e "${RED}No NVIDIA GPU detected${NC}"
    fi
}

# Create new config from template
create_config() {
    echo -e "${YELLOW}Select model type:${NC}"
    echo "1) LLaMA-3 LoRA (Recommended for beginners)"
    echo "2) Mistral LoRA"
    echo "3) Full Fine-tuning"
    echo "4) QLoRA (4-bit)"
    read -r model_choice
    
    echo -e "${YELLOW}Enter name for your config (e.g., my_training.yml):${NC}"
    read -r config_name
    
    cat > "$config_name" << 'EOF'
# Axolotl Training Configuration
# Generated by axolotl_helper.sh

# Model Configuration
base_model: NousResearch/Llama-3.2-1B  # Change to your preferred model

# LoRA Configuration (remove for full fine-tuning)
load_in_8bit: true
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true

# Dataset Configuration
datasets:
  - path: teknium/GPT4-LLM-Cleaned  # Change to your dataset
    type: alpaca  # Format: alpaca, sharegpt, completion, etc.
    
dataset_prepared_path: ./prepared_data
val_set_size: 0.1

# Training Configuration
output_dir: ./outputs/my_model

# Hyperparameters
micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 0.0003
warmup_steps: 100

# Optimization
optimizer: adamw_torch
lr_scheduler: cosine
weight_decay: 0.01

# Memory Optimization
gradient_checkpointing: true
flash_attention: true

# Logging
logging_steps: 10
eval_steps: 50
save_steps: 200
save_total_limit: 3

wandb_project: my_axolotl_project  # Optional: remove if not using W&B
wandb_entity:  # Your W&B username
wandb_name: my_training_run
EOF
    
    echo -e "${GREEN}Config created: $config_name${NC}"
    echo "Edit this file to customize your training settings."
}

# Validate config
validate_config() {
    activate_venv
    echo -e "${YELLOW}Enter path to config file:${NC}"
    read -r config_path
    
    if [ ! -f "$config_path" ]; then
        echo -e "${RED}Config file not found: $config_path${NC}"
        return
    fi
    
    python -c "
import yaml
import sys
try:
    with open('$config_path', 'r') as f:
        config = yaml.safe_load(f)
    print('\033[0;32m✓ Config file is valid YAML\033[0m')
    print('Key settings:')
    print(f'  Base Model: {config.get(\"base_model\", \"Not specified\")}')
    print(f'  Output Dir: {config.get(\"output_dir\", \"Not specified\")}')
    print(f'  Adapter: {config.get(\"adapter\", \"None (Full fine-tuning)\")}')
    print(f'  Epochs: {config.get(\"num_epochs\", \"Not specified\")}')
    print(f'  Learning Rate: {config.get(\"learning_rate\", \"Not specified\")}')
except Exception as e:
    print(f'\033[0;31m✗ Config validation failed: {e}\033[0m')
    sys.exit(1)
"
}

# Clean output directories
clean_outputs() {
    echo -e "${YELLOW}This will remove training outputs. Are you sure? (y/n):${NC}"
    read -r confirm
    
    if [[ "$confirm" = "y" ]]; then
        rm -rf outputs/
        rm -rf prepared_data/
        rm -rf wandb/
        echo -e "${GREEN}Output directories cleaned${NC}"
    else
        echo "Cancelled"
    fi
}

# Main loop
main() {
    while true; do
        show_menu
        echo -e "${YELLOW}Enter your choice:${NC}"
        read -r choice
        
        case $choice in
            1)
                quick_train
                ;;
            2)
                custom_train
                ;;
            3)
                preprocess_data
                ;;
            4)
                run_inference
                ;;
            5)
                merge_lora
                ;;
            6)
                launch_tensorboard
                ;;
            7)
                run_tests
                ;;
            8)
                update_axolotl
                ;;
            9)
                show_gpu_status
                ;;
            10)
                create_config
                ;;
            11)
                validate_config
                ;;
            12)
                clean_outputs
                ;;
            0)
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                ;;
        esac
        
        echo ""
        echo "Press Enter to continue..."
        read -r
    done
}

# Check if running with arguments or interactive
if [ $# -eq 0 ]; then
    main
else
    # Allow direct command execution
    case "$1" in
        train)
            shift
            activate_venv
            axolotl train "$@"
            ;;
        preprocess)
            shift
            activate_venv
            axolotl preprocess "$@"
            ;;
        inference)
            shift
            activate_venv
            axolotl inference "$@"
            ;;
        *)
            echo "Usage: $0 [train|preprocess|inference] [args...]"
            echo "Or run without arguments for interactive menu"
            ;;
    esac
fi