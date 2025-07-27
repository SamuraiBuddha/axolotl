@echo off
REM Windows setup script for BCL fine-tuning

echo Setting up BCL fine-tuning...

REM 1. Install Axolotl in development mode
echo Installing Axolotl...
pip install -e .

REM 2. Install flash-attention (optional but recommended)
echo Installing flash-attention...
pip install flash-attn --no-build-isolation

echo.
echo ==========================================
echo NEXT STEPS:
echo 1. Copy your bcl_training_data.jsonl to this directory
echo 2. Run: accelerate launch -m axolotl.cli.train bcl_finetune_config.yaml
echo ==========================================