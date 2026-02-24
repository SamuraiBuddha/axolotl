#!/bin/bash
# BCL Post-Training Pipeline: merge LoRA → GGUF → quantize → ollama
# Usage: ./bcl_post_train.sh [version_tag]
# Example: ./bcl_post_train.sh v3

set -e

VERSION=${1:-v3}
AXOLOTL_DIR="$HOME/Documents/GitHub/axolotl"
LLAMA_CPP="$HOME/Documents/GitHub/llama.cpp"
VENV="$AXOLOTL_DIR/venv_axolotl/bin/python"
OUTPUT_DIR="$AXOLOTL_DIR/outputs/bcl_qwen_qlora"
CONFIG="$AXOLOTL_DIR/bcl_linux_a4000.yaml"

echo "=== BCL Post-Training Pipeline ($VERSION) ==="
echo ""

# Step 1: Merge LoRA
echo "[1/4] Merging LoRA adapter into base model..."
cd "$AXOLOTL_DIR"
WANDB_MODE=disabled $VENV -m axolotl.cli.merge_lora "$CONFIG" --lora_model_dir="$OUTPUT_DIR"
echo "  ✓ Merged to $OUTPUT_DIR/merged/"

# Step 2: Convert to F16 GGUF
echo "[2/4] Converting to GGUF F16..."
$VENV "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$OUTPUT_DIR/merged" \
    --outfile "$OUTPUT_DIR/bcl-qwen2.5-coder-7b-${VERSION}-f16.gguf" \
    --outtype f16
echo "  ✓ F16 GGUF created"

# Step 3: Quantize to Q5_K_M
echo "[3/4] Quantizing to Q5_K_M..."
"$LLAMA_CPP/build/bin/llama-quantize" \
    "$OUTPUT_DIR/bcl-qwen2.5-coder-7b-${VERSION}-f16.gguf" \
    "$OUTPUT_DIR/bcl-qwen2.5-coder-7b-${VERSION}-Q5_K_M.gguf" \
    Q5_K_M
echo "  ✓ Q5_K_M quantized"

# Step 4: Update ollama
echo "[4/4] Updating ollama model..."
cat > "$OUTPUT_DIR/Modelfile" <<'MODELFILE'
FROM ./GGUF_PLACEHOLDER

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"

SYSTEM """You are BCL-Coder, an expert in Building Code Language (BCL) — a domain-specific language that extends Python with physics operators for automated building code compliance checking. You translate building regulations into executable, verifiable BCL rules.

BCL rules follow this structure:
- `rule <name>:` defines a rule
- `where:` specifies conditions when the rule applies
- `must:` defines mandatory constraints
- `should:` defines recommended constraints
- `safety_factor:` applies a safety margin
- `reference:` cites the code section

You know NEC, IBC, IRC, IMC, IPC, IECC, NFPA, ASHRAE, ASCE, ACI, AISC, ASME and international building codes. You produce precise, physics-grounded BCL code."""

TEMPLATE """{{ if .System }}{{ .System }}{{ end }}

### Instruction:
{{ .Prompt }}

### Response:
{{ .Response }}"""
MODELFILE

# Replace placeholder with actual GGUF filename
sed -i "s|GGUF_PLACEHOLDER|bcl-qwen2.5-coder-7b-${VERSION}-Q5_K_M.gguf|" "$OUTPUT_DIR/Modelfile"

cd "$OUTPUT_DIR"
ollama create bcl-coder -f Modelfile
echo "  ✓ ollama model updated"

# Summary
echo ""
echo "=== Done! ==="
F16_SIZE=$(du -h "$OUTPUT_DIR/bcl-qwen2.5-coder-7b-${VERSION}-f16.gguf" | cut -f1)
Q5_SIZE=$(du -h "$OUTPUT_DIR/bcl-qwen2.5-coder-7b-${VERSION}-Q5_K_M.gguf" | cut -f1)
echo "  F16:    $OUTPUT_DIR/bcl-qwen2.5-coder-7b-${VERSION}-f16.gguf ($F16_SIZE)"
echo "  Q5_K_M: $OUTPUT_DIR/bcl-qwen2.5-coder-7b-${VERSION}-Q5_K_M.gguf ($Q5_SIZE)"
echo "  ollama: bcl-coder (updated)"
echo ""
echo "Test with: ollama run bcl-coder \"Write a BCL rule for NFPA 13 sprinkler spacing in light hazard\""
