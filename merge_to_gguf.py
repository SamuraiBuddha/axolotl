"""
Merge LoRA adapter with base model and convert to GGUF
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("BCL Model Merge and GGUF Conversion Pipeline")
print("=" * 60)

# Paths
base_model_name = "Qwen/Qwen2.5-Coder-1.5B"
adapter_path = "./outputs/bcl_physics_small"  # Your training output
merged_path = "./outputs/bcl_physics_merged"

# Step 1: Load and merge
print("\n1. Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("2. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("3. Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("4. Merging adapter with base model...")
model = model.merge_and_unload()

print("5. Saving merged model...")
model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)

print(f"\n✓ Merged model saved to: {merged_path}")

# Step 2: Convert to GGUF
print("\n" + "=" * 60)
print("GGUF Conversion Instructions")
print("=" * 60)

print("""
To convert to GGUF format:

1. Clone llama.cpp if you haven't:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp

2. Install requirements:
   pip install -r requirements.txt

3. Convert to GGUF:
   python convert_hf_to_gguf.py \\
     {merged_path} \\
     --outfile ./bcl-physics-1.5b.gguf \\
     --outtype f16

4. (Optional) Quantize for smaller size:
   ./llama-quantize ./bcl-physics-1.5b.gguf ./bcl-physics-1.5b-q4_k_m.gguf q4_k_m

5. Test in LM Studio:
   - Copy the GGUF file to your LM Studio models folder
   - Load and test with BCL prompts
""".format(merged_path=os.path.abspath(merged_path)))

# Create a test script
test_script = '''"""
Test the merged BCL model before GGUF conversion
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./outputs/bcl_physics_merged"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

test_prompts = [
    {
        "instruction": "Convert this legal building code requirement to BCL format",
        "input": "Fire doors must have a minimum 90-minute fire resistance rating in high-rise buildings"
    },
    {
        "instruction": "Explain the physics and safety reasoning behind this BCL rule",
        "input": "must: sprinkler_pressure(end_of_line) >= 15.psi"
    },
    {
        "instruction": "Complete this BCL rule with all required constraints",
        "input": "rule emergency_lighting:\\n    where: occupancy.type = 'assembly'\\n    # Complete this"
    }
]

print("Testing merged BCL model...\\n")
for i, prompt in enumerate(test_prompts):
    text = f"{prompt['instruction']}\\n\\nInput: {prompt['input']}\\n\\nOutput:"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Test {i+1}:")
    print(f"Instruction: {prompt['instruction']}")
    print(f"Response: {response[:200]}...")
    print("-" * 60)
'''

with open("test_merged_model.py", "w") as f:
    f.write(test_script)

print("\n✓ Created test_merged_model.py")
print("\nRun this after training completes to merge the model!")
