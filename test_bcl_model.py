from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model and tokenizer
base_model_name = "Qwen/Qwen2.5-Coder-7B"
adapter_path = "./outputs/bcl_finetune"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

# Test prompts
test_prompts = [
    "Convert this legal building code requirement to BCL format: Buildings over 6 stories must have fire-rated stairwells",
    "Explain the physics and safety reasoning behind this BCL rule: emergency_lighting.duration >= 90.minutes",
    "Complete this BCL rule with all required constraints:\nrule fire_door_rating:\n    where:\n        door.type = 'fire_rated'"
]

print("\nTesting BCL model...\n")
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.95
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt[:50]}...")
    print(f"Response: {response}\n")
    print("-" * 80)
