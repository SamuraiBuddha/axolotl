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

# Better generation parameters
gen_config = {
    "max_new_tokens": 300,
    "temperature": 0.8,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,  # Prevents repetition
    "pad_token_id": tokenizer.eos_token_id
}

# Test prompts
test_prompts = [
    {
        "instruction": "Convert this legal building code requirement to BCL format",
        "input": "Emergency exits must be clearly marked with illuminated signs visible from 100 feet"
    },
    {
        "instruction": "Explain the physics and safety reasoning behind this BCL rule", 
        "input": "rule stairwell_pressurization with constraint: pressure_differential(stairwell, floor) >= 0.05.inches_water"
    },
    {
        "instruction": "Complete this BCL rule with all required constraints",
        "input": "rule elevator_shaft_ventilation:\n    where:\n        building.height > 75.feet\n    # Complete this rule"
    }
]

print("\nTesting improved BCL model...\n")
for test in test_prompts:
    # Format as training data expected
    prompt = f"{test['instruction']}\n\nInput: {test['input']}\n\nOutput:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_config
        )
    
    # Only get the generated part
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"Instruction: {test['instruction']}")
    print(f"Input: {test['input'][:50]}...")
    print(f"Generated BCL:\n{response}\n")
    print("-" * 80)

print("\nDEBUG: Testing with different temperatures...")
test_temps = [0.5, 0.7, 1.0]
prompt = "Convert this legal building code requirement to BCL format\n\nInput: Smoke detectors required in all sleeping areas\n\nOutput:"

for temp in test_temps:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=temp, do_sample=True, repetition_penalty=1.2)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nTemperature {temp}:\n{response[:200]}...")
