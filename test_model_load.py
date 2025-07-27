"""
Test if we can load the model with GPU
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

print("\nTrying to load model...")
try:
    # Try loading with different strategies
    model_name = "Qwen/Qwen2.5-Coder-7B"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print("SUCCESS! Model loaded")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Test inference
    text = "def hello_world():"
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTest generation:\n{result}")
    
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    print("\nTrying without device_map...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        model = model.cuda()
        print("SUCCESS with manual .cuda()")
    except Exception as e2:
        print(f"Also failed: {e2}")
