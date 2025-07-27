import json

# Read the original data
with open('bcl_training_data.jsonl', 'r') as f:
    lines = f.readlines()

# Transform and write the fixed data
with open('bcl_training_data_fixed.jsonl', 'w') as f:
    for line in lines:
        data = json.loads(line)
        
        # Create instruction based on type
        if data['type'] == 'legal_to_bcl':
            instruction = "Convert this legal building code requirement to BCL format"
        elif data['type'] == 'bcl_to_physics':
            instruction = "Explain the physics and safety reasoning behind this BCL rule"
        elif data['type'] == 'bcl_completion':
            instruction = "Complete this BCL rule with all required constraints"
        else:
            instruction = "Process this BCL-related task"
        
        # Create the alpaca format
        fixed_data = {
            'instruction': instruction,
            'input': data['input'],
            'output': data['output']
        }
        
        f.write(json.dumps(fixed_data) + '\n')

print("Fixed training data saved to bcl_training_data_fixed.jsonl")
