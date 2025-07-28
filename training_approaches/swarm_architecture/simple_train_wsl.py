#!/usr/bin/env python3
"""
Simple training script without axolotl
Uses native Hugging Face transformers for training
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json

def load_training_data(file_path):
    """Load JSONL training data"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Convert to simple format
            messages = item['messages']
            text = f"{messages[0]['content']}\n{messages[1]['content']}"
            data.append({'text': text})
    return Dataset.from_list(data)

def train_simple_model(model_name="gpt2", data_path="intent_parser/training_data.jsonl", output_dir="intent_parser/simple_output"):
    """Train a model using simple Hugging Face approach"""
    print(f"Training {model_name} on {data_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare data
    dataset = load_training_data(data_path)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=True, truncation=True, max_length=256)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=False,
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")

def main():
    """Train intent parser with simple approach"""
    # Check if we have training data
    if not os.path.exists("intent_parser/training_data.jsonl"):
        print("Error: No training data found. Run generate_swarm_training_data.py first")
        return
        
    print("=== Simple Swarm Training (No Axolotl) ===")
    print("This uses basic Hugging Face training as a fallback")
    
    # Train intent parser with GPT-2 (small model)
    train_simple_model(
        model_name="gpt2",  # 124M params, similar to SmolLM
        data_path="intent_parser/training_data.jsonl",
        output_dir="intent_parser/simple_output"
    )
    
    print("\nTraining complete! Test with:")
    print("from transformers import pipeline")
    print("generator = pipeline('text-generation', model='intent_parser/simple_output')")
    print("result = generator('User: create a file\\nAssistant:', max_length=50)")

if __name__ == "__main__":
    main()
