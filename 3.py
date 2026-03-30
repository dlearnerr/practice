!pip uninstall peft -y
!pip uninstall transformers -y

!pip install transformers==4.40.0 peft==0.10.0 datasets accelerate


from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token   # Fix padding issue

model = AutoModelForCausalLM.from_pretrained(model_name)

# Small custom dataset
data = {
    "text": [
        "Q: What is AI?\nA: Artificial Intelligence is the simulation of human intelligence.",
        "Q: What is ML?\nA: Machine Learning is a subset of AI that learns from data."
    ]
}

dataset = Dataset.from_dict(data)

# Tokenization with labels (IMPORTANT FIX)
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )
    tokens["labels"] = tokens["input_ids"]   # Required for loss
    return tokens

dataset = dataset.map(tokenize)

# LoRA configuration
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Text generation function
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# BEFORE fine-tuning
print("Before Fine-Tuning:")
print(generate("Q: What is AI?\nA:"))

# Train model
trainer.train()

# AFTER fine-tuning
print("\nAfter Fine-Tuning:")
print(generate("Q: What is AI?\nA:"))
