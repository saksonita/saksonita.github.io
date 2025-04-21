import torch
from datasets import Dataset
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Step 2: Load your training data
with open('saksonita_training_data.json', 'r') as f:
    training_data = json.load(f)

# Convert to Hugging Face dataset format
dataset = Dataset.from_list(training_data)

# Step 3: Set up quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Step 4: Load the base model (using smallest Llama 2 for efficiency)
# model_id = "meta-llama/Llama-2-7b-chat-hf" 
model_id = "meta-llama/Llama-3.1-8B-Instruct"
# You need proper access credentials
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Step 5: Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "gate_proj", 
        "up_proj", 
        "down_proj"
    ]
)

# Apply LoRA config to the model
model = get_peft_model(model, lora_config)

# Step 6: Define training arguments
training_args = TrainingArguments(
    output_dir="./saksonita-llama2-assistant",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500,  # Adjust based on dataset size
    save_strategy="steps",
    save_steps=100,
    report_to="tensorboard",
    save_total_limit=3,
    fp16=True,
)

# Step 7: Set up the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    dataset_text_field="instruction",
    max_seq_length=1024,
)

# Step 8: Train the model
trainer.train()

# Step 9: Save the trained model
trainer.save_model("./saksonita-llama3.1-8B-Instruct-assistant-final")