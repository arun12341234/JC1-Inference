import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import DPOTrainer  # Direct Preference Optimization for RLHF
import os

MODEL_NAME = "meta-llama/Llama-3-8B"  # Change to JC1 model path
DATASET_PATH = "data/dataset/train.jsonl"
CHECKPOINT_PATH = "checkpoints/jc1-rlhf"

# Enable DeepSpeed
ds_config = {
    "train_batch_size": 4,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 8,
    "fp16": {"enabled": True},  # Mixed precision training
    "zero_optimization": {"stage": 3}  # Zero Redundancy Optimizer (ZeRO)
}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# RLHF Training Arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_PATH,
    per_device_train_batch_size=1,
    save_steps=500,
    save_total_limit=5,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=True,
    dataloader_num_workers=4,
    report_to="none"
)

# Trainer with Direct Preference Optimization (DPO)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=DATASET_PATH,
    tokenizer=tokenizer
)

# 🚀 Train the Model
def train():
    print("🚀 Starting RLHF fine-tuning...")
    trainer.train()
    model.save_pretrained(CHECKPOINT_PATH)
    tokenizer.save_pretrained(CHECKPOINT_PATH)
    print(f"✅ Model saved at {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()
