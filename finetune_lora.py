import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    TrainerCallback
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# 1. Configuration
MODEL_ID = "CausalNLP/gpt2-hf_multilingual-20"
DATASET_ID = "CausalNLP/gpt2small_full_training_data"
TARGET_LOSS = 3.09  # Stopping threshold: English/French parity level

# 2. Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load in bfloat16 for H100 optimization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# 3. LoRA Configuration
# We apply LoRA to all attention layers to allow the model to 
# "rediscover" the bridge between fragments and semantic clusters.
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    # Now targeting both Attention AND MLP for complete CLT interpretation
    target_modules=["c_attn", "c_fc", "c_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Load & Process General Arabic Data
dataset = load_dataset(DATASET_ID, split="arb_Arab")
# 5% is enough for evaluation given the H100's speed
dataset = dataset.train_test_split(test_size=0.05, seed=42)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"],
    num_proc=8
)

# 5. Custom Parity Stopping Logic
class ParityStoppingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            current_loss = metrics["eval_loss"]
            print(f"\n[Step {state.global_step}] Current Eval Loss: {current_loss:.4f}")
            if current_loss <= TARGET_LOSS:
                print(f"!!! Target Loss {TARGET_LOSS} reached. Stopping training.")
                control.should_training_stop = True

# 6. Training Arguments (H100 Optimized)
training_args = TrainingArguments(
    output_dir="./arabic_parity_lora_h100",
    per_device_train_batch_size=64, # High throughput for H100
    per_device_eval_batch_size=64,
    bf16=True,                      # Essential for H100
    tf32=True,                      # Fast matrix multiplication
    learning_rate=2e-4,
    num_train_epochs=3,             # We will likely stop much earlier
    evaluation_strategy="steps",
    eval_steps=50,                  # Check parity frequently
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none"
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    callbacks=[ParityStoppingCallback()]
)

# 8. Run Training
trainer.train()

# 9. Save final adapter for CLT Diffing
model.save_pretrained("./arabic_parity_adapter")
print("Training complete. Use this adapter with your CLT for model diffing.")