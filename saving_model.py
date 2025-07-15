import torch
from model import GPT, GPTConfig  # Update this import if needed

# Load checkpoint
ckpt = torch.load("/home/aharrasse/out_50/ckpt.pt", map_location="cpu")
model_args = ckpt["model_args"]

# Build and load model
model = GPT(GPTConfig(**model_args))
model.load_state_dict(ckpt["model"])
model.eval()

# Save the model in Hugging Face format
model.save_pretrained("gpt2-multilingual-50")

# Link your tokenizer from the Hub
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained("abir-hr196/tokenizer2")
tokenizer.save_pretrained("gpt2-multilingual-50")
