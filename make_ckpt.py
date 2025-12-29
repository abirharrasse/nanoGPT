import torch
from transformers import GPT2LMHeadModel
import os
# Download from HF
hf_model = GPT2LMHeadModel.from_pretrained("CausalNLP/gpt2-hf_multilingual-20")
state_dict = hf_model.state_dict()

# Reverse transformations
for key in list(state_dict.keys()):
    if any(name in key for name in [".c_attn.weight", ".c_proj.weight", ".c_fc.weight"]):
        state_dict[key] = state_dict[key].t()

# Remove HF prefix
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace('transformer.', '')
    new_state_dict[new_k] = v

# Save - NO _orig_mod. prefix needed
checkpoint = {
    'model': new_state_dict,  # clean state dict
    'optimizer': None,
    'model_args': {
        'n_layer': 12, 'n_head': 12, 'n_embd': 768,
        'block_size': 1024, 'bias': False, 'vocab_size': 119547, 'dropout': 0.0
    },
    'iter_num': 0,
    'best_val_loss': 1e9,
    'config': {}
}

import os

# Create directory first
out_dir = '/home/abir19/out_multilingual-20'
os.makedirs(out_dir, exist_ok=True)
torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))