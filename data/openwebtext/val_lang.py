# import os
# from tqdm import tqdm
# import numpy as np
# from datasets import load_dataset
# from transformers import PreTrainedTokenizerFast

# num_proc = 8
# print("Loading tokenizer...")
# tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/abir19/my_non_english_tokenizer")
# print(f"Tokenizer loaded. EOS token ID: {tokenizer.eos_token_id}")

# TOTAL_TOKENS = 8_000_000_000
# OTHER_TOKENS = TOTAL_TOKENS
# OTHER_LANGS = ["deu_Latn", "arb_Arab", "cmn_Hani", "fra_Latn"]
# TOKENS_PER_OTHER = OTHER_TOKENS // len(OTHER_LANGS)
# LANG_TARGETS = {lang: TOKENS_PER_OTHER for lang in OTHER_LANGS}

# def tokenize_batch(batch):
#     ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
#     return {"ids": [x + [tokenizer.eos_token_id] for x in ids], "len": [len(x) + 1 for x in ids]}

# if __name__ == "__main__":
#     print("Loading dataset...")
#     other_dataset = load_dataset("CausalNLP/gpt2small_full_training_data", num_proc=num_proc)
#     print("Dataset loaded")

#     all_tokenized = {}

#     for lang in OTHER_LANGS:
#         print(f"\n{'='*50}")
#         print(f"Processing {lang}")
#         print(f"Dataset size: {len(other_dataset[lang]):,} examples")
#         print(f"Tokenizing {lang}...")
        
#         tokenized = other_dataset[lang].map(
#             tokenize_batch,
#             batched=True,
#             batch_size=1000,
#             num_proc=num_proc,
#             remove_columns=["text"],
#             desc=f"Tokenizing {lang}",
#             load_from_cache_file=True
#         )
#         all_tokenized[lang] = tokenized
#         print(f"Finished tokenizing {lang}")

#     VAL_FRAC = 0.01
#     val_targets = {lang: max(1, int(target * VAL_FRAC)) for lang, target in LANG_TARGETS.items()}

#     print(f"\n{'='*50}")
#     print("Target val tokens per language:")
#     for lang, target in val_targets.items():
#         print(f"  {lang}: {target:,}")

#     output_dir = "./val_data_per_lang"
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"\nOutput directory: {output_dir}")

#     for lang, target_tokens in val_targets.items():
#         print(f"\n{'='*50}")
#         print(f"Creating val set for {lang}...")
#         dset = all_tokenized[lang].shuffle(seed=42)
#         print("Shuffling complete")
        
#         print(f"Checking {lang} for None values...")
#         none_count = 0
#         for i, example in enumerate(dset):
#             if i >= 1000:
#                 break
#             if example["ids"] is None:
#                 none_count += 1
#                 print(f"Found None at index {i}")
#                 if none_count >= 5:
#                     break
        
#         if none_count > 0:
#             print(f"WARNING: Found {none_count}+ None values in first 1000 examples")
#             continue
#         print("No None values found")
        
#         val_path = os.path.join(output_dir, f"{lang}_val.bin")
#         print(f"Creating memory-mapped array at {val_path}...")
#         val_arr = np.memmap(val_path, dtype=np.uint32, mode="w+", shape=(target_tokens,))
#         idx = 0

#         print(f"Filling array...")
#         for example in tqdm(dset, desc=f"Selecting {lang}"):
#             ids = example["ids"]
#             if ids is None or not ids:
#                 continue
            
#             n = min(len(ids), target_tokens - idx)
#             val_arr[idx:idx + n] = ids[:n]
#             idx += n
#             if idx >= target_tokens:
#                 break

#         val_arr.flush()
#         print(f"Saved {lang} val set with {idx:,} tokens to {val_path}")
    
#     print(f"\n{'='*50}")
#     print("Done!")

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import functools

print = functools.partial(print, flush=True)
num_proc = 8

print("Loading tokenizer...")
tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/abir19/my_non_english_tokenizer")
print(f"Tokenizer loaded. EOS token ID: {tokenizer.eos_token_id}")

TOTAL_TOKENS = 8_000_000_000
OTHER_LANGS = ["deu_Latn", "arb_Arab", "cmn_Hani", "fra_Latn"]
TOKENS_PER_OTHER = TOTAL_TOKENS // len(OTHER_LANGS)
LANG_TARGETS = {lang: TOKENS_PER_OTHER for lang in OTHER_LANGS}

def tokenize_batch(batch):
    ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
    return {"ids": [x + [tokenizer.eos_token_id] for x in ids], "len": [len(x) + 1 for x in ids]}

if __name__ == "__main__":
    print("Loading dataset...")
    other_dataset = load_dataset("CausalNLP/gpt2small_full_training_data", num_proc=num_proc)
    print("Dataset loaded")

    all_tokenized = {}

    for lang in OTHER_LANGS:
        print(f"\n{'='*50}")
        print(f"Processing {lang}")
        print(f"Dataset size: {len(other_dataset[lang]):,} examples")
        print(f"Tokenizing {lang}...")
        
        tokenized = other_dataset[lang].map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=["text"],
            desc=f"Tokenizing {lang}",
            load_from_cache_file=True
        )
        all_tokenized[lang] = tokenized
        print(f"Finished tokenizing {lang}")

    VAL_FRAC = 0.01
    val_targets = {lang: int(target * VAL_FRAC) for lang, target in LANG_TARGETS.items()}

    print(f"\n{'='*50}")
    print("Target val tokens per language:")
    for lang, target in val_targets.items():
        print(f"  {lang}: {target:,}")

    output_dir = "./val_data_per_lang"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    for lang, target_tokens in val_targets.items():
        print(f"\n{'='*50}")
        print(f"Creating val set for {lang}...")
        dset = all_tokenized[lang].shuffle(seed=42)
        print("Shuffling complete")
        
        val_path = os.path.join(output_dir, f"{lang}_val.bin")
        print(f"Creating memory-mapped array at {val_path}...")
        val_arr = np.memmap(val_path, dtype=np.uint32, mode="w+", shape=(target_tokens,))
        
        print(f"Filling array with sharded batches...")
        total_batches = 512
        idx = 0
        
        for batch_idx in tqdm(range(total_batches), desc=f"{lang}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            
            for ids in batch['ids']:
                if ids is None or len(ids) == 0:
                    continue
                
                n = min(len(ids), target_tokens - idx)
                val_arr[idx:idx + n] = ids[:n]
                idx += n
                
                if idx >= target_tokens:
                    break
            
            if idx >= target_tokens:
                break

        val_arr.flush()
        print(f"Saved {lang} val set with {idx:,} tokens to {val_path}")
    
    print(f"\n{'='*50}")
    print("Done!")