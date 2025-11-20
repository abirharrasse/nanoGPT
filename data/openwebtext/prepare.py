# import os
# from tqdm import tqdm
# import numpy as np
# from datasets import load_dataset
# from transformers import PreTrainedTokenizerFast
# import tempfile
# import shutil

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

# def concat_bin_files(file_list, out_path):
#     print(f"Concatenating {len(file_list)} files into {out_path}...")
#     total_len = sum(os.path.getsize(f) // 4 for f in file_list)
#     print(f"Total tokens: {total_len:,}")
#     out_arr = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(total_len,))
#     idx = 0
#     for f in file_list:
#         arr = np.memmap(f, dtype=np.uint32, mode="r")
#         out_arr[idx:idx + arr.shape[0]] = arr[:]
#         idx += arr.shape[0]
#     out_arr.flush()
#     print(f"Concatenation complete")

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
#     train_targets = {lang: target - val_targets[lang] for lang, target in LANG_TARGETS.items()}

#     print(f"\n{'='*50}")
#     print("Target tokens per language:")
#     for lang in LANG_TARGETS:
#         print(f"  {lang}: train={train_targets[lang]:,}, val={val_targets[lang]:,}")

#     temp_dir = tempfile.mkdtemp()
#     print(f"\nTemp directory: {temp_dir}")
#     lang_train_files = []
#     lang_val_files = []

#     for lang in LANG_TARGETS.keys():
#         print(f"\n{'='*50}")
#         print(f"Selecting train/val for {lang}...")
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
        
#         token_count_train = 0
#         token_count_val = 0
        
#         train_path = os.path.join(temp_dir, f"{lang}_train.bin")
#         val_path = os.path.join(temp_dir, f"{lang}_val.bin")
#         lang_train_files.append(train_path)
#         lang_val_files.append(val_path)
        
#         print(f"Creating memory-mapped arrays...")
#         train_arr = np.memmap(train_path, dtype=np.uint32, mode="w+", shape=(train_targets[lang],))
#         val_arr = np.memmap(val_path, dtype=np.uint32, mode="w+", shape=(val_targets[lang],))
#         train_idx = 0
#         val_idx = 0
        
#         print(f"Filling arrays...")
#         for example in tqdm(dset, desc=f"Selecting {lang}"):
#             ids = example["ids"]
#             if ids is None or not ids:
#                 continue
            
#             if token_count_val < val_targets[lang]:
#                 n = min(len(ids), val_targets[lang] - token_count_val)
#                 val_arr[val_idx:val_idx + n] = ids[:n]
#                 val_idx += n
#                 token_count_val += n
#                 if n < len(ids):
#                     ids = ids[n:]
#                 else:
#                     continue
            
#             if token_count_train < train_targets[lang]:
#                 n = min(len(ids), train_targets[lang] - token_count_train)
#                 train_arr[train_idx:train_idx + n] = ids[:n]
#                 train_idx += n
#                 token_count_train += n
#                 if token_count_train >= train_targets[lang]:
#                     break
        
#         train_arr.flush()
#         val_arr.flush()
#         print(f"{lang} - train: {token_count_train:,} tokens, val: {token_count_val:,} tokens")

#     print(f"\n{'='*50}")
#     train_filename = os.path.join(os.path.dirname(__file__), "train.bin")
#     val_filename = os.path.join(os.path.dirname(__file__), "val.bin")
    
#     concat_bin_files(lang_train_files, train_filename)
#     concat_bin_files(lang_val_files, val_filename)
    
#     print(f"\nSaved {train_filename} and {val_filename}")
#     print("Cleaning up temp directory...")
#     shutil.rmtree(temp_dir)
#     print("Done!")

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import tempfile
import shutil
import functools

print = functools.partial(print, flush=True)
num_proc = 8
tokenizer = PreTrainedTokenizerFast.from_pretrained("abir-hr196/non_english_tokenizer")

TOTAL_TOKENS = 8_000_000_000
OTHER_LANGS = ["deu_Latn", "arb_Arab", "cmn_Hani", "fra_Latn"]
TOKENS_PER_OTHER = TOTAL_TOKENS // len(OTHER_LANGS)
LANG_TARGETS = {lang: TOKENS_PER_OTHER for lang in OTHER_LANGS}

def tokenize_batch(batch):
    ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
    return {"ids": [x + [tokenizer.eos_token_id] for x in ids], "len": [len(x) + 1 for x in ids]}

def concat_bin_files(file_list, out_path):
    total_len = sum(os.path.getsize(f) // 4 for f in file_list)
    out_arr = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(total_len,))
    idx = 0
    for f in file_list:
        arr = np.memmap(f, dtype=np.uint32, mode="r")
        out_arr[idx:idx + arr.shape[0]] = arr[:]
        idx += arr.shape[0]
    out_arr.flush()

if __name__ == "__main__":
    print("Loading dataset...")
    other_dataset = load_dataset("CausalNLP/gpt2small_full_training_data", num_proc=num_proc)

    all_tokenized = {}
    for lang in OTHER_LANGS:
        print(f"Tokenizing {lang}...")
        all_tokenized[lang] = other_dataset[lang].map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=["text"],
            desc=f"Tokenizing {lang}"
        )

    VAL_FRAC = 0.01
    val_targets = {lang: int(target * VAL_FRAC) for lang, target in LANG_TARGETS.items()}
    train_targets = {lang: target - val_targets[lang] for lang, target in LANG_TARGETS.items()}

    temp_dir = tempfile.mkdtemp()
    lang_train_files = []
    lang_val_files = []

    for lang in LANG_TARGETS:
        print(f"\nProcessing {lang}...")
        dset = all_tokenized[lang].shuffle(seed=42)
        
        train_path = os.path.join(temp_dir, f"{lang}_train.bin")
        val_path = os.path.join(temp_dir, f"{lang}_val.bin")
        train_arr = np.memmap(train_path, dtype=np.uint32, mode="w+", shape=(train_targets[lang],))
        val_arr = np.memmap(val_path, dtype=np.uint32, mode="w+", shape=(val_targets[lang],))
        
        # Use sharding for faster batch processing
        total_batches = 1024
        train_idx = 0
        val_idx = 0
        
        for batch_idx in tqdm(range(total_batches), desc=f"{lang}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            
            for ids in batch['ids']:
                # Fill val first
                if val_idx < val_targets[lang]:
                    n = min(len(ids), val_targets[lang] - val_idx)
                    val_arr[val_idx:val_idx+n] = ids[:n]
                    val_idx += n
                    ids = ids[n:]
                
                # Then train
                if train_idx < train_targets[lang] and len(ids) > 0:
                    n = min(len(ids), train_targets[lang] - train_idx)
                    train_arr[train_idx:train_idx+n] = ids[:n]
                    train_idx += n
                
                if train_idx >= train_targets[lang]:
                    break
            
            if train_idx >= train_targets[lang]:
                break
        
        train_arr.flush()
        val_arr.flush()
        lang_train_files.append(train_path)
        lang_val_files.append(val_path)
        print(f"{lang}: train={train_idx:,}, val={val_idx:,}")

    concat_bin_files(lang_train_files, os.path.join(os.path.dirname(__file__), "train.bin"))
    concat_bin_files(lang_val_files, os.path.join(os.path.dirname(__file__), "val.bin"))
    shutil.rmtree(temp_dir)
    print("Done!")


