import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import tempfile
import shutil
import functools
from multiprocessing import Pool

print = functools.partial(print, flush=True)
num_proc = 16  # INCREASED from 8
tokenizer = PreTrainedTokenizerFast.from_pretrained("CausalNLP/gpt2-hf_multilingual-20")

# Finetuning setup: ~2.5B tokens with 80/20 mix
TOTAL_TOKENS = 2_500_000_000

# 80% Arabic, 20% split among others
ARABIC_TOKENS = int(TOTAL_TOKENS * 0.8)
OTHER_TOKENS_EACH = int(TOTAL_TOKENS * 0.05)

LANG_TARGETS = {
    "arb_Arab": ARABIC_TOKENS,
    "eng": OTHER_TOKENS_EACH,
    "deu_Latn": OTHER_TOKENS_EACH,
    "fra_Latn": OTHER_TOKENS_EACH,
    "cmn_Hani": OTHER_TOKENS_EACH,
}

def tokenize_batch(batch):
    ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
    return {"ids": [x + [tokenizer.eos_token_id] for x in ids], "len": [len(x) + 1 for x in ids]}

def concat_bin_files(file_list, out_path):
    """Optimized concatenation with preallocated array"""
    total_len = sum(os.path.getsize(f) // 4 for f in file_list)
    out_arr = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(total_len,))
    idx = 0
    for f in tqdm(file_list, desc="Concatenating"):
        arr = np.memmap(f, dtype=np.uint32, mode="r")
        chunk_size = arr.shape[0]
        out_arr[idx:idx + chunk_size] = arr[:]
        idx += chunk_size
        del arr  # explicitly free memory
    out_arr.flush()
    del out_arr

if __name__ == "__main__":
    print("Loading dataset...")
    # Load with streaming for memory efficiency
    dataset = load_dataset("CausalNLP/gpt2small_full_training_data", num_proc=num_proc)

    all_tokenized = {}
    for lang in LANG_TARGETS.keys():
        print(f"Tokenizing {lang}...")
        all_tokenized[lang] = dataset[lang].map(
            tokenize_batch,
            batched=True,
            batch_size=2000,  # INCREASED from 1000
            num_proc=num_proc,
            remove_columns=["text"],
            desc=f"Tokenizing {lang}",
            writer_batch_size=2000  # NEW: write in larger batches
        )

    VAL_FRAC = 0.01
    val_targets = {lang: int(target * VAL_FRAC) for lang, target in LANG_TARGETS.items()}
    train_targets = {lang: target - val_targets[lang] for lang, target in LANG_TARGETS.items()}

    print("\nToken allocation:")
    for lang, train_tok in train_targets.items():
        print(f"  {lang}: train={train_tok:,} ({train_tok/TOTAL_TOKENS*100:.1f}%), val={val_targets[lang]:,}")

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
        
        # OPTIMIZED: use fewer, larger batches
        total_batches = 256  # REDUCED from 1024
        train_idx = 0
        val_idx = 0
        
        # OPTIMIZED: pre-convert to numpy format once
        dset_numpy = dset.with_format('numpy')
        
        for batch_idx in tqdm(range(total_batches), desc=f"{lang}"):
            batch = dset_numpy.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
            
            # OPTIMIZED: process all ids at once using numpy operations
            all_ids = np.concatenate(batch['ids'])
            
            # Fill val first
            if val_idx < val_targets[lang]:
                n = min(len(all_ids), val_targets[lang] - val_idx)
                val_arr[val_idx:val_idx+n] = all_ids[:n]
                val_idx += n
                all_ids = all_ids[n:]
            
            # Then train
            if train_idx < train_targets[lang] and len(all_ids) > 0:
                n = min(len(all_ids), train_targets[lang] - train_idx)
                train_arr[train_idx:train_idx+n] = all_ids[:n]
                train_idx += n
            
            if train_idx >= train_targets[lang]:
                break
        
        train_arr.flush()
        val_arr.flush()
        del train_arr, val_arr  # free memory
        
        lang_train_files.append(train_path)
        lang_val_files.append(val_path)
        print(f"{lang}: train={train_idx:,}, val={val_idx:,}")

    print("\nConcatenating files...")
    concat_bin_files(lang_train_files, os.path.join(os.path.dirname(__file__), "fine_train.bin"))
    concat_bin_files(lang_val_files, os.path.join(os.path.dirname(__file__), "fine_val.bin"))
    
    print("\nCleaning up...")
    shutil.rmtree(temp_dir)
    
    print("\nDone! Files created:")
    train_size = os.path.getsize(os.path.join(os.path.dirname(__file__), "fine_train.bin"))
    val_size = os.path.getsize(os.path.join(os.path.dirname(__file__), "fine_val.bin"))
    print(f"  fine_train.bin: {train_size // 4:,} tokens")
    print(f"  fine_val.bin: {val_size // 4:,} tokens")