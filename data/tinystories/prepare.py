# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset  # huggingface datasets
from transformers import PreTrainedTokenizerFast
import random
import tempfile
import shutil

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

# Initialize the tokenizer
# Make sure you have the correct tokenizer files locally or access to the model hub
# If you get an error, try running: pip install transformers

tokenizer = PreTrainedTokenizerFast.from_pretrained("abir-hr196/tokenizer2")

# Token distribution
TOTAL_TOKENS = 533_978_966
EN_TOKENS = int(TOTAL_TOKENS * 0.2)  # 6.3B
OTHER_TOKENS = TOTAL_TOKENS - EN_TOKENS  # 700M
OTHER_LANGS = ["deu", "arb", "cmn", "fra"]
TOKENS_PER_OTHER = OTHER_TOKENS // len(OTHER_LANGS)  # 175M each
LANG_TARGETS = {"eng": EN_TOKENS}
LANG_TARGETS.update({lang: TOKENS_PER_OTHER for lang in OTHER_LANGS})


if __name__ == "__main__":
    # Load English from Tinystories, others from CausalNLP
    print("Loading English (Tinystories)...")
    en_dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc_load_dataset, trust_remote_code=True)["train"]
    print("Loading other languages (CausalNLP)...")
    other_dataset = load_dataset("CausalNLP/multilingual_tinystories_data_2", num_proc=num_proc_load_dataset, trust_remote_code=True)

    # Helper: batched tokenization
    def tokenize_batch(batch):
        ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
        # Add EOS token to each
        return {"ids": [x + [tokenizer.eos_token_id] for x in ids], "len": [len(x) + 1 for x in ids]}

    # Prepare a dict for all splits
    all_tokenized = {}

    # English
    print("Tokenizing English (batched)...")
    en_tokenized = en_dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing eng"
    )
    all_tokenized["eng"] = en_tokenized

    # Other languages
    for lang in OTHER_LANGS:
        print(f"Tokenizing {lang} (batched)...")
        dset = other_dataset[lang]
        tokenized = dset.map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=["text"],
            desc=f"Tokenizing {lang}"
        )
        all_tokenized[lang] = tokenized

    # Now, for each split, sample enough examples to reach the target token count
    VAL_FRAC = 0.01  # 1% for validation (increased from 0.05%)
    val_targets = {lang: max(1, int(target * VAL_FRAC)) for lang, target in LANG_TARGETS.items()}
    train_targets = {lang: target - val_targets[lang] for lang, target in LANG_TARGETS.items()}

    temp_dir = tempfile.mkdtemp()
    lang_train_files = []
    lang_val_files = []

    for lang in LANG_TARGETS.keys():
        print(f"Selecting train/val for {lang}... case 20% ----------")
        dset = all_tokenized[lang].shuffle(seed=42)
        token_count_train = 0
        token_count_val = 0
        train_path = os.path.join(temp_dir, f"{lang}_train_tiny_20.bin")
        val_path = os.path.join(temp_dir, f"{lang}_val_tiny_20.bin")
        lang_train_files.append(train_path)
        lang_val_files.append(val_path)
        train_arr = np.memmap(train_path, dtype=np.uint32, mode="w+", shape=(train_targets[lang],))
        val_arr = np.memmap(val_path, dtype=np.uint32, mode="w+", shape=(val_targets[lang],))
        train_idx = 0
        val_idx = 0
        for example in tqdm(dset, desc=f"Selecting {lang}"):
            ids = example["ids"]
            # Fill val set first
            if token_count_val < val_targets[lang]:
                n = min(len(ids), val_targets[lang] - token_count_val)
                val_arr[val_idx:val_idx+n] = ids[:n]
                val_idx += n
                token_count_val += n
                if n < len(ids):
                    ids = ids[n:]
                else:
                    continue
            # Then fill train set
            if token_count_train < train_targets[lang]:
                n = min(len(ids), train_targets[lang] - token_count_train)
                train_arr[train_idx:train_idx+n] = ids[:n]
                train_idx += n
                token_count_train += n
                if token_count_train >= train_targets[lang]:
                    break
        train_arr.flush()
        val_arr.flush()
        print(f"{lang} train: {token_count_train:,} tokens, val: {token_count_val:,} tokens")

    # Concatenate all train and val files into final output
    def concat_bin_files(file_list, out_path):
        total_len = sum(os.path.getsize(f)//4 for f in file_list)  # uint32 = 4 bytes
        out_arr = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(total_len,))
        idx = 0
        for f in file_list:
            arr = np.memmap(f, dtype=np.uint32, mode="r")
            n = arr.shape[0]
            out_arr[idx:idx+n] = arr[:]
            idx += n
        out_arr.flush()

    filename = os.path.join(os.path.dirname(__file__), "train_tiny_20.bin")
    val_filename = os.path.join(os.path.dirname(__file__), "val_tiny_20.bin")
    concat_bin_files(lang_train_files, filename)
    concat_bin_files(lang_val_files, val_filename)
    print(f"Saved {filename} and {val_filename}")

    # Clean up temp files
    shutil.rmtree(temp_dir)

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')






# if __name__ == "__main__":
#     # --- Helper: Batched tokenization ---
#     def tokenize_batch(batch):
#         ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
#         return {
#             "ids": [x + [tokenizer.eos_token_id] for x in ids],
#             "len": [len(x) + 1 for x in ids],
#         }

#     # --- Helper: Concatenate binary files ---
#     def concat_bin_files(file_list, out_path):
#         total_len = sum(os.path.getsize(f) // 4 for f in file_list)  # uint32 = 4 bytes
#         out_arr = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(total_len,))
#         idx = 0
#         for f in file_list:
#             arr = np.memmap(f, dtype=np.uint32, mode="r")
#             n = arr.shape[0]
#             out_arr[idx:idx+n] = arr[:]
#             idx += n
#         out_arr.flush()

#     # --- Load datasets ---
#     print("📥 Loading English dataset (TinyStories)...")
#     en_dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc_load_dataset, trust_remote_code=True)["train"]

#     print("📥 Loading other languages (CausalNLP)...")
#     other_dataset = load_dataset("CausalNLP/multilingual_tinystories_data_2", num_proc=num_proc_load_dataset, trust_remote_code=True)
#     OTHER_LANGS = ["deu", "arb", "cmn", "fra"]

#     all_tokenized = {}

#     # --- Tokenize English and compute total token count ---
#     print("🔡 Tokenizing English...")
#     en_tokenized = en_dataset.map(
#         tokenize_batch,
#         batched=True,
#         batch_size=1000,
#         num_proc=num_proc,
#         remove_columns=["text"],
#         desc="Tokenizing eng",
#     )
#     all_tokenized["eng"] = en_tokenized
#     en_total_tokens = sum(en_tokenized["len"])
#     print(f"✅ English token count: {en_total_tokens:,}")

#     # --- Compute total token budget and distribute ---
#     TOTAL_TOKENS = int(en_total_tokens / 0.9)
#     TOKENS_PER_OTHER = (TOTAL_TOKENS - en_total_tokens) // len(OTHER_LANGS)
#     LANG_TARGETS = {"eng": en_total_tokens}
#     LANG_TARGETS.update({lang: TOKENS_PER_OTHER for lang in OTHER_LANGS})

#     print("\n📊 Token Budget Breakdown:")
#     print(f"Total tokens (90% English): {TOTAL_TOKENS:,}")
#     for lang in ["eng"] + OTHER_LANGS:
#         print(f"  {lang}: {LANG_TARGETS[lang]:,} tokens")

#     # --- Tokenize other languages ---
#     for lang in OTHER_LANGS:
#         print(f"🔡 Tokenizing {lang}...")
#         dset = other_dataset[lang]
#         tokenized = dset.map(
#             tokenize_batch,
#             batched=True,
#             batch_size=1000,
#             num_proc=num_proc,
#             remove_columns=["text"],
#             desc=f"Tokenizing {lang}",
#         )
#         all_tokenized[lang] = tokenized

#     # --- Train/Val Split ---
#     VAL_FRAC = 0.01
#     val_targets = {lang: max(1, int(t * VAL_FRAC)) for lang, t in LANG_TARGETS.items()}
#     train_targets = {lang: LANG_TARGETS[lang] - val_targets[lang] for lang in LANG_TARGETS}

#     temp_dir = tempfile.mkdtemp()
#     lang_train_files, lang_val_files = [], []

#     for lang in LANG_TARGETS:
#         print(f"\n✂️ Splitting train/val for {lang}...")
#         dset = all_tokenized[lang].shuffle(seed=42)
#         token_count_train = 0
#         token_count_val = 0

#         train_path = os.path.join(temp_dir, f"{lang}_train_tiny_90.bin")
#         val_path = os.path.join(temp_dir, f"{lang}_val_tiny_90.bin")
#         train_arr = np.memmap(train_path, dtype=np.uint32, mode="w+", shape=(train_targets[lang],))
#         val_arr = np.memmap(val_path, dtype=np.uint32, mode="w+", shape=(val_targets[lang],))
#         train_idx, val_idx = 0, 0

#         for example in tqdm(dset, desc=f"Selecting {lang}"):
#             ids = example["ids"]

#             # Fill validation set
#             if token_count_val < val_targets[lang]:
#                 n = min(len(ids), val_targets[lang] - token_count_val)
#                 val_arr[val_idx:val_idx + n] = ids[:n]
#                 val_idx += n
#                 token_count_val += n
#                 ids = ids[n:] if n < len(ids) else []

#             # Fill training set
#             if token_count_train < train_targets[lang] and ids:
#                 n = min(len(ids), train_targets[lang] - token_count_train)
#                 train_arr[train_idx:train_idx + n] = ids[:n]
#                 train_idx += n
#                 token_count_train += n
#                 if token_count_train >= train_targets[lang]:
#                     break

#         train_arr.flush()
#         val_arr.flush()
#         lang_train_files.append(train_path)
#         lang_val_files.append(val_path)

#         print(f"✅ {lang} — train: {token_count_train:,}, val: {token_count_val:,}")

#     # --- Final concat ---
#     out_train_path = os.path.join(os.path.dirname(__file__), "train_tiny_90.bin")
#     out_val_path = os.path.join(os.path.dirname(__file__), "val_tiny_90.bin")
#     concat_bin_files(lang_train_files, out_train_path)
#     concat_bin_files(lang_val_files, out_val_path)
#     print(f"\n💾 Saved final bin files:")
#     print(f"  Train: {out_train_path}")
#     print(f"  Val:   {out_val_path}")

#     # --- Cleanup ---
#     shutil.rmtree(temp_dir)
#     print("🧹 Temporary files cleaned up.")
