
import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

num_proc = 8
tokenizer = PreTrainedTokenizerFast.from_pretrained("abir-hr196/tokenizer2")

# Total tokens target (small for validation)
TOTAL_TOKENS = 8_000_000_000  # example total tokens overall (adjust as needed)
EN_TOKENS = int(TOTAL_TOKENS * 0.7)  # 70% English
OTHER_TOKENS = TOTAL_TOKENS - EN_TOKENS
OTHER_LANGS = ["deu_Latn", "arb_Arab", "cmn_Hani", "fra_Latn"]
TOKENS_PER_OTHER = OTHER_TOKENS // len(OTHER_LANGS)
LANG_TARGETS = {"eng": EN_TOKENS}
LANG_TARGETS.update({lang: TOKENS_PER_OTHER for lang in OTHER_LANGS})

def tokenize_batch(batch):
    ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
    # Add EOS token to each example
    return {"ids": [x + [tokenizer.eos_token_id] for x in ids], "len": [len(x) + 1 for x in ids]}

if __name__ == "__main__":
    # Load datasets
    print("Loading English (openwebtext)...")
    en_dataset = load_dataset("openwebtext", num_proc=num_proc)["train"]

    print("Loading other languages (CausalNLP)...")
    other_dataset = load_dataset("CausalNLP/gpt2small_full_training_data", num_proc=num_proc)

    all_tokenized = {}

    # Tokenize English
    print("Tokenizing English...")
    en_tokenized = en_dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing eng"
    )
    all_tokenized["eng"] = en_tokenized

    # Tokenize other languages
    for lang in OTHER_LANGS:
        print(f"Tokenizing {lang}...")
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

    # Now sample validation tokens per language (small val set)
    VAL_FRAC = 0.01  # 1% for validation
    val_targets = {lang: max(1, int(target * VAL_FRAC)) for lang, target in LANG_TARGETS.items()}

    output_dir = "./val_data_per_lang"
    os.makedirs(output_dir, exist_ok=True)

    for lang, target_tokens in val_targets.items():
        print(f"Creating val set for {lang} with target {target_tokens} tokens...")
        dset = all_tokenized[lang].shuffle(seed=42)
        token_count = 0

        val_path = os.path.join(output_dir, f"{lang}_val.bin")
        val_arr = np.memmap(val_path, dtype=np.uint32, mode="w+", shape=(target_tokens,))
        idx = 0

        for example in tqdm(dset, desc=f"Selecting val tokens for {lang}"):
            ids = example["ids"]
            n = min(len(ids), target_tokens - token_count)
            val_arr[idx:idx + n] = ids[:n]
            idx += n
            token_count += n
            if token_count >= target_tokens:
                break

        val_arr.flush()
        print(f"Saved {lang} val set with {token_count} tokens to {val_path}")
