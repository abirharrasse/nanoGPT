import sys
import os
from itertools import islice
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi
import functools

print = functools.partial(print, flush=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_training_corpus(dataset_name, split_name, num_samples, text_column="text", os.getenv('HF_TOKEN')=None):
    count = 0
    print(f"Starting to stream {split_name}...")
    streamed_dataset = load_dataset(dataset_name, split=split_name, streaming=True, token=os.getenv('HF_TOKEN'))
    
    for item in islice(streamed_dataset, num_samples):
        if item and text_column in item and item[text_column]:
            yield item[text_column]
            count += 1
            if count % 100000 == 0:
                print(f"Processed {count} documents...")
    
    print(f"Total documents: {count}")

def train_tokenizer(dataset_name, split_name, num_samples, text_column, vocab_size, save_path, os.getenv('HF_TOKEN')=None):
    os.makedirs(save_path, exist_ok=True)
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<pad>", "<unk>", "<s>", "</s>"],
        min_frequency=2,
        show_progress=True,
    )
    
    print(f"Training {split_name} tokenizer with vocab_size={vocab_size}...")
    corpus = get_training_corpus(dataset_name, split_name, num_samples, text_column, os.getenv('HF_TOKEN'))
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    
    # Wrap and save properly
    os.getenv('HF_TOKEN') = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
    )
    os.getenv('HF_TOKEN').save_pretrained(save_path)
    print(f"Saved to {save_path}")

def measure_fertility(tokenizer_path, dataset_name, split, num_samples, os.getenv('HF_TOKEN')=None):
    print(f"Measuring fertility for {split}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    dataset = load_dataset(dataset_name, split=split, streaming=True, token=os.getenv('HF_TOKEN'))
    
    total_tokens = 0
    total_words = 0
    count = 0
    
    for item in islice(dataset, num_samples):
        text = item["text"]
        if not text:
            continue
        total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
        total_words += len(text.split())
        count += 1
        if count % 10000 == 0:
            print(f"  Processed {count}/{num_samples} samples...")
    
    print(f"  Total: {total_tokens:,} tokens / {total_words:,} words")
    return total_tokens / total_words

def push_to_hf(repo_id, local_path, os.getenv('HF_TOKEN')):
    print(f"Pushing to {repo_id}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(local_path)
    tokenizer.push_to_hub(repo_id, token=os.getenv('HF_TOKEN'))
    print(f"Pushed successfully")

if __name__ == "__main__":
    HF_READ_TOKEN = 'os.getenv('HF_TOKEN')'
    HF_WRITE_TOKEN = 'os.getenv('HF_TOKEN')'
    
    DATASET_NAME = "CausalNLP/gpt2small_full_training_data"
    TRAIN_SAMPLES = 200000
    TEST_SAMPLES = 100000
    VOCAB_SIZES = [8000, 16000, 32000, 64000, 128000]
    
    languages = {
        "arabic": "arb_Arab",
        "english": "eng",
        "german": "deu_Latn",
        "french": "fra_Latn",
        "chinese": "cmn_Hani"
    }
    
    print("="*60)
    print("STARTING TOKENIZER TRAINING AND EVALUATION")
    print("="*60)
    
    results = {}
    
    for lang_name, split in languages.items():
        print(f"\n{'='*60}")
        print(f"LANGUAGE: {lang_name.upper()} ({split})")
        print(f"{'='*60}")
        results[lang_name] = {}
        
        for vocab_size in VOCAB_SIZES:
            print(f"\n--- Vocab Size: {vocab_size} ---")
            save_path = f"./{lang_name}_tokenizer_{vocab_size}"
            repo_id = f"abir-hr196/{lang_name}_tokenizer_{vocab_size}"
            
            if os.path.exists(os.path.join(save_path, "tokenizer_config.json")):
                print(f"✓ Tokenizer exists at {save_path}, skipping training")
            else:
                print(f"✗ Training new tokenizer...")
                train_tokenizer(DATASET_NAME, split, TRAIN_SAMPLES, "text", vocab_size, save_path, HF_READ_TOKEN)
            
            fertility = measure_fertility(save_path, DATASET_NAME, split, TEST_SAMPLES, HF_READ_TOKEN)
            results[lang_name][vocab_size] = fertility
            
            print(f"→ {lang_name} | vocab={vocab_size} | fertility={fertility:.3f}")
            
            if HF_WRITE_TOKEN:
                push_to_hf(repo_id, save_path, HF_WRITE_TOKEN)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for lang_name in languages:
        print(f"\n{lang_name.upper()}:")
        for vocab_size in VOCAB_SIZES:
            print(f"  {vocab_size:6d}: {results[lang_name][vocab_size]:.3f}")