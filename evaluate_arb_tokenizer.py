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

def get_training_corpus(dataset_name, splits, num_samples_per_split, text_column="text", os.getenv('HF_TOKEN')=None):
    """Get training corpus from multiple splits"""
    for split in splits:
        count = 0
        print(f"Streaming {split}...")
        streamed_dataset = load_dataset(dataset_name, split=split, streaming=True, token=os.getenv('HF_TOKEN'))
        
        for item in islice(streamed_dataset, num_samples_per_split):
            if item and text_column in item and item[text_column]:
                yield item[text_column]
                count += 1
                if count % 50000 == 0:
                    print(f"  {split}: {count} documents...")

def train_multilingual_tokenizer(dataset_name, splits, num_samples_per_split, vocab_size, save_path, os.getenv('HF_TOKEN')=None):
    os.makedirs(save_path, exist_ok=True)
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<pad>", "<unk>", "<s>", "</s>"],
        min_frequency=2,
        show_progress=True,
    )
    
    print(f"Training multilingual tokenizer (vocab_size={vocab_size})...")
    corpus = get_training_corpus(dataset_name, splits, num_samples_per_split, "text", os.getenv('HF_TOKEN'))
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    
    os.getenv('HF_TOKEN') = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
    )
    os.getenv('HF_TOKEN').save_pretrained(save_path)
    print(f"Saved to {save_path}")

def compare_tokenizers(mono_path, dataset_name, split, num_samples, os.getenv('HF_TOKEN')=None):
    """Compare monolingual vs multilingual tokenizer on same data"""
    print(f"Comparing tokenizers on {split}...")
    
    mono_tok = PreTrainedTokenizerFast.from_pretrained(mono_path)
    multi_tok = PreTrainedTokenizerFast.from_pretrained("CausalNLP/gpt2-os.getenv('HF_TOKEN')-90")
    dataset = load_dataset(dataset_name, split=split, streaming=True, token=os.getenv('HF_TOKEN'))
    
    mono_tokens = 0
    multi_tokens = 0
    total_chars = 0
    count = 0
    
    for item in islice(dataset, num_samples):
        text = item["text"]
        if not text:
            continue
        
        mono_tokens += len(mono_tok.encode(text, add_special_tokens=False))
        multi_tokens += len(multi_tok.encode(text, add_special_tokens=False))
        total_chars += len(text.replace(" ", ""))
        count += 1
        
        if count % 5000 == 0:
            print(f"  Processed {count}/{num_samples}...")
    
    print(f"  Monolingual: {mono_tokens:,} tokens ({mono_tokens/total_chars:.3f} tok/char)")
    print(f"  Multilingual: {multi_tokens:,} tokens ({multi_tokens/total_chars:.3f} tok/char)")
    print(f"  Fragmentation ratio: {multi_tokens/mono_tokens:.2f}x")
    
    return {
        "mono_tokens": mono_tokens,
        "multi_tokens": multi_tokens,
        "fragmentation_ratio": multi_tokens / mono_tokens,
        "mono_char_fertility": mono_tokens / total_chars,
        "multi_char_fertility": multi_tokens / total_chars
    }

if __name__ == "__main__":
    HF_READ_TOKEN = 'os.getenv('HF_TOKEN')'
    HF_WRITE_TOKEN = 'os.getenv('HF_TOKEN')'
    
    DATASET_NAME = "CausalNLP/gpt2small_full_training_data"
    TRAIN_SAMPLES_PER_LANG = 40000  # 40k per language = 200k total
    TEST_SAMPLES = 20000
    VOCAB_SIZE = 32000
    
    languages = {
        "arabic": "arb_Arab",
        "english": "eng",
        "german": "deu_Latn",
        "french": "fra_Latn",
        "chinese": "cmn_Hani"
    }
    
    print("="*60)
    print("MULTILINGUAL TOKENIZER CANNIBALIZATION TEST")
    print("="*60)
    

    
    # Compare each language
    results = {}
    for lang_name, split in languages.items():
        print(f"\n{'='*60}")
        print(f"TESTING: {lang_name.upper()}")
        print(f"{'='*60}")
        
        mono_path = f"./{lang_name}_tokenizer_{VOCAB_SIZE}"
        
        if not os.path.exists(os.path.join(mono_path, "tokenizer_config.json")):
            print(f"⚠ Monolingual tokenizer not found at {mono_path}, skipping...")
            continue
        
        results[lang_name] = compare_tokenizers(
            mono_path,
            DATASET_NAME,
            split,
            TEST_SAMPLES,
            HF_READ_TOKEN
        )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: CANNIBALIZATION EFFECTS")
    print("="*60)
    for lang_name in languages:
        if lang_name in results:
            r = results[lang_name]
            print(f"\n{lang_name.upper()}:")
            print(f"  Fragmentation: {r['fragmentation_ratio']:.2f}x worse in multilingual")
            print(f"  Char/Token (mono): {1/r['mono_char_fertility']:.3f}")
            print(f"  Char/Token (multi): {1/r['multi_char_fertility']:.3f}")