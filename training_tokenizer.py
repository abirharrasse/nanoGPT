import os
from itertools import islice
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_training_corpus(dataset_name, splits_info, text_column="text", os.getenv('HF_TOKEN')=None):
    print("Starting to yield text for tokenizer training...")
    count = 0

    for split_name, num_samples in splits_info.items():
        print(f"Streaming split: {split_name} (taking up to {num_samples} samples)")
        
        streamed_dataset = load_dataset(
            dataset_name,
            split=split_name,
            streaming=True,
            token=os.getenv('HF_TOKEN')
        )

        for item in islice(streamed_dataset, num_samples):
            if item and text_column in item and item[text_column]:
                yield item[text_column]
                count += 1
                if count % 100000 == 0:
                    print(f"Processed {count} documents...")

    print(f"Finished yielding. Total documents processed: {count}")


def train_and_save_tokenizer(dataset_name, splits_info, text_column, vocab_size, save_path, os.getenv('HF_TOKEN')=None):
    os.makedirs(save_path, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()

    special_tokens = ["<|endoftext|>", "<pad>", "<unk>", "<s>", "</s>"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    print(f"Starting tokenizer training with vocab_size={vocab_size}...")
    corpus_iterator = get_training_corpus(dataset_name, splits_info, text_column, os.getenv('HF_TOKEN'))
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    print("Tokenizer training complete.")

    # Wrap and assign special tokens
    os.getenv('HF_TOKEN') = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
    )
    
    os.getenv('HF_TOKEN').save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path}")


if __name__ == "__main__":
HF_TOKEN = os.getenv('HF_TOKEN')
HF_WRITE_TOKEN = os.getenv('HF_WRITE_TOKEN')
    
    DATASET_NAME = "CausalNLP/gpt2small_full_training_data"
    TEXT_COLUMN = "text"

    SPLITS_AND_SAMPLES = {
        "deu_Latn": 200000,
        "fra_Latn": 200000,
        "arb_Arab": 200000,
        "cmn_Hani": 200000,
    }

    VOCAB_SIZE = 100000
    TOKENIZER_SAVE_PATH = "./my_non_english_tokenizer"
    HF_REPO_ID = "abir-hr196/non_english_tokenizer"

    print("--- Tokenizer Training Script ---")
    train_and_save_tokenizer(
        dataset_name=DATASET_NAME,
        splits_info=SPLITS_AND_SAMPLES,
        text_column=TEXT_COLUMN,
        vocab_size=VOCAB_SIZE,
        save_path=TOKENIZER_SAVE_PATH,
        os.getenv('HF_TOKEN')=HF_READ_TOKEN
    )
    
    # Verify special tokens
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_SAVE_PATH)
    print(f"\nVerification:")
    print(f"  EOS token: {tok.eos_token} (id: {tok.eos_token_id})")
    print(f"  PAD token: {tok.pad_token} (id: {tok.pad_token_id})")
    
    # Push to HF
    tok.push_to_hub(HF_REPO_ID, token=HF_WRITE_TOKEN)
    print(f"Tokenizer pushed to https://huggingface.co/{HF_REPO_ID}")