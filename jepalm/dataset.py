"""Real text dataset using HuggingFace datasets and BERT tokenizer."""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """Tokenizes real text from HuggingFace datasets.

    Loads text, tokenizes with BERT tokenizer, and returns fixed-length
    sequences for JEPA-LM training.
    """

    def __init__(self, config, split="train"):
        self.config = config
        self.max_seq_len = config.enc_max_seq_len

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        config.enc_vocab_size = len(self.tokenizer)
        config.pad_token_id = self.tokenizer.pad_token_id
        config.mask_token_id = self.tokenizer.mask_token_id

        # Load dataset
        print(f"Loading dataset: {config.dataset_name} ({config.dataset_config})...")
        from datasets import load_dataset
        dataset = load_dataset(config.dataset_name, config.dataset_config, split=split)

        # Tokenize and chunk into fixed-length sequences
        print("Tokenizing and chunking...")
        self.sequences = []
        buffer = []

        for item in dataset:
            text = item["text"]
            if not text.strip():
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.max_seq_len:
                seq = buffer[:self.max_seq_len]
                buffer = buffer[self.max_seq_len:]
                self.sequences.append(torch.tensor(seq, dtype=torch.long))

                if config.max_samples and len(self.sequences) >= config.max_samples:
                    break

            if config.max_samples and len(self.sequences) >= config.max_samples:
                break

        # Add [CLS] and [SEP] tokens
        cls_id = self.tokenizer.cls_token_id or 101
        sep_id = self.tokenizer.sep_token_id or 102
        for i in range(len(self.sequences)):
            seq = self.sequences[i]
            self.sequences[i] = torch.cat([
                torch.tensor([cls_id]),
                seq[:self.max_seq_len - 2],
                torch.tensor([sep_id]),
            ])

        print(f"Created {len(self.sequences)} sequences of length {self.max_seq_len}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"input_ids": self.sequences[idx]}


class SyntheticTextDataset(Dataset):
    """Quick synthetic dataset for testing (no download needed)."""

    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(1, self.vocab_size, (self.seq_len,))
        return {"input_ids": tokens}


def get_dataset(config, split="train"):
    """Get dataset based on config.

    Returns TextDataset for real data, SyntheticTextDataset for testing.
    """
    if config.dataset_name == "synthetic":
        return SyntheticTextDataset(
            vocab_size=config.enc_vocab_size,
            seq_len=config.enc_max_seq_len,
            num_samples=config.max_samples,
        )
    else:
        return TextDataset(config, split=split)
