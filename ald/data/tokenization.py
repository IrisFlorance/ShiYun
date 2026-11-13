from collections import Counter

import torch


class CharacterTokenizer:
    pad_id = 0
    eos_id = 1
    unk_id = 2

    def __init__(self, vocabulary):
        self.vocabulary = list(vocabulary)
        self.lookup = {character: i + 3 for i, character in enumerate(vocabulary)}

    @classmethod
    def fit(cls, texts):
        counts = Counter(character for text in texts for character in text)
        return cls(sorted(counts, key=lambda character: (-counts[character], character))[:4096])

    def __len__(self):
        return len(self.vocabulary) + 3

    def encode(self, text, limit):
        return [self.lookup.get(character, self.unk_id) for character in text[:limit]]

    def decode(self, ids):
        return "".join(self.vocabulary[i - 3] if i >= 3 else "�" for i in ids if i > 1)


class QwenTokenizer:
    def __init__(self, config):
        from transformers import AutoTokenizer

        self.inner = AutoTokenizer.from_pretrained(
            config["model"],
            revision=config["revision"],
            trust_remote_code=False,
        )
        self.pad_id = self.inner.pad_token_id
        self.eos_id = self.inner.eos_token_id
        if self.pad_id is None:
            self.pad_id = self.eos_id
        if self.eos_id is None:
            raise ValueError("Qwen decoder tokenizer must have an EOS token.")

    def encode(self, text, limit):
        return self.inner.encode(text, add_special_tokens=False, truncation=True, max_length=limit)

    def decode(self, ids):
        return self.inner.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def decoder_tokenizer(config, vocabulary=None):
    if config["backend"] == "mock":
        return CharacterTokenizer(vocabulary)
    return QwenTokenizer(config["decoder"])


def targets(texts, tokenizer, limit, device):
    if limit < 2:
        raise ValueError("Decoder max_length must leave room for EOS.")
    encoded = [tokenizer.encode(text, limit - 1) + [tokenizer.eos_id] for text in texts]
    width = max(map(len, encoded))
    tokens = torch.full((len(texts), width), tokenizer.pad_id, dtype=torch.long)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for index, ids in enumerate(encoded):
        tokens[index, : len(ids)] = torch.tensor(ids)
        mask[index, : len(ids)] = True
    return tokens.to(device), mask.to(device)
