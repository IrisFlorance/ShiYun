from hashlib import sha256

import torch
from torch import nn

from ald.utils.runtime import generator


class MockEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_length = config["encoder"]["max_length"]
        self.width = config["encoder"]["width"]
        rng = generator("cpu", config["seed"] + 101)
        self.register_buffer("table", torch.randn(1024, self.width, generator=rng))
        self.register_buffer("positions", torch.randn(self.max_length, self.width, generator=rng))
        self.signature = {
            "backend": "mock",
            "version": 1,
            "seed": config["seed"],
            "width": self.width,
            "max_length": self.max_length,
            "mode": "unicode_mod1023_position",
        }

    @torch.no_grad()
    def forward(self, texts):
        values = [
            [1 + ord(char) % 1023 for char in text[: self.max_length]] or [1] for text in texts
        ]
        width = max(map(len, values))
        ids = torch.tensor(
            [row + [0] * (width - len(row)) for row in values], device=self.table.device
        )
        hidden = self.table[ids] + 0.15 * self.positions[:width]
        return hidden, ids.ne(0)


class QwenEncoder(nn.Module):
    def __init__(self, config, runtime):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        options = {"revision": config["revision"], "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], **options)
        self.tokenizer.padding_side = "right"
        self.model = (
            AutoModel.from_pretrained(
                config["model"],
                dtype=runtime.dtype,
                attn_implementation="sdpa",
                **options,
            )
            .requires_grad_(False)
            .eval()
            .to(runtime.device)
        )
        self.width = self.model.config.hidden_size
        self.max_length = config["max_length"]
        self.device = runtime.device
        self.signature = {
            "backend": "qwen",
            "model": config["model"],
            "revision": getattr(self.model.config, "_commit_hash", None) or config["revision"],
            "tokenizer_sha256": sha256(
                self.tokenizer.backend_tokenizer.to_str().encode()
            ).hexdigest(),
            "tokenizer_special_tokens": self.tokenizer.special_tokens_map,
            "max_length": self.max_length,
            "dtype": str(runtime.dtype),
            "mode": "last_hidden/right_padding/no_instruction",
        }

    @torch.no_grad()
    def forward(self, texts):
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.model(**batch, use_cache=False).last_hidden_state
        return hidden, batch.attention_mask.bool()


def make_encoder(config, runtime):
    if config["backend"] == "mock":
        return MockEncoder(config).eval().to(runtime.device)
    if config["backend"] == "qwen":
        return QwenEncoder(config["encoder"], runtime)
    raise ValueError(f"Unknown backend {config['backend']!r}")


def semantic_embedding(hidden, mask):
    # Model-card pooling for Qwen3 Embedding. All encoders here use right padding.
    return torch.nn.functional.normalize(
        hidden[torch.arange(len(hidden), device=hidden.device), mask.sum(1) - 1].float(),
        dim=-1,
    )
