import torch
from torch import nn
from torch.nn import functional as F


class MockLM(nn.Module):
    def __init__(self, vocab_size, width):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, embeds, attention_mask, cache=None, use_cache=False):
        hidden, state = self.recurrent(embeds, cache)
        return self.head(hidden), state if use_cache else None


class QwenLM(nn.Module):
    def __init__(self, config, runtime, training):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            config["model"],
            revision=config["revision"],
            dtype=runtime.dtype,
            attn_implementation="sdpa",
            trust_remote_code=False,
        ).requires_grad_(False)
        self.width = self.model.config.hidden_size
        self.max_positions = self.model.config.max_position_embeddings
        if config["lora_rank"]:
            from peft import LoraConfig, get_peft_model

            rank = config["lora_rank"]
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=rank,
                    lora_alpha=2 * rank,
                    target_modules=["q_proj", "v_proj"],
                    task_type="CAUSAL_LM",
                    lora_dropout=0.0,
                ),
            )
        if training and config["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

    @property
    def embedding(self):
        return self.model.get_input_embeddings()

    def forward(self, embeds, attention_mask, cache=None, use_cache=False):
        result = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask.long(),
            past_key_values=cache,
            use_cache=use_cache,
        )
        return result.logits, result.past_key_values


def sample_token(logits, temperature, top_p, rng):
    logits = logits.float()
    if temperature == 0:
        return logits.argmax(-1)
    values, indices = (logits / temperature).sort(descending=True)
    probabilities = values.softmax(-1)
    remove = probabilities.cumsum(-1) - probabilities >= top_p
    probabilities = probabilities.masked_fill(remove, 0)
    selected = torch.multinomial(probabilities, 1, generator=rng)
    return indices.gather(-1, selected).squeeze(-1)


class LatentDecoder(nn.Module):
    def __init__(self, config, tokenizer, runtime, *, training=False):
        super().__init__()
        options = config["decoder"]
        if config["backend"] == "mock":
            width = options["width"]
            self.lm = MockLM(len(tokenizer), width)
            self.lm.embedding.requires_grad_(False)
            self.max_positions = None
        else:
            self.lm = QwenLM(options, runtime, training)
            width = self.lm.width
            self.max_positions = self.lm.max_positions
        self.prefix = nn.Sequential(
            nn.Linear(config["latent"]["dim"], width),
            nn.Tanh(),
            nn.Linear(width, width),
        )
        # Initialize continuous token embeddings at the scale of the pretrained vocabulary.
        if config["backend"] != "mock":
            scale = self.lm.embedding.weight[:256].detach().float().std().item()
            nn.init.normal_(self.prefix[-1].weight, std=scale / width**0.5)
            nn.init.zeros_(self.prefix[-1].bias)
        self.tokenizer = tokenizer
        self.to(runtime.device)

    def prefix_embeddings(self, latent):
        return self.prefix(latent.float()).to(dtype=self.lm.embedding.weight.dtype)

    def teacher_logits(self, latent, ids, mask):
        prefix = self.prefix_embeddings(latent)
        text = self.lm.embedding(ids[:, :-1])
        embeds = torch.cat([prefix, text], dim=1)
        visible = torch.cat([torch.ones_like(prefix[:, :, 0], dtype=torch.bool), mask[:, :-1]], 1)
        logits, _ = self.lm(embeds, visible)
        return logits[:, prefix.shape[1] - 1 :]

    def nll(self, latent, ids, mask):
        logits = self.teacher_logits(latent, ids, mask)
        losses = F.cross_entropy(
            logits.flatten(0, 1).float(),
            ids.masked_fill(~mask, -100).flatten(),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(ids)
        return losses.sum(1), mask.sum(1)

    @torch.no_grad()
    def generate(self, latent, *, rng, max_tokens=96, temperature=0.8, top_p=0.95, use_cache=True):
        if max_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("Invalid max_tokens, temperature, or top_p.")
        embeds = self.prefix_embeddings(latent)
        if self.max_positions is not None and max_tokens + embeds.shape[1] > self.max_positions:
            raise ValueError("Prefix and generation exceed decoder position limit.")
        visible = torch.ones_like(embeds[:, :, 0], dtype=torch.bool)
        cache = None
        outputs = [[] for _ in latent]
        done = torch.zeros(len(latent), dtype=torch.bool, device=latent.device)
        for _ in range(max_tokens):
            logits, cache = self.lm(embeds, visible, cache, use_cache)
            token_logits = logits[:, -1].clone()
            if self.tokenizer.pad_id != self.tokenizer.eos_id:
                token_logits[:, self.tokenizer.pad_id] = -torch.inf
            new_ids = sample_token(token_logits, temperature, top_p, rng)
            new_ids = new_ids.masked_fill(done, self.tokenizer.eos_id)
            done |= new_ids.eq(self.tokenizer.eos_id)
            for sequence, index in zip(outputs, new_ids.tolist(), strict=True):
                if index != self.tokenizer.eos_id:
                    sequence.append(index)
            if done.all():
                break
            new_embedding = self.lm.embedding(new_ids[:, None])
            embeds = new_embedding if use_cache else torch.cat((embeds, new_embedding), 1)
            visible = F.pad(visible, (0, 1), value=True)
        return [self.tokenizer.decode(sequence) for sequence in outputs]
