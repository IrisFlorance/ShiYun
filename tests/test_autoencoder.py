from pathlib import Path

import pytest
import torch

from ald.config import load_config
from ald.data.cache import pad_features
from ald.data.tokenization import CharacterTokenizer, targets
from ald.models.encoder import make_encoder
from ald.training.autoencoder import make_autoencoder
from ald.utils.runtime import Runtime, generator, seed_everything


@pytest.fixture
def mock_model():
    config = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
    runtime = Runtime(config)
    seed_everything(config["seed"])
    encoder = make_encoder(config, runtime)
    texts = ["窗外正在下雨。", "晚霞落在船尾。"]
    vocabulary = CharacterTokenizer.fit(texts).vocabulary
    model = make_autoencoder(config, encoder.width, vocabulary, runtime)
    return config, runtime, encoder, model, texts


def test_encode_decode_gradients_and_frozen_parameters(mock_model):
    config, runtime, encoder, model, texts = mock_model
    hidden, attention = encoder(texts)
    latent = model.projector(hidden, attention)
    assert latent.shape == (2, 4, 12)
    ids, mask = targets(texts, model.decoder.tokenizer, 48, runtime.device)
    assert model.decoder.teacher_logits(latent, ids, mask).shape[:2] == ids.shape
    embedding = model.decoder.lm.embedding.weight.detach().clone()
    loss, _, _ = model.loss(hidden, attention, ids, mask)
    loss.backward()
    for module in (model.projector, model.decoder.prefix):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.step()
    assert not hidden.requires_grad
    assert model.decoder.lm.embedding.weight.grad is None
    torch.testing.assert_close(embedding, model.decoder.lm.embedding.weight)


def test_attention_mask_and_causal_targets(mock_model):
    _, runtime, encoder, model, texts = mock_model
    hidden, mask = encoder([texts[0][:3], texts[1]])
    reference = model.projector(hidden, mask)
    changed = hidden.masked_fill(~mask[..., None], 1200)
    torch.testing.assert_close(reference, model.projector(changed, mask))
    short, _ = pad_features([hidden[0, :3]], runtime.device)
    assert short.shape[1] == 3

    ids, target_mask = targets(
        [texts[0][:3], texts[1]], model.decoder.tokenizer, 48, runtime.device
    )
    changed_ids = ids.clone()
    changed_ids[0, ~target_mask[0]] = 3
    nll, count = model.decoder.nll(reference, ids, target_mask)
    changed_nll, _ = model.decoder.nll(reference, changed_ids, target_mask)
    torch.testing.assert_close(nll, changed_nll)
    assert count[0] == 4  # Three characters and a real EOS.

    future = ids.clone()
    future[:, 2:] = 3
    torch.testing.assert_close(
        model.decoder.teacher_logits(reference, ids, target_mask)[:, :3],
        model.decoder.teacher_logits(reference, future, target_mask)[:, :3],
    )


def test_latent_response_and_cached_seeded_decoding(mock_model):
    _, runtime, encoder, model, texts = mock_model
    model.eval()
    hidden, mask = encoder(texts)
    latent = model.projector(hidden, mask)
    ids, target_mask = targets(texts, model.decoder.tokenizer, 48, runtime.device)
    logits = model.decoder.teacher_logits(latent, ids, target_mask)
    swapped = model.decoder.teacher_logits(latent.flip(0), ids, target_mask)
    assert (logits[:, 0] - swapped[:, 0]).abs().max() > 1e-4

    def run(seed, cache=True):
        return model.decoder.generate(
            latent, rng=generator(runtime.device, seed), max_tokens=12, use_cache=cache
        )

    assert run(311) == run(311, cache=False) == run(311)
    assert run(311) != run(312)
