import os

import pytest
import torch

from ald.config import load_config
from ald.data.tokenization import targets
from ald.models.encoder import make_encoder
from ald.training.autoencoder import make_autoencoder
from ald.utils.runtime import Runtime, seed_everything

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("ALD_QWEN_TEST") != "1", reason="Enable local Qwen/CUDA tests explicitly."
    ),
]


def test_qwen_long_context_backward_and_cached_prefix(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real-model integration test")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    config = load_config("configs/default.yaml")
    runtime = Runtime(config)
    seed_everything(config["seed"])
    runtime.reset_peak()
    encoder = make_encoder(config, runtime)
    text = "雨一直下，路上的车灯亮起来。\n窗边有一只空杯。\n" * 60
    hidden, attention = encoder([text])
    assert hidden.shape[1] == config["encoder"]["max_length"]
    assert not hidden.requires_grad and not any(p.requires_grad for p in encoder.parameters())
    del encoder
    torch.cuda.empty_cache()

    model = make_autoencoder(config, hidden.shape[-1], None, runtime, training=True)
    ids, mask = targets(
        [text], model.decoder.tokenizer, config["decoder"]["max_length"], runtime.device
    )
    assert ids.shape[1] == config["decoder"]["max_length"]
    assert not model.decoder.lm.embedding.weight.requires_grad
    with runtime.autocast():
        loss, _, _ = model.loss(hidden, attention, ids, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.projector.queries.grad.abs().max() > 0
    assert model.decoder.prefix[0].weight.grad.abs().max() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().max() > 0
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    assert model.decoder.lm.embedding.weight.grad is None

    model.eval()
    with torch.no_grad():
        latent = model.projector(hidden, attention)
        prefix = model.decoder.prefix_embeddings(latent)
        prefix_mask = torch.ones(prefix.shape[:2], device=runtime.device, dtype=torch.bool)
        _, cache = model.decoder.lm(prefix, prefix_mask, use_cache=True)
        assert cache is not None
        first_token = model.decoder.lm.embedding(ids[:, :1])
        next_mask = torch.nn.functional.pad(prefix_mask, (0, 1), value=True)
        cached, _ = model.decoder.lm(first_token, next_mask, cache, use_cache=True)
        full, _ = model.decoder.lm(torch.cat((prefix, first_token), dim=1), next_mask)
        torch.testing.assert_close(cached[:, -1].float(), full[:, -1].float(), rtol=0.06, atol=0.5)
        alternate, _ = model.decoder.lm(model.decoder.prefix_embeddings(-latent), prefix_mask)
        original, _ = model.decoder.lm(prefix, prefix_mask)
        assert (alternate[:, -1].float() - original[:, -1].float()).abs().mean() > 0.1

    print(f"Qwen maximum-context CUDA peak: {runtime.memory()}")
