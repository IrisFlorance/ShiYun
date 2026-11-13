from pathlib import Path

import pytest
import torch
from torch import nn

from ald.config import load_config
from ald.training.loop import run_identity, train
from ald.utils.files import load_tensors
from ald.utils.runtime import Runtime, seed_everything


def test_resume_at_optimizer_boundary(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
    config["autoencoder_train"].update(epochs=1, batch_size=2, accumulation=2, checkpoint_every=1)
    options = config["autoencoder_train"]
    runtime = Runtime(config)
    identity = run_identity(config, "autoencoder", runtime, {})
    data = [torch.tensor([index / 10]) for index in range(12)]

    def run(path, *, resume=False, interrupt=False):
        seed_everything(config["seed"])
        model = nn.Linear(1, 1)
        steps = 0

        def loss(rows):
            nonlocal steps
            steps += 1
            if interrupt and steps == 3:
                raise RuntimeError("interrupted")
            x = torch.stack(rows)
            value = (model(x + torch.randn_like(x)) - x).square().mean()
            return value, value.detach() * len(rows), len(rows)

        return train(
            model,
            (data, data),
            options,
            runtime,
            identity,
            path,
            loss,
            lambda _: {"val_loss": 1.0},
            resume=resume,
        )

    recovered = tmp_path / "recovered"
    with pytest.raises(RuntimeError, match="interrupted"):
        run(recovered, interrupt=True)
    partial = load_tensors(recovered / "last.pt")
    assert partial["epoch"] == 0 and partial["partial_epoch"]["next_batch"] == 2
    reference = tmp_path / "reference"
    assert run(recovered, resume=True) == run(reference)
    actual = load_tensors(recovered / "last.pt")
    expected = load_tensors(reference / "last.pt")
    for name, parameter in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], parameter, rtol=0, atol=0)
