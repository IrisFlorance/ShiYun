import torch

from ald.models.diffusion import DiffusionSchedule, LatentDenoiser, LatentNormalizer
from ald.utils.files import load_tensors, save_tensors
from ald.utils.runtime import generator


def test_forward_formula_and_velocity_inverse():
    schedule = DiffusionSchedule(32)
    clean = torch.ones((2, 4, 12))
    noise = torch.full_like(clean, -0.25)
    t = torch.tensor([0, 31])
    alpha = schedule.alpha_bar[t, None, None]
    expected = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    noisy = schedule.add_noise(clean, noise, t)
    torch.testing.assert_close(noisy, expected)
    recovered, epsilon = schedule.predict(noisy, schedule.velocity(clean, noise, t), t)
    torch.testing.assert_close(recovered, clean)
    torch.testing.assert_close(epsilon, noise)


def test_sampling_modes_and_seed():
    torch.set_num_threads(2)
    model = LatentDenoiser((4, 12), {"width": 32, "depth": 1, "conditional": True}).eval()
    schedule = DiffusionSchedule(32)
    source = torch.zeros(2, 4, 12)

    def draw(seed, **kwargs):
        return schedule.sample(model, batch=2, steps=10, rng=generator("cpu", seed), **kwargs)

    unconditional = draw(5)
    assert unconditional.shape == source.shape and unconditional.isfinite().all()
    torch.testing.assert_close(unconditional, draw(5))
    assert not torch.equal(unconditional, draw(6))
    assert torch.equal(source, draw(5, source=source, strength=0))
    assert not torch.equal(source, draw(5, source=source, strength=0.4))
    assert not torch.equal(unconditional, draw(5, condition=source))
    torch.testing.assert_close(unconditional, draw(5, condition=source, cfg=0))


def test_train_and_inference_normalization_are_identical(tmp_path):
    samples = torch.randn(16, 4, 12)
    samples[:, 0, 0] = 5
    normalizer = LatentNormalizer.fit(samples)
    path = tmp_path / "state.pt"
    save_tensors(path, normalizer.state_dict())
    restored = LatentNormalizer(**load_tensors(path))
    normalized = normalizer.normalize(samples)
    torch.testing.assert_close(normalized, restored.normalize(samples))
    torch.testing.assert_close(samples, restored.denormalize(normalized))
    assert normalized[:, 0, 0].eq(0).all()
