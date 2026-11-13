import math

import torch
from torch import nn


class LatentNormalizer(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    @classmethod
    def fit(cls, train_latents, floor=0.03):
        mean = train_latents.mean(dim=0)
        scale = train_latents.std(dim=0, unbiased=False).clamp_min(floor)
        return cls(mean, scale)

    def normalize(self, latent):
        return (latent.float() - self.mean) / self.scale

    def denormalize(self, latent):
        return latent.float() * self.scale + self.mean


def timestep_embedding(timesteps, width):
    rates = torch.exp(
        -math.log(10000)
        * torch.arange(width // 2, device=timesteps.device)
        / max(1, width // 2 - 1)
    )
    angles = timesteps.float()[:, None] * rates[None, :]
    return torch.cat([angles.cos(), angles.sin()], 1)


class LatentDenoiser(nn.Module):
    def __init__(self, shape, config):
        super().__init__()
        tokens, dim = shape
        width = config["width"]
        if width % 4:
            raise ValueError("Denoiser width must be divisible by four.")
        self.shape = (tokens, dim)
        self.conditional = config["conditional"]
        self.width = width
        self.input = nn.Linear(dim, width)
        self.position = nn.Parameter(torch.randn(tokens, width) * 0.02)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        if self.conditional:
            self.condition = nn.Linear(dim, width)
            self.null = nn.Parameter(torch.zeros(1, tokens, width))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width,
                    4,
                    width * 3,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config["depth"])
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))

    def forward(self, noisy, timestep, condition=None, dropped=None):
        if condition is not None and not self.conditional:
            raise ValueError("An unconditional model does not take condition tensors.")
        x = (
            self.input(noisy)
            + self.position
            + self.time(timestep_embedding(timestep, self.width))[:, None]
        )
        if self.conditional:
            addition = self.null if condition is None else self.condition(condition)
            if dropped is not None:
                addition = torch.where(dropped[:, None, None], self.null, addition)
            x = x + addition
        for layer in self.layers:
            x = layer(x)
        return self.output(x).float()


class DiffusionSchedule(nn.Module):
    """Cosine forward process and deterministic DDIM, with velocity prediction."""

    def __init__(self, timesteps):
        super().__init__()
        if timesteps < 2:
            raise ValueError("Diffusion needs at least two timesteps.")
        self.timesteps = timesteps
        grid = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos((grid + 0.008) / 1.008 * math.pi / 2).square()
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
        self.register_buffer("alpha_bar", (1 - beta).cumprod(0).float())

    def coefficients(self, t):
        alpha = self.alpha_bar[t].reshape(-1, 1, 1)
        return alpha.sqrt(), (1 - alpha).sqrt()

    def add_noise(self, clean, noise, t):
        a, b = self.coefficients(t)
        return a * clean + b * noise

    def velocity(self, clean, noise, t):
        a, b = self.coefficients(t)
        return a * noise - b * clean

    def predict(self, noisy, velocity, t):
        a, b = self.coefficients(t)
        return a * noisy - b * velocity, b * noisy + a * velocity

    @torch.no_grad()
    def sample(
        self, model, *, batch, steps, rng, source=None, strength=1.0, condition=None, cfg=1.0
    ):
        if not 0 <= strength <= 1 or not 1 <= steps <= self.timesteps or batch < 1:
            raise ValueError("Invalid DDIM strength, steps, or batch.")
        if not math.isfinite(cfg) or cfg < 0:
            raise ValueError("CFG must be finite and nonnegative.")
        shape = (batch, *model.shape)
        if source is not None and source.shape != shape:
            raise ValueError("Source shape differs from diffusion model shape.")
        if condition is not None and condition.shape != shape:
            raise ValueError("Condition shape differs from diffusion model shape.")
        device = self.alpha_bar.device
        if source is not None and strength == 0:
            return source
        start = self.timesteps - 1 if source is None else round(strength * (self.timesteps - 1))
        timeline = torch.linspace(start, 0, steps=min(steps, start + 1)).round().long().tolist()
        noisy = torch.randn(shape, device=device, generator=rng)
        if source is not None:
            noisy = self.add_noise(source, noisy, torch.tensor([start], device=device))
        for index, current in enumerate(timeline):
            timestep = torch.full((batch,), current, device=device, dtype=torch.long)
            velocity = model(noisy, timestep, condition)
            if condition is not None and cfg != 1:
                null = model(noisy, timestep)
                velocity = null + cfg * (velocity - null)
            clean, noise = self.predict(noisy, velocity, timestep)
            if index + 1 == len(timeline):
                return clean
            next_t = torch.tensor([timeline[index + 1]], device=device)
            noisy = self.add_noise(clean, noise, next_t)
        raise AssertionError("DDIM timeline cannot be empty.")
