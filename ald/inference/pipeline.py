from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from ald.data.corpus import normalize
from ald.models.encoder import make_encoder, semantic_embedding
from ald.models.projector import LatentProjector
from ald.training.autoencoder import load_autoencoder
from ald.training.diffusion import load_diffusion
from ald.training.loop import restore_parameters
from ald.utils.files import digest, load_tensors
from ald.utils.runtime import Runtime, generator


@dataclass
class Encoding:
    latent: torch.Tensor
    semantic: torch.Tensor


class GenerationPipeline:
    def __init__(self, ae_path, diffusion_path=None, *, device=None, conditional_path=None):
        self.ae_path = Path(ae_path)
        payload = load_tensors(ae_path)
        identity = payload["identity"]
        if identity["stage"] != "autoencoder":
            raise ValueError("Expected an autoencoder checkpoint.")
        self.config = identity["config"]
        self.runtime = Runtime(self.config, device)
        self.ae_hash = digest(ae_path)
        self.projector = (
            LatentProjector(identity["encoder_width"], self.config["latent"])
            .to(self.runtime.device)
            .eval()
        )
        restore_parameters(
            self.projector,
            {
                name.removeprefix("projector."): value
                for name, value in payload["model"].items()
                if name.startswith("projector.")
            },
        )
        self.encoder_signature = identity["encoder_signature"]
        self.dataset_id = identity["dataset_id"]
        self.encoder = None
        self.autoencoder = None
        self.prior = None
        self.conditional = None
        self.paths = {"autoencoder": str(ae_path), "diffusion": None, "conditional": None}
        for path in (diffusion_path, conditional_path):
            if path:
                bundle = load_diffusion(path, self.runtime, self.ae_hash)
                kind = "conditional" if bundle[0].conditional else "diffusion"
                if self.paths[kind] is not None:
                    raise ValueError(f"Multiple checkpoints supplied for {kind}.")
                self.paths[kind] = str(path)
                if kind == "conditional":
                    self.conditional = bundle
                else:
                    self.prior = bundle
        self.runtime.reset_peak()

    @property
    def description(self):
        return {
            "backend": self.config["backend"],
            "decoder": self.config["decoder"]["model"],
            "encoder": self.encoder_signature,
            "checkpoints": self.paths,
            "device": str(self.runtime.device),
            "dtype": str(self.runtime.dtype),
        }

    @torch.no_grad()
    def encode(self, texts):
        cleaned = [normalize(text) for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Input texts must be nonempty.")
        device = self.runtime.device
        if self.autoencoder is not None and device.type == "cuda":
            self.autoencoder.cpu()
            torch.cuda.empty_cache()
        if self.encoder is None:
            self.encoder = make_encoder(self.config, self.runtime)
            expected = {
                key: value for key, value in self.encoder_signature.items() if key != "dtype"
            }
            current = {
                key: value for key, value in self.encoder.signature.items() if key != "dtype"
            }
            if expected != current:
                self.encoder = None
                raise ValueError("Loaded encoder differs from training cache provenance.")
        self.encoder.to(device)
        encodings = []
        size = self.config["encoder"]["batch_size"]
        try:
            for offset in range(0, len(cleaned), size):
                hidden, mask = self.encoder(cleaned[offset : offset + size])
                latents = self.projector(hidden, mask).cpu()
                embeddings = semantic_embedding(hidden, mask).cpu()
                encodings.extend(Encoding(z, e) for z, e in zip(latents, embeddings, strict=True))
        finally:
            if device.type == "cuda":
                self.encoder.cpu()
                torch.cuda.empty_cache()
        return encodings

    @torch.no_grad()
    def transform(self, source, *, mode="sdedit", seed=0, strength=0.5, steps=None, cfg=1.0):
        if mode == "direct":
            if source is None:
                raise ValueError("Direct generation requires an encoded source.")
            return source
        if mode not in {"sdedit", "unconditional", "conditional"}:
            raise ValueError(f"Unknown mode {mode!r}.")
        bundle = self.conditional if mode == "conditional" else self.prior
        if bundle is None:
            raise ValueError(f"No compatible diffusion checkpoint for {mode}.")
        if source is None and mode != "unconditional":
            raise ValueError(f"{mode} requires an encoded source.")
        if mode == "sdedit" and strength == 0:
            return source
        model, schedule, normalizer, identity = bundle
        if steps is None:
            steps = identity["config"]["diffusion"]["steps"]
        device = self.runtime.device
        z = normalizer.normalize(source[None].to(device)) if mode != "unconditional" else None
        sampled = schedule.sample(
            model,
            batch=1,
            steps=steps,
            rng=generator(device, seed),
            strength=strength,
            source=z if mode == "sdedit" else None,
            condition=z if mode == "conditional" else None,
            cfg=cfg,
        )
        return normalizer.denormalize(sampled)[0].cpu()

    @torch.no_grad()
    def decode(self, latent, *, seed=0, temperature=0.8, top_p=0.95, max_tokens=96):
        runtime = self.runtime
        if self.autoencoder is None:
            self.autoencoder, _ = load_autoencoder(self.ae_path, runtime)
        self.autoencoder.to(runtime.device)
        output = self.autoencoder.decoder.generate(
            latent[None].to(runtime.device),
            rng=generator(runtime.device, seed),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return output[0]

    def generate(
        self,
        text,
        *,
        mode="sdedit",
        seed=0,
        count=1,
        strength=0.5,
        steps=None,
        cfg=1.0,
        temperature=0.8,
        top_p=0.95,
        max_tokens=96,
    ):
        if count < 1:
            raise ValueError("count must be positive.")
        source = self.encode([text])[0].latent if mode != "unconditional" else None
        rows = []
        for index in range(count):
            sample_seed = seed + index
            latent = self.transform(
                source, mode=mode, seed=sample_seed, strength=strength, steps=steps, cfg=cfg
            )
            generated = self.decode(
                latent,
                seed=sample_seed,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            rows.append(
                {
                    "text": generated,
                    "seed": sample_seed,
                    "mode": mode,
                    "similarity": latent_similarity(source, latent),
                    "input_latent": source.tolist() if source is not None else None,
                    "output_latent": latent.tolist(),
                }
            )
        return rows


def latent_similarity(left, right):
    if left is None:
        return None
    return F.cosine_similarity(left.flatten(), right.flatten(), dim=0).item()
