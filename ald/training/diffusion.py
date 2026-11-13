from pathlib import Path

import torch

from ald.data.cache import FeatureDataset, pad_features
from ald.models.diffusion import DiffusionSchedule, LatentDenoiser, LatentNormalizer
from ald.models.projector import LatentProjector
from ald.training.loop import batches, restore_parameters, run_identity, train
from ald.utils.files import digest, load_tensors, save_tensors
from ald.utils.runtime import Runtime, generator, seed_everything


def diffusion_modules(config, runtime):
    shape = (config["latent"]["tokens"], config["latent"]["dim"])
    denoiser = LatentDenoiser(shape, config["diffusion"]).to(runtime.device)
    schedule = DiffusionSchedule(config["diffusion"]["timesteps"]).to(runtime.device)
    return denoiser, schedule


@torch.no_grad()
def export_latents(config, cache, ae_path, output):
    saved = load_tensors(ae_path)
    ae = saved["identity"]
    if ae["stage"] != "autoencoder":
        raise ValueError("Expected an autoencoder checkpoint.")
    for key in ("tokens", "dim", "width"):
        if ae["config"]["latent"][key] != config["latent"][key]:
            raise ValueError(f"Latent config differs from autoencoder: {key}")
    if ae["config"]["backend"] != config["backend"]:
        raise ValueError("Backend differs from autoencoder.")
    runtime = Runtime(config)
    projector = LatentProjector(ae["encoder_width"], config["latent"]).to(runtime.device).eval()
    restore_parameters(
        projector,
        {
            name.removeprefix("projector."): value
            for name, value in saved["model"].items()
            if name.startswith("projector.")
        },
    )
    conditional = config["diffusion"]["conditional"]
    latent_data = {"ae_sha256": digest(ae_path), "conditional": conditional, "splits": {}}
    for split in ("train", "val", "test"):
        data = FeatureDataset(cache, split, "poem")
        manifest = data.manifest
        if (
            manifest["encoder"] != ae["encoder_signature"]
            or manifest["dataset_id"] != ae["dataset_id"]
        ):
            raise ValueError("Feature cache encoder or dataset differs from the autoencoder.")
        if conditional and (
            manifest["view_fraction"] != config["diffusion"]["view_fraction"]
            or manifest["view_seed"] != config["seed"]
        ):
            raise ValueError("Build the matching view cache with --views first.")
        latent_data["cache_id"] = manifest["cache_id"]
        main, views = [], []
        for rows in batches(data, config["encoder"]["batch_size"]):
            main.append(
                projector(*pad_features([row["hidden"] for row in rows], runtime.device)).cpu()
            )
            if conditional:
                views.append(
                    projector(
                        *pad_features([row["view_hidden"] for row in rows], runtime.device)
                    ).cpu()
                )
        shape = (0, config["latent"]["tokens"], config["latent"]["dim"])
        latent_data["splits"][split] = {"clean": torch.cat(main) if main else torch.empty(shape)}
        if conditional:
            latent_data["splits"][split]["condition"] = (
                torch.cat(views) if views else torch.empty(shape)
            )
    save_tensors(output, latent_data)
    return latent_data


def load_diffusion(path, runtime, ae_sha256):
    saved = load_tensors(path)
    identity = saved["identity"]
    if identity["stage"] != "diffusion" or identity["ae_sha256"] != ae_sha256:
        raise ValueError("Diffusion checkpoint belongs to a different autoencoder.")
    model, schedule = diffusion_modules(identity["config"], runtime)
    restore_parameters(model, saved["model"])
    normalizer = LatentNormalizer(**identity["normalization"]).to(runtime.device)
    return model.eval(), schedule, normalizer, identity


def distribution_stats(latents):
    flat = latents.flatten(1).float()
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "between_example_variance": flat.var(dim=0, unbiased=False).mean().item(),
        "mean_pair_distance": torch.pdist(flat).mean().item() if len(flat) > 1 else None,
    }


def fit_diffusion(config, cache, ae_path, out, *, resume=False):
    out = Path(out)
    if (out / "last.pt").exists() and not resume:
        raise FileExistsError("This diffusion run already exists. Use --resume or a new output.")
    exported_path = out / "latents.pt"
    if resume:
        latent_data = load_tensors(exported_path)
        from ald.utils.files import read_json

        cache_info = read_json(Path(cache) / "manifest.json")
        if (
            latent_data["ae_sha256"] != digest(ae_path)
            or latent_data["cache_id"] != cache_info["cache_id"]
            or latent_data["conditional"] != config["diffusion"]["conditional"]
        ):
            raise ValueError("Resume requires the original autoencoder and latent cache.")
    else:
        latent_data = export_latents(config, cache, ae_path, exported_path)
    runtime = Runtime(config)
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()
    runtime.reset_peak()
    seed_everything(config["seed"])
    data = latent_data["splits"]
    if min(len(data[split]["clean"]) for split in ("train", "val")) < 1:
        raise ValueError("Diffusion requires train and validation poems.")
    normalizer = LatentNormalizer.fit(data["train"]["clean"])
    # Keep bulk tensors on CPU throughout diffusion training.
    datasets = []
    for split in ("train", "val"):
        clean = normalizer.normalize(data[split]["clean"])
        if latent_data["conditional"]:
            view = normalizer.normalize(data[split]["condition"])
            datasets.append(list(zip(clean, view, strict=True)))
        else:
            datasets.append([(example, None) for example in clean])
    dropout = config["diffusion"]["condition_dropout"]
    if not 0 <= dropout < 1:
        raise ValueError("condition_dropout must be in [0, 1).")
    model, schedule = diffusion_modules(config, runtime)
    identity = run_identity(
        config,
        "diffusion",
        runtime,
        {
            "ae_sha256": latent_data["ae_sha256"],
            "cache_id": latent_data["cache_id"],
            "latent_sha256": digest(exported_path),
            "normalization": {
                key: tensor.tolist() for key, tensor in normalizer.state_dict().items()
            },
            "target": "velocity/cosine_ddim_v1",
        },
    )
    options = config["diffusion_train"]

    def batch_values(rows):
        clean = torch.stack([row[0] for row in rows]).to(runtime.device)
        condition = (
            torch.stack([row[1] for row in rows]).to(runtime.device)
            if latent_data["conditional"]
            else None
        )
        return clean, condition

    def loss_at(clean, condition, t, noise, drop=None):
        noisy = schedule.add_noise(clean, noise, t)
        velocity = model(noisy, t, condition, drop)
        target = schedule.velocity(clean, noise, t)
        v_loss = (velocity - target).square().mean()
        _, estimated_noise = schedule.predict(noisy, velocity, t)
        epsilon_loss = (estimated_noise - noise).square().mean()
        return v_loss, epsilon_loss, target.square().mean()

    def train_loss(rows):
        clean, condition = batch_values(rows)
        t = torch.randint(schedule.timesteps, (len(clean),), device=runtime.device)
        drop = (
            torch.rand(len(clean), device=runtime.device) < dropout
            if condition is not None
            else None
        )
        loss, _, _ = loss_at(clean, condition, t, torch.randn_like(clean), drop)
        return loss, loss.detach() * len(clean), len(clean)

    def validate(dataset):
        rng = generator(runtime.device, config["seed"] + 202)
        totals = {label: [0.0, 0.0, 0.0] for label in ("low", "middle", "high")}
        for rows in batches(dataset, options["batch_size"]):
            clean, condition = batch_values(rows)
            noise = torch.randn(clean.shape, device=runtime.device, generator=rng)
            for label, fraction in zip(totals, (0.1, 0.5, 0.9), strict=True):
                timestep = round(fraction * (schedule.timesteps - 1))
                t = torch.full((len(clean),), timestep, device=runtime.device, dtype=torch.long)
                v_loss, eps_loss, baseline = loss_at(clean, condition, t, noise)
                totals[label][0] += v_loss.item() * len(clean)
                totals[label][1] += eps_loss.item() * len(clean)
                totals[label][2] += baseline.item() * len(clean)
        report = {
            label: {
                "velocity_mse": v / len(dataset),
                "epsilon_mse": eps / len(dataset),
                "zero_velocity_mse": zero / len(dataset),
            }
            for label, (v, eps, zero) in totals.items()
        }
        sample = schedule.sample(model, batch=8, steps=config["diffusion"]["steps"], rng=rng)
        return {
            "val_loss": sum(row["velocity_mse"] for row in report.values()) / len(report),
            "noise_bands": report,
            "sample_latents": distribution_stats(sample),
            "val_latents": distribution_stats(torch.stack([row[0] for row in dataset[:256]])),
        }

    return train(
        model, datasets, options, runtime, identity, out, train_loss, validate, resume=resume
    )
