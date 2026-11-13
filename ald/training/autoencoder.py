from collections import defaultdict

import torch

from ald.data.cache import FeatureDataset, pad_features
from ald.data.tokenization import CharacterTokenizer, decoder_tokenizer, targets
from ald.models.autoencoder import LatentAutoencoder
from ald.models.decoder import LatentDecoder
from ald.models.projector import LatentProjector
from ald.training.loop import batches, restore_parameters, run_identity, train
from ald.utils.files import load_tensors
from ald.utils.runtime import Runtime, generator, seed_everything


def make_autoencoder(config, width, vocabulary, runtime, *, training=False):
    tokenizer = decoder_tokenizer(config, vocabulary)
    projector = LatentProjector(width, config["latent"]).to(runtime.device)
    decoder = LatentDecoder(config, tokenizer, runtime, training=training)
    return LatentAutoencoder(projector, decoder, config)


def load_autoencoder(path, runtime):
    saved = load_tensors(path)
    identity = saved["identity"]
    if identity["stage"] != "autoencoder":
        raise ValueError(f"Not an autoencoder checkpoint: {path}")
    config = identity["config"]
    seed_everything(config["seed"])
    model = make_autoencoder(config, identity["encoder_width"], identity["vocabulary"], runtime)
    restore_parameters(model, saved["model"])
    return model.eval(), identity


def fit_autoencoder(config, cache, out, *, resume=False):
    runtime = Runtime(config)
    runtime.reset_peak()
    seed_everything(config["seed"])
    train_set = FeatureDataset(cache, "train", "poem")
    val_set = FeatureDataset(cache, "val", "poem")
    manifest = train_set.manifest
    encoder_metadata = manifest["encoder"]
    expected = {"backend": config["backend"], "max_length": config["encoder"]["max_length"]}
    if config["backend"] == "mock":
        expected.update(seed=config["seed"], width=config["encoder"]["width"])
    else:
        expected["model"] = config["encoder"]["model"]
        if config["encoder"]["revision"] != "main":
            expected["revision"] = config["encoder"]["revision"]
    if any(encoder_metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Encoder cache provenance differs from the requested configuration.")
    vocabulary = None
    if config["backend"] == "mock":
        vocabulary = CharacterTokenizer.fit(row["text"] for row in train_set).vocabulary
    identity = run_identity(
        config,
        "autoencoder",
        runtime,
        {
            "dataset_id": manifest["dataset_id"],
            "cache_id": manifest["cache_id"],
            "encoder_signature": manifest["encoder"],
            "encoder_width": manifest["encoder_width"],
            "vocabulary": vocabulary,
        },
    )
    model = make_autoencoder(config, manifest["encoder_width"], vocabulary, runtime, training=True)
    options = config["autoencoder_train"]

    def batch_values(rows):
        features, attention = pad_features([row["hidden"] for row in rows], runtime.device)
        ids, mask = targets(
            [row["text"] for row in rows],
            model.decoder.tokenizer,
            config["decoder"]["max_length"],
            runtime.device,
        )
        return features, attention, ids, mask

    def train_loss(rows):
        return model.loss(*batch_values(rows))

    def validate(dataset):
        total = count = seen = 0
        summed = squares = 0
        forms = defaultdict(lambda: [0.0, 0, 0])
        for rows in batches(dataset, options["batch_size"]):
            features, attention, ids, mask = batch_values(rows)
            latent = model.projector(features, attention)
            nll, tokens = model.decoder.nll(latent, ids, mask)
            flat = latent.flatten(1).double()
            summed = summed + flat.sum(0)
            squares = squares + flat.square().sum(0)
            seen += len(rows)
            total += nll.sum().item()
            count += tokens.sum().item()
            for row, row_loss, row_tokens in zip(rows, nll.tolist(), tokens.tolist(), strict=True):
                bucket = forms[row.get("form", "unspecified")]
                bucket[0] += row_loss
                bucket[1] += row_tokens
                bucket[2] += 1
        mean = summed / seen
        variance = (squares / seen - mean.square()).clamp_min(0)
        diagnostic = {
            "val_loss": total / count,
            "latent_mean": mean.mean().item(),
            "latent_variance": variance.mean().item(),
            "active_dimensions": int((variance > 1e-4).sum().item()),
            "latent_variance_samples": seen,
            "val_nll_by_form": {
                name: {"nll": loss / tokens, "tokens": tokens, "samples": samples}
                for name, (loss, tokens, samples) in forms.items()
            },
        }
        examples = dataset if len(dataset) >= 2 else train_set
        rows = [examples[index] for index in range(min(2, len(examples)))]
        features, attention, ids, mask = batch_values(rows)
        latent = model.projector(features, attention)
        reference, tokens = model.decoder.nll(latent, ids, mask)
        zeroed, _ = model.decoder.nll(torch.zeros_like(latent), ids, mask)
        diagnostic["zero_nll_delta"] = ((zeroed.sum() - reference.sum()) / tokens.sum()).item()
        if len(rows) >= 2:
            swapped, _ = model.decoder.nll(latent.flip(0), ids, mask)
            diagnostic["swap_nll_delta"] = ((swapped.sum() - reference.sum()) / tokens.sum()).item()
            diagnostic["swap_source"] = "val" if examples is dataset else "train"
        gen_options = {
            "rng": generator(runtime.device, config["seed"]),
            "temperature": 0,
            "max_tokens": min(24, config["decoder"]["max_length"]),
        }
        diagnostic["reconstruction"] = model.decoder.generate(latent, **gen_options)
        diagnostic["targets"] = [row["text"] for row in rows]
        if len(rows) >= 2:
            diagnostic["swapped_reconstruction"] = model.decoder.generate(
                latent.flip(0), **gen_options
            )
        return diagnostic

    return train(
        model,
        (train_set, val_set),
        options,
        runtime,
        identity,
        out,
        train_loss,
        validate,
        resume=resume,
    )
