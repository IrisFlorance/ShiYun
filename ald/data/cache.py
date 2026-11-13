from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

import torch
from torch.utils.data import Dataset

from ald.data.corpus import SPLITS
from ald.models.encoder import make_encoder, semantic_embedding
from ald.utils.files import digest, fingerprint, load_tensors, read_json, read_jsonl
from ald.utils.files import save_tensors, write_json
from ald.utils.runtime import Runtime, versions


def view_text(text, fraction, rng):
    if not 0 < fraction <= 1:
        raise ValueError("view_fraction must be in (0, 1].")
    flattened = " ".join(text.split())
    keep = max(1, round(len(flattened) * fraction))
    start = rng.randrange(len(flattened) - keep + 1)
    return flattened[start : start + keep]


def validate_prepared(directory):
    manifest = read_json(directory / "manifest.json")
    for split in SPLITS:
        if digest(directory / f"{split}.jsonl") != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Prepared split changed since manifest creation: {split}")
    return manifest


@torch.no_grad()
def build_cache(config, prepared, output, *, conditional=False):
    import random

    prepared, output = Path(prepared), Path(output)
    dataset = validate_prepared(prepared)
    runtime = Runtime(config)
    runtime.reset_peak()
    encoder = make_encoder(config, runtime)
    options = config["encoder"]
    if min(options["batch_size"], options["shard_size"]) < 1:
        raise ValueError("Encoder batch_size and shard_size must be positive.")
    signature = {
        "dataset_id": dataset["dataset_id"],
        "encoder": encoder.signature,
        "view_fraction": config["diffusion"]["view_fraction"] if conditional else None,
        "view_seed": config["seed"] if conditional else None,
        "view_algorithm": "flatten_random_contiguous_crop_v1" if conditional else None,
        "cache_format": 1,
    }
    cache_id = fingerprint(signature)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["cache_id"] != cache_id:
            raise ValueError("Cache provenance differs. Use a new output directory.")
        for files in manifest["splits"].values():
            for entry in files:
                if digest(output / entry["file"]) != entry["sha256"]:
                    raise ValueError(f"Damaged cache shard: {entry['file']}")
        return manifest

    manifest = {
        "cache_id": cache_id,
        **signature,
        "encoder_width": encoder.width,
        "versions": versions(),
        "splits": {},
    }
    run = uuid4().hex[:12]
    for split in SPLITS:
        entries = []
        rows = read_jsonl(prepared / f"{split}.jsonl")
        records = []
        rng = random.Random(f"{config['seed']}:{split}")

        def write_shard():
            filename = f"{run}-{split}-{len(entries):05d}.pt"
            destination = output / filename
            save_tensors(destination, records)
            entries.append({"file": filename, "count": len(records), "sha256": digest(destination)})
            records.clear()
            if len(entries) % 10 == 0:
                done = sum(entry["count"] for entry in entries)
                print(f"features {split}: {done}/{dataset['splits'][split]['count']}", flush=True)

        while batch := take(rows, options["batch_size"]):
            texts = [row["text"] for row in batch]
            hidden, mask = encoder(texts)
            semantics = semantic_embedding(hidden, mask).cpu()
            lengths = mask.sum(1).tolist()
            hidden = hidden.cpu()
            if conditional:
                views = [view_text(text, signature["view_fraction"], rng) for text in texts]
                views_hidden, views_mask = encoder(views)
                view_lengths = views_mask.sum(1).tolist()
                views_hidden = views_hidden.cpu()
            for i, (row, length) in enumerate(zip(batch, lengths, strict=True)):
                record = {
                    **row,
                    "hidden": hidden[i, :length].clone(),
                    "semantic": semantics[i].clone(),
                }
                if conditional:
                    record.update(
                        view=views[i], view_hidden=views_hidden[i, : view_lengths[i]].clone()
                    )
                records.append(record)
                if len(records) == options["shard_size"]:
                    write_shard()
        if records:
            write_shard()
        manifest["splits"][split] = entries
    manifest["memory"] = runtime.memory()
    write_json(manifest_path, manifest)
    return manifest


def take(iterator, count):
    from itertools import islice

    return list(islice(iterator, count))


class FeatureDataset(Dataset):
    def __init__(self, directory, split, domain=None):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        self.entries = self.manifest["splits"][split]
        self.ends = []
        self.loaded = OrderedDict()
        self.verified = set()
        self.indices = None
        count = 0
        for entry in self.entries:
            count += entry["count"]
            self.ends.append(count)
        if domain is not None:
            self.indices = []
            start = 0
            for i, end in enumerate(self.ends):
                self.indices.extend(
                    start + j for j, row in enumerate(self._shard(i)) if row["domain"] == domain
                )
                start = end

    def shuffle_indices(self, rng):
        ranges = [[] for _ in self.ends]
        global_indices = self.indices if self.indices is not None else range(len(self))
        for position, global_index in enumerate(global_indices):
            ranges[bisect_right(self.ends, global_index)].append(position)
        for shard in torch.randperm(len(ranges), generator=rng).tolist():
            values = ranges[shard]
            for index in torch.randperm(len(values), generator=rng).tolist():
                yield values[index]

    def _shard(self, index):
        if index not in self.loaded:
            path = self.directory / self.entries[index]["file"]
            if index not in self.verified:
                if digest(path) != self.entries[index]["sha256"]:
                    raise ValueError(f"Cache shard hash mismatch: {path}")
                self.verified.add(index)
            self.loaded[index] = load_tensors(path)
            if len(self.loaded) > 2:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]

    def __len__(self):
        return (
            len(self.indices) if self.indices is not None else (self.ends[-1] if self.ends else 0)
        )

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.indices is not None:
            index = self.indices[index]
        shard = bisect_right(self.ends, index)
        return self._shard(shard)[index - (self.ends[shard - 1] if shard else 0)]


def pad_features(tensors, device):
    lengths = torch.tensor([len(tensor) for tensor in tensors], device=device)
    values = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True).to(device)
    mask = torch.arange(values.shape[1], device=device)[None, :] < lengths[:, None]
    return values, mask
