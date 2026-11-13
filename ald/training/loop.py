import math
from pathlib import Path
import random

import numpy as np
import torch

from ald.utils.files import load_tensors, save_tensors, write_jsonl
from ald.utils.runtime import generator, seed_everything, versions


def parameter_state(model):
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable
    }


def restore_parameters(model, state):
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    if expected != state.keys():
        raise ValueError(f"Checkpoint parameters differ: {sorted(expected ^ state.keys())[:10]}")
    model.load_state_dict(state, strict=False)


def random_state(runtime):
    numpy = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
        "cuda": torch.cuda.get_rng_state(runtime.device) if runtime.device.type == "cuda" else None,
    }


def restore_random(state, runtime):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    kind, keys, *rest = state["numpy"]
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), *rest))
    if state["cuda"] is not None and runtime.device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], runtime.device)


def batches(dataset, size, *, seed=None, start_batch=0):
    if size < 1:
        raise ValueError("batch size must be positive.")
    if seed is None:
        indices = range(len(dataset))
    else:
        # Randomize batches and rows without keeping the whole feature cache in RAM.
        rng = generator("cpu", seed)
        indices = (
            dataset.shuffle_indices(rng)
            if hasattr(dataset, "shuffle_indices")
            else torch.randperm(len(dataset), generator=rng).tolist()
        )
    from itertools import islice

    iterator = islice(iter(indices), start_batch * size, None)
    while chunk := list(islice(iterator, size)):
        yield [dataset[i] for i in chunk]


def train(model, datasets, options, runtime, identity, out, train_loss, validate, *, resume=False):
    out = Path(out)
    if min(map(len, datasets)) < 1:
        raise ValueError("Training and validation each need at least one poem after grouping.")
    if (
        min(options["accumulation"], options["epochs"], options["patience"]) < 1
        or options["lr"] <= 0
    ):
        raise ValueError("Training accumulation, epochs, patience and lr must be positive.")
    last_path = out / "last.pt"
    seed = identity["config"]["seed"]
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=options["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.dtype == torch.float16)
    history = []
    best = float("inf")
    stale = 0
    start = 0
    partial = None
    if resume:
        saved = load_tensors(last_path)
        if saved["identity"] != identity:
            raise ValueError(
                "Resume requires the same model, data, device type, precision, and options (epochs may increase)."
            )
        restore_parameters(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_random(saved["rng"], runtime)
        start, best, stale, history = (
            saved["epoch"],
            saved["best"],
            saved["stale"],
            saved["history"],
        )
        partial = saved.get("partial_epoch")
    elif last_path.exists():
        raise FileExistsError(
            f"{last_path} already exists. Use --resume or a new output directory."
        )
    else:
        seed_everything(seed + 1000)

    train_set, val_set = datasets
    microbatches = math.ceil(len(train_set) / options["batch_size"])
    accumulation = options["accumulation"]
    log_every = options.get("log_every", 0)
    checkpoint_every = options.get("checkpoint_every", 0)

    def snapshot(completed_epochs):
        return {
            "identity": identity,
            "model": parameter_state(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": random_state(runtime),
            "history": history,
            "epoch": completed_epochs,
            "best": best,
            "stale": stale,
            "versions": versions(),
            "planned_epochs": options["epochs"],
        }

    for epoch in range(start, options["epochs"]):
        if stale >= options["patience"]:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        numerator = denominator = 0
        first_batch = 0
        if partial is not None:
            first_batch = partial["next_batch"]
            numerator, denominator = partial["numerator"], partial["denominator"]
            if not 0 < first_batch < microbatches or first_batch % accumulation:
                raise ValueError("Checkpoint is outside a valid optimizer boundary.")
            partial = None
        epoch_batches = batches(
            train_set, options["batch_size"], seed=seed + epoch, start_batch=first_batch
        )
        for step, batch in enumerate(epoch_batches, start=first_batch):
            group_size = min(accumulation, microbatches - (step // accumulation) * accumulation)
            with runtime.autocast():
                loss, total, count = train_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss. Resume from the last complete epoch after diagnosis."
                )
            scaler.scale(loss / group_size).backward()
            numerator += float(total)
            denominator += int(count)
            if (step + 1) % accumulation == 0 or step + 1 == microbatches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    params, 1.0, error_if_nonfinite=not scaler.is_enabled()
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update = (step + accumulation) // accumulation
                if checkpoint_every and update % checkpoint_every == 0 and step + 1 < microbatches:
                    payload = snapshot(epoch)
                    payload["partial_epoch"] = {
                        "next_batch": step + 1,
                        "numerator": numerator,
                        "denominator": denominator,
                    }
                    save_tensors(last_path, payload)
                    print(
                        f"epoch {epoch + 1} batch {step + 1}: recovery checkpoint saved", flush=True
                    )
            if log_every and (step + 1) % log_every == 0:
                print(
                    f"epoch {epoch + 1} batch {step + 1}/{microbatches}: "
                    f"train={numerator / denominator:.4f} memory={runtime.memory()}",
                    flush=True,
                )
        model.eval()
        with torch.no_grad(), runtime.autocast():
            metrics = validate(val_set)
        record = {
            "epoch": epoch + 1,
            "train_loss": numerator / denominator,
            **metrics,
            **runtime.memory(),
        }
        if not math.isfinite(record["val_loss"]):
            raise FloatingPointError("Non-finite validation loss.")
        improved = record["val_loss"] < best
        best = min(best, record["val_loss"])
        stale = 0 if improved else stale + 1
        history.append(record)
        payload = snapshot(epoch + 1)
        # Best is written before last so an interrupted commit can safely replay the epoch.
        if improved:
            artifact = {
                key: payload[key]
                for key in ("identity", "model", "epoch", "versions", "planned_epochs")
            }
            save_tensors(out / "best.pt", artifact)
        save_tensors(last_path, payload)
        write_jsonl(out / "metrics.jsonl", history)
        print(
            f"epoch {epoch + 1}: train={record['train_loss']:.4f} val={record['val_loss']:.4f}",
            flush=True,
        )
    return history


def run_identity(config, stage, runtime, provenance):
    # Save resolved config, excluding only requested epoch count from resume compatibility.
    normalized = {
        **config,
        f"{stage}_train": {
            key: value for key, value in config[f"{stage}_train"].items() if key != "epochs"
        },
    }
    return {
        "stage": stage,
        "config": normalized,
        "device_type": runtime.device.type,
        "dtype": str(runtime.dtype),
        **provenance,
    }
