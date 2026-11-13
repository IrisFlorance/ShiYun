from collections import Counter, defaultdict
import csv
from itertools import combinations
from pathlib import Path
import random
import re

import torch

from ald.data.cache import FeatureDataset, pad_features
from ald.inference.pipeline import latent_similarity
from ald.training.diffusion import distribution_stats
from ald.utils.files import atomic_write, write_json, write_jsonl


def grams(text, width):
    text = "".join(ch for ch in text.casefold() if ch.isalnum())
    return [text[i : i + width] for i in range(max(0, len(text) - width + 1))]


def jaccard(left, right):
    return len(left & right) / len(left | right) if left or right else 0.0


class PoemIndex:
    def __init__(self, cache, expected_signature, dataset_id):
        dataset = FeatureDataset(cache, "train", "poem")
        if (
            dataset.manifest["encoder"] != expected_signature
            or dataset.manifest["dataset_id"] != dataset_id
        ):
            raise ValueError("Retrieval cache differs from the autoencoder's training corpus.")
        if not len(dataset):
            raise ValueError("Retrieval needs training poems.")
        self.rows = [(row["id"], row["text"]) for row in dataset]
        self.embeddings = torch.stack([row["semantic"] for row in dataset])
        self.ngrams = [set(grams(text, 4)) for _, text in self.rows]
        self.index = defaultdict(list)
        for index, values in enumerate(self.ngrams):
            for value in values:
                self.index[value].append(index)

    def retrieve(self, embedding):
        scores = self.embeddings @ embedding.float()
        index = scores.argmax().item()
        poem_id, text = self.rows[index]
        return {"poem_id": poem_id, "text": text, "retrieval_cosine": scores[index].item()}

    def nearest_overlap(self, candidate):
        counts = Counter(index for value in candidate for index in self.index.get(value, ()))
        if not counts:
            return 0.0
        return max(
            common / (len(candidate) + len(self.ngrams[index]) - common)
            for index, common in counts.items()
        )


def measure(source, output, index):
    trigrams = grams(output, 3)
    candidate = set(grams(output, 4))
    return {
        "chars": len(output),
        "lines": len(output.splitlines()),
        "empty": float(not output.strip()),
        "repeated_trigram_ratio": 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0,
        "source_4gram_jaccard": jaccard(candidate, set(grams(source, 4))),
        "train_4gram_nearest": index.nearest_overlap(candidate),
    }


def summarize(rows):
    methods = defaultdict(list)
    for row in rows:
        methods[row["method"]].append(row)
    summary = {}
    for method, group in methods.items():
        metrics = {
            key: sum(row["metrics"][key] for row in group) / len(group)
            for key in group[0]["metrics"]
        }
        by_source = defaultdict(list)
        for row in group:
            by_source[row["source_id"]].append(row["text"])
        diversity = [
            1 - jaccard(set(grams(left, 3)), set(grams(right, 3))) if left != right else 0.0
            for outputs in by_source.values()
            for left, right in combinations(outputs, 2)
        ]
        summary[method] = {
            "count": len(group),
            "distinct_output_ratio": len({row["text"] for row in group}) / len(group),
            "seed_pair_3gram_distance": sum(diversity) / len(diversity) if diversity else None,
            **metrics,
        }
    return summary


def write_blind(out, rows, seed):
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    fields = [
        "candidate_id",
        "source",
        "candidate",
        "fluency_1_5",
        "imagery_1_5",
        "aesthetic_relation_1_5",
        "perceived_copy_1_5",
        "notes",
    ]

    def write(path):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=fields)
            csv_writer.writeheader()
            for i, row in enumerate(ordered, 1):
                csv_writer.writerow(
                    {
                        "candidate_id": f"P{i:05d}",
                        "source": row["source"],
                        "candidate": row["text"],
                    }
                )

    atomic_write(out / "blind_rating.csv", write)
    write_jsonl(
        out / "blind_key.jsonl",
        ({"candidate_id": f"P{i:05d}", **row} for i, row in enumerate(ordered, 1)),
    )


def pca(vectors):
    flat = vectors.flatten(1).float()
    centered = flat - flat.mean(0)
    _, singular, basis = torch.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ basis[:2].T
    if coordinates.shape[1] < 2:
        coordinates = torch.nn.functional.pad(coordinates, (0, 2 - coordinates.shape[1]))
    fraction = singular[:2].square() / singular.square().sum().clamp_min(1e-12)
    return coordinates, fraction.tolist()


def scatter(points, path):
    import html

    labels = sorted({row["group"] for row in points})
    palette = ["#246a80", "#e58937", "#9054a0", "#2f8b60", "#59576c"]
    bounds = [
        (min(row[key] for row in points), max(row[key] for row in points)) for key in ("x", "y")
    ]

    def axis(value, index):
        low, high = bounds[index]
        return (value - low) / max(1e-9, high - low)

    circles = []
    for row in points:
        x, y = 25 + 515 * axis(row["x"], 0), 340 - 310 * axis(row["y"], 1)
        color = palette[labels.index(row["group"]) % len(palette)]
        title = html.escape(f"{row['group']} {row['id']}")
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" opacity=".7"><title>{title}</title></circle>'
        )
    legend = [
        f'<text x="555" y="{32 + i * 22}" fill="{palette[i % len(palette)]}">{html.escape(label)}</text>'
        for i, label in enumerate(labels)
    ]
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="740" height="365">'
        '<rect width="100%" height="100%" fill="white"/>' + "".join(circles + legend) + "</svg>",
        encoding="utf-8",
    )


@torch.no_grad()
def inspect_latents(engine, cache, inputs, encodings, samples, output, *, umap=False):
    ids = [(row["id"], "probe") for row in inputs]
    latents = [entry.latent for entry in encodings]
    domain_latents = defaultdict(list)
    for split in ("train", "val", "test"):
        for domain in ("poem", "prose"):
            dataset = FeatureDataset(cache, split, domain)
            for start in range(0, min(len(dataset), 48), 8):
                rows = [dataset[i] for i in range(start, min(start + 8, 48, len(dataset)))]
                features = pad_features([row["hidden"] for row in rows], engine.runtime.device)
                projected = engine.projector(*features).cpu()
                ids.extend((row["id"], domain) for row in rows)
                latents.extend(projected)
                domain_latents[domain].extend(projected)
    ids.extend((str(i), "unconditional") for i in range(len(samples)))
    latents.extend(samples)
    vectors = torch.stack(latents)
    coordinates, fraction = pca(vectors)

    def export(name, coords):
        points = [
            {"id": ident, "group": group, "x": x, "y": y}
            for (ident, group), (x, y) in zip(ids, coords.tolist(), strict=True)
        ]
        write_json(output / f"{name}.json", points)
        scatter(points, output / f"{name}.svg")

    export("pca", coordinates)
    if umap:
        from umap import UMAP

        if len(vectors) < 4:
            raise ValueError("UMAP requires at least four latent samples.")
        coordinates = UMAP(
            random_state=23, n_jobs=1, n_neighbors=min(15, len(vectors) - 1), init="random"
        ).fit_transform(vectors.flatten(1).numpy())
        export("umap", coordinates)
    report = {
        "pca_explained_variance_ratio": fraction,
        "domains": {
            name: distribution_stats(torch.stack(items)) for name, items in domain_latents.items()
        },
    }
    if engine.prior is not None:
        normalizer = engine.prior[2].cpu()
        report["normalized_domain_rms"] = {
            name: normalizer.normalize(torch.stack(items)).square().mean().sqrt().item()
            for name, items in domain_latents.items()
        }
        normalizer.to(engine.runtime.device)
    return report


def perturb(text):
    return re.sub(r"[^\w\s]", "", " ".join(text.split())).strip() or text


@torch.no_grad()
def conditional_view_control(engine, cache, seed, steps, max_tokens):
    heldout = FeatureDataset(cache, "test", "poem")
    metadata = heldout.manifest
    if len(heldout) < 2 or metadata["view_fraction"] is None:
        return {"status": "requires at least two held-out poems and a --views cache"}
    identity = engine.conditional[3]
    if metadata["view_fraction"] != identity["config"]["diffusion"]["view_fraction"]:
        return {"status": "view fraction differs from conditional training"}
    rows = [heldout[i] for i in range(min(len(heldout), 12))]
    features = pad_features([row["view_hidden"] for row in rows], engine.runtime.device)
    sources = engine.projector(*features).cpu()
    controls = []
    for i, row in enumerate(rows):
        retained = set(grams(row["view"], 4))
        target = set(grams(row["text"], 4))
        visible = target & retained
        hidden = target - retained
        for label, source in (("matched_view", sources[i]), ("shuffled_view", sources[i - 1])):
            latent = engine.transform(source, mode="conditional", seed=seed, steps=steps)
            text = engine.decode(latent, seed=seed, temperature=0, max_tokens=max_tokens)
            output = set(grams(text, 4))
            controls.append(
                {
                    "poem_id": row["id"],
                    "condition": label,
                    "view": row["view"],
                    "target": row["text"],
                    "output": text,
                    "visible_target_4gram_recall": len(output & visible) / len(visible)
                    if visible
                    else None,
                    "hidden_target_4gram_recall": len(output & hidden) / len(hidden)
                    if hidden
                    else None,
                }
            )
    return {"status": "evaluated", "rows": controls}


def evaluate(
    engine,
    cache,
    inputs,
    out,
    *,
    seeds,
    strengths,
    main_strength=0.5,
    steps=None,
    temperature=0.8,
    top_p=0.95,
    max_tokens=64,
    umap=False,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not seeds or len(inputs) < 2 or len({row["id"] for row in inputs}) != len(inputs):
        raise ValueError("Evaluation needs at least two unique input IDs and one seed.")
    if engine.prior is None:
        raise ValueError("The A–F comparison needs an unconditional diffusion checkpoint.")
    index = PoemIndex(cache, engine.encoder_signature, engine.dataset_id)
    text = [row["text"] for row in inputs]
    encoded = engine.encode(text + [perturb(source) for source in text])
    encodings, altered = encoded[: len(text)], encoded[len(text) :]
    permutation = list(range(len(text)))
    random.Random(seeds[0]).shuffle(permutation)
    shuffled = dict(zip(permutation, permutation[1:] + permutation[:1], strict=True))
    variants = [
        ("B_direct", "direct", 0),
        ("C_unconditional", "unconditional", 1),
        ("D_sdedit", "sdedit", main_strength),
        ("E_shuffled", "sdedit", main_strength),
    ]
    variants.extend(
        (f"F_strength_{strength:g}", "sdedit", strength)
        for strength in dict.fromkeys(strengths)
        if strength != main_strength
    )
    if engine.conditional is not None:
        variants += [
            ("G_conditional", "conditional", 1),
            ("G_conditional_shuffled", "conditional", 1),
        ]
    rows = []
    samples = []
    cached_unconditional = {}
    decoding = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    for i, (record, encoding) in enumerate(zip(inputs, encodings, strict=True)):
        source = record["text"]

        def record_result(method, output, **fields):
            rows.append(
                {
                    "source_id": record["id"],
                    "source": source,
                    "method": method,
                    "text": output,
                    "metrics": measure(source, output, index),
                    **fields,
                }
            )

        retrieved = index.retrieve(encoding.semantic)
        retrieved_text = retrieved.pop("text")
        record_result("A_retrieval", retrieved_text, **retrieved)
        for seed in seeds:
            for method, mode, strength in variants:
                used = shuffled[i] if method in {"E_shuffled", "G_conditional_shuffled"} else i
                if mode == "unconditional" and seed in cached_unconditional:
                    latent, output = cached_unconditional[seed]
                else:
                    latent = engine.transform(
                        encodings[used].latent, mode=mode, strength=strength, seed=seed, steps=steps
                    )
                    output = engine.decode(latent, seed=seed, **decoding)
                    if mode == "unconditional":
                        cached_unconditional[seed] = latent, output
                        samples.append(latent)
                record_result(
                    method,
                    output,
                    seed=seed,
                    strength=strength,
                    condition_id=inputs[used]["id"] if mode != "unconditional" else None,
                    source_latent_cosine=latent_similarity(encoding.latent, latent),
                )
    write_jsonl(out / "generations.jsonl", rows)
    write_blind(out, rows, seeds[0])
    representation = inspect_latents(engine, cache, inputs, encodings, samples, out, umap=umap)
    representation["perturbation_cosine"] = {
        record["id"]: latent_similarity(left.latent, right.latent)
        for record, left, right in zip(inputs, encodings, altered, strict=True)
    }
    left, right = encodings[0].latent, encodings[1].latent
    interpolation = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        z = left.lerp(right, fraction)
        interpolation.append(
            {
                "fraction": fraction,
                "to_first_cosine": latent_similarity(left, z),
                "text": engine.decode(z, seed=seeds[0], temperature=0, max_tokens=max_tokens),
            }
        )
    representation["interpolation"] = interpolation
    if engine.conditional:
        representation["conditional_view_control"] = conditional_view_control(
            engine, cache, seeds[0], steps, max_tokens
        )
    grouped = {(row["source_id"], row.get("seed"), row["method"]): row["text"] for row in rows}
    changes = {
        f"{left}_vs_{right}": sum(
            grouped[row["id"], seed, left] != grouped[row["id"], seed, right]
            for row in inputs
            for seed in seeds
        )
        / (len(inputs) * len(seeds))
        for left, right in [("D_sdedit", "B_direct"), ("D_sdedit", "E_shuffled")]
    }
    if engine.conditional:
        changes["G_conditional_vs_shuffled"] = sum(
            grouped[row["id"], seed, "G_conditional"]
            != grouped[row["id"], seed, "G_conditional_shuffled"]
            for row in inputs
            for seed in seeds
        ) / (len(inputs) * len(seeds))
    report = {
        "model": engine.description,
        "dataset_id": engine.dataset_id,
        "settings": {
            "seeds": seeds,
            "strengths": strengths,
            "main_strength": main_strength,
            "steps": steps,
            **decoding,
        },
        "methods": summarize(rows),
        "paired_output_change_rate": changes,
        "representation": representation,
        "memory": engine.runtime.memory(),
    }
    write_json(out / "report.json", report)
    return report
