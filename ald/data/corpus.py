from collections import Counter, defaultdict
from hashlib import sha256
import math
from pathlib import Path
import random
import re
import unicodedata

from ald.utils.files import digest, fingerprint, read_jsonl, write_json, write_jsonl

SPLITS = ("train", "val", "test")


def normalize(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def read_source(path, domain):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        # A dedicated separator permits multiline works with empty stanza lines.
        rows = ({"text": piece} for piece in re.split(r"(?m)^\s*===\s*$", text))
    else:
        raise ValueError(f"Expected .jsonl or .txt: {path}")
    for number, row in enumerate(rows, 1):
        if not isinstance(row.get("text"), str):
            raise ValueError(f"{path}: record {number} requires a string text field.")
        metadata = {key: normalize(str(row.get(key) or "")) for key in ("work_id", "author")}
        for key in ("title", "form", "source", "source_revision", "license"):
            if row.get(key):
                metadata[key] = normalize(str(row[key]))
        yield {"text": normalize(row["text"]), "domain": domain, **metadata}


def shingles(text):
    joined = "".join(character for character in text.casefold() if character.isalnum())
    return {joined[i : i + 4] for i in range(max(1, len(joined) - 3))} if joined else {text}


def prepare(
    sources,
    output,
    *,
    seed=31415,
    min_chars=6,
    max_chars=3000,
    near=0.85,
    group_authors=True,
    stratify_form=False,
    train_poems=None,
):
    if not 0 < near <= 1 or not 0 < min_chars <= max_chars:
        raise ValueError("Invalid near-duplicate threshold or length limits.")
    if train_poems is not None and train_poems < 1:
        raise ValueError("train_poems must be positive.")
    output = Path(output)
    rows = []
    parents = []
    groups = {}
    exact = {}
    features = []
    removed = Counter()

    def root(number):
        while number != parents[number]:
            parents[number] = parents[parents[number]]
            number = parents[number]
        return number

    def union(left, right):
        parents[root(left)] = root(right)

    for path, domain in sources:
        for row in read_source(path, domain):
            if not min_chars <= len(row["text"]) <= max_chars:
                removed["length"] += 1
                continue
            number = len(rows)
            parents.append(number)
            text_key = re.sub(r"\s+", "", row["text"].casefold())
            if text_key in exact:
                union(number, exact[text_key])
                removed["exact"] += 1
                duplicate = True
            else:
                exact[text_key] = number
                duplicate = False
            for field in ("work_id", "author") if group_authors else ("work_id",):
                if row[field]:
                    key = (field, row[field].casefold())
                    union(number, groups.setdefault(key, number))
            row["id"] = sha256(f"{domain}\0{row['text']}".encode()).hexdigest()[:24]
            rows.append((row, duplicate))
            features.append(set() if duplicate else shingles(row["text"]))

    # Qualifying Jaccard pairs share a prefix token under a common ordering.
    # Index the rarest prefix tokens, then check exact Jaccard on candidates.
    frequencies = Counter(token for tokens in features for token in tokens)
    index = defaultdict(list)
    for number, tokens in enumerate(features):
        if not tokens:
            continue
        limit = len(tokens) - math.ceil(near * len(tokens) - 1e-10) + 1
        prefix = sorted(tokens, key=lambda token: (frequencies[token], token))[:limit]
        candidates = {previous for token in prefix for previous in index[token]}
        duplicate = False
        for previous in sorted(candidates):
            other = features[previous]
            if min(len(tokens), len(other)) < near * max(len(tokens), len(other)):
                continue
            common = len(tokens & other)
            if common / (len(tokens) + len(other) - common) >= near:
                union(number, previous)
                duplicate = True
        if duplicate:
            rows[number] = (rows[number][0], True)
            removed["near"] += 1
        for token in prefix:
            index[token].append(number)

    clusters = defaultdict(list)
    for number, (row, duplicate) in enumerate(rows):
        if not duplicate:
            clusters[root(number)].append(row)

    if not any(row["domain"] == "poem" for group in clusters.values() for row in group):
        raise ValueError("No poems passed preprocessing.")
    ordered = list(clusters.values())
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if stratify_form:
        ordered.sort(key=len, reverse=True)
    else:
        ordered.sort(key=lambda group: any(row["domain"] == "poem" for row in group), reverse=True)
    splits = {name: [] for name in SPLITS}
    targets = {"train": 0.8, "val": 0.1, "test": 0.1}
    poems = Counter()
    prose = Counter()
    strata = {name: Counter() for name in SPLITS}

    def stratum(row):
        form = row.get("form", "")
        if form in {"new_poem", "prose_poem"}:
            form = "modern"
        return row["domain"], form

    for group in ordered:
        poem_count = sum(row["domain"] == "poem" for row in group)
        counts = poems if poem_count else prose
        if stratify_form:
            composition = Counter(map(stratum, group))
            primary = composition.most_common(1)[0][0]
            split = min(
                SPLITS,
                key=lambda name: (
                    (strata[name][primary] + composition[primary] / 2) / targets[name],
                    counts[name] / targets[name],
                ),
            )
            strata[split].update(composition)
        else:
            split = min(SPLITS, key=lambda name: (counts[name] / targets[name], -targets[name]))
        splits[split].extend(group)
        poems[split] += poem_count
        prose[split] += len(group) - poem_count
    output.mkdir(parents=True, exist_ok=True)
    if train_poems is not None:
        train_rows = splits["train"]
        modern = [
            row
            for row in train_rows
            if row["domain"] == "poem" and row.get("form") in {"new_poem", "prose_poem"}
        ]
        remaining = [
            row
            for row in train_rows
            if row["domain"] == "poem" and row.get("form") not in {"new_poem", "prose_poem"}
        ]
        quota = train_poems - len(modern)
        if not 0 <= quota <= len(remaining):
            raise ValueError(
                f"Cannot select {train_poems} train poems from {len(modern) + len(remaining)} "
                f"eligible samples with {len(modern)} modern samples."
            )
        splits["train"] = (
            modern
            + rng.sample(remaining, quota)
            + [row for row in train_rows if row["domain"] == "prose"]
        )
        for records in splits.values():
            rng.shuffle(records)
    for split, records in splits.items():
        write_jsonl(output / f"{split}.jsonl", records)
    manifest = {
        "format": 1,
        "seed": seed,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "near_jaccard": near,
        "group_authors": group_authors,
        "removed": dict(removed),
        "sources": [
            {"path": str(path), "domain": domain, "sha256": digest(path)}
            for path, domain in sources
        ],
        "splits": {
            name: {
                "count": len(records),
                "domains": dict(Counter(r["domain"] for r in records)),
                "sha256": digest(output / f"{name}.jsonl"),
            }
            for name, records in splits.items()
        },
    }
    if stratify_form:
        manifest["stratify_form"] = True
        for name, records in splits.items():
            manifest["splits"][name]["forms"] = dict(Counter(r.get("form", "") for r in records))
    if train_poems is not None:
        manifest["train_poems"] = train_poems
    manifest["dataset_id"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


def describe(rows):
    counts = Counter()
    lengths = Counter()
    stanzas = Counter()
    for row in rows:
        domain = row["domain"]
        counts[domain] += 1
        lengths[domain, min(len(row["text"]) // 25, 40)] += 1
        stanzas[domain, row["text"].count("\n\n") + 1] += 1
    return {
        "counts": dict(counts),
        "length_bins": {
            domain: {f"{n * 25}+": count for (kind, n), count in lengths.items() if kind == domain}
            for domain in counts
        },
        "stanza_counts": {
            domain: {n: count for (kind, n), count in stanzas.items() if kind == domain}
            for domain in counts
        },
    }
