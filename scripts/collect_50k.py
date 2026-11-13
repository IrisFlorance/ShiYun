"""Build the 50k experiment corpus from a pinned classical snapshot and audited modern rows."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
from urllib.parse import quote
from zipfile import ZipFile

from ald.data.corpus import normalize
from ald.utils.files import digest, read_json, read_jsonl, write_json, write_jsonl
from scripts.collect_public_corpus import CLASSICAL_COMMIT, CLASSICAL_REPO

SOURCES = {
    "tang": (27000, r"全唐诗/poet\.tang\.\d+\.json"),
    "song_poetry": (27000, r"全唐诗/poet\.song\.(?:[1-6]?\d000|0)\.json"),
    "song_ci": (15500, r"宋词/ci\.song\.\d+\.json"),
    "shijing": (300, r"诗经/shijing\.json"),
    "chuci": (90, r"楚辞/chuci\.json"),
}
ANONYMOUS = {"佚名", "無名氏", "无名氏", "不詳", "不详", "无名", "無名"}
EDITORIAL_MARKUP = re.compile(r"[\[\]〔〕〖〗【】《》〈〉（）()]|詩題\s*[:：]|首句\s*[:：]")
CLAUSE_BOUNDARY = re.compile(r"[，。！？；、：,.!?;:\n]+")
HAN = re.compile(r"[\u3400-\u9fff]")


def verse_candidate(text):
    """Conservative screen for verse lines and obvious editorial or dialogue material."""
    if "\n" not in text or EDITORIAL_MARKUP.search(text):
        return False
    lengths = [len(HAN.findall(clause)) for clause in CLAUSE_BOUNDARY.split(text)]
    return max(lengths, default=0) <= 24


def classical_rows(archive, names, form):
    for entry, filename in names:
        url = f"{CLASSICAL_REPO}/{CLASSICAL_COMMIT}/{quote(filename)}"
        for i, record in enumerate(json.loads(archive.read(entry))):
            lines = record.get("paragraphs") or record.get("content")
            if isinstance(lines, str):
                lines = re.split(r"(?<=[。！？])", lines)
            if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
                continue
            text = normalize("\n".join(lines))
            title = record.get("title") or record.get("rhythmic") or ""
            if (
                not 16 <= len(text) <= 430
                or any(char in text for char in "□�")
                or title.startswith(("存目", "佚句"))
                or not verse_candidate(text)
            ):
                continue
            author = record.get("author") or ""
            yield {
                "text": text,
                "title": title,
                "author": "" if author in ANONYMOUS else author,
                "form": form,
                "work_id": f"chinese-poetry:{filename}:{i}",
                "source": url,
                "source_revision": CLASSICAL_COMMIT,
                "license": "Public-domain original; upstream MIT",
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--modern", type=Path, default=Path("data/raw/public-v1"))
    args = parser.parse_args()
    rng = random.Random(31415)
    sources = read_json(args.modern / "sources.json")
    for filename in ("poems", "prose"):
        if digest(args.modern / f"{filename}.jsonl") != sources[f"{filename}_sha256"]:
            raise ValueError(f"Audited source hash differs: {filename}")
    modern = [
        row
        for row in read_jsonl(args.modern / "poems.jsonl")
        if row["form"] in {"new_poem", "prose_poem"}
    ]
    rows = [
        row
        for row in modern
        if row["form"] == "new_poem"
        and verse_candidate(row["text"])
        and not row["work_id"].startswith(("wikisource:晨光集:", "wikisource:黑潮集:"))
    ]
    modern_excluded = Counter(row["form"] for row in modern) - Counter(row["form"] for row in rows)
    prose = list(read_jsonl(args.modern / "prose.jsonl"))
    with ZipFile(args.archive) as archive:
        names = [(name, name.split("/", 1)[1]) for name in archive.namelist() if "/" in name]
        license_file = next(entry for entry, name in names if name == "LICENSE")
        license_data = archive.read(license_file)
        if b"MIT License" not in license_data or not license_file.startswith(
            f"chinese-poetry-{CLASSICAL_COMMIT}/"
        ):
            raise ValueError("Expected the pinned chinese-poetry MIT archive.")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "LICENSE.chinese-poetry").write_bytes(license_data)
        for form, (target, pattern) in SOURCES.items():
            matched = sorted((entry, name) for entry, name in names if re.fullmatch(pattern, name))
            pool = list(classical_rows(archive, matched, form))
            rng.shuffle(pool)
            selected = pool[:target]
            if form in {"tang", "song_poetry", "song_ci"} and len(selected) != target:
                raise ValueError(f"Insufficient {form} rows: {len(selected)} / {target}")
            rows.extend(selected)
            print(f"{form}: {len(selected)} / {len(pool)} eligible", flush=True)
    write_jsonl(args.out / "poems.jsonl", rows)
    write_jsonl(args.out / "prose.jsonl", prose)
    write_json(
        args.out / "sources.json",
        {
            "classical_commit": CLASSICAL_COMMIT,
            "archive_sha256": digest(args.archive),
            "modern_manifest_sha256": digest(args.modern / "sources.json"),
            "modern_manifest": str(args.modern / "sources.json"),
            "modern_excluded": dict(modern_excluded),
            "selection": "verse_lines_v1; excludes yuan theatre and prose poem excerpts",
            "forms": dict(Counter(row["form"] for row in rows)),
            "poems_sha256": digest(args.out / "poems.jsonl"),
            "prose_sha256": digest(args.out / "prose.jsonl"),
        },
    )


if __name__ == "__main__":
    main()
