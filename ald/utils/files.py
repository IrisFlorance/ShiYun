from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def digest(path):
    with Path(path).open("rb") as handle:
        return sha256_stream(handle)


def sha256_stream(handle):
    value = sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(
        path,
        lambda temp: temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        ),
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    def write(temp):
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    atomic_write(path, write)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected an object")
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            yield row


def save_tensors(path, payload):
    atomic_write(path, lambda temp: torch.save(payload, temp))


def load_tensors(path):
    return torch.load(path, map_location="cpu", weights_only=True)
