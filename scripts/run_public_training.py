"""Run the first real-weight experiment as resumable, separate CUDA processes."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from ald.utils.files import read_json, write_json

VERSE_FORMS = {"tang", "song_poetry", "song_ci", "shijing", "chuci", "new_poem"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/public-v1"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints/public-v1"))
    parser.add_argument("--experiments", type=Path, default=Path("experiments/public-v1"))
    parser.add_argument("--config", default="configs/public_v1.yaml")
    parser.add_argument("--conditional-config", default="configs/public_v1_conditional.yaml")
    parser.add_argument("--train-poems", type=int)
    parser.add_argument("--poetry-only", action="store_true")
    args = parser.parse_args()
    root = args.checkpoints
    experiment = args.experiments
    data, cache = root / "prepared", root / "cache"
    ae, ddim, conditional = root / "ae", root / "ddim", root / "conditional"
    status_path = experiment / "status.json"
    status = read_json(status_path) if status_path.exists() else {}

    def run(stage, script, *options):
        if status.get(stage, {}).get("state") == "complete":
            print(f"{stage}: already complete", flush=True)
            return
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-u",
            "-m",
            f"scripts.{script}",
            *map(str, options),
        ]
        log_path = experiment / f"{stage}.log"
        record = {"state": "running", "command": command, "started": now(), "log": str(log_path)}
        status[stage] = record
        write_json(status_path, status)
        print(f"{stage}: started at {record['started']}", flush=True)
        with (
            log_path.open("a", encoding="utf-8") as log,
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            ) as process,
        ):
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            exit_code = process.wait()
        record.update(
            state="complete" if exit_code == 0 else "failed",
            finished=now(),
            exit_code=exit_code,
        )
        write_json(status_path, status)
        if exit_code:
            raise SystemExit(exit_code)

    run(
        "prepare",
        "prepare_data",
        "--poems",
        args.raw / "poems.jsonl",
        *([] if args.poetry_only else ["--prose", args.raw / "prose.jsonl"]),
        "--out",
        data,
        "--stratify-form",
        "--min-chars",
        12,
        "--max-chars",
        520,
        *(["--train-poems", args.train_poems] if args.train_poems else []),
    )
    splits = read_json(data / "manifest.json")["splits"]
    for name, minimum in (("train", 800), ("val", 75), ("test", 75)):
        forms = splits[name]["forms"]
        if args.poetry_only and (
            set(splits[name]["domains"]) != {"poem"} or not set(forms) <= VERSE_FORMS
        ):
            raise RuntimeError(f"{name}: unexpected prose or theatre in a poetry-only run.")
        if splits[name]["domains"].get("poem", 0) < minimum:
            raise RuntimeError(f"{name}: insufficient poems for this experiment")
        modern_minimum = 2 if args.poetry_only and name == "test" else 4
        if forms.get("new_poem", 0) + forms.get("prose_poem", 0) < modern_minimum:
            raise RuntimeError(
                f"{name}: too few modern poems. Inspect author groups before training."
            )
    if args.train_poems and splits["train"]["domains"]["poem"] != args.train_poems:
        raise RuntimeError("Prepared train poem count differs from requested count.")
    config = args.config
    run("cache", "cache_embeddings", "--config", config, "--data", data, "--out", cache, "--views")
    run(
        "autoencoder",
        "train_autoencoder",
        "--config",
        config,
        "--cache",
        cache,
        "--out",
        ae,
        *(["--resume"] if (ae / "last.pt").exists() else []),
    )
    run(
        "unconditional",
        "train_diffusion",
        "--config",
        config,
        "--cache",
        cache,
        "--ae",
        ae / "best.pt",
        "--out",
        ddim,
        *(["--resume"] if (ddim / "last.pt").exists() else []),
    )
    run(
        "conditional",
        "train_diffusion",
        "--config",
        args.conditional_config,
        "--cache",
        cache,
        "--ae",
        ae / "best.pt",
        "--out",
        conditional,
        *(["--resume"] if (conditional / "last.pt").exists() else []),
    )
    run(
        "evaluation",
        "evaluate",
        "--ae",
        ae / "best.pt",
        "--diffusion",
        ddim / "best.pt",
        "--conditional",
        conditional / "best.pt",
        "--cache",
        cache,
        "--out",
        experiment / "evaluation",
        "--seeds",
        71,
        72,
        "--max-tokens",
        48,
    )


def now():
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
