import argparse
import html
import json
from pathlib import Path

from ald.data.corpus import SPLITS, describe, prepare
from ald.utils.files import read_jsonl, write_json


def plot(stats, path):
    rows = []
    for split, summary in stats.items():
        for domain, counts in summary["length_bins"].items():
            for key, count in sorted(counts.items(), key=lambda pair: int(pair[0][:-1])):
                rows.append((f"{split} / {domain} / {key}", count))
    ceiling = max((count for _, count in rows), default=1)
    bars = []
    for i, (label, count) in enumerate(rows):
        y = 36 + i * 23
        bars.append(f'<text x="5" y="{y + 12}">{html.escape(label)}</text>')
        bars.append(
            f'<rect x="190" y="{y}" width="{300 * count / ceiling:.2f}" height="16" fill="#5969a9"/>'
        )
        bars.append(f'<text x="{198 + 300 * count / ceiling:.2f}" y="{y + 12}">{count}</text>')
    body = "".join(bars)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="560" height="{45 + 23 * len(rows)}">'
        f'<text x="5" y="20">Text length by split and domain</text>{body}</svg>',
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, nargs="+", required=True)
    parser.add_argument("--prose", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--min-chars", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--near", type=float, default=0.85)
    parser.add_argument("--work-only", action="store_true")
    parser.add_argument("--train-poems", type=int)
    parser.add_argument(
        "--stratify-form",
        action="store_true",
        help="Balance recorded forms during group splitting.",
    )
    args = parser.parse_args()
    sources = [(path, "poem") for path in args.poems] + [(path, "prose") for path in args.prose]
    manifest = prepare(
        sources,
        args.out,
        seed=args.seed,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        near=args.near,
        group_authors=not args.work_only,
        stratify_form=args.stratify_form,
        train_poems=args.train_poems,
    )
    stats = {split: describe(read_jsonl(args.out / f"{split}.jsonl")) for split in SPLITS}
    write_json(args.out / "stats.json", stats)
    plot(stats, args.out / "lengths.svg")
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
