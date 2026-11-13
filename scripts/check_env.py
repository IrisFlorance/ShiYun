import argparse
import json
from pathlib import Path

from ald.utils.runtime import environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = json.dumps(environment(), ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
