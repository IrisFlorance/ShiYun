import argparse

from ald.config import load_config
from ald.data.cache import build_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--views", action="store_true", help="Cache self-supervised poem views as well."
    )
    args = parser.parse_args()
    manifest = build_cache(load_config(args.config), args.data, args.out, conditional=args.views)
    print("cache_id =", manifest["cache_id"], "memory =", manifest["memory"])


if __name__ == "__main__":
    main()
