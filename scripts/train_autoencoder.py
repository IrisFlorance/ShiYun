import argparse

from ald.config import load_config
from ald.training.autoencoder import fit_autoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    fit_autoencoder(load_config(args.config), args.cache, args.out, resume=args.resume)


if __name__ == "__main__":
    main()
