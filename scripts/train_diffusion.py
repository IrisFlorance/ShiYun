import argparse

from ald.config import load_config
from ald.training.diffusion import fit_diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    fit_diffusion(load_config(args.config), args.cache, args.ae, args.out, resume=args.resume)


if __name__ == "__main__":
    main()
