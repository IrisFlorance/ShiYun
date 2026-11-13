from pathlib import Path

from ald.config import load_config
from ald.data.cache import build_cache
from ald.data.corpus import prepare
from ald.evaluation.compare import evaluate
from ald.inference.pipeline import GenerationPipeline
from ald.training.autoencoder import fit_autoencoder
from ald.training.diffusion import fit_diffusion
from ald.utils.files import read_jsonl, write_json


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the full offline ALD CPU smoke experiment.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--umap", action="store_true")
    args = parser.parse_args()
    out = args.out
    config = load_config("configs/smoke.yaml")
    conditional = load_config("configs/smoke_conditional.yaml")
    prepared, features, ae, prior, cond = (
        out / name for name in ("data", "cache", "ae", "ddim", "conditional")
    )
    dataset = prepare(
        [("data/mock/poems.jsonl", "poem"), ("data/mock/prose.jsonl", "prose")], prepared
    )
    build_cache(config, prepared, features, conditional=True)
    fit_autoencoder(config, features, ae)
    fit_diffusion(config, features, ae / "best.pt", prior)
    fit_diffusion(conditional, features, ae / "best.pt", cond)
    engine = GenerationPipeline(
        ae / "best.pt", prior / "best.pt", conditional_path=cond / "best.pt"
    )
    result = engine.generate("窗外正在下雨。", seed=71, count=2, max_tokens=24)
    write_json(out / "generation.json", result)
    inputs = list(read_jsonl("data/mock/probes.jsonl"))
    evaluate(
        engine,
        features,
        inputs,
        out / "evaluation",
        seeds=[71, 72],
        strengths=[0.2, 0.5, 0.8],
        max_tokens=24,
        umap=args.umap,
    )
    print("Smoke complete:", out, dataset["dataset_id"])


if __name__ == "__main__":
    main()
