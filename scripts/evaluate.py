import argparse
import json

from ald.evaluation.compare import evaluate
from ald.inference.pipeline import GenerationPipeline
from ald.utils.files import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae", required=True)
    parser.add_argument("--diffusion", required=True)
    parser.add_argument("--conditional")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--inputs", default="data/mock/probes.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[71, 72, 73])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    parser.add_argument("--main-strength", type=float, default=0.5)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--umap", action="store_true")
    args = parser.parse_args()
    engine = GenerationPipeline(
        args.ae, args.diffusion, device=args.device, conditional_path=args.conditional
    )
    result = evaluate(
        engine,
        args.cache,
        list(read_jsonl(args.inputs)),
        args.out,
        seeds=args.seeds,
        strengths=args.strengths,
        main_strength=args.main_strength,
        steps=args.steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        umap=args.umap,
    )
    print(json.dumps(result["methods"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
