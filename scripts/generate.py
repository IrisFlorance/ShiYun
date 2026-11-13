import argparse
import json
import sys

from ald.inference.pipeline import GenerationPipeline
from ald.utils.files import write_json


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae", required=True)
    parser.add_argument("--diffusion")
    parser.add_argument("--conditional")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--text")
    parser.add_argument(
        "--mode", choices=["direct", "unconditional", "sdedit", "conditional"], default="sdedit"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument(
        "--out", help="Write the generation settings and full latent tensors as JSON."
    )
    args = parser.parse_args()
    if args.text is None and args.mode != "unconditional":
        if sys.stdin.isatty():
            parser.error("Supply --text or pipe UTF-8 text.")
        text = sys.stdin.read()
    else:
        text = args.text
    pipeline = GenerationPipeline(
        args.ae, args.diffusion, device=args.device, conditional_path=args.conditional
    )
    rows = pipeline.generate(
        text,
        mode=args.mode,
        seed=args.seed,
        count=args.count,
        strength=args.strength,
        steps=args.steps,
        cfg=args.cfg,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    if args.out:
        write_json(
            args.out,
            {
                "model": pipeline.description,
                "options": vars(args),
                "results": rows,
                "memory": pipeline.runtime.memory(),
            },
        )
    print(
        json.dumps(
            [
                {key: value for key, value in row.items() if not key.endswith("_latent")}
                for row in rows
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
