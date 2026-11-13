import argparse
import os
from pathlib import Path

from ald.inference.pipeline import GenerationPipeline


def heatmap(latent, label):
    if latent is None:
        return ""
    maximum = max((abs(value) for row in latent for value in row), default=1)
    maximum = max(maximum, 1e-9)
    cells = []
    for i, row in enumerate(latent):
        for j, value in enumerate(row):
            color = "#286c96" if value >= 0 else "#bc5249"
            cells.append(
                f'<rect x="{j}" y="{i}" width="1" height="1" fill="{color}" '
                f'fill-opacity="{min(1, abs(value) / maximum):.3f}">'
                f"<title>token {i + 1}, dim {j + 1}, {value:.4f}</title></rect>"
            )
    return (
        f"<div>{label}（±{maximum:.3f}）</div>"
        f'<svg role="img" aria-label="{label}" width="100%" height="112" '
        f'viewBox="0 0 {len(latent[0])} {len(latent)}" preserveAspectRatio="none">'
        + "".join(cells)
        + "</svg>"
    )


def build_app(root, default_ae=None, default_diffusion=None, device=None):
    import gradio as gr
    import torch

    from ald.utils.files import digest, load_tensors

    ae_choices, diffusion_choices, ae_hashes = [], [], {}
    for path in sorted(Path(root).rglob("best.pt")):
        try:
            identity = load_tensors(path)["identity"]
        except (OSError, RuntimeError, ValueError, KeyError):
            continue
        if identity["stage"] == "autoencoder":
            ae_choices.append(str(path))
            ae_hashes[str(path)] = digest(path)
        elif identity["stage"] == "diffusion":
            diffusion_choices.append(
                (str(path), identity["ae_sha256"], identity["config"]["diffusion"]["conditional"])
            )
    if default_ae:
        default_ae = str(Path(default_ae))
        if default_ae not in ae_choices:
            ae_choices.append(default_ae)
            ae_hashes[default_ae] = digest(default_ae)
    if default_diffusion:
        default_diffusion = str(Path(default_diffusion))
        if default_diffusion not in [name for name, _, _ in diffusion_choices]:
            identity = load_tensors(default_diffusion)["identity"]
            diffusion_choices.append(
                (
                    default_diffusion,
                    identity["ae_sha256"],
                    identity["config"]["diffusion"]["conditional"],
                )
            )

    def choices(ae):
        return [
            (path, conditional)
            for path, ae_hash, conditional in diffusion_choices
            if ae_hashes.get(ae) == ae_hash
        ]

    selected_ae = default_ae or next(iter(ae_choices), None)
    compatible = choices(selected_ae)
    selected_diff = default_diffusion or next(
        (path for path, conditional in compatible if not conditional), None
    )
    if selected_diff not in [path for path, _ in compatible]:
        selected_diff = None
    engine = None
    active = None

    def generate(
        text, ae, diffusion, mode, seed, count, strength, steps, temp, top_p, limit, compare
    ):
        nonlocal engine, active
        try:
            if not ae:
                raise ValueError("请先训练并选择一个自编码器 checkpoint。")
            if active != (ae, diffusion):
                engine = active = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                engine = GenerationPipeline(ae, diffusion, device=device)
                active = (ae, diffusion)
            options = {
                "mode": mode,
                "seed": int(seed),
                "count": int(count),
                "strength": strength,
                "steps": int(steps),
                "temperature": temp,
                "top_p": top_p,
                "max_tokens": int(limit),
            }
            rows = engine.generate(text, **options)
            outputs = "\n\n".join(f"[seed {row['seed']}]\n{row['text']}" for row in rows)
            comparison = ""
            if compare and mode != "direct" and text.strip():
                direct = engine.generate(text, **{**options, "mode": "direct"})
                comparison = "\n\n".join(f"[seed {row['seed']}]\n{row['text']}" for row in direct)
            first = rows[0]
            model_info = {
                **engine.description,
                "candidates": [
                    {"seed": row["seed"], "latent_cosine": row["similarity"]} for row in rows
                ],
                "memory": engine.runtime.memory(),
            }
            return (
                outputs,
                comparison,
                heatmap(first["input_latent"], "输入 latent"),
                heatmap(first["output_latent"], "输出 latent"),
                model_info,
            )
        except (ValueError, OSError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    def update_diffusion(ae):
        allowed = choices(ae)
        return gr.update(
            choices=[path for path, _ in allowed],
            value=next(
                (path for path, conditional in allowed if not conditional),
                next((path for path, _ in allowed), None),
            ),
        )

    with gr.Blocks(title="Aesthetic Latent Diffusion") as interface:
        gr.Markdown("# Aesthetic Latent Diffusion")
        text = gr.Textbox(label="输入文本", lines=5, value="窗外正在下雨。")
        with gr.Row():
            ae = gr.Dropdown(ae_choices, value=selected_ae, label="自编码器 checkpoint")
            diffusion = gr.Dropdown(
                [path for path, _ in compatible], value=selected_diff, label="扩散 checkpoint"
            )
            mode = gr.Dropdown(
                [
                    ("输入引导 SDEdit", "sdedit"),
                    ("直接解码", "direct"),
                    ("无条件扩散", "unconditional"),
                    ("多视图条件扩散", "conditional"),
                ],
                value="sdedit" if selected_diff else "direct",
                label="生成模式",
            )
        with gr.Row():
            seed = gr.Number(value=71, precision=0, label="随机种子")
            count = gr.Slider(1, 8, value=2, step=1, label="生成数量")
            strength = gr.Slider(0, 1, value=0.5, step=0.01, label="扩散强度（SDEdit）")
            steps = gr.Slider(1, 100, value=24, step=1, label="扩散步数")
        with gr.Row():
            temp = gr.Slider(0, 2, value=0.8, step=0.01, label="Temperature")
            top_p = gr.Slider(0.01, 1, value=0.95, step=0.01, label="Top-p")
            limit = gr.Slider(1, 512, value=96, step=1, label="最大输出 token")
        button = gr.Button("生成", variant="primary")
        poems = gr.Textbox(label="生成候选", lines=10)
        with gr.Accordion("实验对照", open=False):
            compare = gr.Checkbox(label="同时生成无扩散候选", value=True)
            direct = gr.Textbox(label="无扩散候选", lines=8)
            gr.Markdown("热图显示第一个候选。输入和输出分别使用各自的色标。")
            input_map = gr.HTML()
            output_map = gr.HTML()
            info = gr.JSON(label="模型、checkpoint 与 latent 余弦相似度")
        ae.change(update_diffusion, ae, diffusion)
        button.click(
            generate,
            [text, ae, diffusion, mode, seed, count, strength, steps, temp, top_p, limit, compare],
            [poems, direct, input_map, output_map, info],
            api_name="generate",
            concurrency_limit=1,
        )
    return interface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--ae")
    parser.add_argument("--diffusion")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    # Gradio probes its own HTTP endpoint during startup.
    bypass = os.environ.get("no_proxy") or os.environ.get("NO_PROXY", "")
    local_bypass = ",".join(filter(None, [bypass, "127.0.0.1", "localhost"]))
    os.environ["NO_PROXY"] = local_bypass
    os.environ["no_proxy"] = local_bypass
    interface = build_app(args.checkpoints, args.ae, args.diffusion, args.device)
    interface.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, show_error=True
    )


if __name__ == "__main__":
    main()
