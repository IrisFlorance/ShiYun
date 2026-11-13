from ald.config import load_config, merge
from ald.data.cache import build_cache
from ald.data.corpus import prepare
from ald.evaluation.compare import evaluate
from ald.inference.pipeline import GenerationPipeline
from ald.training.autoencoder import fit_autoencoder
from ald.training.diffusion import fit_diffusion
from ald.utils.files import load_tensors

import torch


def test_full_training_resume_and_generation(tmp_path):
    data, cache = tmp_path / "prepared", tmp_path / "cache"
    prepare([("data/mock/poems.jsonl", "poem"), ("data/mock/prose.jsonl", "prose")], data)
    config = merge(
        load_config("configs/smoke.yaml"),
        {
            "autoencoder_train": {"epochs": 1},
            "diffusion_train": {"epochs": 1},
        },
    )
    first = build_cache(config, data, cache, conditional=True)
    assert first == build_cache(config, data, cache, conditional=True)
    ae = tmp_path / "ae"
    fit_autoencoder(config, cache, ae)
    extended = merge(config, {"autoencoder_train": {"epochs": 2}})
    history = fit_autoencoder(extended, cache, ae, resume=True)
    assert len(history) == 2
    assert history[0]["swap_source"] == "val"
    uninterrupted = tmp_path / "uninterrupted"
    fit_autoencoder(extended, cache, uninterrupted)
    actual = load_tensors(ae / "last.pt")
    expected = load_tensors(uninterrupted / "last.pt")
    assert actual["history"] == expected["history"]
    for name, param in actual["model"].items():
        torch.testing.assert_close(param, expected["model"][name], rtol=0, atol=0)

    prior = tmp_path / "prior"
    fit_diffusion(config, cache, ae / "best.pt", prior)
    resumed = fit_diffusion(
        merge(config, {"diffusion_train": {"epochs": 2}}), cache, ae / "best.pt", prior, resume=True
    )
    assert len(resumed) == 2
    conditional = tmp_path / "conditional"
    fit_diffusion(
        merge(config, {"diffusion": {"conditional": True}}), cache, ae / "best.pt", conditional
    )
    engine = GenerationPipeline(
        ae / "best.pt", prior / "best.pt", conditional_path=conditional / "best.pt"
    )
    text = "窗外正在下雨。"
    first = engine.generate(text, seed=331, count=2, max_tokens=12, temperature=0.8)
    assert first == engine.generate(text, seed=331, count=2, max_tokens=12, temperature=0.8)
    assert all(isinstance(row["text"], str) for row in first)
    assert first[0]["text"] != first[1]["text"]
    assert first[0]["input_latent"] != first[0]["output_latent"]
    direct = engine.generate(text, mode="direct", seed=331, max_tokens=12)
    zero = engine.generate(text, mode="sdedit", strength=0, seed=331, max_tokens=12)
    assert direct[0]["text"] == zero[0]["text"]
    assert zero[0]["input_latent"] == zero[0]["output_latent"]
    guided = engine.generate(text, mode="conditional", seed=331, max_tokens=12)
    assert guided[0]["output_latent"] != first[0]["output_latent"]

    probes = [{"id": "rain", "text": text}, {"id": "boat", "text": "最后一班船开走了。"}]
    report = evaluate(
        engine,
        cache,
        probes,
        tmp_path / "evaluation",
        seeds=[331],
        strengths=[0.2, 0.8],
        max_tokens=8,
    )
    assert set(report["methods"]) == {
        "A_retrieval",
        "B_direct",
        "C_unconditional",
        "D_sdedit",
        "E_shuffled",
        "F_strength_0.2",
        "F_strength_0.8",
        "G_conditional",
        "G_conditional_shuffled",
    }
    blind = (tmp_path / "evaluation/blind_rating.csv").read_text(encoding="utf-8-sig")
    assert "aesthetic_relation_1_5" in blind and "G_conditional" not in blind
    assert (tmp_path / "evaluation/pca.svg").is_file()
