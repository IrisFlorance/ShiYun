from ald.config import load_config
from ald.utils.runtime import Runtime, environment


def test_smoke_configuration():
    config = load_config("configs/smoke.yaml")
    assert config["backend"] == "mock"
    assert Runtime(config).device.type == "cpu"
    assert environment()["dependencies"]["torch"]


def test_conditional_and_low_vram_configs():
    assert load_config("configs/conditional.yaml")["diffusion"]["conditional"]
    assert load_config("configs/low_vram.yaml")["decoder"]["max_length"] == 128
