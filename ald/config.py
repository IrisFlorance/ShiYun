from copy import deepcopy
from pathlib import Path

import yaml


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, visited=()):
    path = Path(path).resolve()
    if path in visited:
        raise ValueError(f"Circular config inheritance: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    parent = config.pop("extends", None)
    if parent:
        return merge(load_config(path.parent / parent, (*visited, path)), config)
    return config
