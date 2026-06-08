from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(data), fh, indent=2, sort_keys=True)


def deep_copy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(config))


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def with_overrides(config: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    copied = deep_copy_config(config)
    return deep_update(copied, updates)


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is expected in this project.
        np = None

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
