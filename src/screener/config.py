"""Configuration loading.

The default configuration ships with the package (``resources/config.yaml``).
A user file can override any subset of keys: dictionaries are merged
recursively, so a custom file only needs the values that change.
"""

from __future__ import annotations

import copy
import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

RESOURCES = resources.files("screener") / "resources"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the default config, optionally overridden by a user YAML file.

    ``path`` falls back to the ``SCREENER_CONFIG`` environment variable.
    """
    with (RESOURCES / "config.yaml").open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    path = path or os.environ.get("SCREENER_CONFIG")
    if path:
        with Path(path).open("r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        config = _deep_merge(config, user)

    # Environment overrides for secrets / contact details.
    if ua := os.environ.get("SEC_USER_AGENT"):
        config["smart_money"]["sec_user_agent"] = ua
    config["smart_money"]["openfigi_api_key"] = os.environ.get("OPENFIGI_API_KEY")
    return config


def resource_path(*parts: str) -> Path:
    """Path to a bundled resource file (universes, funds list...)."""
    target = RESOURCES
    for part in parts:  # Traversable.joinpath(*parts) needs Python 3.11+
        target = target / part
    return Path(str(target))
