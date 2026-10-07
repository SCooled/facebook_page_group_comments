from __future__ import annotations

from pathlib import Path
from typing import Any

from providers.base import Provider

ADAPTERS = {
    "openai": "providers.openai_adapter:OpenAIProvider",
    "gemini": "providers.gemini_adapter:GeminiProvider",
    "instinct": "providers.instinct_adapter:InstinctProvider",
    "human": "providers.human_adapter:HumanProvider",
    "mock": "providers.mock_adapter:MockProvider",
}


def make_provider(name: str, cfg: dict[str, Any], overrides: dict[str, Any] | None = None) -> Provider:
    """Instantiate a provider by its config.yaml key. Adapters are imported lazily so a
    missing SDK for one provider never blocks the others."""
    try:
        settings = dict(cfg["providers"][name])
    except KeyError:
        raise KeyError(f"provider {name!r} not in config.yaml providers: {list(cfg['providers'])}")
    if overrides:
        settings.update(overrides)
    if settings["adapter"] == "human":
        settings.setdefault("responses_dir", str(Path(cfg["_root"]) / cfg["human_responses_dir"]))
    module_name, cls_name = ADAPTERS[settings["adapter"]].split(":")
    module = __import__(module_name, fromlist=[cls_name])
    return getattr(module, cls_name)(name, settings)
