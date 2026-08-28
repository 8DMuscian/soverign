"""Model registry — YAML-based model configuration management.

Add, remove, replace, or list LLM models by editing YAML files in
the ``models/`` directory.  Zero code changes required.

Usage::

    from sovereign.models import ModelRegistry

    registry = ModelRegistry()          # auto-discovers models/*.yaml
    registry.list_models()              # -> [ModelConfig, ...]
    registry.get_model("Qwen Coder 3B") # -> ModelConfig
    registry.get_default()              # -> ModelConfig (default=True)
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# MODEL CONFIG
# ────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """A single LLM model's configuration."""
    name: str
    id: str
    description: str
    base_url: str
    api_key: str = "not-needed"
    capabilities: list[str] = field(default_factory=lambda: ["code_generation"])
    requires_vision: bool = False
    default: bool = False
    max_tokens: int = 2048
    temperature: float = 0.1
    max_model_len: int = 4096

    # ── helpers ──────────────────────────────────────────────

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities

    @property
    def is_vision(self) -> bool:
        return self.has_capability("vision") or self.requires_vision


# ────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ────────────────────────────────────────────────────────────

class ModelRegistry:
    """Auto-discovers ``*.yaml`` files in the ``models/`` directory.

    Each YAML file describes one LLM model.  The registry provides
    lookup by name, default-model resolution, and listing.

    Parameters
    ----------
    models_dir : str | Path | None
        Directory containing model YAML files.  Defaults to
        ``<package>/models/`` (i.e. ``sovereign/models/``).
    """

    def __init__(self, models_dir: str | Path | None = None) -> None:
        if models_dir is None:
            models_dir = Path(__file__).parent
        self.models_dir = Path(models_dir)
        self._models: dict[str, ModelConfig] = {}
        self._load_all()

    # ── public API ───────────────────────────────────────────

    def list_models(self) -> list[ModelConfig]:
        """Return all registered models (unsorted)."""
        return list(self._models.values())

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Look up a model by its display *name* (case-insensitive)."""
        return self._models.get(name.lower().strip())

    def get_default(self) -> Optional[ModelConfig]:
        """Return the model marked ``default: true``, or ``None``."""
        for m in self._models.values():
            if m.default:
                return m
        # Fallback: first model if no explicit default
        if self._models:
            return next(iter(self._models.values()))
        return None

    def has_model(self, name: str) -> bool:
        return name.lower().strip() in self._models

    def summary(self) -> str:
        """Human-readable list of all models."""
        if not self._models:
            return "No models registered."
        lines = []
        for m in self._models.values():
            tag = " (default)" if m.default else ""
            vision = " [vision]" if m.is_vision else ""
            lines.append(f"  - {m.name}{tag}{vision}: {m.description}")
        return "\n".join(lines)

    # ── endpoint verification ────────────────────────────────

    @staticmethod
    def query_server(base_url: str, timeout: float = 5.0) -> list[str]:
        """Query a vLLM OpenAI-compatible server's ``/v1/models`` endpoint.

        Returns the list of served model IDs (empty on failure).
        Never raises — returns ``[]`` if the server is unreachable.
        """
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url + "/models"
        else:
            url = url + "/v1/models"

        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        except Exception as exc:
            logger.debug("Could not query %s: %s", url, exc)
            return []

    @staticmethod
    def check_model(models: list[str], expected: str) -> bool:
        """True if *expected* is present (exact) among served model IDs."""
        return expected in models

    @staticmethod
    def best_match(models: list[str], expected: str) -> Optional[str]:
        """Return the best served model ID for *expected*.

        Exact match wins; otherwise a case-insensitive / short-name /
        tail match on repo-style names (e.g. ``Qwen/Qwen2.5-Coder-3B``).
        """
        if expected in models:
            return expected

        exp_lower = expected.lower()
        for m in models:
            if m.lower() == exp_lower:
                return m

        # Match the last path segment (repo tail) case-insensitively
        tail = expected.split("/")[-1].lower()
        for m in models:
            if m.split("/")[-1].lower() == tail:
                return m

        return None

    # ── reload (for hot-reload after YAML changes) ──────────

    def reload(self) -> None:
        """Re-read all YAML files from disk."""
        self._models.clear()
        self._load_all()

    # ── internal ─────────────────────────────────────────────

    def _load_all(self) -> None:
        if not self.models_dir.exists():
            logger.warning("Models directory not found: %s", self.models_dir)
            return

        for yaml_file in sorted(self.models_dir.glob("*.yaml")):
            try:
                config = self._load_yaml(yaml_file)
                self._models[config.name.lower().strip()] = config
                logger.debug("Loaded model: %s from %s", config.name, yaml_file.name)
            except Exception as exc:
                logger.error("Failed to load model from %s: %s", yaml_file.name, exc)

        for yml_file in sorted(self.models_dir.glob("*.yml")):
            try:
                config = self._load_yaml(yml_file)
                self._models[config.name.lower().strip()] = config
            except Exception as exc:
                logger.error("Failed to load model from %s: %s", yml_file.name, exc)

    def _load_yaml(self, path: Path) -> ModelConfig:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")

        required = ("name", "id", "base_url")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        return ModelConfig(
            name=str(data["name"]),
            id=str(data["id"]),
            description=str(data.get("description", "")),
            base_url=str(data["base_url"]),
            api_key=str(data.get("api_key", "not-needed")),
            capabilities=list(data.get("capabilities", ["code_generation"])),
            requires_vision=bool(data.get("requires_vision", False)),
            default=bool(data.get("default", False)),
            max_tokens=int(data.get("max_tokens", 2048)),
            temperature=float(data.get("temperature", 0.1)),
            max_model_len=int(data.get("max_model_len", 4096)),
        )
