"""Agent registry — YAML-based agent configuration management.

Add, remove, replace, or list agents by editing YAML files in the
``agents/`` directory.  Zero Python code changes required.

Usage::

    from sovereign.agents import AgentRegistry

    registry = AgentRegistry()            # auto-discovers agents/*.yaml
    registry.list_agents()                # -> [AgentConfig, ...]
    registry.get_agent("Data Processor") # -> AgentConfig
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# AGENT CONFIG
# ────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """A single agent's configuration loaded from YAML."""
    name: str
    description: str
    category: str
    version: str
    enabled: bool
    icon: str
    supported_file_types: list[str]
    requires_vision: bool
    sandbox_image: str | None
    system_prompt: str
    security: dict
    model: str | None = None  # optional model name override

    @property
    def is_registry_agent(self) -> bool:
        return self.category == "registry"


# ────────────────────────────────────────────────────────────
# AGENT REGISTRY
# ────────────────────────────────────────────────────────────

class AgentRegistry:
    """Auto-discovers ``*.yaml`` files in the ``agents/`` directory.

    Each YAML file describes one agent.  The registry provides
    lookup by name, category filtering, and listing.

    Parameters
    ----------
    agents_dir : str | Path | None
        Directory containing agent YAML files.  Defaults to
        ``<package>/agents/`` (i.e. ``sovereign/agents/``).
    """

    def __init__(self, agents_dir: str | Path | None = None) -> None:
        if agents_dir is None:
            agents_dir = Path(__file__).parent
        self.agents_dir = Path(agents_dir)
        self._agents: dict[str, AgentConfig] = {}
        self._load_all()

    # ── public API ───────────────────────────────────────────

    def list_agents(self, enabled_only: bool = True) -> list[AgentConfig]:
        """Return all registered agents."""
        agents = list(self._agents.values())
        if enabled_only:
            agents = [a for a in agents if a.enabled]
        return agents

    def list_categories(self) -> list[str]:
        """Return unique categories across all enabled agents."""
        return sorted({a.category for a in self.list_agents()})

    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Look up an agent by its display *name* (case-insensitive)."""
        return self._agents.get(name.lower().strip())

    def get_by_category(self, category: str) -> list[AgentConfig]:
        """Return all enabled agents in a given category."""
        cat = category.lower().strip()
        return [a for a in self.list_agents() if a.category == cat]

    def has_agent(self, name: str) -> bool:
        return name.lower().strip() in self._agents

    def summary(self) -> str:
        """Human-readable list of all agents."""
        if not self._agents:
            return "No agents registered."
        lines = []
        for a in self._agents.values():
            status = "enabled" if a.enabled else "DISABLED"
            vision = " [vision]" if a.requires_vision else ""
            model_tag = f" model={a.model}" if a.model else ""
            # Use ASCII-safe icon on Windows
            icon = a.icon if sys.platform != "win32" else f"[{a.category[:3].upper()}]"
            lines.append(
                f"  {icon} {a.name} ({a.category}) [{status}]{vision}{model_tag}"
            )
            lines.append(f"     {a.description}")
        return "\n".join(lines)

    # ── reload ───────────────────────────────────────────────

    def reload(self) -> None:
        """Re-read all YAML files from disk."""
        self._agents.clear()
        self._load_all()

    # ── internal ─────────────────────────────────────────────

    def _load_all(self) -> None:
        if not self.agents_dir.exists():
            logger.warning("Agents directory not found: %s", self.agents_dir)
            return

        for yaml_file in sorted(self.agents_dir.glob("*.yaml")):
            try:
                config = self._load_yaml(yaml_file)
                self._agents[config.name.lower().strip()] = config
                logger.debug("Loaded agent: %s from %s", config.name, yaml_file.name)
            except Exception as exc:
                logger.error("Failed to load agent from %s: %s", yaml_file.name, exc)

        for yml_file in sorted(self.agents_dir.glob("*.yml")):
            try:
                config = self._load_yaml(yml_file)
                self._agents[config.name.lower().strip()] = config
            except Exception as exc:
                logger.error("Failed to load agent from %s: %s", yml_file.name, exc)

    def _load_yaml(self, path: Path) -> AgentConfig:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")

        required = ("name", "description", "category", "system_prompt")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        security_data = data.get("security", {})
        if not isinstance(security_data, dict):
            security_data = {}

        return AgentConfig(
            name=str(data["name"]),
            description=str(data["description"]),
            category=str(data["category"]),
            version=str(data.get("version", "1.0.0")),
            enabled=bool(data.get("enabled", True)),
            icon=str(data.get("icon", "🤖")),
            supported_file_types=list(data.get("supported_file_types", [])),
            requires_vision=bool(data.get("requires_vision", False)),
            sandbox_image=data.get("sandbox_image"),
            system_prompt=str(data["system_prompt"]),
            security=security_data,
            model=data.get("model"),
        )
