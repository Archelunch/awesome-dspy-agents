from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from awesome_dspy_agents.runtime import EmitIteration, PatternOutcome, PatternRunRequest


class AgentPattern(Protocol):
    """Protocol all agent patterns must implement."""

    name: str

    def describe(self) -> str:
        """Return a short Markdown description of the pattern."""
        ...

    def default_config_path(self) -> Path | None:
        """Return the default config.yaml path for this pattern, if any."""
        ...

    def available_configs(self) -> Iterable[Path]:
        """Return a list of scenario/config files under this pattern directory."""
        ...

    def available_tools(self) -> Iterable[str]:
        """Return a list of tool names this pattern uses by default (if any)."""
        ...

    def run(
        self,
        request: PatternRunRequest,
        on_iteration: EmitIteration | None = None,
    ) -> PatternOutcome:
        """Execute one Pattern run."""
        ...


@dataclass(frozen=True)
class DiscoveryIssue:
    pattern: str
    message: str


@dataclass(frozen=True)
class PatternCatalog:
    patterns: dict[str, AgentPattern] = field(default_factory=dict)
    issues: tuple[DiscoveryIssue, ...] = ()


def discover_patterns(root: Path) -> PatternCatalog:
    """Discover Pattern adapters and retain every discovery failure."""
    patterns: dict[str, AgentPattern] = {}
    issues = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        candidate = entry / "pattern.py"
        if not candidate.exists():
            continue
        # Dynamic import
        module_name = f"awesome_dspy_agents.patterns.{entry.name}.pattern"
        try:
            mod = __import__(module_name, fromlist=["get_pattern"])  # type: ignore
            get_pattern = getattr(mod, "get_pattern", None)
            if get_pattern is None:
                issues.append(DiscoveryIssue(entry.name, "get_pattern() is missing"))
                continue
            pattern: AgentPattern = get_pattern()
            if pattern.name != entry.name:
                issues.append(
                    DiscoveryIssue(
                        entry.name,
                        f"declared name '{pattern.name}' does not match directory",
                    )
                )
                continue
            if pattern.name in patterns:
                issues.append(DiscoveryIssue(entry.name, "duplicate Pattern name"))
                continue
            patterns[pattern.name] = pattern
        except Exception as error:
            issues.append(
                DiscoveryIssue(entry.name, f"{type(error).__name__}: {error}")
            )
    return PatternCatalog(patterns, tuple(issues))
