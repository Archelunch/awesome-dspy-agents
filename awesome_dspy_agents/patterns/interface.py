from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Protocol


class AgentPattern(Protocol):
    """Protocol all agent patterns must implement."""

    name: str

    def describe(self) -> str:
        """Return a short Markdown description of the pattern."""
        ...

    def default_config_path(self) -> Optional[Path]:
        """Return the default config.yaml path for this pattern, if any."""
        ...

    def available_configs(self) -> Iterable[Path]:
        """Return a list of scenario/config files under this pattern directory."""
        ...

    def available_tools(self) -> Iterable[str]:
        """Return a list of tool names this pattern uses by default (if any)."""
        ...

    def available_scripts(self) -> Dict[str, str]:
        """Return mapping of script-name -> description for optimization/eval scripts."""
        ...

    def run(
        self,
        topic: str,
        config_path: Path,
        overrides: Optional[Dict[str, Any]] = None,
        on_iteration: Optional[Callable[[int, Dict[str, Any], str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the pattern and return a machine-friendly summary of results."""
        ...


def find_patterns(root: Path) -> Dict[str, AgentPattern]:
    """Scan `root` for pattern packages exposing get_pattern()."""
    patterns: Dict[str, AgentPattern] = {}
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
                continue
            pattern: AgentPattern = get_pattern()
            patterns[entry.name] = pattern
        except Exception:
            # Skip broken patterns; CLI can show warnings later
            continue
    return patterns
