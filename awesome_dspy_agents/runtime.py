from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import dspy

from awesome_dspy_agents.config import AppConfig, build_lm, load_config
from awesome_dspy_agents.tools.registry import (
    FileAccessPolicy,
    ToolListener,
    ToolExecutor,
    default_catalog,
    tool_executor_scope,
)


@dataclass(frozen=True)
class PatternRunRequest:
    """Everything required to execute one isolated Pattern run."""

    topic: str
    config_path: Path
    overrides: Mapping[str, Any] = field(default_factory=dict)
    allowed_paths: tuple[Path, ...] = ()
    on_tool_event: Optional[ToolListener] = None


@dataclass(frozen=True)
class IterationEvent:
    """One observable exchange emitted by a Pattern run."""

    iteration: int
    exchange: Any
    history: str


@dataclass(frozen=True)
class RuntimeIssue:
    """A non-fatal failure that occurred while observing a Pattern run."""

    kind: str
    message: str
    iteration: Optional[int] = None


@dataclass(frozen=True)
class PatternOutcome:
    """The final typed outcome of a Pattern run."""

    final_answer: str
    justification: str
    iterations_used: int
    stopped_early: bool
    history: List[Any]
    issues: List[RuntimeIssue] = field(default_factory=list)


def exchange_to_dict(exchange: Any) -> Dict[str, Any]:
    if is_dataclass(exchange) and not isinstance(exchange, type):
        return asdict(exchange)
    if isinstance(exchange, Mapping):
        return dict(exchange)
    raise TypeError(f"Unsupported exchange type: {type(exchange).__name__}")

EmitIteration = Callable[[IterationEvent], None]
ExecutePattern = Callable[[PatternRunRequest, EmitIteration], PatternOutcome]
ExecuteConfiguredPattern = Callable[
    [PatternRunRequest, AppConfig, EmitIteration], PatternOutcome
]

logger = logging.getLogger(__name__)


class PatternRuntime:
    """Own the lifecycle shared by every Pattern run."""

    def run(
        self,
        request: PatternRunRequest,
        *,
        execute: ExecutePattern,
        on_iteration: Optional[EmitIteration] = None,
        lm: Optional[Any] = None,
    ) -> PatternOutcome:
        issues: List[RuntimeIssue] = []

        def emit(event: IterationEvent) -> None:
            if on_iteration is not None:
                try:
                    on_iteration(event)
                except Exception as error:
                    logger.warning(
                        "Iteration listener failed at iteration %s: %s",
                        event.iteration,
                        error,
                    )
                    issues.append(
                        RuntimeIssue(
                            kind="iteration_listener_failed",
                            message=str(error),
                            iteration=event.iteration,
                        )
                    )

        lm_scope = dspy.context(lm=lm) if lm is not None else nullcontext()
        tool_executor = ToolExecutor(
            default_catalog,
            FileAccessPolicy.from_paths(request.allowed_paths),
            request.on_tool_event,
        )
        with lm_scope, tool_executor_scope(tool_executor):
            outcome = execute(request, emit)
        return replace(outcome, issues=[*outcome.issues, *issues])

    def run_configured(
        self,
        request: PatternRunRequest,
        *,
        execute: ExecuteConfiguredPattern,
        on_iteration: Optional[EmitIteration] = None,
        base_config_path: Optional[Path] = None,
    ) -> PatternOutcome:
        """Validate configuration, scope its default LM, and execute a Pattern."""

        config = load_config(
            str(request.config_path),
            dict(request.overrides),
            base_path=str(base_config_path) if base_config_path is not None else None,
        )
        lm = build_lm(config.default_lm) if config.default_lm is not None else None
        return self.run(
            request,
            execute=lambda current_request, emit: execute(
                current_request, config, emit
            ),
            on_iteration=on_iteration,
            lm=lm,
        )
