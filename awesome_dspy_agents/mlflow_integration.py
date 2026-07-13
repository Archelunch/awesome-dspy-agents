from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


class MLflowIntegrationError(RuntimeError):
    """Raised when MLflow was requested but cannot be configured."""


@dataclass(frozen=True)
class MLflowConfig:
    """Configuration for one MLflow-instrumented pattern run."""

    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str = "awesome-dspy-agents"
    run_name: str | None = None
    tags: Mapping[str, str] | None = None


class MLflowRun:
    """Log application-level data alongside DSPy's automatically captured traces."""

    def __init__(self, mlflow: Any) -> None:
        self._mlflow = mlflow

    def log_outcome(self, record: Mapping[str, Any]) -> None:
        result = record.get("result", {})
        history = result.get("history", [])
        tool_events = record.get("tool_events_by_iter", {})
        tool_call_count = sum(len(events) for events in tool_events.values())

        self._mlflow.log_metrics(
            {
                "iterations_used": int(result.get("iterations_used", len(history))),
                "stopped_early": int(bool(result.get("stopped_early", False))),
                "tool_event_count": tool_call_count,
                "runtime_issue_count": len(result.get("issues", [])),
            }
        )
        self._mlflow.log_dict(dict(record), "session.json")


def _load_mlflow() -> Any:
    try:
        return importlib.import_module("mlflow")
    except ImportError as error:
        raise MLflowIntegrationError(
            "MLflow support is not installed. Run "
            "`poetry install -E mlflow` or `pip install 'dspy-agents[mlflow]'`."
        ) from error


@contextmanager
def mlflow_run(
    config: MLflowConfig,
    *,
    pattern: str,
    topic: str,
    config_path: str,
    version: str,
) -> Iterator[MLflowRun | None]:
    """Create an MLflow run and enable DSPy tracing when configured."""

    if not config.enabled:
        yield None
        return

    mlflow = _load_mlflow()
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    mlflow.dspy.autolog()

    tags = {
        "awesome_dspy_agents.pattern": pattern,
        "awesome_dspy_agents.version": version,
        **dict(config.tags or {}),
    }
    with mlflow.start_run(run_name=config.run_name, tags=tags):
        mlflow.log_params(
            {
                "pattern": pattern,
                "topic": topic,
                "config_path": config_path,
            }
        )
        yield MLflowRun(mlflow)
