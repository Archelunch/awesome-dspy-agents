from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from awesome_dspy_agents.mlflow_integration import (
    MLflowConfig,
    mlflow_run,
    mlflow_span,
)


class MLflowIntegrationTests(unittest.TestCase):
    def test_disabled_integration_does_not_import_mlflow(self) -> None:
        with patch("importlib.import_module") as import_module:
            with mlflow_run(
                MLflowConfig(),
                pattern="debate",
                topic="Topic",
                config_path="config.yaml",
                version="1.0",
            ) as run:
                self.assertIsNone(run)

        import_module.assert_not_called()

    def test_enabled_integration_configures_and_logs_run(self) -> None:
        mlflow = MagicMock()
        mlflow.start_run.return_value.__enter__.return_value = MagicMock()
        span = MagicMock()
        mlflow.start_span.return_value.__enter__.return_value = span
        config = MLflowConfig(
            enabled=True,
            tracking_uri="http://localhost:5000",
            experiment_name="agents",
            run_name="test-run",
            tags={"environment": "test"},
        )

        with patch("importlib.import_module", return_value=mlflow):
            with mlflow_run(
                config,
                pattern="debate",
                topic="Topic",
                config_path="config.yaml",
                version="1.0",
            ) as run:
                assert run is not None
                with mlflow_span(
                    "agent.affirmative",
                    span_type="AGENT",
                    inputs={"topic": "Topic"},
                    attributes={"agent.role": "affirmative"},
                ) as active_span:
                    self.assertIs(active_span, span)
                    active_span.set_outputs({"argument": "Answer"})
                run.log_outcome(
                    {
                        "result": {
                            "iterations_used": 2,
                            "stopped_early": True,
                            "issues": [],
                            "history": [{}, {}],
                        },
                        "tool_events_by_iter": {1: [{}, {}]},
                    }
                )

        mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mlflow.set_experiment.assert_called_once_with("agents")
        mlflow.dspy.autolog.assert_called_once_with()
        mlflow.start_run.assert_called_once_with(
            run_name="test-run",
            tags={
                "awesome_dspy_agents.pattern": "debate",
                "awesome_dspy_agents.version": "1.0",
                "environment": "test",
            },
        )
        mlflow.log_metrics.assert_called_once_with(
            {
                "iterations_used": 2,
                "stopped_early": 1,
                "tool_event_count": 2,
                "runtime_issue_count": 0,
            }
        )
        mlflow.log_dict.assert_called_once()
        mlflow.start_span.assert_called_once_with(
            name="agent.affirmative",
            span_type=mlflow.entities.SpanType.AGENT,
        )
        span.set_inputs.assert_called_once_with({"topic": "Topic"})
        span.set_attributes.assert_called_once_with(
            {"agent.role": "affirmative"}
        )
        span.set_outputs.assert_called_once_with({"argument": "Answer"})

    def test_span_is_noop_outside_enabled_run(self) -> None:
        with patch("importlib.import_module") as import_module:
            with mlflow_span("agent", span_type="AGENT") as span:
                self.assertIsNone(span)

        import_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
