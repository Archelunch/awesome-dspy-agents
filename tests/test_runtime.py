from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault(
    "DSPY_CACHEDIR", f"{tempfile.gettempdir()}/dspy-agents-test-cache"
)

import dspy

from awesome_dspy_agents.runtime import (
    PatternOutcome,
    PatternRunRequest,
    PatternRuntime,
)


class PatternRuntimeTests(unittest.TestCase):
    def test_pattern_run_returns_typed_outcome(self) -> None:
        runtime = PatternRuntime()
        request = PatternRunRequest(topic="A topic", config_path=Path("config.yaml"))

        outcome = runtime.run(
            request,
            execute=lambda _request, _emit: PatternOutcome(
                final_answer="Answer",
                justification="Because",
                iterations_used=2,
                stopped_early=True,
                history=[{"iteration": 1}],
            ),
        )

        self.assertIsInstance(outcome, PatternOutcome)
        self.assertEqual(outcome.final_answer, "Answer")
        self.assertEqual(outcome.history, [{"iteration": 1}])

    def test_listener_failure_is_recorded_without_aborting_the_run(self) -> None:
        runtime = PatternRuntime()
        request = PatternRunRequest(topic="A topic", config_path=Path("config.yaml"))

        def execute(_request, emit):
            from awesome_dspy_agents.runtime import IterationEvent

            emit(
                IterationEvent(
                    iteration=1, exchange={"answer": "draft"}, history="draft"
                )
            )
            return PatternOutcome(
                final_answer="Answer",
                justification="Because",
                iterations_used=1,
                stopped_early=False,
                history=[],
            )

        def broken_listener(_event):
            raise RuntimeError("renderer disconnected")

        outcome = runtime.run(request, execute=execute, on_iteration=broken_listener)

        self.assertEqual(outcome.final_answer, "Answer")
        self.assertEqual(len(outcome.issues), 1)
        self.assertEqual(outcome.issues[0].kind, "iteration_listener_failed")
        self.assertEqual(outcome.issues[0].iteration, 1)

    def test_language_model_is_scoped_to_the_pattern_run(self) -> None:
        runtime = PatternRuntime()
        request = PatternRunRequest(topic="A topic", config_path=Path("config.yaml"))
        previous_lm = dspy.settings.lm
        run_lm = object()

        def execute(_request, _emit):
            self.assertIs(dspy.settings.lm, run_lm)
            return PatternOutcome("Answer", "Because", 1, False, [])

        runtime.run(request, execute=execute, lm=run_lm)

        self.assertIs(dspy.settings.lm, previous_lm)


if __name__ == "__main__":
    unittest.main()
