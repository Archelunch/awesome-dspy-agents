from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault(
    "DSPY_CACHEDIR", f"{tempfile.gettempdir()}/dspy-agents-test-cache"
)

from awesome_dspy_agents.patterns.addition_by_subtraction.pattern import (
    AdditionBySubtractionPattern,
)
from awesome_dspy_agents.patterns.debate.pattern import DebatePattern
from awesome_dspy_agents.runtime import PatternOutcome, PatternRunRequest


class _FakeProgram:
    def __init__(self, prediction, **settings):
        self.prediction = prediction
        self.settings = settings

    def __call__(self, **_inputs):
        return self.prediction


class PatternInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "config.yaml"
        self.config_path.write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_debate_uses_typed_runtime_interface(self) -> None:
        prediction = SimpleNamespace(
            final_answer="Debate answer",
            justification="Debate justification",
            iterations_used=4,
            stopped_early=True,
            history=[{"affirmative": "yes", "negative": "no"}],
        )
        programs = []

        def build_program(**settings):
            program = _FakeProgram(prediction, **settings)
            programs.append(program)
            return program

        with patch(
            "awesome_dspy_agents.patterns.debate.pattern.MADFramework",
            side_effect=build_program,
        ):
            result = DebatePattern().run(
                PatternRunRequest(
                    topic="Topic",
                    config_path=self.config_path,
                    overrides={"debate": {"max_iterations": 4}},
                )
            )

        self.assertEqual(programs[0].settings["max_iterations"], 4)
        self.assertIsInstance(result, PatternOutcome)
        self.assertEqual(result.final_answer, "Debate answer")
        self.assertEqual(result.history, prediction.history)

    def test_addition_by_subtraction_uses_typed_runtime_interface(self) -> None:
        prediction = SimpleNamespace(
            final_answer="Refined answer",
            justification="ABS justification",
            iterations_used=3,
            stopped_early=False,
            history=[{"addition": "draft", "subtraction": "refined"}],
        )
        programs = []

        def build_program(**settings):
            program = _FakeProgram(prediction, **settings)
            programs.append(program)
            return program

        with patch(
            "awesome_dspy_agents.patterns.addition_by_subtraction.pattern.ABSFramework",
            side_effect=build_program,
        ):
            result = AdditionBySubtractionPattern().run(
                PatternRunRequest(
                    topic="Topic",
                    config_path=self.config_path,
                    overrides={"abs": {"max_iterations": 3}},
                )
            )

        self.assertEqual(programs[0].settings["max_iterations"], 3)
        self.assertIsInstance(result, PatternOutcome)
        self.assertEqual(result.final_answer, "Refined answer")
        self.assertEqual(result.history, prediction.history)

    def test_debate_can_select_the_consensus_free_protocol(self) -> None:
        prediction = SimpleNamespace(
            final_answer="Consensus-free answer",
            justification="Full trajectory",
            iterations_used=1,
            stopped_early=False,
            history=("proposal", "revision"),
        )
        programs = []

        def build_program(**settings):
            program = _FakeProgram(prediction, **settings)
            programs.append(program)
            return program

        with patch(
            "awesome_dspy_agents.patterns.debate.pattern.ConsensusFreeDebate",
            side_effect=build_program,
        ):
            result = DebatePattern().run(
                PatternRunRequest(
                    topic="Topic",
                    config_path=self.config_path,
                    overrides={
                        "debate": {
                            "protocol": "consensus_free",
                            "agent_count": 3,
                        }
                    },
                )
            )

        self.assertEqual(programs[0].settings["agent_count"], 3)
        self.assertEqual(result.final_answer, "Consensus-free answer")
        self.assertEqual(result.history, ["proposal", "revision"])


if __name__ == "__main__":
    unittest.main()
