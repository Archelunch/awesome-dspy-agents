from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("DSPY_CACHEDIR", f"{tempfile.gettempdir()}/dspy-agents-test-cache")

import dspy

from awesome_dspy_agents.patterns.addition_by_subtraction.pattern import (
    ABSFramework,
    AdditionBySubtractionExchange,
)
from awesome_dspy_agents.patterns.debate.pattern import DebateExchange, MADFramework


class _Addition(dspy.Module):
    def forward(self, **_inputs):
        return dspy.Prediction(candidate_response="draft", reasoning="reason")


class _Subtraction(dspy.Module):
    def forward(self, **_inputs):
        return dspy.Prediction(refined_response="refined", feedback="feedback")


class _Debater(dspy.Module):
    def __init__(self, affirmative: bool) -> None:
        super().__init__()
        self.affirmative = affirmative

    def forward(self, **_inputs):
        if self.affirmative:
            return dspy.Prediction(argument="yes", reasoning="for")
        return dspy.Prediction(counter_argument="no", reasoning="against")


class _Judge(dspy.Module):
    def extract_solution(self, **_inputs):
        return dspy.Prediction(final_answer="answer", justification="because")


class WorkflowStateTests(unittest.TestCase):
    def test_addition_by_subtraction_is_reusable_and_emits_typed_exchanges(self) -> None:
        events = []
        framework = ABSFramework(max_iterations=1, on_iteration=events.append)
        framework.addition = _Addition()
        framework.subtraction = _Subtraction()

        first = framework(context="", instruction="one")
        second = framework(context="", instruction="two")

        self.assertEqual(len(first.history), 1)
        self.assertEqual(len(second.history), 1)
        self.assertIsNot(first.history, second.history)
        self.assertIsInstance(events[0].exchange, AdditionBySubtractionExchange)
        self.assertEqual(framework.history, [])  # DSPy base history remains untouched.

    def test_debate_is_reusable_and_emits_typed_exchanges(self) -> None:
        events = []
        framework = MADFramework(
            max_iterations=1, adaptive_break=False, on_iteration=events.append
        )
        framework.affirmative = _Debater(True)
        framework.negative = _Debater(False)
        framework.judge = _Judge()

        first = framework(debate_topic="one")
        second = framework(debate_topic="two")

        self.assertEqual(len(first.history), 1)
        self.assertEqual(len(second.history), 1)
        self.assertIsNot(first.history, second.history)
        self.assertIsInstance(events[0].exchange, DebateExchange)
        self.assertFalse(hasattr(framework, "debate_history"))


if __name__ == "__main__":
    unittest.main()
