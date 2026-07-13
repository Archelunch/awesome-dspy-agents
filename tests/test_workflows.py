from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault(
    "DSPY_CACHEDIR", f"{tempfile.gettempdir()}/dspy-agents-test-cache"
)

import dspy

from awesome_dspy_agents.patterns.addition_by_subtraction.pattern import (
    ABSFramework,
    AdditionBySubtractionExchange,
)
from awesome_dspy_agents.patterns.debate.pattern import DebateExchange, MADFramework


class _Addition(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(self, **_inputs):
        self.calls.append(_inputs)
        return dspy.Prediction(candidate_response="draft", reasoning="reason")


class _Subtraction(dspy.Module):
    def forward(self, **_inputs):
        return dspy.Prediction(
            refined_response="refined",
            feedback="feedback",
            removals=["repetition"],
            removal_reasons=["duplicate"],
            preserved_facts=["supported fact"],
        )


class _Debater(dspy.Module):
    def __init__(self, affirmative: bool) -> None:
        super().__init__()
        self.affirmative = affirmative
        self.calls = []

    def forward(self, **_inputs):
        self.calls.append(_inputs)
        if self.affirmative:
            return dspy.Prediction(argument="yes", reasoning="for")
        return dspy.Prediction(counter_argument="no", reasoning="against")


class _Judge(dspy.Module):
    def extract_solution(self, **_inputs):
        return dspy.Prediction(final_answer="answer", justification="because")


class WorkflowStateTests(unittest.TestCase):
    def test_addition_by_subtraction_is_reusable_and_emits_typed_exchanges(
        self,
    ) -> None:
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
        self.assertEqual(
            [event.kind for event in first.trajectory.events],
            ["proposal", "revision"],
        )
        self.assertEqual(first.trajectory.events[1].delta.removed, ("repetition",))
        self.assertEqual(framework.history, [])  # DSPy base history remains untouched.

    def test_addition_receives_the_latest_refined_state_not_only_a_transcript(
        self,
    ) -> None:
        framework = ABSFramework(max_iterations=2)
        addition = _Addition()
        framework.addition = addition
        framework.subtraction = _Subtraction()

        framework(context="", instruction="improve this")

        self.assertEqual(addition.calls[0]["current_response"], "")
        self.assertEqual(addition.calls[1]["current_response"], "refined")
        self.assertEqual(addition.calls[1]["previous_feedback"], "feedback")

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
        self.assertEqual(
            [event.kind for event in first.trajectory.events],
            ["proposal", "critique", "judgment"],
        )
        self.assertFalse(hasattr(framework, "debate_history"))

    def test_classic_debate_preserves_its_legacy_history_prompt(self) -> None:
        framework = MADFramework(max_iterations=2, adaptive_break=False)
        affirmative = _Debater(True)
        framework.affirmative = affirmative
        framework.negative = _Debater(False)
        framework.judge = _Judge()

        framework(debate_topic="topic")

        second_round_history = affirmative.calls[1]["debate_history"]
        self.assertIn("--- Iteration 1 ---", second_round_history)
        self.assertNotIn("[event-", second_round_history)


if __name__ == "__main__":
    unittest.main()
