from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault(
    "DSPY_CACHEDIR", f"{tempfile.gettempdir()}/dspy-agents-test-cache"
)

import dspy

from awesome_dspy_agents.predictor import PredictorModule, build_predictor
from awesome_dspy_agents.tools.registry import get_current_agent


class _ToolProvider:
    def __init__(self) -> None:
        self.requested_names = []

    def build_dspy_tools(self, names):
        self.requested_names.append(names)
        return [lambda value: value]


class _AgentProbe(dspy.Module):
    def forward(self, **_inputs):
        return dspy.Prediction(agent=get_current_agent())


class PredictorConstructionTests(unittest.TestCase):
    def test_builds_each_supported_dspy_predictor(self) -> None:
        tools = _ToolProvider()

        predict = build_predictor("question -> answer", "predict", role="test")
        chain = build_predictor("question -> answer", "chain_of_thought", role="test")
        react = build_predictor(
            "question -> answer",
            "react",
            role="test",
            tool_names=["echo"],
            tool_provider=tools,
            react_max_iters=4,
        )

        self.assertIsInstance(predict.predictor, dspy.Predict)
        self.assertIsInstance(chain.predictor, dspy.ChainOfThought)
        self.assertIsInstance(react.predictor, dspy.ReAct)
        self.assertEqual(tools.requested_names, [["echo"]])

    def test_rejects_an_unknown_predictor_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported predictor kind"):
            build_predictor("question -> answer", "unknown", role="test")

    def test_assigns_language_model_to_nested_predictors(self) -> None:
        lm = object()

        predictor = build_predictor("question -> answer", "predict", role="test", lm=lm)

        self.assertTrue(predictor.predictors())
        self.assertTrue(all(item.lm is lm for item in predictor.predictors()))

    def test_scopes_agent_context_and_restores_it_after_execution(self) -> None:
        predictor = PredictorModule(_AgentProbe(), role="addition")

        result = predictor()

        self.assertEqual(result.agent, "addition")
        self.assertEqual(get_current_agent(), "unknown")


if __name__ == "__main__":
    unittest.main()
