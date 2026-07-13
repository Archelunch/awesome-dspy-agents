from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DSPY_CACHEDIR", f"{tempfile.gettempdir()}/dspy-agents-test-cache")

import dspy

from awesome_dspy_agents.evaluation import (
    EvaluationDataset,
    EvaluationExample,
    EvaluationRunner,
    OptimizationRunner,
    compare_answers,
    exact_answer,
)


class _AnswerProgram(dspy.Module):
    def forward(self, question: str):
        return dspy.Prediction(answer="4" if "2 + 2" in question else "unknown")


class _SavingProgram(_AnswerProgram):
    def save(self, path: str, **_kwargs):
        Path(path).write_text('{"optimized": true}', encoding="utf-8")


class _Optimizer:
    def __init__(self) -> None:
        self.trainset = None

    def compile(self, _student, *, trainset, **_kwargs):
        self.trainset = trainset
        return _SavingProgram()


class EvaluationTests(unittest.TestCase):
    def dataset(self) -> EvaluationDataset:
        return EvaluationDataset(
            "math",
            1,
            (EvaluationExample({"question": "What is 2 + 2?"}, {"answer": "4"}),),
        )

    def test_evaluates_offline_program_with_versioned_dataset(self) -> None:
        report = EvaluationRunner().evaluate(
            _AnswerProgram(), self.dataset(), exact_answer
        )

        self.assertEqual(report.dataset_version, 1)
        self.assertEqual(report.score, 100.0)
        self.assertEqual(report.cases[0].score, 1.0)

    def test_dataset_loader_rejects_unversioned_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps({"name": "bad", "examples": []}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "version"):
                EvaluationDataset.load(path)

    def test_optimizer_compiles_and_saves_program(self) -> None:
        optimizer = _Optimizer()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "optimized.json"

            result = OptimizationRunner().optimize(
                _AnswerProgram(), self.dataset(), optimizer, output, seed=7
            )

            self.assertIsInstance(result, _SavingProgram)
            self.assertTrue(output.exists())
            self.assertEqual(len(optimizer.trainset), 1)

    def test_answer_comparison_metrics_are_applied(self) -> None:
        self.assertEqual(compare_answers("same", "same", "exact"), 1.0)
        self.assertEqual(compare_answers("one two", "one", "jaccard"), 0.5)
        self.assertEqual(compare_answers("abcd", "ab", "length"), 0.5)
        with self.assertRaisesRegex(ValueError, "Unknown comparison metric"):
            compare_answers("a", "b", "missing")


if __name__ == "__main__":
    unittest.main()
