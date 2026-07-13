from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

import dspy


@dataclass(frozen=True)
class EvaluationExample:
    inputs: Mapping[str, Any]
    expected: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dspy(self) -> dspy.Example:
        values = {**self.inputs, **self.expected, "metadata": dict(self.metadata)}
        return dspy.Example(**values).with_inputs(*self.inputs.keys())


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    version: int
    examples: tuple[EvaluationExample, ...]

    @classmethod
    def load(cls, path: Path) -> "EvaluationDataset":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data.get("version"), int) or data["version"] < 1:
            raise ValueError("Evaluation dataset version must be a positive integer")
        examples = tuple(
            EvaluationExample(
                inputs=item["inputs"],
                expected=item["expected"],
                metadata=item.get("metadata", {}),
            )
            for item in data.get("examples", [])
        )
        if not examples:
            raise ValueError("Evaluation dataset must contain at least one example")
        return cls(name=data["name"], version=data["version"], examples=examples)

    def to_dspy(self) -> List[dspy.Example]:
        return [example.to_dspy() for example in self.examples]


Metric = Callable[[dspy.Example, dspy.Prediction], float | bool]


def exact_answer(example: dspy.Example, prediction: dspy.Prediction) -> bool:
    return str(prediction.answer).strip() == str(example.answer).strip()


def normalized_answer(example: dspy.Example, prediction: dspy.Prediction) -> bool:
    normalize = lambda value: " ".join(str(value).lower().split())
    return normalize(prediction.answer) == normalize(example.answer)


def jaccard_answer(example: dspy.Example, prediction: dspy.Prediction) -> float:
    expected = set(str(example.answer).lower().split())
    actual = set(str(prediction.answer).lower().split())
    union = expected | actual
    return len(expected & actual) / len(union) if union else 1.0


def compare_answers(first: str, second: str, metric: str) -> float:
    if metric == "exact":
        return float(first.strip() == second.strip())
    if metric == "length":
        longer = max(len(first), len(second))
        return min(len(first), len(second)) / longer if longer else 1.0
    if metric == "jaccard":
        first_words = set(first.lower().split())
        second_words = set(second.lower().split())
        union = first_words | second_words
        return len(first_words & second_words) / len(union) if union else 1.0
    raise ValueError(f"Unknown comparison metric: {metric}")


@dataclass(frozen=True)
class UsageSummary:
    model_calls: int = 0
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class EvaluationCase:
    inputs: Mapping[str, Any]
    prediction: Mapping[str, Any]
    score: float


@dataclass(frozen=True)
class EvaluationReport:
    dataset: str
    dataset_version: int
    score: float
    elapsed_seconds: float
    usage: UsageSummary
    cases: tuple[EvaluationCase, ...]


def _history_usage(history: Sequence[Mapping[str, Any]]) -> UsageSummary:
    cost = 0.0
    input_tokens = 0
    output_tokens = 0
    for item in history:
        cost += float(item.get("cost") or 0.0)
        usage = item.get("usage") or {}
        input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens += int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
    return UsageSummary(len(history), cost, input_tokens, output_tokens)


class EvaluationRunner:
    def evaluate(
        self,
        program: dspy.Module,
        dataset: EvaluationDataset,
        metric: Metric,
        *,
        num_threads: int = 1,
    ) -> EvaluationReport:
        lm = dspy.settings.lm
        history = getattr(lm, "history", None)
        history_start = len(history) if isinstance(history, list) else 0
        started = time.perf_counter()
        result = dspy.Evaluate(
            devset=dataset.to_dspy(),
            metric=metric,
            num_threads=num_threads,
            display_progress=False,
            display_table=False,
        )(program)
        elapsed = time.perf_counter() - started
        new_history = history[history_start:] if isinstance(history, list) else []
        cases = tuple(
            EvaluationCase(
                inputs=dict(example.inputs().toDict()),
                prediction=prediction.toDict(),
                score=float(score),
            )
            for example, prediction, score in result.results
        )
        return EvaluationReport(
            dataset=dataset.name,
            dataset_version=dataset.version,
            score=float(result.score),
            elapsed_seconds=elapsed,
            usage=_history_usage(new_history),
            cases=cases,
        )


class Optimizer(Protocol):
    def compile(
        self, student: dspy.Module, *, trainset: List[dspy.Example], **kwargs: Any
    ) -> dspy.Module: ...


class OptimizationRunner:
    def optimize(
        self,
        program: dspy.Module,
        dataset: EvaluationDataset,
        optimizer: Optimizer,
        output_path: Path,
        **compile_options: Any,
    ) -> dspy.Module:
        optimized = optimizer.compile(
            program, trainset=dataset.to_dspy(), **compile_options
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        optimized.save(str(output_path), save_program=False)
        return optimized
