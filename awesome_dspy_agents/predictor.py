from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import dspy

from awesome_dspy_agents.tools.registry import (
    ToolProvider,
    reset_current_agent,
    runtime_tool_provider,
    set_current_agent,
)

PredictorKind = Literal["predict", "chain_of_thought", "react"]
SignatureLike = str | type[dspy.Signature]


class PredictorModule(dspy.Module):
    """Execute a DSPy predictor inside one agent's runtime context."""

    def __init__(self, predictor: dspy.Module, *, role: str) -> None:
        super().__init__()
        self.predictor = predictor
        self.role = role

    def forward(self, **inputs: Any) -> dspy.Prediction:
        token = set_current_agent(self.role)
        try:
            return self.predictor(**inputs)
        finally:
            reset_current_agent(token)


def build_predictor(
    signature: SignatureLike,
    kind: PredictorKind | str,
    *,
    role: str,
    lm: dspy.LM | None = None,
    tool_names: Sequence[str] | None = None,
    tool_provider: ToolProvider = runtime_tool_provider,
    react_max_iters: int = 3,
) -> PredictorModule:
    """Build and configure one of the supported DSPy predictor adapters."""

    if kind == "predict":
        predictor: dspy.Module = dspy.Predict(signature)
    elif kind == "chain_of_thought":
        predictor = dspy.ChainOfThought(signature)
    elif kind == "react":
        tools = tool_provider.build_dspy_tools(list(tool_names or []))
        # DSPy accepts string signatures and Tool instances at runtime, although
        # its current annotations only describe signature classes and callables.
        predictor = dspy.ReAct(
            cast(Any, signature), tools=cast(Any, tools), max_iters=react_max_iters
        )
    else:
        raise ValueError(f"Unsupported predictor kind: {kind}")

    configured = PredictorModule(predictor, role=role)
    if lm is not None:
        configured.set_lm(lm)
    return configured
