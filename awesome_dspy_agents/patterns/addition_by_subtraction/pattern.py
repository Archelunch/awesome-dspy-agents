from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import dspy  # type: ignore

from awesome_dspy_agents.config import AppConfig, build_lm, load_config
from awesome_dspy_agents.logging_setup import get_logger
from awesome_dspy_agents.patterns.interface import AgentPattern
from awesome_dspy_agents.predictor import build_predictor
from awesome_dspy_agents.runtime import (
    EmitIteration,
    IterationEvent,
    PatternOutcome,
    PatternRunRequest,
    PatternRuntime,
)
from awesome_dspy_agents.tools.registry import (
    reset_current_iteration,
    set_current_iteration,
)

from .signatures import AdditionAgentSignature, SubtractionAgentSignature

llm_logger = get_logger("mad.llm", "llm_calls.log", max_bytes=2_000_000, backup_count=3)


@dataclass(frozen=True)
class AdditionBySubtractionExchange:
    addition: str
    addition_reasoning: str
    subtraction: str
    feedback: str


class AdditionModule(dspy.Module):
    def __init__(
        self,
        persona: str = "",
        lm: dspy.LM | None = None,
        module_type: str = "predict",
        tool_names: list[str] | None = None,
        react_max_iters: int = 3,
    ):
        super().__init__()
        self.persona = persona
        self.module_type = module_type

        self.predict = build_predictor(
            AdditionAgentSignature,
            module_type,
            role="addition",
            lm=lm,
            tool_names=tool_names,
            react_max_iters=react_max_iters,
        )

    def forward(self, context: str, instruction: str, history: str):
        llm_logger.info(
            "predict",
            role="addition",
            module=self.module_type,
            ctx_len=len(context),
            hist_len=len(history),
        )
        return self.predict(
            context=context,
            instruction=instruction,
            history=history,
            persona=self.persona,
        )


class SubtractionModule(dspy.Module):
    def __init__(
        self,
        persona: str = "",
        lm: dspy.LM | None = None,
        module_type: str = "predict",
        tool_names: list[str] | None = None,
        react_max_iters: int = 3,
    ):
        super().__init__()
        self.persona = persona
        self.module_type = module_type

        self.predict = build_predictor(
            SubtractionAgentSignature,
            module_type,
            role="subtraction",
            lm=lm,
            tool_names=tool_names,
            react_max_iters=react_max_iters,
        )

    def forward(
        self, context: str, instruction: str, history: str, candidate_response: str
    ):
        llm_logger.info(
            "predict",
            role="subtraction",
            module=self.module_type,
            ctx_len=len(context),
            hist_len=len(history),
        )
        return self.predict(
            context=context,
            instruction=instruction,
            history=history,
            candidate_response=candidate_response,
            persona=self.persona,
        )


class ABSFramework(dspy.Module):
    """Addition-by-Subtraction collaboration framework."""

    def __init__(
        self,
        max_iterations: int = 2,
        early_exit: bool = True,
        addition_persona: str = "",
        subtraction_persona: str = "",
        addition_lm: dspy.LM | None = None,
        subtraction_lm: dspy.LM | None = None,
        addition_module_type: str = "predict",
        subtraction_module_type: str = "predict",
        addition_tools: list[str] | None = None,
        subtraction_tools: list[str] | None = None,
        react_max_iters: int = 6,
        on_iteration: EmitIteration | None = None,
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.early_exit = early_exit
        self.on_iteration = on_iteration

        self.addition = AdditionModule(
            persona=addition_persona,
            lm=addition_lm,
            module_type=addition_module_type,
            tool_names=addition_tools,
            react_max_iters=react_max_iters,
        )
        self.subtraction = SubtractionModule(
            persona=subtraction_persona,
            lm=subtraction_lm,
            module_type=subtraction_module_type,
            tool_names=subtraction_tools,
            react_max_iters=react_max_iters,
        )

    @staticmethod
    def format_history(history: list[AdditionBySubtractionExchange]) -> str:
        if not history:
            return "No conversation yet."
        formatted = []
        for i, exchange in enumerate(history, 1):
            formatted.append(f"\n--- Iteration {i} ---")
            formatted.append(f"Addition: {exchange.addition}")
            formatted.append(f"Subtraction: {exchange.subtraction}")
            formatted.append(f"Feedback: {exchange.feedback}")
        return "\n".join(formatted)

    def forward(self, context: str, instruction: str):
        history: list[AdditionBySubtractionExchange] = []
        final_response = ""

        H_context = context
        H_instruction = instruction

        previous_refined_response: str | None = None
        for iteration in range(1, self.max_iterations + 1):
            hist_str = self.format_history(history)
            iter_token = set_current_iteration(iteration)
            try:
                # Addition produces candidate
                add_out = self.addition(
                    context=H_context, instruction=H_instruction, history=hist_str
                )

                # Subtraction refines
                sub_out = self.subtraction(
                    context=H_context,
                    instruction=H_instruction,
                    history=hist_str,
                    candidate_response=add_out.candidate_response,
                )

                exchange = AdditionBySubtractionExchange(
                    addition=add_out.candidate_response,
                    addition_reasoning=add_out.reasoning,
                    subtraction=sub_out.refined_response,
                    feedback=sub_out.feedback,
                )
                history.append(exchange)

                # callback for TUI
                if self.on_iteration is not None:
                    self.on_iteration(
                        IterationEvent(
                            iteration, exchange, self.format_history(history)
                        )
                    )

                # Early exit if no changes
                if (
                    previous_refined_response is not None
                    and previous_refined_response.strip()
                    == sub_out.refined_response.strip()
                    and self.early_exit
                ):
                    final_response = sub_out.refined_response
                    break

                previous_refined_response = sub_out.refined_response
                H_context = context
                H_instruction = instruction
            finally:
                reset_current_iteration(iter_token)

        # Final output: prefer last refined response if exists, else last candidate
        if history:
            final_response = history[-1].subtraction or history[-1].addition

        return dspy.Prediction(
            final_answer=final_response,
            justification="Refined via Addition-by-Subtraction iterative collaboration.",
            history=history,
            iterations_used=len(history),
            stopped_early=self.early_exit and len(history) < self.max_iterations,
        )


class AdditionBySubtractionPattern(AgentPattern):
    name = "addition_by_subtraction"

    def __init__(self) -> None:
        self._root = Path(__file__).resolve().parent

    def describe(self) -> str:
        readme = self._root / "README.md"
        if readme.exists():
            return readme.read_text(encoding="utf-8")
        return "Addition-by-Subtraction collaboration pattern."

    def default_config_path(self) -> Path | None:
        cfg = self._root / "config.yaml"
        return cfg if cfg.exists() else None

    def available_configs(self) -> list[Path]:
        configs: list[Path] = []
        cfg = self.default_config_path()
        if cfg:
            configs.append(cfg)
        scenarios = self._root / "scenarios"
        if scenarios.exists():
            for p in scenarios.glob("*.yaml"):
                configs.append(p)
        return configs

    def available_tools(self) -> list[str]:
        cfg_path = self.default_config_path()
        if cfg_path:
            cfg = load_config(str(cfg_path))
            tools = set()
            add = cfg.agents.get("addition")
            sub = cfg.agents.get("subtraction")
            if add:
                tools.update(add.tools)
            if sub:
                tools.update(sub.tools)
            return sorted(tools)
        return []

    def run(
        self,
        request: PatternRunRequest,
        on_iteration: EmitIteration | None = None,
    ) -> PatternOutcome:

        def execute(
            current_request: PatternRunRequest,
            cfg: AppConfig,
            emit: Callable[[IterationEvent], None],
        ) -> PatternOutcome:
            add_cfg = cfg.agents.get("addition")
            sub_cfg = cfg.agents.get("subtraction")
            framework = ABSFramework(
                max_iterations=cfg.abs.max_iterations,
                early_exit=cfg.abs.early_exit,
                addition_persona=(add_cfg.persona if add_cfg else ""),
                subtraction_persona=(sub_cfg.persona if sub_cfg else ""),
                addition_lm=(build_lm(add_cfg.lm) if add_cfg and add_cfg.lm else None),
                subtraction_lm=(
                    build_lm(sub_cfg.lm) if sub_cfg and sub_cfg.lm else None
                ),
                addition_module_type=(add_cfg.module_type if add_cfg else "predict"),
                subtraction_module_type=(sub_cfg.module_type if sub_cfg else "predict"),
                addition_tools=(add_cfg.tools if add_cfg else []),
                subtraction_tools=(sub_cfg.tools if sub_cfg else []),
                on_iteration=emit,
            )
            final = framework(context="", instruction=current_request.topic)
            return PatternOutcome(
                final_answer=final.final_answer,
                justification=final.justification,
                iterations_used=final.iterations_used,
                stopped_early=final.stopped_early,
                history=final.history,
            )

        return PatternRuntime().run_configured(
            request,
            execute=execute,
            on_iteration=on_iteration,
            base_config_path=self.default_config_path(),
        )


def get_pattern() -> AgentPattern:
    return AdditionBySubtractionPattern()
