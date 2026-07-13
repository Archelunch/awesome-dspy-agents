from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import dspy  # type: ignore

from awesome_dspy_agents.config import AppConfig, build_lm, load_config
from awesome_dspy_agents.logging_setup import get_logger
from awesome_dspy_agents.mlflow_integration import mlflow_span
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

from .signatures import (
    AffirmativeDebater,
    JudgeDiscriminative,
    JudgeExtractive,
    NegativeDebater,
)

llm_logger = get_logger("mad.llm", "llm_calls.log", max_bytes=2_000_000, backup_count=3)


@dataclass(frozen=True)
class DebateExchange:
    affirmative: str
    affirmative_reasoning: str
    negative: str
    negative_reasoning: str
    judge_eval: str | None = None


class DebaterModule(dspy.Module):
    """Base debater module with configurable behavior."""

    def __init__(
        self,
        role: str,
        debate_level: int = 2,
        persona: str = "",
        lm: dspy.LM | None = None,
        module_type: str = "predict",
        tool_names: list[str] | None = None,
        react_max_iters: int = 3,
    ):
        super().__init__()
        self.role = role
        self.debate_level = debate_level
        self.persona = persona
        self.module_type = module_type

        # Select signature based on role
        if role == "affirmative":
            signature = AffirmativeDebater
        else:
            signature = NegativeDebater

        self.predict = build_predictor(
            signature,
            module_type,
            role=role,
            lm=lm,
            tool_names=tool_names,
            react_max_iters=react_max_iters,
        )

    def get_role_instruction(self) -> str:
        """Get role instruction based on debate level."""
        level_instructions = {
            0: "Both sides must reach full consensus on every point.",
            1: "Most debate should be disagreements, but some consensus on minor points.",
            2: "It's not necessary to fully agree. Find the correct answer.",
            3: "Both sides must disagree on every point. No consensus.",
        }

        base_instruction = level_instructions.get(
            self.debate_level, level_instructions[2]
        )

        if self.role == "affirmative":
            return (
                f"You are affirmative side. {base_instruction} Present your viewpoint."
            )
        else:
            return (
                f"You are negative side. {base_instruction} Provide counter-arguments."
            )

    def forward(
        self,
        debate_topic: str,
        debate_history: str,
        affirmative_argument: str = "",
    ):
        role_instruction = self.get_role_instruction()

        if self.role == "affirmative":
            llm_logger.info(
                "predict",
                role="affirmative",
                module=self.module_type,
                topic_len=len(debate_topic),
                history_len=len(debate_history),
            )
            return self.predict(
                debate_topic=debate_topic,
                debate_history=debate_history,
                role_instruction=role_instruction,
                persona=self.persona,
            )
        else:
            llm_logger.info(
                "predict",
                role="negative",
                module=self.module_type,
                topic_len=len(debate_topic),
                history_len=len(debate_history),
            )
            return self.predict(
                debate_topic=debate_topic,
                debate_history=debate_history,
                affirmative_argument=affirmative_argument,
                role_instruction=role_instruction,
                persona=self.persona,
            )


class JudgeModule(dspy.Module):
    """Judge module with discriminative and extractive modes."""

    def __init__(
        self,
        lm_discriminative: dspy.LM | None = None,
        lm_extractive: dspy.LM | None = None,
        module_type: str = "predict",
        tool_names: list[str] | None = None,
        react_max_iters: int = 3,
    ):
        super().__init__()
        self.module_type = module_type

        self.discriminative = build_predictor(
            JudgeDiscriminative,
            module_type,
            role="judge",
            lm=lm_discriminative,
            tool_names=tool_names,
            react_max_iters=react_max_iters,
        )
        self.extractive = build_predictor(
            JudgeExtractive,
            module_type,
            role="judge",
            lm=lm_extractive,
            tool_names=tool_names,
            react_max_iters=react_max_iters,
        )

    def evaluate_debate(
        self,
        debate_topic: str,
        debate_history: str,
        current_iteration: int,
    ) -> dspy.Prediction:
        """Discriminative mode: Decide if solution is found."""
        llm_logger.info(
            "judge_discriminative",
            iter=current_iteration,
            topic_len=len(debate_topic),
            history_len=len(debate_history),
        )
        return self.discriminative(
            debate_topic=debate_topic,
            debate_history=debate_history,
            current_iteration=current_iteration,
        )

    def extract_solution(
        self,
        debate_topic: str,
        debate_history: str,
    ) -> dspy.Prediction:
        """Extractive mode: Extract final answer from debate."""
        llm_logger.info(
            "judge_extractive",
            topic_len=len(debate_topic),
            history_len=len(debate_history),
        )
        return self.extractive(
            debate_topic=debate_topic,
            debate_history=debate_history,
        )


class MADFramework(dspy.Module):
    """Complete Multi-Agent Debate framework."""

    def __init__(
        self,
        max_iterations: int = 3,
        debate_level: int = 2,
        adaptive_break: bool = True,
        affirmative_persona: str = "",
        negative_persona: str = "",
        affirmative_lm: dspy.LM | None = None,
        negative_lm: dspy.LM | None = None,
        judge_lm_discriminative: dspy.LM | None = None,
        judge_lm_extractive: dspy.LM | None = None,
        affirmative_module_type: str = "predict",
        negative_module_type: str = "predict",
        judge_module_type: str = "predict",
        affirmative_tools: list[str] | None = None,
        negative_tools: list[str] | None = None,
        react_max_iters: int = 6,
        judge_tool_names: list[str] | None = None,
        on_iteration: EmitIteration | None = None,
    ):
        super().__init__()

        self.max_iterations = max_iterations
        self.debate_level = debate_level
        self.adaptive_break = adaptive_break
        self.on_iteration = on_iteration

        # Initialize agents
        self.affirmative = DebaterModule(
            "affirmative",
            debate_level,
            persona=affirmative_persona,
            lm=affirmative_lm,
            module_type=affirmative_module_type,
            tool_names=affirmative_tools,
            react_max_iters=react_max_iters,
        )
        self.negative = DebaterModule(
            "negative",
            debate_level,
            persona=negative_persona,
            lm=negative_lm,
            module_type=negative_module_type,
            tool_names=negative_tools,
            react_max_iters=react_max_iters,
        )
        self.judge = JudgeModule(
            lm_discriminative=judge_lm_discriminative,
            lm_extractive=judge_lm_extractive,
            module_type=judge_module_type,
            tool_names=judge_tool_names,
            react_max_iters=react_max_iters,
        )

    @staticmethod
    def format_history(history: list[DebateExchange]) -> str:
        """Format debate history as readable text."""
        if not history:
            return "No debate history yet."

        formatted = []
        for i, exchange in enumerate(history, 1):
            formatted.append(f"\n--- Iteration {i} ---")
            formatted.append(f"Affirmative: {exchange.affirmative}")
            formatted.append(f"Negative: {exchange.negative}")
            if exchange.judge_eval is not None:
                formatted.append(f"Judge: {exchange.judge_eval}")

        return "\n".join(formatted)

    def forward(
        self,
        debate_topic: str,
    ):
        """Run complete debate process."""
        history: list[DebateExchange] = []
        solution_found = False
        final_answer = None

        for iteration in range(1, self.max_iterations + 1):
            history_str = self.format_history(history)
            iter_token = set_current_iteration(iteration)
            try:
                # Affirmative speaks
                with mlflow_span(
                    "agent.affirmative",
                    span_type="AGENT",
                    inputs={
                        "debate_topic": debate_topic,
                        "debate_history": history_str,
                    },
                    attributes={
                        "agent.pattern": "debate",
                        "agent.role": "affirmative",
                        "agent.iteration": iteration,
                        "agent.module_type": getattr(
                            self.affirmative, "module_type", "unknown"
                        ),
                        "agent.answer_owner": False,
                    },
                ) as span:
                    aff_response = self.affirmative(
                        debate_topic=debate_topic,
                        debate_history=history_str,
                    )
                    if span is not None:
                        span.set_outputs(
                            {
                                "argument": aff_response.argument,
                                "reasoning": aff_response.reasoning,
                            }
                        )

                # Negative responds
                with mlflow_span(
                    "agent.negative",
                    span_type="AGENT",
                    inputs={
                        "debate_topic": debate_topic,
                        "debate_history": history_str,
                        "affirmative_argument": aff_response.argument,
                    },
                    attributes={
                        "agent.pattern": "debate",
                        "agent.role": "negative",
                        "agent.iteration": iteration,
                        "agent.module_type": getattr(
                            self.negative, "module_type", "unknown"
                        ),
                        "agent.answer_owner": False,
                    },
                ) as span:
                    neg_response = self.negative(
                        debate_topic=debate_topic,
                        debate_history=history_str,
                        affirmative_argument=aff_response.argument,
                    )
                    if span is not None:
                        span.set_outputs(
                            {
                                "counter_argument": neg_response.counter_argument,
                                "reasoning": neg_response.reasoning,
                            }
                        )

                # Record exchange
                exchange = DebateExchange(
                    affirmative=aff_response.argument,
                    affirmative_reasoning=aff_response.reasoning,
                    negative=neg_response.counter_argument,
                    negative_reasoning=neg_response.reasoning,
                )
                history.append(exchange)

                # Callback after initial exchange
                if self.on_iteration is not None:
                    self.on_iteration(
                        IterationEvent(
                            iteration, exchange, self.format_history(history)
                        )
                    )

                # Judge evaluates
                if self.adaptive_break:
                    history_str = self.format_history(history)
                    with mlflow_span(
                        "agent.judge.evaluate",
                        span_type="AGENT",
                        inputs={
                            "debate_topic": debate_topic,
                            "debate_history": history_str,
                            "current_iteration": iteration,
                        },
                        attributes={
                            "agent.pattern": "debate",
                            "agent.role": "judge",
                            "agent.operation": "evaluate",
                            "agent.iteration": iteration,
                            "agent.module_type": getattr(
                                self.judge, "module_type", "unknown"
                            ),
                            "agent.answer_owner": False,
                        },
                    ) as span:
                        judge_eval = self.judge.evaluate_debate(
                            debate_topic=debate_topic,
                            debate_history=history_str,
                            current_iteration=iteration,
                        )
                        if span is not None:
                            span.set_outputs(
                                {
                                    "solution_found": judge_eval.solution_found,
                                    "confidence": judge_eval.confidence,
                                    "reasoning": judge_eval.reasoning,
                                }
                            )

                    exchange = replace(exchange, judge_eval=judge_eval.reasoning)
                    history[-1] = exchange

                    # Callback after judge evaluation
                    if self.on_iteration is not None:
                        self.on_iteration(
                            IterationEvent(
                                iteration, exchange, self.format_history(history)
                            )
                        )

                    if judge_eval.solution_found and judge_eval.confidence > 0.7:
                        solution_found = True
                        with mlflow_span(
                            "agent.judge.final_answer",
                            span_type="AGENT",
                            inputs={
                                "debate_topic": debate_topic,
                                "debate_history": history_str,
                            },
                            attributes={
                                "agent.pattern": "debate",
                                "agent.role": "judge",
                                "agent.operation": "extract_final_answer",
                                "agent.iteration": iteration,
                                "agent.module_type": getattr(
                                    self.judge, "module_type", "unknown"
                                ),
                                "agent.answer_owner": True,
                            },
                        ) as span:
                            final_answer = self.judge.extract_solution(
                                debate_topic=debate_topic,
                                debate_history=history_str,
                            )
                            if span is not None:
                                span.set_outputs(
                                    {
                                        "final_answer": final_answer.final_answer,
                                        "justification": final_answer.justification,
                                    }
                                )
                        break
            finally:
                reset_current_iteration(iter_token)

        # Extract final answer if not found adaptively
        if not solution_found:
            history_str = self.format_history(history)
            with mlflow_span(
                "agent.judge.final_answer",
                span_type="AGENT",
                inputs={
                    "debate_topic": debate_topic,
                    "debate_history": history_str,
                },
                attributes={
                    "agent.pattern": "debate",
                    "agent.role": "judge",
                    "agent.operation": "extract_final_answer",
                    "agent.iteration": len(history),
                    "agent.module_type": getattr(
                        self.judge, "module_type", "unknown"
                    ),
                    "agent.answer_owner": True,
                },
            ) as span:
                final_prediction = self.judge.extract_solution(
                    debate_topic=debate_topic,
                    debate_history=history_str,
                )
                if span is not None:
                    span.set_outputs(
                        {
                            "final_answer": final_prediction.final_answer,
                            "justification": final_prediction.justification,
                        }
                    )
            final_answer = final_prediction

        if final_answer is None:
            return dspy.Prediction(
                final_answer="",
                justification="",
                history=history,
                iterations_used=len(history),
                stopped_early=solution_found,
            )

        return dspy.Prediction(
            final_answer=final_answer.final_answer,
            justification=final_answer.justification,
            history=history,
            iterations_used=len(history),
            stopped_early=solution_found,
        )


class DebatePattern(AgentPattern):
    name = "debate"

    def __init__(self) -> None:
        self._root = Path(__file__).resolve().parent

    def describe(self) -> str:
        readme = self._root / "README.md"
        if readme.exists():
            return readme.read_text(encoding="utf-8")
        return "Multi-Agent Debate pattern."

    def default_config_path(self) -> Path | None:
        cfg = self._root / "config.yaml"
        return cfg if cfg.exists() else None

    def available_configs(self) -> list[Path]:
        configs: list[Path] = []
        # default
        cfg = self.default_config_path()
        if cfg:
            configs.append(cfg)
        # scenarios directory optional
        scenarios = self._root / "scenarios"
        if scenarios.exists():
            for p in scenarios.glob("*.yaml"):
                configs.append(p)
        return configs

    def available_tools(self) -> list[str]:
        # reflect from default config
        cfg_path = self.default_config_path()
        if cfg_path:
            cfg = load_config(str(cfg_path))
            tools = set()
            aff = cfg.agents.get("affirmative")
            neg = cfg.agents.get("negative")
            if aff:
                tools.update(aff.tools)
            if neg:
                tools.update(neg.tools)
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
            aff_cfg = cfg.agents.get("affirmative")
            neg_cfg = cfg.agents.get("negative")
            framework = MADFramework(
                max_iterations=cfg.debate.max_iterations,
                debate_level=cfg.debate.debate_level,
                adaptive_break=cfg.debate.adaptive_break,
                affirmative_persona=(aff_cfg.persona if aff_cfg else ""),
                negative_persona=(neg_cfg.persona if neg_cfg else ""),
                affirmative_lm=(
                    build_lm(aff_cfg.lm) if aff_cfg and aff_cfg.lm else None
                ),
                negative_lm=(build_lm(neg_cfg.lm) if neg_cfg and neg_cfg.lm else None),
                judge_lm_discriminative=(
                    build_lm(cfg.judge.discriminative_lm)
                    if cfg.judge.discriminative_lm
                    else None
                ),
                judge_lm_extractive=(
                    build_lm(cfg.judge.extractive_lm)
                    if cfg.judge.extractive_lm
                    else None
                ),
                affirmative_module_type=(aff_cfg.module_type if aff_cfg else "predict"),
                negative_module_type=(neg_cfg.module_type if neg_cfg else "predict"),
                judge_module_type=cfg.judge.module_type,
                affirmative_tools=(aff_cfg.tools if aff_cfg else []),
                negative_tools=(neg_cfg.tools if neg_cfg else []),
                judge_tool_names=cfg.judge.tools,
                on_iteration=emit,
            )
            final = framework(debate_topic=current_request.topic)
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
    return DebatePattern()
