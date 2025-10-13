import dspy  # type: ignore
from typing import Optional, Callable, Dict, Any, List
from awesome_dspy_agents.logging_setup import get_logger
from .signatures import (
    AffirmativeDebater,
    NegativeDebater,
    JudgeDiscriminative,
    JudgeExtractive,
)
from awesome_dspy_agents.tools.registry import (
    registry,
    set_current_agent,
    reset_current_agent,
    set_current_iteration,
    reset_current_iteration,
)
from pathlib import Path
from awesome_dspy_agents.patterns.interface import AgentPattern
from awesome_dspy_agents.config import AppConfig, load_config, build_lm


llm_logger = get_logger("mad.llm", "llm_calls.log", max_bytes=2_000_000, backup_count=3)


class DebaterModule(dspy.Module):
    """Base debater module with configurable behavior."""

    def __init__(
        self,
        role: str,
        debate_level: int = 2,
        persona: str = "",
        lm: Optional[dspy.LM] = None,
        module_type: str = "predict",
        tool_names: Optional[List[str]] = None,
        react_max_iters: int = 3,
    ):
        super().__init__()
        self.role = role
        self.debate_level = debate_level
        self.persona = persona
        self.module_type = module_type
        self.tool_names = tool_names or []
        self.react_max_iters = react_max_iters

        # Select signature based on role
        if role == "affirmative":
            signature = AffirmativeDebater
        else:
            signature = NegativeDebater

        if module_type == "predict":
            self.predict = dspy.Predict(signature)
        elif module_type == "chain_of_thought":
            self.predict = dspy.ChainOfThought(signature)
        elif module_type == "react":
            tools = registry.build_dspy_tools(self.tool_names)
            llm_logger.info(
                "react_tools", role=self.role, tools=[str(t) for t in tools]
            )
            self.predict = dspy.ReAct(
                signature, tools=tools, max_iters=self.react_max_iters
            )
        else:
            raise ValueError(f"Unsupported module_type: {module_type}")

        # Optionally set a specific LM for this debater
        if lm is not None:
            # Set LM for this module and its children
            try:
                self.set_lm(lm)
            except Exception:
                # Fallback to setting on the predictor only
                self.predict.set_lm(lm)

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
            token = set_current_agent(self.role)
            out = self.predict(
                debate_topic=debate_topic,
                debate_history=debate_history,
                role_instruction=role_instruction,
                persona=self.persona,
            )
            reset_current_agent(token)
            return out
        else:
            llm_logger.info(
                "predict",
                role="negative",
                module=self.module_type,
                topic_len=len(debate_topic),
                history_len=len(debate_history),
            )
            token = set_current_agent(self.role)
            out = self.predict(
                debate_topic=debate_topic,
                debate_history=debate_history,
                affirmative_argument=affirmative_argument,
                role_instruction=role_instruction,
                persona=self.persona,
            )
            reset_current_agent(token)
            return out


class JudgeModule(dspy.Module):
    """Judge module with discriminative and extractive modes."""

    def __init__(
        self,
        lm_discriminative: Optional[dspy.LM] = None,
        lm_extractive: Optional[dspy.LM] = None,
        module_type: str = "predict",
        tool_names: Optional[List[str]] = None,
        react_max_iters: int = 3,
    ):
        super().__init__()
        self.module_type = module_type
        self.tool_names = tool_names or []
        self.react_max_iters = react_max_iters

        if module_type == "predict":
            self.discriminative = dspy.Predict(JudgeDiscriminative)
            self.extractive = dspy.Predict(JudgeExtractive)
        elif module_type == "chain_of_thought":
            self.discriminative = dspy.ChainOfThought(JudgeDiscriminative)
            self.extractive = dspy.ChainOfThought(JudgeExtractive)
        elif module_type == "react":
            tools = registry.build_dspy_tools(self.tool_names)
            llm_logger.info("judge_react_tools", tools=[str(t) for t in tools])
            self.discriminative = dspy.ReAct(
                JudgeDiscriminative, tools=tools, max_iters=self.react_max_iters
            )
            self.extractive = dspy.ReAct(
                JudgeExtractive, tools=tools, max_iters=self.react_max_iters
            )
        else:
            raise ValueError(f"Unsupported judge module_type: {module_type}")

        # Optionally set specific LMs for judge sub-modules
        if lm_discriminative is not None:
            try:
                self.discriminative.set_lm(lm_discriminative)
            except Exception:
                pass
        if lm_extractive is not None:
            try:
                self.extractive.set_lm(lm_extractive)
            except Exception:
                pass

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
        affirmative_lm: Optional[dspy.LM] = None,
        negative_lm: Optional[dspy.LM] = None,
        judge_lm_discriminative: Optional[dspy.LM] = None,
        judge_lm_extractive: Optional[dspy.LM] = None,
        affirmative_module_type: str = "predict",
        negative_module_type: str = "predict",
        judge_module_type: str = "predict",
        affirmative_tools: Optional[List[str]] = None,
        negative_tools: Optional[List[str]] = None,
        react_max_iters: int = 6,
        judge_tool_names: Optional[List[str]] = None,
        on_iteration: Optional[Callable[[int, Dict[str, Any], str], None]] = None,
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

        # Debate state
        self.debate_history: List[Dict[str, Any]] = []

    def format_history(self) -> str:
        """Format debate history as readable text."""
        if not self.debate_history:
            return "No debate history yet."

        formatted = []
        for i, exchange in enumerate(self.debate_history, 1):
            formatted.append(f"\n--- Iteration {i} ---")
            formatted.append(f"Affirmative: {exchange['affirmative']}")
            formatted.append(f"Negative: {exchange['negative']}")
            if "judge_eval" in exchange:
                formatted.append(f"Judge: {exchange['judge_eval']}")

        return "\n".join(formatted)

    def forward(
        self,
        debate_topic: str,
    ):
        """Run complete debate process."""
        self.debate_history = []
        solution_found = False
        final_answer = None

        for iteration in range(1, self.max_iterations + 1):
            history_str = self.format_history()
            iter_token = set_current_iteration(iteration)
            try:
                # Affirmative speaks
                aff_response = self.affirmative(
                    debate_topic=debate_topic,
                    debate_history=history_str,
                )

                # Negative responds
                neg_response = self.negative(
                    debate_topic=debate_topic,
                    debate_history=history_str,
                    affirmative_argument=aff_response.argument,
                )

                # Record exchange
                exchange = {
                    "affirmative": aff_response.argument,
                    "affirmative_reasoning": aff_response.reasoning,
                    "negative": neg_response.counter_argument,
                    "negative_reasoning": neg_response.reasoning,
                }
                self.debate_history.append(exchange)

                # Callback after initial exchange
                if self.on_iteration is not None:
                    try:
                        self.on_iteration(iteration, exchange, self.format_history())
                    except Exception:
                        pass

                # Judge evaluates
                if self.adaptive_break:
                    history_str = self.format_history()
                    judge_token = set_current_agent("judge")
                    try:
                        judge_eval = self.judge.evaluate_debate(
                            debate_topic=debate_topic,
                            debate_history=history_str,
                            current_iteration=iteration,
                        )
                    finally:
                        reset_current_agent(judge_token)

                    exchange["judge_eval"] = judge_eval.reasoning

                    # Callback after judge evaluation
                    if self.on_iteration is not None:
                        try:
                            self.on_iteration(iteration, exchange, self.format_history())
                        except Exception:
                            pass

                    if judge_eval.solution_found and judge_eval.confidence > 0.7:
                        solution_found = True
                        judge_token = set_current_agent("judge")
                        try:
                            final_answer = self.judge.extract_solution(
                                debate_topic=debate_topic,
                                debate_history=history_str,
                            )
                        finally:
                            reset_current_agent(judge_token)
                        break
            finally:
                reset_current_iteration(iter_token)

        # Extract final answer if not found adaptively
        if not solution_found:
            history_str = self.format_history()
            final_prediction = self.judge.extract_solution(
                debate_topic=debate_topic,
                debate_history=history_str,
            )
            final_answer = final_prediction

        if final_answer is None:
            return dspy.Prediction(
                final_answer="",
                justification="",
                debate_history=self.debate_history,
                iterations_used=len(self.debate_history),
                adaptive_break_triggered=solution_found,
            )

        return dspy.Prediction(
            final_answer=final_answer.final_answer,
            justification=final_answer.justification,
            debate_history=self.debate_history,
            iterations_used=len(self.debate_history),
            adaptive_break_triggered=solution_found,
        )


class DebatePattern(AgentPattern):
    name = "debate"

    def __init__(self) -> None:
        self._root = Path(__file__).resolve().parent

    def describe(self) -> str:
        readme = self._root / "README.md"
        if readme.exists():
            try:
                return readme.read_text(encoding="utf-8")
            except Exception:
                return "Multi-Agent Debate pattern."
        return "Multi-Agent Debate pattern."

    def default_config_path(self) -> Optional[Path]:
        cfg = self._root / "config.yaml"
        return cfg if cfg.exists() else None

    def available_configs(self) -> List[Path]:
        configs: List[Path] = []
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

    def available_tools(self) -> List[str]:
        # reflect from default config
        cfg_path = self.default_config_path()
        try:
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
        except Exception:
            pass
        return []

    def available_scripts(self) -> Dict[str, str]:
        # Placeholder for optimization scripts discovery later
        return {}

    def run(
        self,
        topic: str,
        config_path: Path,
        overrides: Optional[Dict[str, Any]] = None,
        on_iteration: Optional[Callable[[int, Dict[str, Any], str], None]] = None,
    ) -> Dict[str, Any]:
        cfg: AppConfig = load_config(str(config_path))

        # apply simple overrides for debate.* keys
        if overrides:
            debate_over = overrides.get("debate")
            if isinstance(debate_over, dict):
                if "max_iterations" in debate_over:
                    cfg.debate.max_iterations = int(debate_over["max_iterations"])  # type: ignore[assignment]
                if "debate_level" in debate_over:
                    cfg.debate.debate_level = int(debate_over["debate_level"])  # type: ignore[assignment]
                if "adaptive_break" in debate_over:
                    cfg.debate.adaptive_break = bool(debate_over["adaptive_break"])  # type: ignore[assignment]

        # Configure default lm if provided
        if cfg.default_lm is not None:
            dspy.configure(lm=build_lm(cfg.default_lm))

        # Build per-agent LMs
        aff_cfg = cfg.agents.get("affirmative")
        neg_cfg = cfg.agents.get("negative")

        aff_lm = build_lm(aff_cfg.lm) if aff_cfg and aff_cfg.lm else None
        neg_lm = build_lm(neg_cfg.lm) if neg_cfg and neg_cfg.lm else None

        judge_disc_lm = (
            build_lm(cfg.judge.discriminative_lm)
            if cfg.judge.discriminative_lm
            else None
        )
        judge_ext_lm = (
            build_lm(cfg.judge.extractive_lm) if cfg.judge.extractive_lm else None
        )

        # Create framework
        framework = MADFramework(
            max_iterations=cfg.debate.max_iterations,
            debate_level=cfg.debate.debate_level,
            adaptive_break=cfg.debate.adaptive_break,
            affirmative_persona=(aff_cfg.persona if aff_cfg else ""),
            negative_persona=(neg_cfg.persona if neg_cfg else ""),
            affirmative_lm=aff_lm,
            negative_lm=neg_lm,
            judge_lm_discriminative=judge_disc_lm,
            judge_lm_extractive=judge_ext_lm,
            affirmative_module_type=(aff_cfg.module_type if aff_cfg else "predict"),
            negative_module_type=(neg_cfg.module_type if neg_cfg else "predict"),
            judge_module_type=cfg.judge.module_type,
            affirmative_tools=(aff_cfg.tools if aff_cfg else []),
            negative_tools=(neg_cfg.tools if neg_cfg else []),
            judge_tool_names=(cfg.judge.tools if getattr(cfg.judge, "tools", None) is not None else []),
        )

        # attach iteration callback
        if on_iteration is not None:
            framework.on_iteration = on_iteration

        final = framework(debate_topic=topic)

        return {
            "final_answer": final.final_answer,
            "justification": final.justification,
            "iterations_used": final.iterations_used,
            "adaptive_break_triggered": final.adaptive_break_triggered,
            "history": final.debate_history,
        }


def get_pattern() -> AgentPattern:
    return DebatePattern()
