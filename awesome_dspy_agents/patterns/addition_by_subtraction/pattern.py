from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import dspy  # type: ignore

from awesome_dspy_agents.config import AppConfig, build_lm, load_config
from awesome_dspy_agents.logging_setup import get_logger
from awesome_dspy_agents.patterns.interface import AgentPattern
from awesome_dspy_agents.tools.registry import (registry, reset_current_agent,
                                                reset_current_iteration,
                                                set_current_agent,
                                                set_current_iteration)

from .signatures import AdditionAgentSignature, SubtractionAgentSignature

llm_logger = get_logger("mad.llm", "llm_calls.log", max_bytes=2_000_000, backup_count=3)


class AdditionModule(dspy.Module):
    def __init__(
        self,
        persona: str = "",
        lm: Optional[dspy.LM] = None,
        module_type: str = "predict",
        tool_names: Optional[List[str]] = None,
        react_max_iters: int = 3,
    ):
        super().__init__()
        self.persona = persona
        self.module_type = module_type
        self.tool_names = tool_names or []
        self.react_max_iters = react_max_iters

        if module_type == "predict":
            self.predict = dspy.Predict(AdditionAgentSignature)
        elif module_type == "chain_of_thought":
            self.predict = dspy.ChainOfThought(AdditionAgentSignature)
        elif module_type == "react":
            tools = registry.build_dspy_tools(self.tool_names)
            llm_logger.info(
                "abs_react_tools", role="addition", tools=[str(t) for t in tools]
            )
            self.predict = dspy.ReAct(
                AdditionAgentSignature, tools=tools, max_iters=self.react_max_iters
            )
        else:
            raise ValueError(f"Unsupported module_type: {module_type}")

        if lm is not None:
            try:
                self.set_lm(lm)
            except Exception:
                self.predict.set_lm(lm)

    def forward(self, context: str, instruction: str, history: str):
        llm_logger.info(
            "predict",
            role="addition",
            module=self.module_type,
            ctx_len=len(context),
            hist_len=len(history),
        )
        token = set_current_agent("addition")
        try:
            out = self.predict(
                context=context,
                instruction=instruction,
                history=history,
                persona=self.persona,
            )
        finally:
            reset_current_agent(token)
        return out


class SubtractionModule(dspy.Module):
    def __init__(
        self,
        persona: str = "",
        lm: Optional[dspy.LM] = None,
        module_type: str = "predict",
        tool_names: Optional[List[str]] = None,
        react_max_iters: int = 3,
    ):
        super().__init__()
        self.persona = persona
        self.module_type = module_type
        self.tool_names = tool_names or []
        self.react_max_iters = react_max_iters

        if module_type == "predict":
            self.predict = dspy.Predict(SubtractionAgentSignature)
        elif module_type == "chain_of_thought":
            self.predict = dspy.ChainOfThought(SubtractionAgentSignature)
        elif module_type == "react":
            tools = registry.build_dspy_tools(self.tool_names)
            llm_logger.info(
                "abs_react_tools", role="subtraction", tools=[str(t) for t in tools]
            )
            self.predict = dspy.ReAct(
                SubtractionAgentSignature, tools=tools, max_iters=self.react_max_iters
            )
        else:
            raise ValueError(f"Unsupported module_type: {module_type}")

        if lm is not None:
            try:
                self.set_lm(lm)
            except Exception:
                self.predict.set_lm(lm)

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
        token = set_current_agent("subtraction")
        try:
            out = self.predict(
                context=context,
                instruction=instruction,
                history=history,
                candidate_response=candidate_response,
                persona=self.persona,
            )
        finally:
            reset_current_agent(token)
        return out


class ABSFramework(dspy.Module):
    """Addition-by-Subtraction collaboration framework."""

    def __init__(
        self,
        max_iterations: int = 2,
        early_exit: bool = True,
        addition_persona: str = "",
        subtraction_persona: str = "",
        addition_lm: Optional[dspy.LM] = None,
        subtraction_lm: Optional[dspy.LM] = None,
        addition_module_type: str = "predict",
        subtraction_module_type: str = "predict",
        addition_tools: Optional[List[str]] = None,
        subtraction_tools: Optional[List[str]] = None,
        react_max_iters: int = 6,
        on_iteration: Optional[Callable[[int, Dict[str, Any], str], None]] = None,
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

        self.history: List[Dict[str, Any]] = []

    def format_history(self) -> str:
        if not self.history:
            return "No conversation yet."
        formatted = []
        for i, ex in enumerate(self.history, 1):
            formatted.append(f"\n--- Iteration {i} ---")
            if "addition" in ex:
                formatted.append(f"Addition: {ex['addition']}")
            if "subtraction" in ex:
                formatted.append(f"Subtraction: {ex['subtraction']}")
            if "feedback" in ex:
                formatted.append(f"Feedback: {ex['feedback']}")
        return "\n".join(formatted)

    def forward(self, context: str, instruction: str):
        self.history = []
        final_response = ""

        H_context = context
        H_instruction = instruction

        prev_R = None
        for iteration in range(1, self.max_iterations + 1):
            hist_str = self.format_history()
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

                exchange = {
                    "addition": add_out.candidate_response,
                    "addition_reasoning": add_out.reasoning,
                    "subtraction": sub_out.refined_response,
                    "feedback": sub_out.feedback,
                }
                self.history.append(exchange)

                # callback for TUI
                if self.on_iteration is not None:
                    try:
                        self.on_iteration(iteration, exchange, self.format_history())
                    except Exception:
                        pass

                # Early exit if no changes
                if (
                    prev_R is not None
                    and prev_R.strip() == add_out.candidate_response.strip()
                    and self.early_exit
                ):
                    final_response = add_out.candidate_response
                    break

                prev_R = add_out.candidate_response
                H_context = context
                H_instruction = instruction
            finally:
                reset_current_iteration(iter_token)

        # Final output: prefer last refined response if exists, else last candidate
        if self.history:
            last = self.history[-1]
            final_response = last.get("subtraction") or last.get("addition") or ""

        return dspy.Prediction(
            final_answer=final_response,
            justification="Refined via Addition-by-Subtraction iterative collaboration.",
            history=self.history,
            iterations_used=len(self.history),
            adaptive_break_triggered=self.early_exit
            and len(self.history) < self.max_iterations,
        )


class AdditionBySubtractionPattern(AgentPattern):
    name = "addition_by_subtraction"

    def __init__(self) -> None:
        self._root = Path(__file__).resolve().parent

    def describe(self) -> str:
        readme = self._root / "README.md"
        if readme.exists():
            try:
                return readme.read_text(encoding="utf-8")
            except Exception:
                return "Addition-by-Subtraction collaboration pattern."
        return "Addition-by-Subtraction collaboration pattern."

    def default_config_path(self) -> Optional[Path]:
        cfg = self._root / "config.yaml"
        return cfg if cfg.exists() else None

    def available_configs(self) -> List[Path]:
        configs: List[Path] = []
        cfg = self.default_config_path()
        if cfg:
            configs.append(cfg)
        scenarios = self._root / "scenarios"
        if scenarios.exists():
            for p in scenarios.glob("*.yaml"):
                configs.append(p)
        return configs

    def available_tools(self) -> List[str]:
        cfg_path = self.default_config_path()
        try:
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
        except Exception:
            pass
        return []

    def available_scripts(self) -> Dict[str, str]:
        return {}

    def run(
        self,
        topic: str,
        config_path: Path,
        overrides: Optional[Dict[str, Any]] = None,
        on_iteration: Optional[Callable[[int, Dict[str, Any], str], None]] = None,
    ) -> Dict[str, Any]:
        # In ABS, interpret topic as the instruction; allow empty context by default
        cfg: AppConfig = load_config(str(config_path))

        # Apply overrides for abs.* keys optionally
        if overrides:
            abs_over = overrides.get("abs")
            if isinstance(abs_over, dict):
                if "max_iterations" in abs_over:
                    cfg.abs.max_iterations = int(abs_over["max_iterations"])  # type: ignore[assignment]
                if "early_exit" in abs_over:
                    cfg.abs.early_exit = bool(abs_over["early_exit"])  # type: ignore[assignment]

        if cfg.default_lm is not None:
            dspy.configure(lm=build_lm(cfg.default_lm))

        add_cfg = cfg.agents.get("addition")
        sub_cfg = cfg.agents.get("subtraction")

        add_lm = build_lm(add_cfg.lm) if add_cfg and add_cfg.lm else None
        sub_lm = build_lm(sub_cfg.lm) if sub_cfg and sub_cfg.lm else None

        framework = ABSFramework(
            max_iterations=cfg.abs.max_iterations,
            early_exit=cfg.abs.early_exit,
            addition_persona=(add_cfg.persona if add_cfg else ""),
            subtraction_persona=(sub_cfg.persona if sub_cfg else ""),
            addition_lm=add_lm,
            subtraction_lm=sub_lm,
            addition_module_type=(add_cfg.module_type if add_cfg else "predict"),
            subtraction_module_type=(sub_cfg.module_type if sub_cfg else "predict"),
            addition_tools=(add_cfg.tools if add_cfg else []),
            subtraction_tools=(sub_cfg.tools if sub_cfg else []),
        )

        if on_iteration is not None:
            framework.on_iteration = on_iteration

        # For simplicity, treat topic as instruction; context is empty
        final = framework(context="", instruction=topic)

        return {
            "final_answer": final.final_answer,
            "justification": final.justification,
            "iterations_used": final.iterations_used,
            "adaptive_break_triggered": final.adaptive_break_triggered,
            "history": final.history,
        }


def get_pattern() -> AgentPattern:
    return AdditionBySubtractionPattern()
