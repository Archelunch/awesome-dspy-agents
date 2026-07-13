"""Per-run tool catalog, access policy, execution, and telemetry."""

from __future__ import annotations

import ast
import base64
import inspect
import logging
import math
import operator
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    Protocol,
)

import dspy
from attachments.dspy import Attachments  # type: ignore

from awesome_dspy_agents.logging_setup import get_logger
from awesome_dspy_agents.tools.ascii_to_png import AsciiToPngConverter

tools_logger = get_logger("mad.tools", "tools.log", max_bytes=1_000_000, backup_count=3)
logger = logging.getLogger(__name__)

_current_agent: ContextVar[str] = ContextVar("mad_current_agent", default="unknown")
_current_iteration: ContextVar[int] = ContextVar("mad_current_iteration", default=0)


def set_current_agent(role: str):
    return _current_agent.set(role)


def reset_current_agent(token: Any) -> None:
    _current_agent.reset(token)


def get_current_agent() -> str:
    return _current_agent.get()


def set_current_iteration(iteration: int):
    return _current_iteration.set(iteration)


def reset_current_iteration(token: Any) -> None:
    _current_iteration.reset(token)


def get_current_iteration() -> int:
    return _current_iteration.get()


@dataclass(frozen=True)
class ToolEvent:
    kind: str
    tool: str
    agent: str
    iteration: int
    details: Mapping[str, Any]


ToolListener = Callable[[ToolEvent], None]
_ambient_tool_listeners: ContextVar[tuple[ToolListener, ...]] = ContextVar(
    "dspy_agents_tool_listeners", default=()
)


@contextmanager
def tool_event_listener_scope(listener: ToolListener) -> Iterator[None]:
    """Observe tool events in the current execution context."""

    token = _ambient_tool_listeners.set((*_ambient_tool_listeners.get(), listener))
    try:
        yield
    finally:
        _ambient_tool_listeners.reset(token)


@dataclass(frozen=True)
class FileAccessPolicy:
    """Resolve filesystem access against explicit roots; no roots means deny."""

    roots: tuple[Path, ...] = ()

    @classmethod
    def from_paths(cls, roots: Sequence[Path | str]) -> FileAccessPolicy:
        return cls(tuple(Path(root).expanduser().resolve() for root in roots))

    def allows(self, path: Path | str) -> bool:
        if not self.roots:
            return False
        try:
            resolved = Path(path).expanduser().resolve()
            return any(
                resolved == root or resolved.is_relative_to(root) for root in self.roots
            )
        except (OSError, RuntimeError):
            return False

    def require(self, path: Path | str) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not self.allows(resolved):
            raise PermissionError(f"access denied by file policy: {resolved}")
        return resolved


ToolFactory = Callable[[FileAccessPolicy], Callable[..., Any]]


class ToolCatalog:
    """Immutable-by-convention catalog of named tool factories."""

    def __init__(self, definitions: Mapping[str, ToolFactory] | None = None) -> None:
        self._definitions = dict(definitions or {})

    def register(self, name: str, factory: ToolFactory) -> None:
        if name in self._definitions:
            raise ValueError(f"Tool '{name}' is already registered")
        self._definitions[name] = factory

    def build(self, name: str, policy: FileAccessPolicy) -> Callable[..., Any]:
        try:
            return self._definitions[name](policy)
        except KeyError as error:
            raise KeyError(f"Unknown tool '{name}'") from error

    def names(self) -> list[str]:
        return sorted(self._definitions)


class ToolExecutor:
    """Build DSPy tools with per-run policy and telemetry."""

    def __init__(
        self,
        catalog: ToolCatalog,
        policy: FileAccessPolicy,
        listener: ToolListener | None = None,
    ) -> None:
        self.catalog = catalog
        self.policy = policy
        self.listener = listener

    def get(self, name: str) -> Callable[..., Any]:
        return self._instrument(name, self.catalog.build(name, self.policy))

    def build_dspy_tools(self, names: list[str]) -> list[dspy.Tool]:
        return [dspy.Tool(self.get(name)) for name in names]

    def _emit(
        self,
        event: ToolEvent,
        *,
        ambient_details: Mapping[str, Any] | None = None,
    ) -> None:
        ambient_event = (
            ToolEvent(
                event.kind,
                event.tool,
                event.agent,
                event.iteration,
                ambient_details,
            )
            if ambient_details is not None
            else event
        )
        for listener in _ambient_tool_listeners.get():
            if self.listener is not None and listener is self.listener:
                continue
            try:
                listener(ambient_event)
            except Exception as error:
                logger.warning("Tool listener failed for %s: %s", event.tool, error)
        if self.listener is not None:
            try:
                self.listener(event)
            except Exception as error:
                logger.warning("Tool listener failed for %s: %s", event.tool, error)

    def _instrument(
        self, name: str, function: Callable[..., Any]
    ) -> Callable[..., Any]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            agent = get_current_agent()
            iteration = get_current_iteration()
            call_details = {
                "args_preview": str(args)[:200],
                "kwargs_preview": str(kwargs)[:200],
            }
            tools_logger.info(
                "tool_call", tool=name, agent=agent, iteration=iteration, **call_details
            )
            self._emit(ToolEvent("tool_call", name, agent, iteration, call_details))
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                details = {"error": str(error)}
                tools_logger.info(
                    "tool_error", tool=name, agent=agent, iteration=iteration, **details
                )
                self._emit(ToolEvent("tool_error", name, agent, iteration, details))
                raise
            details = {
                "result_type": type(result).__name__,
                "size": len(result) if isinstance(result, str) else None,
            }
            tools_logger.info(
                "tool_result", tool=name, agent=agent, iteration=iteration, **details
            )
            self._emit(
                ToolEvent("tool_result", name, agent, iteration, details),
                ambient_details={**details, "result_preview": str(result)[:2000]},
            )
            return result

        wrapped.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]
        return wrapped


class ToolProvider(Protocol):
    def build_dspy_tools(self, names: list[str]) -> list[dspy.Tool]: ...


_current_executor: ContextVar[ToolExecutor | None] = ContextVar(
    "dspy_agents_tool_executor", default=None
)


@contextmanager
def tool_executor_scope(executor: ToolExecutor) -> Iterator[None]:
    token = _current_executor.set(executor)
    try:
        yield
    finally:
        _current_executor.reset(token)


class RuntimeToolProvider:
    def build_dspy_tools(self, names: list[str]) -> list[dspy.Tool]:
        executor = _current_executor.get()
        if executor is None:
            raise RuntimeError("Tools must be built inside a Pattern run")
        return executor.build_dspy_tools(names)


runtime_tool_provider = RuntimeToolProvider()


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_INTEGER_BITS = 4096


def _require_bounded_number(value: int | float) -> int | float:
    if isinstance(value, int) and value.bit_length() > _MAX_INTEGER_BITS:
        raise ValueError("arithmetic result is too large")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("arithmetic result is not finite")
    return value


def _require_bounded_operation(
    operation: ast.operator, left: int | float, right: int | float
) -> None:
    if not isinstance(left, int) or not isinstance(right, int):
        return
    if isinstance(operation, ast.Mult):
        projected_bits = left.bit_length() + right.bit_length()
    elif isinstance(operation, ast.Pow) and right >= 0:
        projected_bits = max(1, left.bit_length()) * right
    else:
        return
    if projected_bits > _MAX_INTEGER_BITS:
        raise ValueError("arithmetic result is too large")


def _evaluate_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_arithmetic(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return _require_bounded_number(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_arithmetic(node.left)
        right = _evaluate_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent is too large")
        _require_bounded_operation(node.op, left, right)
        result = _BINARY_OPERATORS[type(node.op)](left, right)
        return _require_bounded_number(result)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        result = _UNARY_OPERATORS[type(node.op)](_evaluate_arithmetic(node.operand))
        return _require_bounded_number(result)
    raise ValueError("only arithmetic expressions are allowed")


def math_eval(expression: str) -> str:
    """Evaluate an arithmetic expression without executing Python code."""
    if len(expression) > 500:
        return "error: expression is too long"
    try:
        return str(_evaluate_arithmetic(ast.parse(expression, mode="eval")))
    except Exception as error:
        return f"error: {error}"


def word_count(text: str) -> str:
    """Count words in the input text."""
    return str(len(text.split()))


def ascii_to_png(text: str) -> dspy.Image:
    """Render ASCII text to a PNG data URL."""
    converter = AsciiToPngConverter(font_size=16, padding=20)
    image = converter.convert_text_to_image_hq(text, scale_factor=2)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return dspy.Image(url=f"data:image/png;base64,{encoded}")


def _list_files(policy: FileAccessPolicy) -> Callable[[str], str]:
    def list_files(directory: str) -> str:
        """List files immediately inside an allowed directory."""
        try:
            base = policy.require(directory)
            if not base.exists():
                return f"error: path does not exist: {base}"
            if base.is_file():
                return str(base)
            return "\n".join(
                str(path.resolve())
                for path in base.iterdir()
                if path.is_file() and policy.allows(path)
            )
        except Exception as error:
            return f"error: {error}"

    return list_files


def _read_file_attachment(policy: FileAccessPolicy) -> Callable[[str], Attachments]:
    def read_file_attachment(path: str) -> Attachments:
        """Return an attachment for an allowed local file."""
        resolved = policy.require(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")
        return Attachments(str(resolved))  # type: ignore[call-arg]

    return read_file_attachment


def _write_file(policy: FileAccessPolicy) -> Callable[[str, str], str]:
    def write_file(path: str, content: str) -> str:
        """Write UTF-8 text to an allowed path."""
        try:
            resolved = policy.require(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return str(resolved)
        except Exception as error:
            return f"error: {error}"

    return write_file


def _constant_tool(function: Callable[..., Any]) -> ToolFactory:
    return lambda _policy: function


default_catalog = ToolCatalog(
    {
        "math_eval": _constant_tool(math_eval),
        "word_count": _constant_tool(word_count),
        "ascii_to_png": _constant_tool(ascii_to_png),
        "list_files": _list_files,
        "read_file_attachment": _read_file_attachment,
        "write_file": _write_file,
    }
)
