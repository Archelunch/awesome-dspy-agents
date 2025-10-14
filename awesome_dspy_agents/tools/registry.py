"""
Global tools registry for DSPy agents.

Provides:
- ToolRegistry: register callables by name and build dspy.Tool list
- Agent context helpers: set_current_agent/reset_current_agent for logging
- Built-in tools: math_eval, word_count, ascii_to_png

This module centralizes tool registration so all patterns can use the same
registry and logging.
"""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import base64
import inspect
import os
from contextvars import ContextVar
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List

import dspy  # type: ignore
from attachments.dspy import Attachments  # type: ignore

from awesome_dspy_agents.logging_setup import get_logger
from awesome_dspy_agents.tools.ascii_to_png import AsciiToPngConverter

tools_logger = get_logger("mad.tools", "tools.log", max_bytes=1_000_000, backup_count=3)


# Agent/iteration context for tool calls
_current_agent: ContextVar[str] = ContextVar("mad_current_agent", default="unknown")
_current_iteration: ContextVar[int] = ContextVar("mad_current_iteration", default=0)


def set_current_agent(role: str):
    return _current_agent.set(role)


def reset_current_agent(token: Any) -> None:
    try:
        _current_agent.reset(token)
    except Exception:
        pass


def get_current_agent() -> str:
    try:
        return _current_agent.get()
    except Exception:
        return "unknown"


def set_current_iteration(iteration: int):
    return _current_iteration.set(iteration)


def reset_current_iteration(token: Any) -> None:
    try:
        _current_iteration.reset(token)
    except Exception:
        pass


def get_current_iteration() -> int:
    try:
        return _current_iteration.get()
    except Exception:
        return 0


class ToolRegistry:
    """Simple registry mapping string names to callable tools.

    Wraps callables with dspy.Tool lazily to keep registration straightforward.
    """

    def __init__(self) -> None:
        self._functions: Dict[str, Callable[..., Any]] = {}
        self._listeners: List[Callable[[str, Dict[str, Any]], None]] = []
        # sandbox roots: absolute directories allowed for file tools
        self._sandbox_roots: List[Path] = []

    def add_listener(self, listener: Callable[[str, Dict[str, Any]], None]) -> None:
        """Subscribe to tool events.

        Listener signature: (event: str, payload: Dict[str, Any]) -> None

        Events emitted:
        - "tool_call": before a tool is executed
            payload keys: tool, agent, iteration, args_preview, kwargs_preview
        - "tool_result": after a tool returns
            payload keys: tool, agent, iteration, result_type, size
        - "tool_error": when a tool raises an exception
            payload keys: tool, agent, iteration, error
        """
        if listener not in self._listeners:
            self._listeners.append(listener)

    def set_sandbox_roots(self, roots: List[str]) -> None:
        self._sandbox_roots = [Path(os.path.expanduser(r)).resolve() for r in roots]

    def _in_sandbox(self, path: Path) -> bool:
        if not self._sandbox_roots:
            return True
        try:
            p = path.resolve()
        except Exception:
            return False
        return any(
            str(p).startswith(str(root) + os.sep) or p == root
            for root in self._sandbox_roots
        )

    def remove_listener(self, listener: Callable[[str, Dict[str, Any]], None]) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        for listener in list(self._listeners):
            try:
                listener(event, payload)
            except Exception:
                # Never let listeners break tool execution
                pass

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise TypeError("Tool must be callable")

        # Wrap with logging
        @wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            agent = get_current_agent()
            iteration = get_current_iteration()
            try:
                preview_args = str(args)[:200]
                preview_kwargs = str(kwargs)[:200]
                tools_logger.info(
                    "tool_call",
                    agent=agent,
                    tool=name,
                    iteration=iteration,
                    args_preview=preview_args,
                    kwargs_preview=preview_kwargs,
                )
                self._emit(
                    "tool_call",
                    {
                        "tool": name,
                        "agent": agent,
                        "iteration": iteration,
                        "args_preview": preview_args,
                        "kwargs_preview": preview_kwargs,
                    },
                )
            except Exception:
                pass
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                try:
                    tools_logger.info(
                        "tool_error",
                        agent=agent,
                        tool=name,
                        iteration=iteration,
                        error=str(e),
                    )
                    self._emit(
                        "tool_error",
                        {
                            "tool": name,
                            "agent": agent,
                            "iteration": iteration,
                            "error": str(e),
                        },
                    )
                except Exception:
                    pass
                raise
            try:
                result_repr = type(result).__name__
                size_hint = None
                if isinstance(result, str):
                    size_hint = len(result)
                tools_logger.info(
                    "tool_result",
                    agent=agent,
                    tool=name,
                    iteration=iteration,
                    result_type=result_repr,
                    size=size_hint,
                )
                self._emit(
                    "tool_result",
                    {
                        "tool": name,
                        "agent": agent,
                        "iteration": iteration,
                        "result_type": result_repr,
                        "size": size_hint,
                    },
                )
            except Exception:
                pass
            return result

        # Preserve original callable signature for DSPy Tool introspection
        try:
            _wrapped.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        except Exception:
            pass

        self._functions[name] = _wrapped

    def get(self, name: str) -> Callable[..., Any]:
        return self._functions[name]

    def names(self) -> List[str]:
        return sorted(self._functions.keys())

    def build_dspy_tools(self, names: List[str]) -> List[dspy.Tool]:  # type: ignore[name-defined]
        tools: List[dspy.Tool] = []  # type: ignore[name-defined]
        for n in names:
            fn = self._functions.get(n)
            if fn is None:
                raise KeyError(f"Unknown tool '{n}'")
            tools.append(dspy.Tool(fn))  # type: ignore[attr-defined]
        return tools


# Global registry instance and built-in example tools
registry = ToolRegistry()


def math_eval(expression: str) -> str:
    """Evaluate a simple Python math expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"error: {e}"


def word_count(text: str) -> str:
    """Count words in the input text."""
    return str(len(text.split()))


def ascii_to_png(text: str) -> dspy.Image:
    """Render ASCII text to PNG for internal analysis.

    Returns a data-URL (base64) image suitable for DSPy Image fields.
    """
    converter = AsciiToPngConverter(font_size=16, padding=20)
    img = converter.convert_text_to_image_hq(text, scale_factor=2)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    tools_logger.info("ascii_to_png", rendered_bytes=len(png_bytes))
    return dspy.Image(url=f"data:image/png;base64,{b64}")


def list_files(directory: str) -> str:
    """Return newline-separated absolute file paths inside a directory.

    If `directory` is a file, returns the absolute file path.
    Expands '~' and resolves relative paths.
    """
    try:
        base = Path(os.path.expanduser(directory)).resolve()
    except Exception as e:
        return f"error: {e}"

    if not base.exists():
        return f"error: path does not exist: {base}"
    if base.is_file():
        # sandbox check for file
        if not registry._in_sandbox(base):  # type: ignore[attr-defined]
            return f"error: access denied by sandbox: {base}"
        return str(base)

    try:
        files = []
        for p in base.iterdir():
            if p.is_file():
                rp = p.resolve()
                if registry._in_sandbox(rp):  # type: ignore[attr-defined]
                    files.append(str(rp))
        return "\n".join(files)
    except Exception as e:
        return f"error: {e}"


def read_file_attachment(path: str) -> Attachments:
    """Return an Attachments object for the given file path.

    This integrates with DSPy by returning an "Attachments"-typed object that
    models can consume via signatures that accept Attachments.
    """
    p = Path(os.path.expanduser(path)).resolve()
    # sandbox check
    if not registry._in_sandbox(p):  # type: ignore[attr-defined]
        raise PermissionError(f"access denied by sandbox: {p}")
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return Attachments(str(p))  # type: ignore[call-arg]


def write_file(path: str, content: str) -> str:
    """Write text content to a file (UTF-8). Returns absolute path or error."""
    try:
        p = Path(os.path.expanduser(path)).resolve()
        # sandbox check
        if not registry._in_sandbox(p):  # type: ignore[attr-defined]
            return f"error: access denied by sandbox: {p}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return str(p)
    except Exception as e:
        return f"error: {e}"


# Register built-ins
registry.register("math_eval", math_eval)
registry.register("word_count", word_count)
registry.register("ascii_to_png", ascii_to_png)
registry.register("list_files", list_files)
registry.register("read_file_attachment", read_file_attachment)
registry.register("write_file", write_file)
