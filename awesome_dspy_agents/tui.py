from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console  # type: ignore
from rich.markdown import Markdown  # type: ignore
from rich.panel import Panel  # type: ignore
from rich.table import Table  # type: ignore

console = Console()


def render_header(pattern_name: str, topic: str) -> None:
    console.print(
        Panel.fit(
            f"Running [bold cyan]{pattern_name}[/bold cyan] Pattern",
            style="bold blue",
        )
    )
    console.print(Panel.fit(Markdown(f"**Topic:** {topic}")))


def render_iteration(iteration: int, exchange: Dict[str, Any]) -> None:
    if "judge_eval" in exchange:
        console.print(
            Panel.fit(
                exchange["judge_eval"],
                title=f"Judge @ {iteration}",
                border_style="yellow",
            )
        )
        return

    if "affirmative" in exchange or "negative" in exchange:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Iteration", justify="right", style="cyan", width=10)
        table.add_column("Affirmative", style="green")
        table.add_column("Negative", style="red")
        table.add_row(
            str(iteration),
            exchange.get("affirmative", ""),
            exchange.get("negative", ""),
        )
        console.print(table)
        return

    if "addition" in exchange or "subtraction" in exchange:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Iteration", justify="right", style="cyan", width=10)
        table.add_column("Addition", style="green")
        table.add_column("Subtraction", style="red")
        table.add_row(
            str(iteration),
            exchange.get("addition", ""),
            exchange.get("subtraction", ""),
        )
        console.print(table)
        if exchange.get("feedback"):
            console.print(
                Panel.fit(
                    exchange["feedback"],
                    title=f"Feedback @ {iteration}",
                    border_style="yellow",
                )
            )
        return

    console.print(Panel.fit(str(exchange), title=f"Iteration {iteration}"))


def render_tool_events(
    iteration: int, events: List[Dict[str, Any]], start_idx: int
) -> int:
    new_events = events[start_idx:]
    if not new_events:
        return start_idx
    t = Table(show_header=True, header_style="bold blue")
    t.title = f"Tools used @ iteration {iteration} (+{len(new_events)})"
    t.add_column("Agent", style="cyan")
    t.add_column("Tool", style="magenta")
    t.add_column("Event", style="yellow")
    t.add_column("Details", style="green")
    for ev in new_events:
        agent = ev.get("agent", "unknown")
        tool = ev.get("tool", "?")
        evt = ev.get("event", "")
        details = ""
        if evt == "tool_call":
            details = ev.get("args_preview", "")
        elif evt == "tool_result":
            rt = ev.get("result_type", "")
            sz = ev.get("size")
            details = f"{rt} size={sz}" if sz is not None else rt
        elif evt == "tool_error":
            details = ev.get("error", "")
        t.add_row(str(agent), str(tool), str(evt), str(details))
    console.print(t)
    return start_idx + len(new_events)


def render_final(final_answer: str, justification: str) -> None:
    console.rule("Final Decision")
    console.print(
        Panel.fit(justification, title="Justification", border_style="magenta")
    )
    console.print(Panel.fit(final_answer, title="Final Answer", border_style="green"))
