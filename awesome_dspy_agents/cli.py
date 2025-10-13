# pyright: reportMissingTypeStubs=false
from pathlib import Path
from typing import Optional, Dict, Any, List

import dspy
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
import questionary

from awesome_dspy_agents.patterns.interface import find_patterns
from awesome_dspy_agents.config import AppConfig, build_lm
from awesome_dspy_agents.tools.registry import registry


# --- Setup ---

app = typer.Typer(
    name="dspy-agents",
    help="CLI to manage and run DSPy multi-agent patterns.",
    add_completion=True,
    rich_markup_mode="markdown",
)

console = Console()
PATTERNS_DIR = Path(__file__).parent / "patterns"


# --- Helper Functions ---


def _install_default_lm(cfg: AppConfig) -> None:
    if cfg.default_lm is None:
        return
    default_lm = build_lm(cfg.default_lm)
    dspy.configure(lm=default_lm)


def _iteration_callback(iteration: int, exchange: dict, history: str) -> None:
    # When called the first time (no judge_eval), print the exchange table.
    # When called the second time (with judge_eval), only print the judge panel.
    if "judge_eval" not in exchange:
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

    if "judge_eval" in exchange:
        console.print(
            Panel.fit(
                exchange["judge_eval"],
                title=f"Judge @ {iteration}",
                border_style="yellow",
            )
        )


def _pattern_map():
    return find_patterns(PATTERNS_DIR)


# --- CLI Commands ---


@app.command("list")
def list_cmd():
    """List all available agent patterns."""
    console.print(Panel.fit("Available DSPy Agent Patterns", style="bold blue"))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Pattern", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Configs", justify="right", style="magenta")
    table.add_column("Tools", style="yellow")

    patterns = _pattern_map()
    for name, pat in patterns.items():
        # short description from first non-empty line
        desc_text = pat.describe() or ""
        first_line = ""
        for line in desc_text.splitlines():
            if line.strip():
                first_line = line.strip().lstrip("# ")
                break
        if len(first_line) > 90:
            first_line = first_line[:87] + "..."

        # configs and tools summary
        cfgs = list(pat.available_configs())
        cfg_count = str(len(cfgs))
        tools = list(pat.available_tools())
        if tools:
            tools_preview = ", ".join(tools[:3]) + (" …" if len(tools) > 3 else "")
        else:
            tools_preview = "-"

        table.add_row(name, first_line or "-", cfg_count, tools_preview)

    console.print(table)


@app.command()
def describe(
    pattern_name: str = typer.Argument(
        ..., help="The name of the pattern to describe."
    ),
):
    """Show details about a specific pattern."""
    patterns = _pattern_map()
    pat = patterns.get(pattern_name)
    if not pat:
        console.print(
            f"[bold red]Error:[/bold red] Pattern '{pattern_name}' not found."
        )
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(f"Pattern: [bold cyan]{pattern_name}[/bold cyan]", style="bold blue")
    )

    desc = pat.describe()
    if desc:
        console.print(Panel(Markdown(desc), title="Description"))

    cfg = pat.default_config_path()
    if cfg and cfg.exists():
        with open(cfg, "r") as f:
            console.print(Panel(f.read(), title="Default Configuration (config.yaml)"))
    else:
        console.print("No config.yaml found for this pattern.")


@app.command()
def run(
    pattern_name: str = typer.Argument(..., help="The name of the pattern to run."),
    topic: Optional[str] = typer.Argument(
        None, help="The topic or task for the pattern."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a YAML config file."
    ),
    set: Optional[list[str]] = typer.Option(
        None, "--set", help="Override config values, e.g. debate.max_iterations=5"
    ),
):
    """Run a specific agent pattern."""
    patterns = _pattern_map()
    pat = patterns.get(pattern_name)
    if not pat:
        console.print(
            f"[bold red]Error:[/bold red] Pattern '{pattern_name}' not found."
        )
        raise typer.Exit(code=1)
    # Determine config path
    if config:
        config_path = config
    else:
        cfg = pat.default_config_path()
        if not cfg:
            console.print(
                f"[bold red]Error:[/bold red] No default config for '{pattern_name}'. Use --config"
            )
            raise typer.Exit(code=1)
        config_path = cfg

    if not config_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Config file not found at {config_path}"
        )
        raise typer.Exit(code=1)

    if not topic:
        topic = typer.prompt("Please enter the topic/task")

    # Parse --set overrides into nested dict
    overrides: dict = {}
    if set:
        for kv in set:
            if "=" not in kv:
                continue
            key, val = kv.split("=", 1)
            # build nested dict e.g. debate.max_iterations -> {debate: {max_iterations: val}}
            cursor = overrides
            parts = key.split(".")
            for p in parts[:-1]:
                cursor = cursor.setdefault(p, {})
            cursor[parts[-1]] = val

    console.print(
        Panel.fit(
            f"Running [bold cyan]{pattern_name}[/bold cyan] Pattern", style="bold blue"
        )
    )
    console.print(Panel.fit(Markdown(f"**Topic:** {topic}")))

    # Tool event capture for TUI
    tool_events_by_iter: Dict[int, List[Dict[str, Any]]] = {}
    printed_index_by_iter: Dict[int, int] = {}

    def _tool_listener(event: str, payload: Dict[str, Any]) -> None:
        try:
            iteration = int(payload.get("iteration", 0))
        except Exception:
            iteration = 0
        if iteration <= 0:
            return
        tool_events_by_iter.setdefault(iteration, []).append({
            "event": event,
            **payload,
        })

    def _on_iteration(iteration: int, exchange: dict, history: str) -> None:
        # Print debate exchange or judge panel as before
        if "judge_eval" not in exchange:
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
        else:
            console.print(
                Panel.fit(
                    exchange["judge_eval"],
                    title=f"Judge @ {iteration}",
                    border_style="yellow",
                )
            )

        # Print any new tool events for this iteration
        events = tool_events_by_iter.get(iteration, [])
        start_idx = printed_index_by_iter.get(iteration, 0)
        new_events = events[start_idx:]
        if new_events:
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
            printed_index_by_iter[iteration] = start_idx + len(new_events)

    # Register listener and run
    registry.add_listener(_tool_listener)
    try:
        result = pat.run(
            topic=topic,
            config_path=config_path,
            overrides=overrides,
            on_iteration=_on_iteration,
        )
    except Exception as e:
        console.print(f"[bold red]Error during run:[/bold red] {e}")
        raise typer.Exit(code=1)
    finally:
        registry.remove_listener(_tool_listener)

    console.rule("Final Decision")
    console.print(
        Panel.fit(
            result.get("justification", ""),
            title="Justification",
            border_style="magenta",
        )
    )
    console.print(
        Panel.fit(
            result.get("final_answer", ""), title="Final Answer", border_style="green"
        )
    )


@app.command("tools")
def tools(
    pattern_name: Optional[str] = typer.Option(
        None, "--pattern", help="Filter tools for a specific pattern"
    ),
):
    """List available tools (global or for a given pattern)."""
    console.print(Panel.fit("Available Tools", style="bold blue"))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Source", style="green")

    if pattern_name:
        pat = _pattern_map().get(pattern_name)
        if not pat:
            console.print(
                f"[bold red]Error:[/bold red] Pattern '{pattern_name}' not found."
            )
            raise typer.Exit(code=1)
        for t in pat.available_tools():
            table.add_row(t, f"pattern:{pattern_name}")
    else:
        for t in registry.names():
            table.add_row(t, "global")

    console.print(table)


@app.command("configs")
def configs(
    pattern_name: str = typer.Argument(..., help="Pattern to list configs for"),
):
    """List available configs/scenarios for a pattern."""
    pat = _pattern_map().get(pattern_name)
    if not pat:
        console.print(
            f"[bold red]Error:[/bold red] Pattern '{pattern_name}' not found."
        )
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"Configs for [bold cyan]{pattern_name}[/bold cyan]", style="bold blue"
        )
    )
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="green")
    for p in pat.available_configs():
        table.add_row(p.stem, str(p))
    console.print(table)


@app.command("scripts")
def scripts(
    pattern_name: str = typer.Argument(
        ..., help="Pattern to list optimization/eval scripts for"
    ),
):
    """List optimization/evaluation scripts provided by a pattern."""
    pat = _pattern_map().get(pattern_name)
    if not pat:
        console.print(
            f"[bold red]Error:[/bold red] Pattern '{pattern_name}' not found."
        )
        raise typer.Exit(code=1)
    scripts_map = pat.available_scripts()
    console.print(
        Panel.fit(
            f"Scripts for [bold cyan]{pattern_name}[/bold cyan]", style="bold blue"
        )
    )
    if not scripts_map:
        console.print("No scripts available.")
        return
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Script", style="cyan")
    table.add_column("Description", style="green")
    for k, v in scripts_map.items():
        table.add_row(k, v)
    console.print(table)


@app.command("interactive")
def interactive():
    """Guided run: choose pattern, config, topic, and overrides interactively."""
    patterns = _pattern_map()
    if not patterns:
        console.print("No patterns found.")
        raise typer.Exit(code=1)

    names = sorted(patterns.keys())
    pattern_name = questionary.select("Select a pattern", choices=names).ask()
    pat = patterns.get(pattern_name)
    if not pat:
        console.print(
            f"[bold red]Error:[/bold red] Pattern '{pattern_name}' not found."
        )
        raise typer.Exit(code=1)

    configs = list(pat.available_configs())
    config_path: Optional[Path] = None
    if configs:
        console.print(Panel.fit("Available configs:", style="bold blue"))
        for p in configs:
            console.print(f"- {p.stem}: {p}")
        choice = typer.prompt("Config name (leave empty for default)", default="")
        if choice:
            for p in configs:
                if p.stem == choice:
                    config_path = p
                    break
    if config_path is None:
        config_path = pat.default_config_path()
        if not config_path:
            console.print(
                "No default config found; please provide a config with --config in 'run'."
            )
            raise typer.Exit(code=1)

    topic = typer.prompt("Enter topic/task")

    overrides: dict = {}
    if typer.confirm("Override debate.max_iterations?", default=False):
        val = typer.prompt("New max_iterations", default="5")
        overrides.setdefault("debate", {})["max_iterations"] = val

    console.print(
        Panel.fit(
            f"Running [bold cyan]{pattern_name}[/bold cyan] Pattern", style="bold blue"
        )
    )
    console.print(Panel.fit(Markdown(f"**Topic:** {topic}")))
    # Tool event capture for interactive TUI
    tool_events_by_iter: Dict[int, List[Dict[str, Any]]] = {}
    printed_index_by_iter: Dict[int, int] = {}

    def _tool_listener(event: str, payload: Dict[str, Any]) -> None:
        try:
            iteration = int(payload.get("iteration", 0))
        except Exception:
            iteration = 0
        if iteration <= 0:
            return
        tool_events_by_iter.setdefault(iteration, []).append({
            "event": event,
            **payload,
        })

    def _on_iteration(iteration: int, exchange: dict, history: str) -> None:
        if "judge_eval" not in exchange:
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
        else:
            console.print(
                Panel.fit(
                    exchange["judge_eval"],
                    title=f"Judge @ {iteration}",
                    border_style="yellow",
                )
            )

        events = tool_events_by_iter.get(iteration, [])
        start_idx = printed_index_by_iter.get(iteration, 0)
        new_events = events[start_idx:]
        if new_events:
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
            printed_index_by_iter[iteration] = start_idx + len(new_events)

    registry.add_listener(_tool_listener)
    try:
        result = patterns[pattern_name].run(
            topic=topic,
            config_path=config_path,
            overrides=overrides,
            on_iteration=_on_iteration,
        )
    finally:
        registry.remove_listener(_tool_listener)
    console.rule("Final Decision")
    console.print(
        Panel.fit(
            result.get("justification", ""),
            title="Justification",
            border_style="magenta",
        )
    )
    console.print(
        Panel.fit(
            result.get("final_answer", ""), title="Final Answer", border_style="green"
        )
    )


if __name__ == "__main__":
    app()
