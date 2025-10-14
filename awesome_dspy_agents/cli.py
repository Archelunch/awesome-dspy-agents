# pyright: reportMissingTypeStubs=false
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy  # type: ignore
import questionary  # type: ignore
import typer  # type: ignore
from rich.console import Console  # type: ignore
from rich.markdown import Markdown  # type: ignore
from rich.panel import Panel  # type: ignore
from rich.table import Table  # type: ignore

from awesome_dspy_agents import __version__, tui
from awesome_dspy_agents.config import AppConfig, build_lm
from awesome_dspy_agents.patterns.interface import find_patterns
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


@app.callback()
def _setup(
    allow_path: Optional[List[str]] = typer.Option(
        None,
        "--allow-path",
        help="Allow file tools to access given absolute directories (repeatable)",
    ),
):
    if allow_path:
        registry.set_sandbox_roots(allow_path)


def _coerce_value(s: str):
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _parse_overrides(set: Optional[list[str]]) -> dict:
    overrides: dict = {}
    if not set:
        return overrides
    for kv in set:
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        cursor = overrides
        parts = key.split(".")
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = _coerce_value(val)
    return overrides


def _execute_pattern(
    pattern_name: str,
    topic: str,
    config_path: Path,
    overrides: dict,
    stream: bool,
    json_output: bool,
    save_path: Optional[Path],
) -> Dict[str, Any]:
    patterns = _pattern_map()
    pat = patterns.get(pattern_name)
    if not pat:
        console.print(
            f"[bold red]Error:[/bold red] Pattern '{pattern_name}' not found."
        )
        raise typer.Exit(code=1)

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
        tool_events_by_iter.setdefault(iteration, []).append(
            {
                "event": event,
                **payload,
            }
        )

    def _on_iteration(iteration: int, exchange: dict, history: str) -> None:
        if not json_output:
            tui.render_iteration(iteration, exchange)
            events = tool_events_by_iter.get(iteration, [])
            start_idx = printed_index_by_iter.get(iteration, 0)
            printed_index_by_iter[iteration] = tui.render_tool_events(
                iteration, events, start_idx
            )

    if not json_output:
        tui.render_header(pattern_name, topic)

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

    record = {
        "schema": 1,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": __version__,
        "pattern": pattern_name,
        "topic": topic,
        "config_path": str(config_path),
        "overrides": overrides,
        "tool_events_by_iter": tool_events_by_iter,
        "result": result,
    }

    if save_path is not None:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to save session: {e}")

    if json_output:
        console.print_json(data=record)
    else:
        tui.render_final(
            result.get("final_answer", ""), result.get("justification", "")
        )

    return record


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
    stream: bool = typer.Option(False, "--stream", help="Stream tokens if supported"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON record instead of TUI"
    ),
    save: Optional[Path] = typer.Option(
        None, "--save", help="Save session to JSON file"
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

    overrides = _parse_overrides(set)
    assert topic is not None
    _execute_pattern(
        pattern_name=pattern_name,
        topic=topic,
        config_path=config_path,
        overrides=overrides,
        stream=stream,
        json_output=json_output,
        save_path=save,
    )


@app.command("tools")
def tools(
    pattern_name: Optional[str] = typer.Option(
        None, "--pattern", help="Filter tools for a specific pattern"
    ),
    describe: Optional[str] = typer.Option(
        None, "--describe", help="Describe specific tool"
    ),
):
    """List available tools (global or for a given pattern)."""
    console.print(Panel.fit("Available Tools", style="bold blue"))
    if describe:
        try:
            fn = registry.get(describe)
        except KeyError:
            console.print(f"[bold red]Error:[/bold red] Unknown tool '{describe}'")
            raise typer.Exit(code=1)
        sig = getattr(fn, "__signature__", None)
        doc = getattr(fn, "__doc__", None) or "(no docstring)"
        console.print(
            Panel.fit(str(sig), title=f"Signature: {describe}", border_style="magenta")
        )
        console.print(Panel.fit(doc.strip(), title="Docstring"))
        return

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

    _execute_pattern(
        pattern_name=pattern_name,
        topic=topic,
        config_path=config_path,
        overrides=overrides,
        stream=False,
        json_output=False,
        save_path=None,
    )


@app.command("replay")
def replay(path: Path = typer.Argument(..., help="Path to saved session JSON")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    pattern_name = data.get("pattern", "?")
    topic = data.get("topic", "")
    tui.render_header(pattern_name, topic)
    events_by_iter: Dict[int, List[Dict[str, Any]]] = {
        int(k): v for k, v in (data.get("tool_events_by_iter", {}) or {}).items()
    }
    printed_index_by_iter: Dict[int, int] = {}
    history = (
        data.get("result", {}).get("history")
        or data.get("result", {}).get("debate_history")
        or []
    )
    for i, exchange in enumerate(history, 1):
        tui.render_iteration(i, exchange)
        events = events_by_iter.get(i, [])
        idx = printed_index_by_iter.get(i, 0)
        printed_index_by_iter[i] = tui.render_tool_events(i, events, idx)
    tui.render_final(
        data.get("result", {}).get("final_answer", ""),
        data.get("result", {}).get("justification", ""),
    )


@app.command("compare")
def compare(
    pattern_a: str = typer.Argument(..., help="First pattern"),
    pattern_b: str = typer.Argument(..., help="Second pattern"),
    topic: str = typer.Argument(..., help="Topic"),
    config_a: Optional[Path] = typer.Option(None, "--config-a"),
    config_b: Optional[Path] = typer.Option(None, "--config-b"),
    metric: str = typer.Option("exact", "--metric", help="exact|len|jaccard"),
):
    patterns = _pattern_map()
    cfg_a = config_a or patterns[pattern_a].default_config_path()
    cfg_b = config_b or patterns[pattern_b].default_config_path()
    rec_a = _execute_pattern(
        pattern_name=pattern_a,
        topic=topic,
        config_path=cfg_a,
        overrides={},
        stream=False,
        json_output=False,
        save_path=None,
    )
    rec_b = _execute_pattern(
        pattern_name=pattern_b,
        topic=topic,
        config_path=cfg_b,
        overrides={},
        stream=False,
        json_output=False,
        save_path=None,
    )
    a = (rec_a.get("result", {}).get("final_answer", "") or "").strip()
    b = (rec_b.get("result", {}).get("final_answer", "") or "").strip()

    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("Pattern", style="cyan")
    t.add_column("Final Answer (truncated)", style="green")
    t.add_row(pattern_a, (a[:120] + "…") if len(a) > 120 else a)
    t.add_row(pattern_b, (b[:120] + "…") if len(b) > 120 else b)
    console.print(t)


@app.command("version")
def version_cmd():
    console.print(
        Panel.fit(f"awesome-dspy-agents [bold cyan]{__version__}[/bold cyan]")
    )


if __name__ == "__main__":
    app()
