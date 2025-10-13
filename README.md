# Awesome DSPy Agents

A collection of multi-agent systems implemented with the [DSPy](https://github.com/stanfordnlp/dspy) framework.

## Available Patterns

This section will describe the implemented agent patterns.

| Pattern | Description | Strengths | Weaknesses |
| --- | --- | --- | --- |
| | | | |

## Using the CLI

Install and run with Poetry:

```bash
poetry install
poetry run dspy-agents --help
poetry run dspy-agents --install-completion   # optional shell completion
```

Set provider credentials via environment variables (example):

```bash
export GEMINI_API_KEY=...           # for Gemini
export OPENAI_API_KEY=...           # for OpenAI
```

Discover:

```bash
poetry run dspy-agents list
poetry run dspy-agents describe debate
poetry run dspy-agents configs debate
poetry run dspy-agents tools
poetry run dspy-agents tools --pattern debate
```

Run patterns:

```bash
# default config of the pattern
poetry run dspy-agents run debate "Is RLHF always beneficial?"

# custom config
poetry run dspy-agents run debate -c awesome_dspy_agents/patterns/debate/config.yaml "Debate topic"

# override nested config values at runtime
poetry run dspy-agents run debate "Topic" --set debate.max_iterations=3 --set debate.debate_level=2

# interactive guided run with arrow-key selection
poetry run dspy-agents interactive
```

During runs you will see per-iteration exchanges and judge evaluations, followed by a final decision.

## Contributing

...

## Development Guide

This section documents how to extend and maintain the CLI and pattern ecosystem.

### Project layout

```
awesome_dspy_agents/
  cli.py                       # CLI entrypoint (Typer + Rich)
  config.py                    # AppConfig and LM settings
  tools/
    registry.py                # Global tool registry (shared for all patterns)
    ascii_to_png.py            # Example image tool
  patterns/
    interface.py               # AgentPattern protocol and discovery
    debate/
      pattern.py               # DebatePattern + MADFramework
      signatures.py            # DSPy signatures
      config.yaml              # Default config
```

### Common practices
- Prefer typed configuration via `AppConfig` and validated YAML with env-var expansion (`${VAR}`).
- Keep tool implementations deterministic and side-effect minimal; log via `mad.tools`.
- Keep per-pattern logic inside `patterns/<name>/pattern.py`; expose a `get_pattern()` factory.
- Use Rich tables and panels for readable CLI output; avoid noisy logs by default.
- Support per-agent LM configuration (provider/model/api_base/api_key) per DSPy conventions.

### Adding a new pattern
1. Create a new folder under `awesome_dspy_agents/patterns/<your_pattern>/` with at least:
   - `pattern.py`: implement your DSPy modules and wrap them in a class that implements `AgentPattern`.
   - `config.yaml`: default configuration for the pattern (agents, judge, debate params, etc.).
   - `signatures.py` as needed.
2. In `pattern.py`, implement:
   - `class YourPattern(AgentPattern)` with:
     - `name`: a unique string
     - `describe(self) -> str`: short Markdown description
     - `default_config_path(self) -> Path | None`
     - `available_configs(self) -> Iterable[Path]`: include default + optional `scenarios/*.yaml`
     - `available_tools(self) -> Iterable[str]`: the tool names used by default
     - `available_scripts(self) -> dict[str, str]`: mapping of script name to description (MIPRO/GEPA)
     - `run(self, topic, config_path, overrides=None, on_iteration=None) -> dict`
       - Configure default LM if provided; construct your DSPy program; call `on_iteration` after each step.
   - `def get_pattern() -> AgentPattern: return YourPattern()`
3. The CLI will discover it automatically via `find_patterns()` if `pattern.py` exports `get_pattern()`.

Example `run` implementation sketch:

```python
def run(self, topic, config_path, overrides=None, on_iteration=None):
    cfg = load_config(str(config_path))
    if cfg.default_lm:
        dspy.configure(lm=build_lm(cfg.default_lm))
    # build modules based on cfg; call on_iteration(i, exchange, history)
    final = program(...)
    return {"final_answer": final.final_answer, "justification": final.justification}
```

### Adding tools to the global registry
1. Implement a pure function in `awesome_dspy_agents/tools/*.py`.
2. Register it in `awesome_dspy_agents/tools/registry.py`:
```python
from awesome_dspy_agents.tools.registry import registry

def my_tool(arg1: str) -> str:
    return arg1.upper()

registry.register("my_tool", my_tool)
```
3. Reference the tool by name in pattern configs (for ReAct tools) or in code:
```yaml
agents:
  affirmative:
    module_type: react
    tools: ["my_tool"]
```

Guidelines:
- Keep tool I/O small; return primitives or `dspy.Image` for images.
- Use the registry’s logging; avoid network I/O unless essential.

### Configuration best practices
- Layering: default pattern config -> user-provided file (`-c`) -> CLI overrides (`--set a.b=val`).
- Use env var placeholders in YAML (`${GEMINI_API_KEY}`) to avoid committing secrets.
- For local models (Ollama), set `api_base` in the config and leave `api_key` empty.

### Testing patterns and CLI
- Add snapshot tests for CLI output for stable commands (`list`, `describe`).
- Add integration tests per pattern to ensure end-to-end outputs are structured.
- Prefer deterministic seeds or fixed temperatures for tests.

### Observability & logs
- LLM calls are logged under `mad.llm` rotating files in `patterns/logs/`.
- Tool calls are logged under `mad.tools` rotating files in the same directory.

### Roadmap ideas
- Optional token streaming via `dspy.streamify` with a `--stream` flag.
- Built-in optimization scripts (MIPRO/GEPA) surfaced in `scripts` command.
