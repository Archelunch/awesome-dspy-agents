# Awesome DSPy Agents
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.8.2+-blue.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-orange?logo=buy-me-a-coffee)](https://buymeacoffee.com/mike_pavlukhin)

A collection of multi-agent systems implemented with the [DSPy](https://github.com/stanfordnlp/dspy) framework.

## Installation

### Prerequisites

- Python 3.12 or higher
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Archelunch/awesome-dspy-agents
   cd awesome-dspy-agents
   ```

2. **Install Poetry (if not already installed):**
   ```bash
   pip install poetry
   ```

3. **Install dependencies:**
   ```bash
   poetry install
   ```

4. **Set up environment variables:**
   Set your API keys for the language model providers you want to use:
   ```bash
   export GEMINI_API_KEY=your_gemini_api_key_here
   export OPENAI_API_KEY=your_openai_api_key_here
   # Add other provider keys as needed
   ```

5. **Verify installation:**
   ```bash
   poetry run dspy-agents --help
   ```

6. **Optional: Install shell completion:**
   ```bash
   poetry run dspy-agents --install-completion
   ```

## Table of Contents

- [Awesome DSPy Agents](#awesome-dspy-agents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Table of Contents](#table-of-contents)
  - [Available Patterns](#available-patterns)
      - [Debates](#debates)
      - [Addition by Subtraction](#addition-by-subtraction)
    - [Available Tools](#available-tools)
  - [Using the CLI](#using-the-cli)
  - [Configuration](#configuration)
  - [Examples](#examples)
    - [Addition-by-Subtraction Pattern](#addition-by-subtraction-pattern)
  - [Development Guide](#development-guide)
    - [Project layout](#project-layout)
    - [Common practices](#common-practices)
    - [Adding a new pattern](#adding-a-new-pattern)
    - [Adding tools to the catalog](#adding-tools-to-the-catalog)
    - [Configuration](#configuration-1)
    - [Observability \& logs](#observability--logs)
    - [Roadmap ideas](#roadmap-ideas)

## Available Patterns

These are the built-in patterns. Use the CLI to explore and run them.

| Pattern | Description | Strengths | Weaknesses |
| --- | --- | --- | --- |
| debate | Multi‑Agent Debate with a Judge. Two agents argue iteratively; an optional judge evaluates progress and extracts the final answer. | Strong adversarial reasoning; adaptive early stop; ReAct tool support. | Higher token usage; needs careful judge configuration. |
| addition_by_subtraction | Addition‑by‑Subtraction collaboration. Addition expands details; Subtraction removes redundancy and feeds back; early exit when stable. | Concise refined answers; low iteration count by default (M≤2); ReAct tool support. | Can miss alternative directions; relies on good subtraction feedback. |

Related work:
#### Debates 
Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate: https://arxiv.org/abs/2305.19118

#### Addition by Subtraction
(Perhaps) Beyond Human Translation: Harnessing Multi-Agent Collaboration for Translating Ultra-Long Literary Texts: https://arxiv.org/abs/2305.19118

### Available Tools

Tools are built per Pattern run from a shared catalog. File access is denied unless
explicit roots are provided with `--allow-path`. List tools and inspect details:

```bash
poetry run dspy-agents tools
poetry run dspy-agents tools --describe read_file_attachment
```

- math_eval: Evaluate a simple Python math expression safely; returns string.
- word_count: Count words in text; returns string number.
- ascii_to_png: Render ASCII text into a PNG (`dspy.Image`) for visual reasoning.
- list_files: List absolute file paths under a directory (sandboxed).
- read_file_attachment: Return an `Attachments` object for a local file (sandboxed).
- write_file: Write text to a local file path; returns absolute path (sandboxed).

Sandboxing: File tools are restricted to allowed directories. Use `--allow-path /abs/dir` to opt‑in per run (can repeat).

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
poetry run dspy-agents tools --describe ascii_to_png
```

Run patterns:

```bash
# default config of the pattern
poetry run dspy-agents run debate "Is RLHF always beneficial?"

# custom config
poetry run dspy-agents run debate -c awesome_dspy_agents/patterns/debate/config.yaml "Debate topic"

# override nested config values at runtime (typed casting: bool/int/float)
poetry run dspy-agents run debate "Topic" --set debate.max_iterations=3 --set debate.debate_level=2 --set judge.module_type=react

# interactive guided run with arrow-key selection
poetry run dspy-agents interactive
```

JSON output and session save/replay:

```bash
# Emit machine-readable JSON and save the full session
poetry run dspy-agents run debate "Is RLHF always beneficial?" --json --save runs/rlhf.json

# Replay a saved session locally without model calls
poetry run dspy-agents replay runs/rlhf.json
```

Compare patterns on the same topic:

```bash
poetry run dspy-agents compare debate addition_by_subtraction "What is chain-of-thought?" --metric jaccard
```

Version and sandbox:

```bash
poetry run dspy-agents version
poetry run dspy-agents run addition_by_subtraction "Summarize file" --allow-path . --set abs.max_iterations=2
```

During runs you will see per-iteration exchanges (debate or addition/subtraction), optional judge evaluations, and a final decision. Tool usage is summarized under each iteration.

## Configuration

Configuration is layered and typed:

1. Pattern default config (e.g., `patterns/debate/config.yaml`).
2. User-provided file via `-c/--config`.
3. CLI overrides via `--set a.b=value` (auto‑casts `true/false`, integers, and floats).
4. Environment variable expansion inside YAML values: `${OPENAI_API_KEY}`.

Minimal examples:

```yaml
# debate/config.yaml (excerpt)
default_lm:
  provider: gemini
  model: gemini-2.5-flash-preview-09-2025
  api_key_env: GEMINI_API_KEY

agents:
  affirmative:
    persona: "Optimistic, evidence-driven."
    module_type: react
    tools: ["ascii_to_png", "read_file_attachment", "list_files"]
  negative:
    persona: "Rigorous skeptic."
    module_type: react
    tools: ["ascii_to_png", "read_file_attachment", "list_files"]

judge:
  module_type: react
  tools: ["write_file"]

debate:
  max_iterations: 5
  debate_level: 2
  adaptive_break: true
```

```yaml
# addition_by_subtraction/config.yaml (excerpt)
default_lm:
  provider: gemini
  model: gemini-2.5-flash-preview-09-2025
  api_key_env: GEMINI_API_KEY

agents:
  addition:
    persona: "Expand relevant information and synthesize details."
    module_type: react
    tools: ["ascii_to_png", "read_file_attachment", "list_files", "math_eval", "word_count"]
  subtraction:
    persona: "Remove redundancy and provide clear feedback."
    module_type: react
    tools: ["ascii_to_png", "read_file_attachment", "list_files", "math_eval", "word_count"]

abs:
  max_iterations: 2
  early_exit: true
```

Tips:

- Point to a custom config: `-c path/to/config.yaml`.
- OpenRouter DeepSeek profile: `examples/configs/openrouter-deepseek-v4-flash.yaml`.
- Override nested values at runtime (typed): `--set debate.max_iterations=3 --set judge.module_type=react`.
- Per‑agent LM: set `agents.<name>.lm` block with `provider/model/api_base/api_key(_env)`.
- File tools are sandboxed; add `--allow-path /abs/dir` to enable local file access.

## Examples

Run Debate with a custom judge and fewer iterations:

```bash
poetry run dspy-agents run debate "When to use CoT?" \
  --set debate.max_iterations=3 \
  --set judge.module_type=react
```

Run ABS with early exit disabled and save JSON:

```bash
poetry run dspy-agents run addition_by_subtraction "Summarize the paper" \
  --set abs.early_exit=false --json --save runs/abs.json
```

Use a local file during a run (sandboxed):

```bash
poetry run dspy-agents run addition_by_subtraction "Summarize the attached doc" \
  --allow-path "$PWD" \
  --set agents.addition.tools="[read_file_attachment]"
```

### Addition-by-Subtraction Pattern

This collaboration uses two agents only: Addition (expands and aggregates relevant details) and Subtraction (removes redundancy and provides feedback). It iterates up to `abs.max_iterations` with an early-exit when no further revision is needed.

- Default config: `awesome_dspy_agents/patterns/addition_by_subtraction/config.yaml`
- Supports tools via ReAct (same registry as debate). Tools used are rendered in TUI under each iteration.

Examples:

```bash
poetry run dspy-agents describe addition_by_subtraction
poetry run dspy-agents configs addition_by_subtraction
poetry run dspy-agents run addition_by_subtraction "Summarize the key ideas from the attached document"
# Override ABS parameters
poetry run dspy-agents run addition_by_subtraction "Instruction" --set abs.max_iterations=2 --set abs.early_exit=true
```

TUI displays two columns: Addition and Subtraction, plus a Feedback panel each iteration. Tool usage events are summarized below the panels.

Note: Early exit happens when subsequent additions stabilize. Default maximum iterations M=2 (configurable via `abs.max_iterations`).


## Development Guide

This section documents how to extend and maintain the CLI and pattern ecosystem.

### Quality checks

Install the development dependency group and run the same checks enforced in CI:

```bash
poetry install --with dev
poetry run ruff check .
poetry run ruff format --check .
poetry run pyrefly check --summarize-errors
poetry run pytest -q
```

Use `poetry run ruff check . --fix` and `poetry run ruff format .` to apply safe
automatic lint and formatting fixes locally.

### Project layout

```
awesome_dspy_agents/
  cli.py                       # CLI entrypoint (Typer + Rich)
  config.py                    # AppConfig and LM settings
  runtime.py                   # Typed Pattern run lifecycle
  predictor.py                 # DSPy predictor construction and agent context
  evaluation.py                # Datasets, metrics, reports, and optimization
  tools/
    registry.py                # Tool catalog, per-run executor, and file policy
    ascii_to_png.py            # Example image tool
  patterns/
    interface.py               # AgentPattern protocol and discovery
    debate/
      pattern.py               # DebatePattern + MADFramework
      signatures.py            # DSPy signatures
      config.yaml              # Default config
    addition_by_subtraction/
      pattern.py               # Addition-by-Subtraction Pattern + ABSFramework
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
     - `run(self, request: PatternRunRequest, on_iteration: EmitIteration | None = None) -> PatternOutcome`
       - Execute through `PatternRuntime`; emit typed `IterationEvent` values after each step.
   - `def get_pattern() -> AgentPattern: return YourPattern()`
3. The CLI will discover it automatically via `discover_patterns()` if `pattern.py` exports `get_pattern()`.

Example `run` implementation sketch:

```python
def run(self, request: PatternRunRequest, on_iteration=None) -> PatternOutcome:
    def execute(request, cfg, emit):
        program = build_program(cfg, on_iteration=emit)
        final = program(topic=request.topic)
        return PatternOutcome(
            final_answer=final.final_answer,
            justification=final.justification,
            iterations_used=final.iterations_used,
            stopped_early=final.stopped_early,
            history=final.history,
        )

    return PatternRuntime().run_configured(
        request, execute=execute, on_iteration=on_iteration
    )
```

### Adding tools to the catalog
1. Implement a pure function in `awesome_dspy_agents/tools/*.py`.
2. Add it to `default_catalog` in `awesome_dspy_agents/tools/registry.py`:
```python
from awesome_dspy_agents.tools.registry import default_catalog

def my_tool(arg1: str) -> str:
    return arg1.upper()

default_catalog.register("my_tool", lambda _policy: my_tool)
```
3. Reference the tool by name in pattern configs (for ReAct tools) or in code:
```yaml
agents:
  affirmative:
    module_type: react
    tools: ["my_tool"]
```

Guidelines:
- Keep tool I/O small; return primitives, `dspy.Image` for images or `Attachments` for other files.


### Configuration
- Layering: default pattern config -> user-provided file (`-c`) -> CLI overrides (`--set a.b=val`).
- Use env var placeholders in YAML (`${GEMINI_API_KEY}`) to avoid committing secrets.
- For local models (Ollama, vLLM), set `api_base` and `api_key` in the config.

### Observability & logs
- LLM calls are logged under `mad.llm` rotating files in `patterns/logs/`.
- Tool calls are logged under `mad.tools` rotating files in the same directory.
  
### Roadmap ideas
- [x] Add Addition-by-Subtraction pattern
- [x] Add tools
- [ ] Add MAPS pattern
- [x] Versioned evaluation datasets, metrics, and optimizer seam
- [ ] Integration with MLFlow
- [x] Evaluation profiles for token counts, cost, and latency
- [ ] Batch runs (`run-batch --topics file.txt --concurrency N`).
- [x] Add deterministic tests
- [ ] More examples
- [ ] More tools
- [ ] Improve agents communication 
