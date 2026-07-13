# DSPy Agent Patterns

This context describes how reusable multi-agent reasoning patterns are configured, executed, and observed.

## Language

**Pattern**:
A reusable collaboration strategy that coordinates one or more DSPy agents to produce an answer.
_Avoid_: Workflow, framework

**Pattern run**:
One isolated execution of a Pattern for a topic, including its configuration, iteration events, outcome, and runtime issues.
_Avoid_: Session, job

**Iteration event**:
An observable exchange emitted while a Pattern run progresses.
_Avoid_: Callback payload, update

**Pattern outcome**:
The final answer, justification, history, and completion metadata produced by a Pattern run.
_Avoid_: Result dictionary, response

**Runtime issue**:
A non-fatal failure in observation or presentation that does not invalidate the Pattern outcome.
_Avoid_: Swallowed exception, warning string

**Evaluation dataset**:
A versioned collection of inputs and expected outputs used to measure a DSPy program.
_Avoid_: Test prompts, benchmark file

**Evaluation report**:
The quality score, per-example results, latency, and model usage measured for one program against one Evaluation dataset version.
_Avoid_: Test output, score file
