# Evaluation

Evaluation datasets are versioned JSON files under `datasets/`. Each example separates
program inputs from expected outputs so it can be converted into a `dspy.Example`.

```python
from pathlib import Path

from awesome_dspy_agents.evaluation import (
    EvaluationDataset,
    EvaluationRunner,
    OptimizationRunner,
    exact_answer,
)

dataset = EvaluationDataset.load(Path("evaluation/datasets/smoke.json"))
report = EvaluationRunner().evaluate(program, dataset, exact_answer)

# Optimizers such as MIPROv2 or GEPA satisfy the OptimizationRunner interface.
optimized = OptimizationRunner().optimize(
    program,
    dataset,
    optimizer,
    Path("optimization/program.json"),
    seed=7,
)
```

`EvaluationReport` records dataset identity and version, aggregate and per-example
scores, wall-clock latency, model calls, estimated cost, and token usage.
