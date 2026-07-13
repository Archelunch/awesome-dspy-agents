# Addition by Subtraction

Addition by Subtraction is an asymmetric refinement protocol. The addition role
expands the latest refined response; the subtraction role removes redundancy,
unsupported claims, and irrelevant material while preserving the instruction and
supported facts.

Each round carries explicit state forward:

```text
current refined response + previous feedback
  -> additions + candidate response
  -> removals + preserved facts + refined response + next feedback
```

The framework still returns its backward-compatible typed exchange history. It
also returns an immutable `trajectory` containing proposal and revision events,
their causal links, and the claims preserved at each step.

```python
from awesome_dspy_agents.patterns.addition_by_subtraction.pattern import ABSFramework

refiner = ABSFramework(max_iterations=2)
result = refiner(context=source_material, instruction="Write a concise answer")

print(result.final_answer)
print(result.stop_reason)
```
