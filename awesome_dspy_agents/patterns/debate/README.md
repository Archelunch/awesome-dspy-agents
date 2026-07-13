# Multi-Agent Debate

The debate pattern provides two protocols over a shared immutable deliberation
trajectory.

- `ClassicAdversarialDebate` preserves the sequential affirmative/negative
  protocol.
- `ConsensusFreeDebate` runs independent proposals, selects consequential
  conflicts, performs anti-conformity revisions, and arbitrates over the full
  trajectory.

The consensus-free protocol deliberately does not ask agents to reach agreement
or vote over the last round. Initial proposals never contain peer history,
revisers receive only selected anonymous disagreements, and the arbiter receives
the complete trajectory.

```python
from awesome_dspy_agents.patterns.debate import ConsensusFreeDebate

debate = ConsensusFreeDebate(agent_count=2)
result = debate(problem="Which conclusion is best supported?", context=evidence)

print(result.final_answer)
for event in result.trajectory.events:
    print(event.event_id, event.kind, event.parent_ids)
```

Every optimizable role is a named DSPy predictor:

```python
for name, predictor in debate.named_predictors():
    print(name, predictor.signature.instructions)
```

This allows GEPA to optimize proposer, conflict-selector, reviser, and arbiter
instructions as parts of one DSPy program. The scenario
`scenarios/consensus_free.yaml` selects this protocol through the normal pattern
runtime.

## History policy

The trajectory is the source of truth. A history view decides which immutable
events an agent sees and whether roles or reasoning are revealed. Evidence IDs
remain attached to events even when agent identity is hidden.
