from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault(
    "DSPY_CACHEDIR", f"{tempfile.gettempdir()}/dspy-agents-test-cache"
)

import dspy

from awesome_dspy_agents.patterns.debate.consensus_free import ConsensusFreeDebate


class _Proposer(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, str]] = []

    def forward(self, **inputs):
        self.calls.append(inputs)
        perspective = inputs["perspective"]
        return dspy.Prediction(
            answer=f"answer-{perspective}",
            claims=[f"claim-{perspective}"],
            evidence_ids=[],
            reasoning=f"reason-{perspective}",
        )


class _ConflictSelector(dspy.Module):
    def forward(self, **_inputs):
        return dspy.Prediction(
            selected_event_ids=["event-1", "event-2"],
            conflicts=["The answers disagree"],
        )


class _Reviser(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, str]] = []

    def forward(self, **inputs):
        self.calls.append(inputs)
        return dspy.Prediction(
            revised_answer=f"revised:{inputs['original_answer']}",
            retained_claims=[],
            rejected_claims=[],
            accepted_corrections=[],
            reasoning="reconsidered independently",
        )


class _Arbiter(dspy.Module):
    def forward(self, **_inputs):
        return dspy.Prediction(
            winning_event_id="event-3",
            final_answer="best answer",
            justification="best supported trajectory",
        )


class ConsensusFreeDebateTests(unittest.TestCase):
    def test_exposes_each_optimizable_role_as_a_named_predictor(self) -> None:
        debate = ConsensusFreeDebate(agent_count=2)

        names = {name for name, _predictor in debate.named_predictors()}

        self.assertEqual(
            names,
            {
                "proposer.predictor",
                "conflict_selector.predictor",
                "reviser.predictor",
                "arbiter.predictor",
            },
        )

    def test_runs_independent_proposals_then_revision_and_arbitration(self) -> None:
        events = []
        debate = ConsensusFreeDebate(agent_count=2, on_iteration=events.append)
        proposer = _Proposer()
        reviser = _Reviser()
        debate.proposer = proposer
        debate.conflict_selector = _ConflictSelector()
        debate.reviser = reviser
        debate.arbiter = _Arbiter()

        result = debate(problem="What is correct?", context="Evidence")

        self.assertEqual(result.final_answer, "best answer")
        self.assertEqual(len(proposer.calls), 2)
        self.assertNotIn("history", proposer.calls[0])
        self.assertEqual(len(reviser.calls), 2)
        self.assertNotIn("proposer-", reviser.calls[0]["opposing_arguments"])
        self.assertEqual(
            [event.kind for event in result.trajectory.events],
            ["proposal", "proposal", "revision", "revision", "judgment"],
        )
        self.assertEqual(result.trajectory.events[2].parent_ids, ("event-1", "event-2"))
        self.assertEqual(events[0].exchange.revision_ids, ("event-3", "event-4"))


if __name__ == "__main__":
    unittest.main()
