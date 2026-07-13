from __future__ import annotations

import os
import re
import tempfile
import unittest

os.environ.setdefault(
    "DSPY_CACHEDIR", f"{tempfile.gettempdir()}/dspy-agents-test-cache"
)

import dspy

from awesome_dspy_agents.patterns.debate.consensus_free import ConsensusFreeDebate
from awesome_dspy_agents.patterns.deliberation import EvidenceArtifact
from awesome_dspy_agents.tools.registry import (
    FileAccessPolicy,
    ToolExecutor,
    default_catalog,
)


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


class _ToolUsingProposer(_Proposer):
    def forward(self, **inputs):
        ToolExecutor(default_catalog, FileAccessPolicy()).get("word_count")("one two")
        return super().forward(**inputs)


class _ConflictSelector(dspy.Module):
    def forward(self, **_inputs):
        return dspy.Prediction(
            selected_event_ids=["event-1", "event-2"],
            conflicts=["The answers disagree"],
        )


class _RoutedConflictSelector(dspy.Module):
    def forward(self, **_inputs):
        return dspy.Prediction(
            selected_event_ids=["event-1", "event-2", "event-3"],
            conflicts=["global conflict"],
            conflict_routes={
                "event-1": ["event-2"],
                "event-2": ["event-1"],
                "event-3": ["event-1"],
            },
            conflicts_by_event={"event-1": ["relevant to the first proposal"]},
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
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(self, **_inputs):
        self.calls.append(_inputs)
        candidate_ids = re.findall(
            r"\[(event-\d+) \|[^\]]+\| (?:proposal|revision)\]",
            _inputs["trajectory"],
        )
        return dspy.Prediction(
            winning_event_id=candidate_ids[-1],
            candidate_scores={event_id: 1.0 for event_id in candidate_ids},
            final_answer="best answer",
            justification="best supported trajectory",
        )


class _NoConflictSelector(dspy.Module):
    def forward(self, **_inputs):
        return dspy.Prediction(
            selected_event_ids=[],
            conflicts=[],
            conflict_routes={},
            conflicts_by_event={},
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
            [
                "proposal",
                "proposal",
                "critique",
                "revision",
                "revision",
                "judgment",
            ],
        )
        self.assertEqual(
            result.trajectory.events[3].parent_ids,
            ("event-1", "event-2", "event-3"),
        )
        self.assertEqual(events[0].exchange.revision_ids, ("event-4", "event-5"))

    def test_records_and_validates_document_and_tool_evidence(self) -> None:
        debate = ConsensusFreeDebate(agent_count=2)
        proposer = _Proposer()
        arbiter = _Arbiter()
        debate.proposer = proposer
        debate.conflict_selector = _ConflictSelector()
        debate.reviser = _Reviser()
        debate.arbiter = arbiter

        original_forward = proposer.forward

        def propose_with_evidence(**inputs):
            prediction = original_forward(**inputs)
            prediction.evidence_ids = ["document-1", "invented-source"]
            return prediction

        proposer.forward = propose_with_evidence
        result = debate(
            problem="What is correct?",
            evidence=(
                EvidenceArtifact("document-1", "Document evidence"),
                EvidenceArtifact("tool-1", "Tool output", kind="tool"),
            ),
        )

        self.assertEqual(
            [event.kind for event in result.trajectory.events[:2]],
            ["evidence", "tool_observation"],
        )
        self.assertEqual(result.trajectory.events[2].evidence_ids, ("document-1",))
        self.assertIn("Source: document-1", arbiter.calls[0]["trajectory"])
        self.assertNotIn("invented-source", arbiter.calls[0]["trajectory"])

    def test_routes_only_relevant_conflicts_to_each_reviser(self) -> None:
        debate = ConsensusFreeDebate(agent_count=3)
        reviser = _Reviser()
        debate.proposer = _Proposer()
        debate.conflict_selector = _RoutedConflictSelector()
        debate.reviser = reviser
        debate.arbiter = _Arbiter()

        debate(problem="What is correct?")

        self.assertIn(
            "answer-independent perspective 2", reviser.calls[0]["opposing_arguments"]
        )
        self.assertNotIn(
            "answer-independent perspective 3",
            reviser.calls[0]["opposing_arguments"],
        )
        self.assertEqual(
            reviser.calls[0]["conflicts"], ["relevant to the first proposal"]
        )

    def test_skips_revision_when_selector_finds_no_meaningful_conflict(self) -> None:
        debate = ConsensusFreeDebate(agent_count=2)
        reviser = _Reviser()
        debate.proposer = _Proposer()
        debate.conflict_selector = _NoConflictSelector()
        debate.reviser = reviser
        debate.arbiter = _Arbiter()

        result = debate(problem="What is correct?")

        self.assertEqual(reviser.calls, [])
        self.assertEqual(
            [event.kind for event in result.trajectory.events],
            ["proposal", "proposal", "critique", "judgment"],
        )
        self.assertEqual(result.iterations_used, 0)
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.stop_reason, "no_conflict")
        self.assertEqual(set(result.candidate_scores), {"event-1", "event-2"})

    def test_records_live_agent_tool_results_and_attaches_them_to_proposals(
        self,
    ) -> None:
        debate = ConsensusFreeDebate(agent_count=2)
        debate.proposer = _ToolUsingProposer()
        debate.conflict_selector = _ConflictSelector()
        debate.reviser = _Reviser()
        debate.arbiter = _Arbiter()

        result = debate(problem="What is correct?")

        tool_events = [
            event
            for event in result.trajectory.events
            if event.kind == "tool_observation"
        ]
        proposals = [
            event for event in result.trajectory.events if event.kind == "proposal"
        ]
        self.assertEqual([event.content for event in tool_events], ["2", "2"])
        self.assertEqual(proposals[0].evidence_ids, (tool_events[0].source_id,))
        self.assertEqual(proposals[1].evidence_ids, (tool_events[1].source_id,))


if __name__ == "__main__":
    unittest.main()
