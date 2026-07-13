from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import dspy

from awesome_dspy_agents.patterns.deliberation import (
    DeliberationDelta,
    DeliberationTrajectory,
    EvidenceArtifact,
    HistoryView,
)
from awesome_dspy_agents.predictor import build_predictor
from awesome_dspy_agents.runtime import EmitIteration, IterationEvent

from .consensus_free_signatures import (
    ArbitrateFullTrajectory,
    IndependentProposal,
    ReviseWithoutConformity,
    SelectConflicts,
)


@dataclass(frozen=True)
class ConsensusFreeDebateRound:
    proposal_ids: tuple[str, ...]
    revision_ids: tuple[str, ...]
    conflicts: tuple[str, ...]


class ConsensusFreeDebate(dspy.Module):
    """Independent proposals, anti-conformity revision, and trajectory arbitration."""

    def __init__(
        self,
        *,
        agent_count: int = 2,
        perspectives: Sequence[str] | None = None,
        proposer_module_type: str = "predict",
        conflict_selector_module_type: str = "predict",
        reviser_module_type: str = "predict",
        arbiter_module_type: str = "predict",
        proposer_lm: dspy.LM | None = None,
        conflict_selector_lm: dspy.LM | None = None,
        reviser_lm: dspy.LM | None = None,
        arbiter_lm: dspy.LM | None = None,
        proposer_tools: Sequence[str] | None = None,
        conflict_selector_tools: Sequence[str] | None = None,
        reviser_tools: Sequence[str] | None = None,
        arbiter_tools: Sequence[str] | None = None,
        react_max_iters: int = 6,
        on_iteration: EmitIteration | None = None,
    ) -> None:
        super().__init__()
        if agent_count < 2:
            raise ValueError("Consensus-free debate requires at least two agents")
        if perspectives is not None and len(perspectives) != agent_count:
            raise ValueError("perspectives must contain one value per agent")

        self.agent_count = agent_count
        self.on_iteration = on_iteration
        self.perspectives = tuple(
            perspectives
            or (
                f"independent perspective {index}"
                for index in range(1, agent_count + 1)
            )
        )
        self.proposer = build_predictor(
            IndependentProposal,
            proposer_module_type,
            role="proposer",
            lm=proposer_lm,
            tool_names=proposer_tools,
            react_max_iters=react_max_iters,
        )
        self.conflict_selector = build_predictor(
            SelectConflicts,
            conflict_selector_module_type,
            role="conflict_selector",
            lm=conflict_selector_lm,
            tool_names=conflict_selector_tools,
            react_max_iters=react_max_iters,
        )
        self.reviser = build_predictor(
            ReviseWithoutConformity,
            reviser_module_type,
            role="reviser",
            lm=reviser_lm,
            tool_names=reviser_tools,
            react_max_iters=react_max_iters,
        )
        self.arbiter = build_predictor(
            ArbitrateFullTrajectory,
            arbiter_module_type,
            role="arbiter",
            lm=arbiter_lm,
            tool_names=arbiter_tools,
            react_max_iters=react_max_iters,
        )

    def forward(
        self,
        problem: str,
        context: str = "",
        evidence: Sequence[EvidenceArtifact] = (),
    ) -> dspy.Prediction:
        trajectory = DeliberationTrajectory()
        evidence_event_ids: dict[str, str] = {}
        for artifact in evidence:
            if artifact.evidence_id in evidence_event_ids:
                raise ValueError(f"Duplicate evidence ID: {artifact.evidence_id}")
            trajectory = trajectory.record(
                round_index=0,
                role="evidence",
                kind="tool_observation" if artifact.kind == "tool" else "evidence",
                content=artifact.content,
                source_id=artifact.evidence_id,
            )
            evidence_event_ids[artifact.evidence_id] = trajectory.events[-1].event_id

        evidence_context = trajectory.render(
            HistoryView(event_ids=tuple(evidence_event_ids.values()))
        )
        effective_context = context
        if evidence:
            effective_context = (
                f"{context}\n\nStructured evidence:\n{evidence_context}".strip()
            )

        proposal_ids: list[str] = []
        for index, perspective in enumerate(self.perspectives, start=1):
            # Proposals are deliberately independent: no peer history is provided.
            proposal = self.proposer(
                problem=problem,
                context=effective_context,
                perspective=perspective,
            )
            trajectory = trajectory.record(
                round_index=0,
                role=f"proposer-{index}",
                kind="proposal",
                content=proposal.answer,
                reasoning=proposal.reasoning,
                claims=tuple(proposal.claims),
                evidence_ids=tuple(
                    evidence_id
                    for evidence_id in proposal.evidence_ids
                    if evidence_id in evidence_event_ids
                ),
            )
            proposal_ids.append(trajectory.events[-1].event_id)

        proposal_view = HistoryView(
            event_ids=(*evidence_event_ids.values(), *proposal_ids),
            anonymize_roles=True,
        )
        conflict_result = self.conflict_selector(
            problem=problem,
            proposals=trajectory.render(proposal_view),
        )
        selected_ids = [
            event_id
            for event_id in conflict_result.selected_event_ids
            if event_id in proposal_ids
        ]
        if len(selected_ids) < 2:
            selected_ids = proposal_ids

        conflicts = tuple(conflict_result.conflicts)
        trajectory = trajectory.record(
            round_index=1,
            role="conflict-selector",
            kind="critique",
            content="\n".join(conflicts)
            or "No explicit conflict description returned.",
            parent_ids=tuple(selected_ids),
        )
        conflict_event_id = trajectory.events[-1].event_id
        routes = getattr(conflict_result, "conflict_routes", {}) or {}
        conflicts_by_event = getattr(conflict_result, "conflicts_by_event", {}) or {}

        revision_ids: list[str] = []
        for index, own_event_id in enumerate(proposal_ids, start=1):
            routed_ids = routes.get(own_event_id, selected_ids)
            opposing_ids = [
                event_id
                for event_id in routed_ids
                if event_id in proposal_ids and event_id != own_event_id
            ]
            if not opposing_ids:
                opposing_ids = [
                    event_id for event_id in proposal_ids if event_id != own_event_id
                ]
            opposing_evidence_ids = {
                evidence_id
                for event_id in opposing_ids
                for evidence_id in trajectory.get(event_id).evidence_ids
            }
            visible_evidence_events = [
                event_id
                for evidence_id, event_id in evidence_event_ids.items()
                if evidence_id in opposing_evidence_ids
            ]
            opposing_view = HistoryView(
                event_ids=(*visible_evidence_events, *opposing_ids),
                anonymize_roles=True,
            )
            original = trajectory.get(own_event_id)
            revision = self.reviser(
                problem=problem,
                context=effective_context,
                original_answer=original.content,
                opposing_arguments=trajectory.render(opposing_view),
                conflicts=list(conflicts_by_event.get(own_event_id, conflicts)),
            )
            revised_evidence_ids = getattr(
                revision, "evidence_ids", original.evidence_ids
            )
            trajectory = trajectory.record(
                round_index=1,
                role=f"proposer-{index}",
                kind="revision",
                content=revision.revised_answer,
                reasoning=revision.reasoning,
                claims=tuple(
                    [*revision.retained_claims, *revision.accepted_corrections]
                ),
                delta=DeliberationDelta(
                    preserved=tuple(revision.retained_claims),
                    rejected=tuple(revision.rejected_claims),
                    accepted_corrections=tuple(revision.accepted_corrections),
                ),
                parent_ids=(own_event_id, *opposing_ids, conflict_event_id),
                evidence_ids=tuple(
                    evidence_id
                    for evidence_id in revised_evidence_ids
                    if evidence_id in evidence_event_ids
                ),
            )
            revision_ids.append(trajectory.events[-1].event_id)

        if self.on_iteration is not None:
            self.on_iteration(
                IterationEvent(
                    iteration=1,
                    exchange=ConsensusFreeDebateRound(
                        proposal_ids=tuple(proposal_ids),
                        revision_ids=tuple(revision_ids),
                        conflicts=conflicts,
                    ),
                    history=trajectory.render(),
                )
            )

        arbitration = self.arbiter(
            problem=problem,
            context=effective_context,
            trajectory=trajectory.render(
                HistoryView(anonymize_roles=True, include_reasoning=True)
            ),
        )
        known_candidate_ids = {*proposal_ids, *revision_ids}
        winning_event_id = arbitration.winning_event_id
        judgment_parents = (
            (winning_event_id,)
            if winning_event_id in known_candidate_ids
            else tuple(revision_ids)
        )
        trajectory = trajectory.record(
            round_index=1,
            role="arbiter",
            kind="judgment",
            content=arbitration.final_answer,
            reasoning=arbitration.justification,
            parent_ids=judgment_parents,
        )

        return dspy.Prediction(
            final_answer=arbitration.final_answer,
            justification=arbitration.justification,
            trajectory=trajectory,
            history=trajectory.events,
            iterations_used=1,
            stopped_early=False,
            stop_reason="completed",
            candidate_scores=getattr(arbitration, "candidate_scores", {}),
        )
