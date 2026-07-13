from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EventKind = Literal[
    "evidence",
    "proposal",
    "critique",
    "revision",
    "tool_observation",
    "judgment",
]
EvidenceKind = Literal["context", "document", "tool"]


@dataclass(frozen=True)
class EvidenceArtifact:
    """Evidence supplied to a deliberation with a stable external identifier."""

    evidence_id: str
    content: str
    kind: EvidenceKind = "document"


@dataclass(frozen=True)
class DeliberationDelta:
    """Structured changes made by a proposal, critique, or revision."""

    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    removal_reasons: tuple[str, ...] = ()
    preserved: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()
    accepted_corrections: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeliberationEvent:
    """One immutable contribution to a deliberation trajectory."""

    event_id: str
    round_index: int
    role: str
    kind: EventKind
    content: str
    reasoning: str = ""
    claims: tuple[str, ...] = ()
    delta: DeliberationDelta = DeliberationDelta()
    parent_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    source_id: str = ""


@dataclass(frozen=True)
class HistoryView:
    """A rendered, role-specific view over a trajectory."""

    event_ids: tuple[str, ...] | None = None
    exclude_roles: tuple[str, ...] = ()
    anonymize_roles: bool = False
    include_reasoning: bool = False


@dataclass(frozen=True)
class DeliberationTrajectory:
    """Append-only deliberation state shared by agentic patterns."""

    events: tuple[DeliberationEvent, ...] = ()

    def record(
        self,
        *,
        round_index: int,
        role: str,
        kind: EventKind,
        content: str,
        reasoning: str = "",
        claims: tuple[str, ...] = (),
        delta: DeliberationDelta | None = None,
        parent_ids: tuple[str, ...] = (),
        evidence_ids: tuple[str, ...] = (),
        source_id: str = "",
    ) -> DeliberationTrajectory:
        known_ids = {event.event_id for event in self.events}
        unknown_parents = set(parent_ids) - known_ids
        if unknown_parents:
            unknown = sorted(unknown_parents)
            raise ValueError(f"Unknown trajectory event IDs: {unknown}")

        event = DeliberationEvent(
            event_id=f"event-{len(self.events) + 1}",
            round_index=round_index,
            role=role,
            kind=kind,
            content=content,
            reasoning=reasoning,
            claims=claims,
            delta=delta or DeliberationDelta(),
            parent_ids=parent_ids,
            evidence_ids=evidence_ids,
            source_id=source_id,
        )
        return DeliberationTrajectory(events=(*self.events, event))

    def get(self, event_id: str) -> DeliberationEvent:
        for event in self.events:
            if event.event_id == event_id:
                return event
        raise KeyError(event_id)

    def render(self, view: HistoryView | None = None) -> str:
        view = view or HistoryView()
        visible_ids = set(view.event_ids) if view.event_ids is not None else None
        events = [
            event
            for event in self.events
            if (visible_ids is None or event.event_id in visible_ids)
            and event.role not in view.exclude_roles
        ]
        if not events:
            return "No deliberation history yet."

        role_aliases: dict[str, str] = {}
        lines: list[str] = []
        for event in events:
            role = event.role
            if view.anonymize_roles:
                if role not in role_aliases:
                    role_aliases[role] = f"Agent {chr(65 + len(role_aliases))}"
                role = role_aliases[role]
            lines.append(
                f"[{event.event_id} | round {event.round_index} | "
                f"{role} | {event.kind}]\n{event.content}"
            )
            if event.source_id:
                lines.append(f"Source: {event.source_id}")
            if event.claims:
                lines.append(f"Claims: {'; '.join(event.claims)}")
            if event.evidence_ids:
                lines.append(f"Evidence: {', '.join(event.evidence_ids)}")
            if event.parent_ids:
                lines.append(f"Responds to: {', '.join(event.parent_ids)}")
            delta_lines = (
                ("Added", event.delta.added),
                ("Removed", event.delta.removed),
                ("Removal reasons", event.delta.removal_reasons),
                ("Preserved", event.delta.preserved),
                ("Rejected", event.delta.rejected),
                ("Accepted corrections", event.delta.accepted_corrections),
            )
            for label, values in delta_lines:
                if values:
                    lines.append(f"{label}: {'; '.join(values)}")
            if view.include_reasoning and event.reasoning:
                lines.append(f"Reasoning: {event.reasoning}")
        return "\n\n".join(lines)
