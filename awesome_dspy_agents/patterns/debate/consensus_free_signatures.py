import dspy


class IndependentProposal(dspy.Signature):
    """Solve independently before seeing any other agent's answer."""

    problem: str = dspy.InputField(desc="Problem or instruction to solve")
    context: str = dspy.InputField(desc="Evidence and context available to the agent")
    perspective: str = dspy.InputField(
        desc="Assigned perspective; it must not override evidence"
    )

    answer: str = dspy.OutputField(desc="Independent candidate answer")
    claims: list[str] = dspy.OutputField(desc="Atomic claims supporting the answer")
    evidence_ids: list[str] = dspy.OutputField(
        desc="Identifiers of evidence used by the answer"
    )
    reasoning: str = dspy.OutputField(desc="Concise rationale for the proposal")


class SelectConflicts(dspy.Signature):
    """Retain the most consequential disagreements without resolving them."""

    problem: str = dspy.InputField()
    proposals: str = dspy.InputField(desc="Anonymous independent proposals")

    selected_event_ids: list[str] = dspy.OutputField(
        desc="Proposal or revision event IDs involved in meaningful disagreement"
    )
    conflicts: list[str] = dspy.OutputField(
        desc="Specific contradictory claims, evidence, or conclusions"
    )
    conflict_routes: dict[str, list[str]] = dspy.OutputField(
        desc="For each candidate event ID, the opposing event IDs it should inspect"
    )
    conflicts_by_event: dict[str, list[str]] = dspy.OutputField(
        desc="For each candidate event ID, only its relevant conflict descriptions"
    )


class ReviseWithoutConformity(dspy.Signature):
    """Reassess an answer without assuming that agreement or majority is correct."""

    problem: str = dspy.InputField()
    context: str = dspy.InputField()
    original_answer: str = dspy.InputField()
    opposing_arguments: str = dspy.InputField(
        desc="Selected anonymous disagreements from other agents"
    )
    conflicts: list[str] = dspy.InputField()

    retained_claims: list[str] = dspy.OutputField(
        desc="Original claims retained because they remain supported"
    )
    rejected_claims: list[str] = dspy.OutputField(
        desc="Original claims rejected after concrete counter-evidence"
    )
    accepted_corrections: list[str] = dspy.OutputField(
        desc="Corrections accepted from opposing arguments with justification"
    )
    evidence_ids: list[str] = dspy.OutputField(
        desc="Identifiers of supplied evidence supporting the revised answer"
    )
    revised_answer: str = dspy.OutputField()
    reasoning: str = dspy.OutputField(desc="Why the answer changed or remained stable")


class ArbitrateFullTrajectory(dspy.Signature):
    """Choose the best-supported answer from the entire reasoning trajectory."""

    problem: str = dspy.InputField()
    context: str = dspy.InputField()
    trajectory: str = dspy.InputField(
        desc="Anonymous proposals and revisions from every round"
    )

    winning_event_id: str = dspy.OutputField(
        desc="Event ID containing the strongest answer"
    )
    candidate_scores: dict[str, float] = dspy.OutputField(
        desc="Evidence-grounded score for every proposal or revision considered"
    )
    final_answer: str = dspy.OutputField()
    justification: str = dspy.OutputField(
        desc="Evidence-based reason for selecting this trajectory"
    )
