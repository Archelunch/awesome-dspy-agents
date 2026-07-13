import dspy  # type: ignore


class AdditionAgentSignature(dspy.Signature):
    """Add useful material to the current response without restarting it."""

    context: str = dspy.InputField(desc="Context C")
    instruction: str = dspy.InputField(desc="Instruction I")
    current_response: str = dspy.InputField(desc="Latest refined response")
    previous_feedback: str = dspy.InputField(
        desc="Specific feedback from the previous subtraction"
    )
    persona: str = dspy.InputField(desc="Persona influencing tone and style")

    additions: list[str] = dspy.OutputField(
        desc="Material added to satisfy the instruction or feedback"
    )
    candidate_response: str = dspy.OutputField(desc="Detailed candidate response R'")
    reasoning: str = dspy.OutputField(desc="Concise rationale for the additions")


class SubtractionAgentSignature(dspy.Signature):
    """Remove harmful material while preserving supported facts and constraints."""

    context: str = dspy.InputField(desc="Context C")
    instruction: str = dspy.InputField(desc="Instruction I")
    candidate_response: str = dspy.InputField(desc="Latest candidate response R'")
    preservation_requirements: str = dspy.InputField(
        desc="Instruction and constraints that subtraction must preserve"
    )
    persona: str = dspy.InputField(desc="Persona influencing tone and style")

    removals: list[str] = dspy.OutputField(
        desc="Redundant, unsupported, irrelevant, or harmful material removed"
    )
    removal_reasons: list[str] = dspy.OutputField(
        desc="Reason corresponding to each removal"
    )
    preserved_facts: list[str] = dspy.OutputField(
        desc="Important supported facts explicitly preserved"
    )
    refined_response: str = dspy.OutputField(desc="Concise response after subtraction")
    feedback: str = dspy.OutputField(desc="Feedback to guide the next addition step")
