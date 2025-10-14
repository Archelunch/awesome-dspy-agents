import dspy  # type: ignore


class AdditionAgentSignature(dspy.Signature):
    """Addition agent produces a comprehensive candidate response."""

    context: str = dspy.InputField(desc="Context C")
    instruction: str = dspy.InputField(desc="Instruction I")
    history: str = dspy.InputField(desc="Conversation history H")
    persona: str = dspy.InputField(desc="Persona influencing tone and style")

    candidate_response: str = dspy.OutputField(desc="Detailed candidate response R'")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning for additions")


class SubtractionAgentSignature(dspy.Signature):
    """Subtraction agent removes redundancies and provides feedback."""

    context: str = dspy.InputField(desc="Context C")
    instruction: str = dspy.InputField(desc="Instruction I")
    history: str = dspy.InputField(desc="Conversation history H including latest R'")
    candidate_response: str = dspy.InputField(desc="Latest candidate response R'")
    persona: str = dspy.InputField(desc="Persona influencing tone and style")

    refined_response: str = dspy.OutputField(desc="Concise response after subtraction")
    feedback: str = dspy.OutputField(desc="Feedback to guide the next addition step")
