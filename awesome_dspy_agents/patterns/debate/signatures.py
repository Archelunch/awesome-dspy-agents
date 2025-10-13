import dspy


class AffirmativeDebater(dspy.Signature):
    """Affirmative debater presents initial argument."""

    debate_topic: str = dspy.InputField(desc="Topic to debate")
    debate_history: str = dspy.InputField(desc="Previous debate exchanges")
    role_instruction: str = dspy.InputField(
        desc="Role-specific instructions for affirmative side"
    )
    persona: str = dspy.InputField(
        desc="Persona profile influencing tone, priorities, and style"
    )

    argument: str = dspy.OutputField(desc="Affirmative side's argument")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning")


class NegativeDebater(dspy.Signature):
    """Negative debater provides counter-argument."""

    debate_topic: str = dspy.InputField(desc="Topic to debate")
    debate_history: str = dspy.InputField(desc="Previous debate exchanges")
    affirmative_argument: str = dspy.InputField(desc="Current affirmative argument")
    role_instruction: str = dspy.InputField(
        desc="Role-specific instructions for negative side"
    )
    persona: str = dspy.InputField(
        desc="Persona profile influencing tone, priorities, and style"
    )

    counter_argument: str = dspy.OutputField(desc="Negative side's counter-argument")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning")


class JudgeDiscriminative(dspy.Signature):
    """Judge evaluates if correct solution has been reached."""

    debate_topic: str = dspy.InputField(desc="Original problem/question")
    debate_history: str = dspy.InputField(desc="Complete debate transcript")
    current_iteration: int = dspy.InputField(desc="Current iteration number")

    solution_found: bool = dspy.OutputField(
        desc="True if correct solution identified, False otherwise"
    )
    confidence: float = dspy.OutputField(desc="Confidence in decision (0.0 to 1.0)")
    reasoning: str = dspy.OutputField(desc="Explanation for decision")


class JudgeExtractive(dspy.Signature):
    """Judge extracts final answer from debate history."""

    debate_topic: str = dspy.InputField(desc="Original problem/question")
    debate_history: str = dspy.InputField(desc="Complete debate transcript")

    final_answer: str = dspy.OutputField(desc="Extracted final solution")
    justification: str = dspy.OutputField(
        desc="Why this answer was chosen based on debate"
    )
