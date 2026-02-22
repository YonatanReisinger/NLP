"""
Analysis Agent — second stage of the two-agent QA pipeline.

Receives the context, question, and structured observations from the
Extraction Agent and makes a second LLM call with enhanced decision
logic, focusing on correctly deciding NO ANSWER cases in SQuAD 2.0.
"""

from utils.evaluate_results import NO_ANSWER_MARKER


class AnalysisAgent:
    """Uses a second LLM call to decide the final answer from extraction evidence."""

    SYSTEM_PROMPT = (
        "You are an analysis agent for a question-answering system. "
        "You receive a question, its context passage, and evidence signals from an extraction step. "
        "Your job is to decide the FINAL answer.\n\n"
        "RULES:\n"
        "1. If the extracted answer looks correct and is well-supported by the context, output it.\n"
        "2. If a better matching span was found in the context, prefer the matched span.\n"
        "3. If the evidence suggests the question CANNOT be answered from the context, output exactly: NO ANSWER\n"
        "4. Pay close attention to these NO ANSWER signals:\n"
        "   - The extraction step detected no-answer patterns in the model output\n"
        "   - Low confidence score (below -1.0 is suspicious, below -2.0 is very low)\n"
        "   - The answer is NOT grounded in the context\n"
        "   - No good span match was found (low similarity)\n"
        "5. Respond ONLY with the final answer text or NO ANSWER — no explanations."
    )

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def _build_analysis_prompt(self, context, question, observation):
        """Build a chat-message list for the analysis LLM call."""
        no_answer_str = "Yes" if observation["no_answer_detected"] else "No"
        grounded_str = "Yes" if observation["grounded_in_context"] else "No"
        span_str = observation["best_span"] if observation["best_span"] else "(none)"
        ratio_str = f"{observation['similarity_ratio']:.2f}"
        confidence_str = f"{observation['confidence']:.2f}"

        user_content = (
            f"Question: {question}\n"
            f"Context: {context}\n\n"
            f"--- Extraction Evidence ---\n"
            f"Extracted Answer: {observation['raw_answer']}\n"
            f"Confidence Score: {confidence_str} (range: -inf to 0, higher is better)\n"
            f"Grounded in Context: {grounded_str}\n"
            f"Best Matching Span: {span_str} (similarity: {ratio_str})\n"
            f"No-Answer Signals Detected: {no_answer_str}\n\n"
            f"Decide: Should the final answer be the extracted answer, "
            f"the matched span, or NO ANSWER?"
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def analyze_single(self, context, question, observation):
        """Run analysis on a single observation and return the final answer.

        Args:
            context: the context passage.
            question: the question.
            observation: observation dict from ExtractionAgent.

        Returns:
            str: The final answer.
        """
        messages = self._build_analysis_prompt(context, question, observation)
        raw_outputs, _ = self.model_manager.generate_batch([messages])

        answer = raw_outputs[0].strip().split("\n")[0].strip()
        # Normalize no-answer variants
        if answer.upper() in ("NO ANSWER", "NO_ANSWER", "NOANSWER"):
            answer = NO_ANSWER_MARKER
        return answer
