"""
Analysis Agent (Agent 2) for SQuAD 2.0 QA pipeline.

Provides a second verification pass using the same LLM with a
verification-focused prompt.  Decides whether to ACCEPT or REJECT
Agent 1's proposed answer, improving NO ANSWER detection.
"""


class AnalysisAgent:
    """Verifies proposed answers by asking the LLM to reason about
    whether the answer is truly supported by the context."""

    VERIFICATION_PROMPT = (
        "You are an answer verification agent for reading comprehension. "
        "Given a context paragraph, a question, and a proposed answer, "
        "decide whether the proposed answer is CORRECT (directly stated "
        "in the context) or should be REJECTED.\n\n"
        "Rules:\n"
        "- ACCEPT only if the answer is explicitly stated in the context.\n"
        "- REJECT if the answer is not in the context, is hallucinated, "
        "or if the question is unanswerable from the context.\n"
        "- REJECT if the proposed answer is 'NO ANSWER' but the context "
        "actually contains the answer (false rejection).\n"
        "- Respond with a single word: ACCEPT or REJECT.\n"
    )

    FEW_SHOT_EXAMPLES = [
        # 1. Correct answer → ACCEPT
        {
            "context": "The Eiffel Tower was constructed in 1889 for the "
                       "World's Fair in Paris. It stands 330 metres tall.",
            "question": "When was the Eiffel Tower built?",
            "proposed_answer": "1889",
            "confidence": -0.15,
            "decision": "ACCEPT",
        },
        # 2. Hallucinated answer → REJECT (should be NO ANSWER)
        {
            "context": "The Eiffel Tower was constructed in 1889 for the "
                       "World's Fair in Paris. It stands 330 metres tall.",
            "question": "Who designed the Eiffel Tower?",
            "proposed_answer": "Gustave Eiffel",
            "confidence": -0.85,
            "decision": "REJECT",
        },
        # 3. Tricky: answer seems related but isn't stated → REJECT
        {
            "context": "Alexander Fleming discovered penicillin in 1928. "
                       "The discovery revolutionized medicine and saved "
                       "millions of lives worldwide.",
            "question": "What university did Fleming attend?",
            "proposed_answer": "St Mary's Hospital Medical School",
            "confidence": -0.70,
            "decision": "REJECT",
        },
        # 4. Rescue: answer was wrongly marked NO ANSWER → ACCEPT
        {
            "context": "Alexander Fleming discovered penicillin in 1928. "
                       "The discovery revolutionized medicine and saved "
                       "millions of lives worldwide.",
            "question": "What did Fleming discover?",
            "proposed_answer": "NO ANSWER",
            "confidence": -0.40,
            "decision": "REJECT",
        },
    ]

    def build_verification_messages(self, context, question,
                                    proposed_answer, confidence):
        """Build chat messages for the verification call.

        Returns:
            list[dict]: Messages ready for tokenizer.apply_chat_template().
        """
        messages = [{"role": "system", "content": self.VERIFICATION_PROMPT}]

        for ex in self.FEW_SHOT_EXAMPLES:
            user_msg = (
                f"Context: {ex['context']}\n\n"
                f"Question: {ex['question']}\n\n"
                f"Proposed answer: {ex['proposed_answer']}\n"
                f"Confidence: {ex['confidence']:.2f}"
            )
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ex["decision"]})

        user_msg = (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Proposed answer: {proposed_answer}\n"
            f"Confidence: {confidence:.2f}"
        )
        messages.append({"role": "user", "content": user_msg})

        return messages

    def verify(self, context, question, proposed_answer, confidence,
               model_manager):
        """Run verification and return ACCEPT or REJECT.

        Args:
            context: The passage text.
            question: The question being answered.
            proposed_answer: Agent 1's answer (may be NO ANSWER).
            confidence: Agent 1's confidence score.
            model_manager: ModelManager instance for inference.

        Returns:
            str: "ACCEPT" or "REJECT".
        """
        messages = self.build_verification_messages(
            context, question, proposed_answer, confidence
        )
        raw = model_manager.generate_single(messages).strip().upper()

        if "ACCEPT" in raw:
            return "ACCEPT"
        return "REJECT"
