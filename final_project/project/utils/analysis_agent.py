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
        "🔍 You are an answer verification agent — a critical second pair of eyes "
        "in a reading comprehension QA pipeline.\n\n"
        "🎯 YOUR PRIMARY GOAL:\n"
        "Your most important job is to catch questions that should be answered with "
        "NO ANSWER but were incorrectly given an answer by the first agent. "
        "The SQuAD 2.0 dataset is specifically designed with many unanswerable questions — "
        "questions that look like they belong to the context but whose answers are NOT "
        "actually stated anywhere in the passage. The first agent sometimes gets fooled "
        "by these tricky questions and hallucinates an answer that sounds plausible but "
        "is not in the context. Your job is to catch these mistakes.\n\n"
        "📋 HOW IT WORKS:\n"
        "You receive a context paragraph, a question, and a proposed answer from the "
        "first agent. You must carefully verify whether that proposed answer is truly "
        "supported by the context, and decide: ACCEPT or REJECT.\n\n"
        "✅ ACCEPT — say this ONLY when:\n"
        "- The proposed answer is a text span that appears directly in the context.\n"
        "- The answer actually addresses the question being asked.\n"
        "- You can point to the exact words in the context that match the answer.\n\n"
        "❌ REJECT — say this when ANY of the following are true:\n"
        "- The proposed answer contains information NOT found in the context, "
        "even if the information is factually true in the real world. "
        "This is the most common mistake — the first agent uses its world knowledge "
        "instead of sticking to the context. You must catch this!\n"
        "- The question asks about something the context does not discuss or address.\n"
        "- The context talks about a related topic but does not contain the specific "
        "answer to the specific question being asked.\n"
        "- The proposed answer is vaguely related to the context but does not "
        "actually answer the question.\n"
        "- The confidence score is very low (very negative), which suggests the "
        "first agent was uncertain and may have guessed.\n"
        "- The proposed answer is 'NO ANSWER' but the context clearly does contain "
        "the answer (this is a false rejection that should be rescued).\n\n"
        "⚠️ IMPORTANT REMINDERS:\n"
        "- When in doubt, REJECT. It is much better to say NO ANSWER for a question "
        "that has an answer than to accept a hallucinated answer.\n"
        "- Many questions are deliberately designed to trick the model. The context "
        "might mention a person but not their birthdate, or mention an event but not "
        "its location. Stay vigilant!\n"
        "- Read the context carefully word by word. Do not rely on your general knowledge.\n\n"
        "📝 RESPONSE FORMAT:\n"
        "Respond with a single word: ACCEPT or REJECT. Nothing else."
    )

    FEW_SHOT_EXAMPLES = [
        # 1. Correct answer clearly in context → ACCEPT
        {
            "context": "The Eiffel Tower was constructed in 1889 for the "
                       "World's Fair in Paris. It stands 330 metres tall.",
            "question": "When was the Eiffel Tower built?",
            "proposed_answer": "1889",
            "confidence": -0.15,
            "decision": "ACCEPT",
        },
        # 2. Hallucinated answer — true in real world but NOT in context → REJECT
        {
            "context": "The Eiffel Tower was constructed in 1889 for the "
                       "World's Fair in Paris. It stands 330 metres tall.",
            "question": "Who designed the Eiffel Tower?",
            "proposed_answer": "Gustave Eiffel",
            "confidence": -0.85,
            "decision": "REJECT",
        },
        # 3. Tricky: context mentions the person but not the asked detail → REJECT
        {
            "context": "Alexander Fleming discovered penicillin in 1928. "
                       "The discovery revolutionized medicine and saved "
                       "millions of lives worldwide.",
            "question": "What university did Fleming attend?",
            "proposed_answer": "St Mary's Hospital Medical School",
            "confidence": -0.70,
            "decision": "REJECT",
        },
        # 4. Context talks about related topic but doesn't have this answer → REJECT
        {
            "context": "The Ottoman Empire controlled much of Southeast Europe, "
                       "Western Asia and North Africa between the 14th and early "
                       "20th centuries. It was founded by Osman I.",
            "question": "What was the population of the Ottoman Empire?",
            "proposed_answer": "35 million",
            "confidence": -0.90,
            "decision": "REJECT",
        },
        # 5. Correct answer present in context → ACCEPT
        {
            "context": "The Ottoman Empire controlled much of Southeast Europe, "
                       "Western Asia and North Africa between the 14th and early "
                       "20th centuries. It was founded by Osman I.",
            "question": "Who founded the Ottoman Empire?",
            "proposed_answer": "Osman I",
            "confidence": -0.20,
            "decision": "ACCEPT",
        },
        # 6. Rescue: answer was wrongly marked NO ANSWER → REJECT the NO ANSWER
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
