"""
Builds chat-format prompts for the SQuAD 2.0 QA task.

Uses a carefully crafted system prompt and few-shot examples to guide
the model toward extractive answering and proper NO ANSWER detection.
"""


class PromptBuilder:
    """Responsible for constructing chat-format prompts for the QA model."""

    SYSTEM_PROMPT = (
        "🤖 You are a highly precise extractive question answering system.\n"
        "Your task is to read a context passage carefully and answer the question "
        "by extracting the exact answer span directly from the context.\n\n"
        "📋 RULES — follow these strictly:\n"
        "1. ✅ The answer MUST be a short text span taken DIRECTLY from the context — "
        "copy the relevant words exactly as they appear. Do not change a single word.\n"
        "2. 🚫 Do NOT paraphrase, rephrase, summarize, or generate any text that does "
        "not appear word-for-word in the context passage.\n"
        "3. ❌ If the answer to the question is NOT explicitly stated in the context, "
        "you MUST respond with exactly: NO ANSWER. "
        "Many questions are intentionally designed to be unanswerable from the given context — "
        "this is expected and you should not hesitate to say NO ANSWER when appropriate.\n"
        "4. 🧠 Do NOT guess, infer, use prior knowledge, or assume anything beyond "
        "what is explicitly written in the context. Even if you know the answer from "
        "your training data, if it is not in the context, say NO ANSWER.\n"
        "5. ⚠️ Be extra careful with tricky questions: even if the question is related "
        "to the same topic as the context, if the specific answer is not stated in the "
        "context, you must say NO ANSWER. The context might discuss a related subject "
        "without actually containing the answer.\n"
        "6. 📝 Respond ONLY with the extracted answer or NO ANSWER — "
        "no explanations, no extra words, no prefixes like 'The answer is'."
    )

    # Alternating answerable / unanswerable examples to teach both behaviours
    FEW_SHOT_EXAMPLES = [
        # --- pair 1 (Turing machines) ---
        {
            "context": (
                "Many types of Turing machines are used to define complexity classes, such as "
                "deterministic Turing machines, probabilistic Turing machines, non-deterministic "
                "Turing machines, quantum Turing machines, symmetric Turing machines and alternating "
                "Turing machines. They are all equally powerful in principle, but when resources "
                "(such as time or space) are bounded, some of these may be more powerful than others."
            ),
            "question": "What are two factors that directly effect how powerful a Turing machine can be?",
            "answer": "time or space",
        },
        {
            "context": (
                "Many types of Turing machines are used to define complexity classes, such as "
                "deterministic Turing machines, probabilistic Turing machines, non-deterministic "
                "Turing machines, quantum Turing machines, symmetric Turing machines and alternating "
                "Turing machines. They are all equally powerful in principle, but when resources "
                "(such as time or space) are bounded, some of these may be more powerful than others."
            ),
            "question": "What machines are not equally powerful in principle?",
            "answer": "NO ANSWER",
        },
        # --- pair 2 (Scotland / Normans) ---
        {
            "context": (
                "King David I of Scotland, whose elder brother Alexander I had married "
                "Sybilla of Normandy, was instrumental in introducing Normans and Norman "
                "culture to Scotland, part of the process some scholars call the "
                '"Davidian Revolution".'
            ),
            "question": "Who did Alexander I marry?",
            "answer": "Sybilla of Normandy",
        },
        {
            "context": (
                "King David I of Scotland, whose elder brother Alexander I had married "
                "Sybilla of Normandy, was instrumental in introducing Normans and Norman "
                "culture to Scotland, part of the process some scholars call the "
                '"Davidian Revolution".'
            ),
            "question": "Who did King David I of Scotland marry?",
            "answer": "NO ANSWER",
        },
    ]

    def build_messages(self, context, question):
        """
        Build a chat-message list with system prompt, few-shot examples,
        and the actual question.

        Returns:
            list[dict]: Messages ready for tokenizer.apply_chat_template().
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        for example in self.FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": f"Context: {example['context']}\n\nQuestion: {example['question']}",
            })
            messages.append({
                "role": "assistant",
                "content": example["answer"],
            })

        messages.append({
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}",
        })

        return messages


# Keep the module-level function so existing imports still work
_default_builder = PromptBuilder()


def build_qa_messages(context, question):
    return _default_builder.build_messages(context, question)
