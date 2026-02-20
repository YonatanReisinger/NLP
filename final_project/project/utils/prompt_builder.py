"""
Builds chat-format prompts for the SQuAD 2.0 QA task.

Uses a carefully crafted system prompt and few-shot examples to guide
the model toward extractive answering and proper NO ANSWER detection.
"""

SYSTEM_PROMPT = (
    "You are an extractive question answering system. "
    "Given a context passage and a question, extract the answer directly from the context.\n\n"
    "RULES:\n"
    "1. The answer MUST be a short text span taken DIRECTLY from the context — copy it exactly.\n"
    "2. Do NOT paraphrase, rephrase, or generate text that does not appear in the context.\n"
    "3. If the answer to the question is NOT explicitly stated in the context, "
    "respond with exactly: NO ANSWER\n"
    "4. Do NOT guess, infer, or assume anything beyond what is written.\n"
    "5. Even if the question is related to the topic, if the specific answer is not in the "
    "context, say NO ANSWER.\n"
    "6. Respond ONLY with the extracted answer or NO ANSWER — no explanations, no extra words."
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


def build_qa_messages(context, question):
    """
    Build a chat-message list with system prompt, few-shot examples,
    and the actual question.

    Returns:
        list[dict]: Messages ready for tokenizer.apply_chat_template().
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for example in FEW_SHOT_EXAMPLES:
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
