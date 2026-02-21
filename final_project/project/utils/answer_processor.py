"""
Post-processing for model outputs in SQuAD 2.0 QA.

Handles:
- Detecting "no answer" responses in various phrasings
- Cleaning common prefixes / suffixes from model outputs
- Verifying the answer is grounded in the context passage
"""

import re
import string

NO_ANSWER_MARKER = "NO ANSWER"


class AnswerProcessor:
    """Responsible for cleaning raw model output into a final answer."""

    NO_ANSWER_PATTERNS = [
        r"\bno\s*answer\b",
        r"\bcannot\s+be\s+answered\b",
        r"\bunanswerable\b",
        r"\bnot\s+(?:explicitly\s+)?(?:mentioned|stated|provided|found|given"
        r"|included|specified|indicated|addressed)\b",
        r"\bno\s+(?:information|mention|reference|evidence)\b",
        r"\bcannot\s+(?:find|determine|answer|be determined)\b",
        r"\bdoes\s+not\s+(?:contain|provide|mention|state|include|specify|address)\b",
        r"\bnot\s+(?:enough|sufficient)\s+information\b",
        r"\b(?:context|passage|text)\s+does\s+not\b",
        r"\bcannot\s+be\s+determined\b",
        r"\bnot\s+(?:answerable|possible\s+to\s+(?:answer|determine))\b",
        r"\bi\s+(?:cannot|can't|could\s+not|couldn't)\s+(?:find|determine|answer)\b",
    ]

    PREFIX_PATTERNS = [
        r"^(?:the\s+)?answer\s*(?:is\s*:?\s*|:\s*)",
        r"^based\s+on\s+(?:the\s+)?(?:context|passage|text)\s*,?\s*",
        r"^according\s+to\s+(?:the\s+)?(?:context|passage|text)\s*,?\s*",
        r"^from\s+(?:the\s+)?(?:context|passage|text)\s*,?\s*",
        r"^in\s+(?:the\s+)?(?:context|passage|text)\s*,?\s*",
    ]

    STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "of", "in", "to", "for",
        "with", "on", "at", "from", "by", "as", "into", "through", "during",
        "before", "after", "and", "but", "or", "not", "so", "it", "its",
        "this", "that", "these", "those", "he", "she", "they", "them", "their",
        "we", "us", "our", "you", "your", "i", "me", "my", "which", "who",
        "whom", "what", "where", "when", "how", "than", "then",
    })

    def _is_no_answer_response(self, text):
        """Return True if *text* signals the question is unanswerable."""
        lowered = text.lower().strip()
        return any(re.search(p, lowered) for p in self.NO_ANSWER_PATTERNS)

    def _remove_common_prefixes(self, text):
        """Strip typical LLM preamble from the answer."""
        result = text
        for pattern in self.PREFIX_PATTERNS:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        return result.strip()

    @staticmethod
    def _strip_quotes(text):
        """Remove surrounding quotation marks."""
        if len(text) >= 2:
            if (text[0] == '"' and text[-1] == '"') or \
               (text[0] == "'" and text[-1] == "'"):
                return text[1:-1].strip()
        return text

    @staticmethod
    def _normalize(text):
        """Lowercase, remove punctuation, collapse whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join(text.split())

    def _is_grounded_in_context(self, answer, context):
        """
        Check whether the answer is supported by the context.

        Uses two strategies:
        1. Exact normalised-substring match.
        2. Content-word overlap (ignoring stopwords). At least 50 % of
           content words must appear in the context.
        """
        norm_answer = self._normalize(answer)
        norm_context = self._normalize(context)

        # Strategy 1 — substring
        if norm_answer in norm_context:
            return True

        # Strategy 2 — word overlap
        content_words = [w for w in norm_answer.split() if w not in self.STOPWORDS]
        if not content_words:
            return True  # only stopwords — hard to judge, keep the answer

        matched = sum(1 for w in content_words if w in norm_context)
        return matched / len(content_words) >= 0.5

    def process(self, raw_model_output, context):
        """
        Convert raw model text into a clean final answer.

        Steps:
            1. Take the first line (models sometimes ramble).
            2. Check for no-answer indicators.
            3. Remove common prefixes / quotes / trailing period.
            4. Verify grounding in context.

        Returns:
            str: The cleaned answer or ``NO_ANSWER_MARKER``.
        """
        text = raw_model_output.strip()

        # Use only the first line
        first_line = text.split("\n")[0].strip()

        # Detect no-answer phrasing
        if self._is_no_answer_response(first_line):
            return NO_ANSWER_MARKER

        # Clean up
        answer = self._remove_common_prefixes(first_line)
        answer = self._strip_quotes(answer)
        answer = answer.rstrip(".")
        answer = answer.strip()

        if not answer:
            return NO_ANSWER_MARKER

        # Grounding verification
        if not self._is_grounded_in_context(answer, context):
            return NO_ANSWER_MARKER

        return answer


# Keep the module-level function so existing imports still work
_default_processor = AnswerProcessor()


def process_answer(raw_model_output, context):
    return _default_processor.process(raw_model_output, context)
