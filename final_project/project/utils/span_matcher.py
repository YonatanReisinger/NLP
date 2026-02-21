"""
Fuzzy span matching — snaps a model-generated answer to the closest
actual substring in the context passage.

This corrects two kinds of errors:
1. Minor extraction mistakes (extra/missing words, small rewording).
2. Hallucinated answers that have no close match in the context,
   which get rejected (acts as a second grounding check).
"""

from difflib import SequenceMatcher


class SpanMatcher:
    """Finds the best matching span in the context for a candidate answer."""

    def __init__(self, min_similarity=0.6):
        self.min_similarity = min_similarity

    def find_best_span(self, answer, context):
        """Find the context substring most similar to *answer*.

        Uses a sliding window of varying widths around the answer
        length, scored by difflib.SequenceMatcher.

        Returns:
            tuple[str | None, float]:
                (best_span, similarity_ratio).
                best_span is None when nothing exceeds min_similarity.
        """
        answer_lower = answer.lower().strip()
        context_lower = context.lower()
        words = context.split()

        if not answer_lower or not words:
            return None, 0.0

        answer_word_count = len(answer_lower.split())

        best_span = None
        best_ratio = 0.0

        # Try window sizes from (answer_len - 2) to (answer_len + 3)
        min_win = max(1, answer_word_count - 2)
        max_win = answer_word_count + 3

        for win_size in range(min_win, min(max_win + 1, len(words) + 1)):
            for start in range(len(words) - win_size + 1):
                candidate = " ".join(words[start: start + win_size])
                ratio = SequenceMatcher(
                    None, answer_lower, candidate.lower()
                ).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_span = candidate

        if best_ratio >= self.min_similarity:
            return best_span, best_ratio
        return None, best_ratio

    def snap_or_reject(self, answer, context):
        """Snap *answer* to the best context span, or return None if
        no close match is found (likely hallucination).

        Returns:
            str | None: The snapped span, or None to signal rejection.
        """
        span, ratio = self.find_best_span(answer, context)
        return span
