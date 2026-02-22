"""
Extraction Agent — first stage of the two-agent QA pipeline.

Runs the initial LLM call via PromptBuilder and collects structured
observations (raw answer, confidence, grounding status, span match, etc.)
that the Analysis Agent will use to make the final decision.
"""

from utils.prompt_builder import PromptBuilder
from utils.answer_processor import AnswerProcessor
from utils.confidence_scorer import ConfidenceScorer
from utils.span_matcher import SpanMatcher


class ExtractionAgent:
    """Wraps the first LLM call and gathers evidence signals per question."""

    def __init__(self, model_manager, prompt_builder=None,
                 answer_processor=None, span_matcher=None):
        self.model_manager = model_manager
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.answer_processor = answer_processor or AnswerProcessor()
        self.span_matcher = span_matcher or SpanMatcher(min_similarity=0.6)

    def _collect_observation(self, raw_answer, context, confidence):
        """Build an observation dict for a single item."""
        # Clean answer via answer processor
        cleaned_answer = self.answer_processor.process(raw_answer, context)

        # Check no-answer signals from regex patterns
        no_answer_detected = self.answer_processor._is_no_answer_response(raw_answer)

        # Check grounding
        first_line = raw_answer.strip().split("\n")[0].strip()
        prefix_removed = self.answer_processor._remove_common_prefixes(first_line)
        prefix_removed = self.answer_processor._strip_quotes(prefix_removed)
        prefix_removed = prefix_removed.rstrip(".").strip()
        grounded = self.answer_processor._is_grounded_in_context(
            prefix_removed, context
        ) if prefix_removed else False

        # Find best span match
        best_span, similarity_ratio = None, 0.0
        if prefix_removed:
            best_span, similarity_ratio = self.span_matcher.find_best_span(
                prefix_removed, context
            )

        return {
            "raw_answer": raw_answer,
            "confidence": confidence,
            "no_answer_detected": no_answer_detected,
            "grounded_in_context": grounded,
            "best_span": best_span,
            "similarity_ratio": similarity_ratio,
            "cleaned_answer": cleaned_answer,
        }

    def extract_single(self, context, question):
        """Run extraction on a single question and return an observation dict.

        Args:
            context: the context passage.
            question: the question.

        Returns:
            dict: Observation with all evidence signals.
        """
        messages = self.prompt_builder.build_messages(context, question)
        raw_answers, confidences = self.model_manager.generate_batch([messages])
        return self._collect_observation(raw_answers[0], context, confidences[0])
