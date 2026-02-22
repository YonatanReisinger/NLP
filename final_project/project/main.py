import json
import re
import string
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import time
import torch
torch.set_default_device('cpu')

from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.evaluate_results import NO_ANSWER_MARKER, evaluate_results
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
#  ConfidenceScorer
# ═══════════════════════════════════════════════════════════════════════

class ConfidenceScorer:
    """Scores generation confidence from per-step logits."""

    def __init__(self, no_answer_threshold: float = -1.0) -> None:
        """
        Args:
            no_answer_threshold (float): minimum avg log-prob to consider an answer confident.
        """
        self.no_answer_threshold = no_answer_threshold

    def compute_batch_confidences(self, scores: Tuple[torch.Tensor, ...], sequences: torch.Tensor,
                                     prompt_len: int, eos_token_id: int) -> List[float]:
        """Compute avg log-probability per generated token for each item in the batch.
        Low avg log-prob signals the model is uncertain — likely hallucination.

        Args:
            scores (tuple[torch.Tensor]): per-step logits from model.generate, one tensor per step.
            sequences (torch.Tensor): generated token IDs, shape (batch_size, total_seq_len).
            prompt_len (int): number of prompt tokens (to skip when reading generated tokens).
            eos_token_id (int): end-of-sequence token ID to stop at.

        Returns:
            list[float]: average log-probability for each item in the batch.
        """
        batch_size = sequences.shape[0]
        confidences = []

        for batch_idx in range(batch_size):
            log_probs = []
            for step, step_scores in enumerate(scores):
                token_id = sequences[batch_idx, prompt_len + step].item()
                if token_id == eos_token_id:
                    break
                probs = F.log_softmax(step_scores[batch_idx].float(), dim=-1)
                log_probs.append(probs[token_id].item())

            avg = sum(log_probs) / len(log_probs) if log_probs else float('-inf')
            confidences.append(avg)

        return confidences

    def is_confident(self, confidence: float) -> bool:
        """
        Args:
            confidence (float): avg log-probability of the generated answer.

        Returns:
            bool: True if confidence meets the threshold.
        """
        return confidence >= self.no_answer_threshold


# ═══════════════════════════════════════════════════════════════════════
#  SpanMatcher
# ═══════════════════════════════════════════════════════════════════════

class SpanMatcher:
    """Finds the best matching span in the context for a candidate answer."""

    def __init__(self, min_similarity: float = 0.6) -> None:
        """
        Args:
            min_similarity (float): minimum SequenceMatcher ratio to accept a span.
        """
        self.min_similarity = min_similarity

    def find_best_span(self, answer: str, context: str) -> Tuple[Optional[str], float]:
        """Slide a window over context words and find the span most similar to the answer.
        Uses SequenceMatcher ratio (0-1) to score each candidate window.

        Args:
            answer (str): candidate answer text from the model.
            context (str): the original context passage.

        Returns:
            tuple[str | None, float]: (best matching span, similarity ratio).
                Returns (None, best_ratio) if no span meets min_similarity.
        """
        answer_lower = answer.lower().strip()
        context_lower = context.lower()
        words = context.split()

        if not answer_lower or not words:
            return None, 0.0

        answer_word_count = len(answer_lower.split())

        best_span = None
        best_ratio = 0.0

        # Search windows slightly smaller and larger than the answer length
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

    def snap_or_reject(self, answer: str, context: str) -> Optional[str]:
        """Return the best matching span if above threshold, else None.

        Args:
            answer (str): candidate answer text.
            context (str): the original context passage.

        Returns:
            str | None: the snapped span, or None if no match meets the threshold.
        """
        span, ratio = self.find_best_span(answer, context)
        return span


# ═══════════════════════════════════════════════════════════════════════
#  AnswerProcessor
# ═══════════════════════════════════════════════════════════════════════

class AnswerProcessor:
    """Responsible for cleaning raw model output into a final answer."""

    # Regex patterns that indicate the model is saying "I can't answer this"
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

    # Common LLM prefixes to strip (e.g., "The answer is: ...")
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

    def _is_no_answer_response(self, text: str) -> bool:
        """
        Args:
            text (str): raw model output text.

        Returns:
            bool: True if the text matches any no-answer pattern.
        """
        lowered = text.lower().strip()
        return any(re.search(p, lowered) for p in self.NO_ANSWER_PATTERNS)

    def _remove_common_prefixes(self, text: str) -> str:
        """
        Args:
            text (str): answer text potentially starting with a prefix like "The answer is:".

        Returns:
            str: text with known prefixes stripped.
        """
        result = text
        for pattern in self.PREFIX_PATTERNS:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        return result.strip()

    @staticmethod
    def _strip_quotes(text: str) -> str:
        """
        Args:
            text (str): text potentially wrapped in matching quotes.

        Returns:
            str: text with outer quotes removed if present.
        """
        if len(text) >= 2:
            if (text[0] == '"' and text[-1] == '"') or \
               (text[0] == "'" and text[-1] == "'"):
                return text[1:-1].strip()
        return text

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, remove punctuation, and collapse whitespace.

        Args:
            text (str): raw text.

        Returns:
            str: normalized text.
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join(text.split())

    def _is_grounded_in_context(self, answer: str, context: str) -> bool:
        """Check that the answer actually comes from the context (not hallucinated).
        First tries exact substring match, then falls back to word-overlap ratio.

        Args:
            answer (str): cleaned candidate answer.
            context (str): the original context passage.

        Returns:
            bool: True if the answer is grounded in the context.
        """
        norm_answer = self._normalize(answer)
        norm_context = self._normalize(context)

        # Exact substring match
        if norm_answer in norm_context:
            return True

        # Word-overlap: at least 50% of content words must appear in context
        content_words = [w for w in norm_answer.split() if w not in self.STOPWORDS]
        if not content_words:
            return True

        matched = sum(1 for w in content_words if w in norm_context)
        return matched / len(content_words) >= 0.5

    def process(self, raw_model_output: str, context: str) -> str:
        """Clean raw LLM output into a final answer or NO_ANSWER_MARKER.
        Pipeline: detect no-answer -> strip prefixes/quotes -> verify grounding.

        Args:
            raw_model_output (str): raw text generated by the LLM.
            context (str): the original context passage.

        Returns:
            str: cleaned answer string, or NO_ANSWER_MARKER if unanswerable.
        """
        text = raw_model_output.strip()
        first_line = text.split("\n")[0].strip()

        if self._is_no_answer_response(first_line):
            return NO_ANSWER_MARKER

        # Clean up common LLM formatting artifacts
        answer = self._remove_common_prefixes(first_line)
        answer = self._strip_quotes(answer)
        answer = answer.rstrip(".")
        answer = answer.strip()

        if not answer:
            return NO_ANSWER_MARKER

        # Reject answers that don't appear in the context (anti-hallucination)
        if not self._is_grounded_in_context(answer, context):
            return NO_ANSWER_MARKER

        return answer


# ═══════════════════════════════════════════════════════════════════════
#  PromptBuilder
# ═══════════════════════════════════════════════════════════════════════

class PromptBuilder:
    """Responsible for constructing chat-format prompts for the QA model."""

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

    # Paired few-shot examples: each answerable question is followed by
    # a similar but unanswerable variant, teaching the model to distinguish them.
    FEW_SHOT_EXAMPLES = [
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

    def build_messages(self, context: str, question: str) -> List[Dict[str, str]]:
        """Assemble a chat-format prompt: system instruction + few-shot examples + actual query.

        Args:
            context (str): the context passage for the question.
            question (str): the question to answer.

        Returns:
            list[dict[str, str]]: list of chat messages with 'role' and 'content' keys.
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


# ═══════════════════════════════════════════════════════════════════════
#  ModelManager
# ═══════════════════════════════════════════════════════════════════════

class ModelManager:
    """Responsible for loading the LLM and running inference (single + batch)."""

    def __init__(self, model_name: str = 'meta-llama/Llama-3.2-3B-Instruct') -> None:
        """
        Args:
            model_name (str): HuggingFace model identifier to load.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, token=True
        )
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_single(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response for a single chat-format message list (no confidence scores).

        Args:
            messages (list[dict[str, str]]): chat messages with 'role' and 'content' keys.

        Returns:
            str: decoded model output text.
        """
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=False,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_batch(self, messages_list: List[List[Dict[str, str]]]) -> Tuple[List[str], List[float]]:
        """Generate responses for a batch of prompts.
        Uses left-padding so shorter prompts align correctly in the batch.

        Args:
            messages_list (list[list[dict[str, str]]]): list of chat message lists.

        Returns:
            tuple[list[str], list[float]]: (decoded texts, confidence scores).
        """
        prompt_texts = [
            self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            for msgs in messages_list
        ]

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompt_texts, return_tensors="pt", padding=True, truncation=True
        )
        self.tokenizer.padding_side = original_padding_side

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        prompt_len = inputs["input_ids"].shape[1]

        texts = []
        for i in range(len(messages_list)):
            new_tokens = outputs.sequences[i][prompt_len:]
            texts.append(
                self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            )

        confidence_scorer = ConfidenceScorer()
        confidences = confidence_scorer.compute_batch_confidences(
            outputs.scores, outputs.sequences, prompt_len,
            self.tokenizer.eos_token_id,
        )

        return texts, confidences


# ═══════════════════════════════════════════════════════════════════════
#  ExtractionAgent
# ═══════════════════════════════════════════════════════════════════════

class ExtractionAgent:
    """First stage: runs the LLM call and gathers evidence signals."""

    def __init__(self, model_manager: 'ModelManager', prompt_builder: 'PromptBuilder',
                 answer_processor: 'AnswerProcessor', confidence_scorer: 'ConfidenceScorer',
                 span_matcher: 'SpanMatcher') -> None:
        """
        Args:
            model_manager (ModelManager): handles LLM inference.
            prompt_builder (PromptBuilder): constructs chat-format prompts.
            answer_processor (AnswerProcessor): cleans and validates raw answers.
            confidence_scorer (ConfidenceScorer): evaluates generation confidence.
            span_matcher (SpanMatcher): finds matching spans in context.
        """
        self.model_manager = model_manager
        self.prompt_builder = prompt_builder
        self.answer_processor = answer_processor
        self.confidence_scorer = confidence_scorer
        self.span_matcher = span_matcher

    def _collect_observation(self, raw_answer: str, context: str,
                                confidence: float) -> Dict[str, Any]:
        """Gather all evidence signals for the AnalysisAgent to reason over:
        cleaned answer, confidence, grounding check, and span matching.

        Args:
            raw_answer (str): raw text generated by the LLM.
            context (str): the original context passage.
            confidence (float): avg log-probability from the generation step.

        Returns:
            dict[str, any]: observation with keys: raw_answer, confidence, is_confident,
                no_answer_detected, grounded_in_context, best_span, snapped_span,
                similarity_ratio, cleaned_answer.
        """
        cleaned_answer = self.answer_processor.process(raw_answer, context)
        no_answer_detected = self.answer_processor._is_no_answer_response(raw_answer)

        # Strip LLM artifacts to get the raw candidate answer text
        first_line = raw_answer.strip().split("\n")[0].strip()
        prefix_removed = self.answer_processor._remove_common_prefixes(first_line)
        prefix_removed = self.answer_processor._strip_quotes(prefix_removed)
        prefix_removed = prefix_removed.rstrip(".").strip()
        grounded = self.answer_processor._is_grounded_in_context(
            prefix_removed, context
        ) if prefix_removed else False

        # Confidence threshold gate
        confident = self.confidence_scorer.is_confident(confidence)

        # Span matching — find closest context span via SequenceMatcher
        best_span, similarity_ratio = None, 0.0
        snapped_span = None
        if prefix_removed:
            best_span, similarity_ratio = self.span_matcher.find_best_span(
                prefix_removed, context
            )
            snapped_span = self.span_matcher.snap_or_reject(prefix_removed, context)

        return {
            "raw_answer": raw_answer,
            "confidence": confidence,
            "is_confident": confident,
            "no_answer_detected": no_answer_detected,
            "grounded_in_context": grounded,
            "best_span": best_span,
            "snapped_span": snapped_span,
            "similarity_ratio": similarity_ratio,
            "cleaned_answer": cleaned_answer,
        }

    def extract_single(self, context: str, question: str) -> Dict[str, Any]:
        """Run the first LLM call and return an observation dict with all evidence signals.

        Args:
            context (str): the context passage.
            question (str): the question to answer.

        Returns:
            dict[str, any]: observation dict (see _collect_observation).
        """
        messages = self.prompt_builder.build_messages(context, question)
        raw_answers, confidences = self.model_manager.generate_batch([messages])
        return self._collect_observation(raw_answers[0], context, confidences[0])


# ═══════════════════════════════════════════════════════════════════════
#  AnalysisAgent
# ═══════════════════════════════════════════════════════════════════════

class AnalysisAgent:
    """Second stage: decides the final answer from extraction evidence."""

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

    def __init__(self, model_manager: 'ModelManager') -> None:
        """
        Args:
            model_manager (ModelManager): handles LLM inference.
        """
        self.model_manager = model_manager

    def _build_analysis_prompt(self, context: str, question: str,
                                observation: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format the extraction evidence into a structured prompt for the second LLM call.

        Args:
            context (str): the original context passage.
            question (str): the question being answered.
            observation (dict[str, any]): evidence dict from ExtractionAgent.

        Returns:
            list[dict[str, str]]: chat messages for the analysis LLM call.
        """
        no_answer_str = "Yes" if observation["no_answer_detected"] else "No"
        grounded_str = "Yes" if observation["grounded_in_context"] else "No"
        confident_str = "Yes" if observation["is_confident"] else "No (below threshold — likely hallucination)"
        span_str = observation["best_span"] if observation["best_span"] else "(none)"
        snapped_str = observation["snapped_span"] if observation["snapped_span"] else "(rejected — no close match)"
        ratio_str = f"{observation['similarity_ratio']:.2f}"
        confidence_str = f"{observation['confidence']:.2f}"

        user_content = (
            f"Question: {question}\n"
            f"Context: {context}\n\n"
            f"--- Extraction Evidence ---\n"
            f"Extracted Answer: {observation['raw_answer']}\n"
            f"Confidence Score: {confidence_str} (range: -inf to 0, higher is better)\n"
            f"Passes Confidence Threshold: {confident_str}\n"
            f"Grounded in Context: {grounded_str}\n"
            f"Best Matching Span: {span_str} (similarity: {ratio_str})\n"
            f"Snapped Span (after snap-or-reject): {snapped_str}\n"
            f"No-Answer Signals Detected: {no_answer_str}\n\n"
            f"Decide: Should the final answer be the extracted answer, "
            f"the matched span, or NO ANSWER?"
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def analyze_single(self, context: str, question: str,
                        observation: Dict[str, Any]) -> str:
        """Run the second LLM call to make the final accept/reject decision.

        Args:
            context (str): the original context passage.
            question (str): the question being answered.
            observation (dict[str, any]): evidence dict from ExtractionAgent.

        Returns:
            str: final answer string, or NO_ANSWER_MARKER.
        """
        messages = self._build_analysis_prompt(context, question, observation)
        raw_outputs, _ = self.model_manager.generate_batch([messages])

        # Normalize various "no answer" phrasings to the official marker
        answer = raw_outputs[0].strip().split("\n")[0].strip()
        if answer.upper() in ("NO ANSWER", "NO_ANSWER", "NOANSWER"):
            answer = NO_ANSWER_MARKER
        return answer


# ═══════════════════════════════════════════════════════════════════════
#  SquadQARunner
# ═══════════════════════════════════════════════════════════════════════

class SquadQARunner:
    """Orchestrates the two-agent QA pipeline.

    For each question, runs the full pipeline sequentially:
        1. ExtractionAgent — first LLM call + evidence gathering
        2. AnalysisAgent — second LLM call for final decision
    """

    def __init__(self, model_manager: 'ModelManager', extraction_agent: 'ExtractionAgent',
                 analysis_agent: 'AnalysisAgent') -> None:
        """
        Args:
            model_manager (ModelManager): handles LLM inference.
            extraction_agent (ExtractionAgent): first-stage agent.
            analysis_agent (AnalysisAgent): second-stage agent.
        """
        self.model_manager = model_manager
        self.extraction_agent = extraction_agent
        self.analysis_agent = analysis_agent

    def answer_single(self, context: str, question: str) -> str:
        """Run the full two-agent pipeline for a single question.

        Args:
            context (str): the context passage.
            question (str): the question to answer.

        Returns:
            str: final answer or NO_ANSWER_MARKER.
        """
        observation = self.extraction_agent.extract_single(context, question)
        return self.analysis_agent.analyze_single(context, question, observation)

    def run(self, data_filename: str) -> str:
        """Run the pipeline on all rows in a CSV file.

        Args:
            data_filename (str): path to CSV with 'context' and 'question' columns.

        Returns:
            str: path to the output CSV with a 'final answer' column added.
        """
        df = pd.read_csv(data_filename)
        total = len(df)
        final_answers = []

        for idx, row in df.iterrows():
            context = row["context"]
            question = row["question"]

            # 1. Extraction Agent: first LLM call + evidence gathering
            observation = self.extraction_agent.extract_single(context, question)

            # 2. Analysis Agent: second LLM call for final decision
            answer = self.analysis_agent.analyze_single(context, question, observation)

            final_answers.append(answer)
            print(f"  [{len(final_answers)}/{total}] {answer[:60]}")

        df["final answer"] = final_answers

        out_filename = data_filename.replace('.csv', '-results.csv')
        df.to_csv(out_filename, index=False)
        print(f'final answers recorded into {out_filename}')
        return out_filename


# ── Initialise components ────────────────────────────────────────────
# Wire up the two-agent pipeline: ExtractionAgent → AnalysisAgent
model_manager = ModelManager()
prompt_builder = PromptBuilder()
answer_processor = AnswerProcessor()
confidence_scorer = ConfidenceScorer(no_answer_threshold=-1.0)
span_matcher = SpanMatcher(min_similarity=0.6)

extraction_agent = ExtractionAgent(
    model_manager, prompt_builder, answer_processor, confidence_scorer, span_matcher,
)
analysis_agent = AnalysisAgent(model_manager)

runner = SquadQARunner(model_manager, extraction_agent, analysis_agent)

# Keep legacy references so nothing else breaks
model_name = 'meta-llama/Llama-3.2-3B-Instruct'
tokenizer = model_manager.tokenizer
model = model_manager.model


def generate_answer(messages: List[Dict[str, str]]) -> str:
    """
    Args:
        messages (list[dict[str, str]]): chat messages with 'role' and 'content' keys.

    Returns:
        str: decoded model output text.
    """
    return model_manager.generate_single(messages)


def answer_single_question(context: str, question: str) -> str:
    """
    Args:
        context (str): the context passage.
        question (str): the question to answer.

    Returns:
        str: final answer or NO_ANSWER_MARKER.
    """
    return runner.answer_single(context, question)


def squad_qa(data_filename: str) -> str:
    """
    Args:
        data_filename (str): path to CSV with 'context' and 'question' columns.

    Returns:
        str: path to the output results CSV.
    """
    return runner.run(data_filename)


if __name__ == '__main__':
    start_time = time.time()

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    data = pd.read_csv(config['data'])
    sample = data.sample(n=config['sample_for_solution'])  # for grading will be replaced with 'sample_for_grading'
    sample_filename = config['data'].replace('.csv', '-sample.csv')
    sample.to_csv(sample_filename, index=False)

    out_filename = squad_qa(sample_filename)  # todo: the function you implement

    eval_out = evaluate_results(out_filename, final_answer_column='final answer')
    eval_out_list = [str((k, round(v, 3))) for (k, v) in eval_out.items()]
    print('\n'.join(eval_out_list))

    elapsed_time = time.time() - start_time
    print(f"time: {elapsed_time: .2f} sec")
