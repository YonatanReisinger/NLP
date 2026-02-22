import json
import re
import string
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

    def __init__(self, no_answer_threshold=-1.0):
        self.no_answer_threshold = no_answer_threshold

    def compute_batch_confidences(self, scores, sequences, prompt_len, eos_token_id):
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

    def is_confident(self, confidence):
        return confidence >= self.no_answer_threshold


# ═══════════════════════════════════════════════════════════════════════
#  SpanMatcher
# ═══════════════════════════════════════════════════════════════════════

class SpanMatcher:
    """Finds the best matching span in the context for a candidate answer."""

    def __init__(self, min_similarity=0.6):
        self.min_similarity = min_similarity

    def find_best_span(self, answer, context):
        answer_lower = answer.lower().strip()
        context_lower = context.lower()
        words = context.split()

        if not answer_lower or not words:
            return None, 0.0

        answer_word_count = len(answer_lower.split())

        best_span = None
        best_ratio = 0.0

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
        span, ratio = self.find_best_span(answer, context)
        return span


# ═══════════════════════════════════════════════════════════════════════
#  AnswerProcessor
# ═══════════════════════════════════════════════════════════════════════

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
        lowered = text.lower().strip()
        return any(re.search(p, lowered) for p in self.NO_ANSWER_PATTERNS)

    def _remove_common_prefixes(self, text):
        result = text
        for pattern in self.PREFIX_PATTERNS:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        return result.strip()

    @staticmethod
    def _strip_quotes(text):
        if len(text) >= 2:
            if (text[0] == '"' and text[-1] == '"') or \
               (text[0] == "'" and text[-1] == "'"):
                return text[1:-1].strip()
        return text

    @staticmethod
    def _normalize(text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join(text.split())

    def _is_grounded_in_context(self, answer, context):
        norm_answer = self._normalize(answer)
        norm_context = self._normalize(context)

        if norm_answer in norm_context:
            return True

        content_words = [w for w in norm_answer.split() if w not in self.STOPWORDS]
        if not content_words:
            return True

        matched = sum(1 for w in content_words if w in norm_context)
        return matched / len(content_words) >= 0.5

    def process(self, raw_model_output, context):
        text = raw_model_output.strip()
        first_line = text.split("\n")[0].strip()

        if self._is_no_answer_response(first_line):
            return NO_ANSWER_MARKER

        answer = self._remove_common_prefixes(first_line)
        answer = self._strip_quotes(answer)
        answer = answer.rstrip(".")
        answer = answer.strip()

        if not answer:
            return NO_ANSWER_MARKER

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

    def build_messages(self, context, question):
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

    def __init__(self, model_name='meta-llama/Llama-3.2-3B-Instruct'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, token=True
        )
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_single(self, messages):
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

    def generate_batch(self, messages_list):
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

    def __init__(self, model_manager, prompt_builder, answer_processor,
                 confidence_scorer, span_matcher):
        self.model_manager = model_manager
        self.prompt_builder = prompt_builder
        self.answer_processor = answer_processor
        self.confidence_scorer = confidence_scorer
        self.span_matcher = span_matcher

    def _collect_observation(self, raw_answer, context, confidence):
        cleaned_answer = self.answer_processor.process(raw_answer, context)
        no_answer_detected = self.answer_processor._is_no_answer_response(raw_answer)

        first_line = raw_answer.strip().split("\n")[0].strip()
        prefix_removed = self.answer_processor._remove_common_prefixes(first_line)
        prefix_removed = self.answer_processor._strip_quotes(prefix_removed)
        prefix_removed = prefix_removed.rstrip(".").strip()
        grounded = self.answer_processor._is_grounded_in_context(
            prefix_removed, context
        ) if prefix_removed else False

        # Confidence threshold gate
        confident = self.confidence_scorer.is_confident(confidence)

        # Span matching — snap to closest context span or reject
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

    def extract_single(self, context, question):
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

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def _build_analysis_prompt(self, context, question, observation):
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

    def analyze_single(self, context, question, observation):
        messages = self._build_analysis_prompt(context, question, observation)
        raw_outputs, _ = self.model_manager.generate_batch([messages])

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

    def __init__(self, model_manager, extraction_agent, analysis_agent):
        self.model_manager = model_manager
        self.extraction_agent = extraction_agent
        self.analysis_agent = analysis_agent

    def answer_single(self, context, question):
        observation = self.extraction_agent.extract_single(context, question)
        return self.analysis_agent.analyze_single(context, question, observation)

    def run(self, data_filename):
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


def generate_answer(messages):
    return model_manager.generate_single(messages)


def answer_single_question(context, question):
    return runner.answer_single(context, question)


def squad_qa(data_filename):
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
