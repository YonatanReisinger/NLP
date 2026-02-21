"""
Confidence scoring based on token-level log-probabilities.

When the model is uncertain about a generated answer (low average
log-probability), the answer is likely a hallucination.  This is
especially useful for detecting unanswerable questions — the model
tends to produce lower-confidence tokens when it is forced to guess
rather than extract from the context.
"""

import torch
import torch.nn.functional as F


class ConfidenceScorer:
    """Scores generation confidence from per-step logits."""

    def __init__(self, no_answer_threshold=-1.0):
        self.no_answer_threshold = no_answer_threshold

    def compute_batch_confidences(self, scores, sequences, prompt_len, eos_token_id):
        """Compute mean token log-probability for each sequence in a batch.

        Args:
            scores: tuple of (batch_size, vocab_size) tensors, one per step.
            sequences: (batch_size, seq_len) generated token ids.
            prompt_len: number of prompt tokens (same for every item
                        because the batch is left-padded).
            eos_token_id: id used for padding / end-of-sequence.

        Returns:
            list[float]: average log-probability per sequence.
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

    def is_confident(self, confidence):
        """Return True if the confidence score is above the threshold."""
        return confidence >= self.no_answer_threshold
