from typing import List

import numpy as np


def top_k_sampling(logits: np.ndarray, k: int = 10):
    logits = np.squeeze(logits, axis=0)  # Assuming logits is [1, vocab_size]
    top_inds = np.argsort(logits)[::-1][:k]
    filtered_logits = logits[top_inds]
    top_probs = np.exp(filtered_logits - np.max(filtered_logits))
    top_probs /= np.sum(top_probs)
    chosen_ind = np.random.choice(top_inds, p=top_probs)
    return np.array([chosen_ind])


def top_p_sampling(logits: np.ndarray, p: float = 0.9):
    logits = np.squeeze(logits, axis=0)  # Assuming logits is [1, vocab_size]

    # Sort logits in descending order
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]

    # Calculate cumulative probabilities
    sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
    cumulative_probs = np.cumsum(sorted_probs) / np.sum(sorted_probs)

    # Find indices to remove based on cumulative probability threshold p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]  # Shift to the right
    sorted_indices_to_remove[0] = False  # Always keep the highest probability index

    indices_to_remove = sorted_indices[sorted_indices_to_remove]

    # Set logits of indices to remove to -inf
    logits[indices_to_remove] = float("-inf")

    # Perform multinomial sampling from remaining logits
    remaining_probs = np.exp(logits - np.max(logits))
    remaining_probs /= np.sum(remaining_probs)
    sampled_ind = np.random.choice(np.arange(len(logits)), p=remaining_probs)

    # Return sampled index as numpy array
    return np.array([sampled_ind])


def temperature_scaling(logits: np.ndarray, temperature: float = 1.0):
    scaled_logits = logits / temperature
    return scaled_logits


class DecodeUtils:
    def __init__(self, method: str) -> None:
        self.method = method
        assert self.method in ["greedy", "beam_search"]

    def decode(self, input_ids: List[int]) -> str:
        if self.method == "greedy":
            return self.greedy_decode(input_ids)
        elif self.method == "beam_search":
            return self.beam_search_decode(input_ids)

    def greedy_decode(self, logits: np.ndarray) -> List[int]:
        # logits shape: [batch_size, sequence_length, vocab_size]
        return np.argmax(logits[:, -1], axis=-1).tolist()

    def beam_search_decode(
        self,
        logits: np.ndarray,
        beam_width: int = 3,
        max_len: int = 20,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> List[List[int]]:
        batch_size, sequence_length, vocab_size = logits.size()

        # Initialize the beam search
        beams = [[[np.ndarray([], dtype=np.int64), 0.0]] for _ in range(batch_size)]

        for _ in range(max_len):
            all_candidates = []
            for beam in beams:
                seq, score = beam[-1]
                if len(seq) > 0:
                    input_ids = seq.unsqueeze(0)  # [1, seq_length]
                    input_logits = logits.new_zeros((1, sequence_length + 1, vocab_size))
                    input_logits[:, :sequence_length, :] = logits
                    input_logits[:, sequence_length, :] = logits.gather(2, input_ids).squeeze(0)
                else:
                    input_logits = logits

                # Apply temperature scaling
                scaled_logits = input_logits / temperature

                # Apply top-k sampling
                if top_k > 0:
                    scaled_logits = top_k_sampling(scaled_logits, k=top_k)

                # Apply top-p sampling (nucleus sampling)
                if top_p < 1.0:
                    scaled_logits = top_p_sampling(scaled_logits, p=top_p)

                next_logits = np.log_softmax(scaled_logits[:, -1, :], axis=-1)

                top_scores, top_inds = np.topk(next_logits, beam_width)
                for score, ind in zip(top_scores[0], top_inds[0]):
                    score = score.item() + score
                    all_candidates.append((seq, score, ind))
                ordered = sorted(all_candidates, key=lambda a: a[1], reverse=True)

                # Keep only the top beam_width sequences
                beams = ordered[:beam_width]

        # Extract sequences from beams
        sequences = [beam[0] for beam in beams]

        return sequences
