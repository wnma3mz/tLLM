from typing import List

import torch
import torch.nn.functional as F


def top_k_sampling(logits: torch.Tensor, k: int = 10):
    logits = logits.squeeze(0)  # Assuming logits is [1, vocab_size]
    filtered_logits, top_inds = logits.topk(k)
    top_probs = F.softmax(filtered_logits, dim=-1)
    chosen_ind = torch.multinomial(top_probs, 1)
    return top_inds[chosen_ind].unsqueeze(0)


def top_p_sampling(logits: torch.Tensor, p: float = 0.9):
    logits = logits.squeeze(0)  # Assuming logits is [1, vocab_size]
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float("-inf")
    sampled_ind = torch.multinomial(F.softmax(logits, dim=-1), 1)
    return sampled_ind.unsqueeze(0)


def temperature_scaling(logits: torch.Tensor, temperature: float = 1.0):
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

    def greedy_decode(self, logits: torch.Tensor) -> List[int]:
        # logits shape: [batch_size, sequence_length, vocab_size]
        return torch.argmax(logits[:, -1], dim=-1).tolist()

    def beam_search_decode(
        self,
        logits: torch.Tensor,
        beam_width: int = 3,
        max_len: int = 20,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> List[List[int]]:
        batch_size, sequence_length, vocab_size = logits.size()

        # Initialize the beam search
        beams = [[[torch.tensor([], dtype=torch.long), 0.0]] for _ in range(batch_size)]

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

                next_logits = torch.log_softmax(scaled_logits[:, -1, :], dim=-1)

                top_scores, top_inds = torch.topk(next_logits, beam_width)
                for score, ind in zip(top_scores[0], top_inds[0]):
                    score = score.item() + score
                    all_candidates.append((seq, score, ind))
                ordered = sorted(all_candidates, key=lambda a: a[1], reverse=True)

                # Keep only the top beam_width sequences
                beams = ordered[:beam_width]

        # Extract sequences from beams
        sequences = [beam[0] for beam in beams]

        return sequences
