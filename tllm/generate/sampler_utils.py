from typing import List, Optional

from tllm import BACKEND, BackendEnum
from tllm.schemas import MIX_TENSOR, SamplingParams

if BACKEND == BackendEnum.TORCH:
    import torch
    import torch.nn.functional as F


def top_k_sampling(logits: MIX_TENSOR, k: int = 10) -> MIX_TENSOR:
    # logits: [seq_len, vocab_size]
    filtered_logits, top_indices = torch.topk(logits, k, dim=-1)
    probs = torch.zeros_like(logits).scatter_(-1, top_indices, F.softmax(filtered_logits, dim=-1))
    return probs


def top_p_sampling(logits: MIX_TENSOR, p: float = 0.9) -> MIX_TENSOR:
    # logits: [seq_len, vocab_size]
    # TODO: fix
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

    top_p_mask = cumulative_probs < p
    filtered_logits = torch.where(top_p_mask, sorted_logits, torch.ones_like(sorted_logits) * (-float("inf")))
    filtered_logits = torch.nan_to_num(filtered_logits)
    filtered_probs = F.softmax(filtered_logits, dim=-1)

    next_token = torch.multinomial(filtered_probs, 1).squeeze(-1)
    return sorted_indices.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)


def temperature_scaling(logits: MIX_TENSOR, temperature: float = 1.0) -> MIX_TENSOR:
    return logits / temperature


class SamplerUtils:
    def __init__(self, method: str) -> None:
        self.method = method
        assert self.method in ["greedy", "beam_search", "sampling"]

    def sampling(self, logits: MIX_TENSOR, sampling_params: Optional[SamplingParams] = None) -> List[int]:
        if self.method == "greedy":
            raise ValueError("Greedy decoding is not supported in sampling method")
        elif self.method == "beam_search":
            return self.beam_search_decode(logits, sampling_params)
        elif self.method == "sampling":
            return self.sampling_decode(logits, sampling_params)

    def sampling_decode(self, logits: MIX_TENSOR, sampling_params: SamplingParams) -> List[int]:
        generate_logits = logits[:, -1]
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p
        temperature = sampling_params.temperature
        temperature_scaled_logits = temperature_scaling(generate_logits, temperature)
        if top_k > 0:
            temperature_scaled_logits = top_k_sampling(temperature_scaled_logits, k=top_k)
        else:
            temperature_scaled_logits = F.softmax(temperature_scaled_logits, dim=-1)

        if top_p < 1.0:
            next_token = top_p_sampling(temperature_scaled_logits, p=top_p)

        else:
            next_token = torch.multinomial(temperature_scaled_logits, 1).squeeze(-1)
        return next_token.tolist()

    def beam_search_decode(self, logits: MIX_TENSOR, sampling_params: SamplingParams) -> List[List[int]]:
        max_len = sampling_params.max_tokens
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p
        temperature = sampling_params.temperature
        beam_width = sampling_params.n
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
