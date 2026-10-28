"""Qwen3-Embedding-8B embedding extraction — the exact recipe used at
MEDIATOR training and evaluation time (per the model card): instruct-prefixed
query, left padding, last-token pooling, L2 normalization.

MEDIATOR consumes these 4096-d embeddings as input; texts are lowercased and
stripped before embedding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

TASK_INSTRUCTION = (
    "Instruct: Given a clinical concept string, produce an embedding that preserves exact biomedical concept identity across synonyms, ontology versions, abbreviations, and local hospital wording. Strings that refer to the same medical concept should be close. Strings that are clinically related but not equivalent should remain distinguishable.\nQuery: "
)


def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def clean_text(s: str) -> str:
    return str(s).strip().lower()


class QwenEmbedder:
    """Wraps Qwen/Qwen3-Embedding-8B with the pooling/prompt recipe used to train MEDIATOR."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B",
                 device: str = "cuda", dtype: torch.dtype = torch.float16,
                 max_len: int = 96):
        self.device = device
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 256,
              use_instruct: bool = True) -> torch.Tensor:
        """Return a float32 CPU tensor of shape [len(texts), 4096]."""
        texts = [clean_text(t) for t in texts]
        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            if use_instruct:
                batch = [TASK_INSTRUCTION + t for t in batch]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**enc)
            emb = last_token_pool(outputs.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().float())
        return torch.cat(all_embeddings, dim=0)
